"""Facade around inspected upstream RVC command-line entry points."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..monitoring.gpu_monitor import GPUMetricsMonitor
from ..monitoring.process_monitor import ProcessMonitor, ProcessResult
from ..state import StageResult
from .feature_extraction import (
    FeatureRequest,
    build_extract_f0_command,
    build_extract_features_command,
)
from .index_builder import IndexRequest, build_index_command, validate_index
from .inference import (
    InferenceRequest,
    InferenceResult,
    build_inference_command,
)
from .preprocess import PreprocessRequest, build_preprocess_command
from .repository_inspector import RepositoryInspector, RVCRepositoryInfo
from .training import (
    TrainingRequest,
    batch_size_attempts,
    build_training_command,
    detect_cuda_oom,
    discover_valid_checkpoints,
    validate_artifact,
    write_checkpoint_manifest,
)


class RVCAdapterError(RuntimeError):
    """Base class for actionable RVC adaptation errors."""


class RVCCommandError(RVCAdapterError):
    """Raised when an upstream RVC process fails or emits an invalid artifact."""

    def __init__(
        self,
        message: str,
        *,
        result: ProcessResult | None = None,
        command: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.result = result
        self.command = tuple(command)


class RVCAdapter:
    """Version-aware, subprocess-based adapter for an external RVC checkout.

    The adapter never imports the upstream repository into this process and never
    uses a shell. Callers may inject a :class:`ProcessMonitor` for deterministic
    tests and a cancellation event for Ctrl+C propagation.
    """

    def __init__(
        self,
        repository: Path,
        python_executable: Path,
        *,
        logs_dir: Path | None = None,
        process_monitor: ProcessMonitor | None = None,
        logger: logging.Logger | None = None,
        cancel_event: threading.Event | None = None,
        output_callback: Callable[[str, str], None] | None = None,
        monitoring_interval_seconds: float = 30.0,
        monitor_gpu: bool = True,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.logs_dir = (
            Path(logs_dir).expanduser().resolve()
            if logs_dir is not None
            else (self.repository.parent / "adapter_logs").resolve()
        )
        self.logger = logger or logging.getLogger(__name__)
        self.process_monitor = process_monitor or ProcessMonitor(logger=self.logger)
        self.cancel_event = cancel_event or threading.Event()
        self.output_callback = output_callback
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.monitor_gpu = monitor_gpu
        self._repository_info: RVCRepositoryInfo | None = None
        self._log_sequence = 0

    def inspect_repository(self, *, refresh: bool = False) -> RVCRepositoryInfo:
        """Inspect and cache facts about the actual configured RVC checkout."""

        if refresh or self._repository_info is None:
            self._repository_info = RepositoryInspector(
                self.repository, self.python_executable
            ).inspect()
        return self._repository_info

    def preprocess(self, request: PreprocessRequest) -> StageResult:
        """Run RVC-native preprocessing or return its dry-run command."""

        command = build_preprocess_command(self.inspect_repository(), request)
        result = self._execute(
            "preprocess",
            command,
            timeout_seconds=request.timeout_seconds,
            dry_run=request.dry_run,
            env=request.env,
        )
        return self._to_stage_result(
            result,
            command,
            outputs=(Path(request.experiment_dir).resolve(),),
            dry_run=request.dry_run,
        )

    def extract_f0(self, request: FeatureRequest) -> StageResult:
        """Run an inspected F0 extractor."""

        command = build_extract_f0_command(self.inspect_repository(), request)
        result = self._execute(
            "extract_f0",
            command,
            timeout_seconds=request.timeout_seconds,
            dry_run=request.dry_run,
            env=request.env,
        )
        return self._to_stage_result(
            result,
            command,
            outputs=(Path(request.experiment_dir).resolve(),),
            dry_run=request.dry_run,
        )

    def extract_features(self, request: FeatureRequest) -> StageResult:
        """Run an inspected content-feature extractor."""

        command = build_extract_features_command(self.inspect_repository(), request)
        result = self._execute(
            "extract_features",
            command,
            timeout_seconds=request.timeout_seconds,
            dry_run=request.dry_run,
            env=request.env,
        )
        return self._to_stage_result(
            result,
            command,
            outputs=(Path(request.experiment_dir).resolve(),),
            dry_run=request.dry_run,
        )

    def train(self, request: TrainingRequest) -> StageResult:
        """Train with bounded CUDA-OOM batch fallback and checkpoint evidence."""

        attempts = batch_size_attempts(request)
        attempt_records: list[dict[str, Any]] = []
        final_result: ProcessResult | None = None
        final_command: tuple[str, ...] = ()
        selected_batch_size = attempts[0]
        stage_started_at = time.time()
        if request.dry_run:
            final_command = build_training_command(
                self.inspect_repository(), request, batch_size=selected_batch_size
            )
            return self._to_stage_result(
                None,
                final_command,
                outputs=tuple(
                    path.resolve()
                    for path in (request.expected_model, request.checkpoint_dir)
                    if path is not None
                ),
                dry_run=True,
                extra_metadata={
                    "batch_size_attempts": attempts,
                    "selected_batch_size": selected_batch_size,
                },
            )

        metrics_monitor: GPUMetricsMonitor | None = None
        if self.monitor_gpu:
            metrics_monitor = GPUMetricsMonitor(
                self.logs_dir / "gpu_metrics.csv",
                interval_seconds=self.monitoring_interval_seconds,
                gpu_index=request.gpu_ids[0],
                disk_path=self.logs_dir,
                logger=self.logger,
            )
            metrics_monitor.start()
        try:
            for attempt_number, selected_batch_size in enumerate(attempts, start=1):
                final_command = build_training_command(
                    self.inspect_repository(), request, batch_size=selected_batch_size
                )
                final_result = self._execute(
                    "train",
                    final_command,
                    timeout_seconds=request.timeout_seconds,
                    dry_run=False,
                    env=request.env,
                    raise_on_failure=False,
                    suffix=f"attempt_{attempt_number}_bs_{selected_batch_size}",
                )
                assert final_result is not None
                checkpoint_evidence = discover_valid_checkpoints(
                    request.checkpoint_dir,
                    minimum_bytes=request.minimum_checkpoint_bytes,
                ) if request.checkpoint_dir is not None else ()
                oom = detect_cuda_oom(
                    final_result,
                    (final_result.stderr_log, final_result.stdout_log),
                )
                attempt_records.append(
                    {
                        "attempt": attempt_number,
                        "batch_size": selected_batch_size,
                        "return_code": final_result.return_code,
                        "cuda_oom": oom,
                        "valid_checkpoint_count": len(checkpoint_evidence),
                        "stdout_log": str(final_result.stdout_log),
                        "stderr_log": str(final_result.stderr_log),
                    }
                )
                if final_result.success:
                    break
                has_next = attempt_number < len(attempts)
                if not (oom and has_next):
                    raise self._command_error("train", final_command, final_result)
                self.logger.warning(
                    "CUDA OOM at batch size %s; retrying with batch size %s. "
                    "Validated checkpoints available: %s",
                    selected_batch_size,
                    attempts[attempt_number],
                    len(checkpoint_evidence),
                )
        finally:
            if metrics_monitor is not None:
                metrics_monitor.stop()

        if final_result is None or not final_result.success:
            raise RVCCommandError("RVC training ended without a successful process")

        outputs: list[Path] = []
        metadata: dict[str, Any] = {
            "command": list(final_command),
            "return_code": final_result.return_code,
            "selected_batch_size": selected_batch_size,
            "oom_retry_count": sum(1 for attempt in attempt_records if attempt["cuda_oom"]),
            "attempts": attempt_records,
            "stdout_log": str(final_result.stdout_log),
            "stderr_log": str(final_result.stderr_log),
        }
        checkpoints = ()
        if request.checkpoint_dir is not None:
            checkpoints = discover_valid_checkpoints(
                request.checkpoint_dir,
                minimum_bytes=request.minimum_checkpoint_bytes,
            )
            manifest = write_checkpoint_manifest(
                Path(request.checkpoint_dir) / "checkpoint_manifest.json",
                checkpoints,
                batch_size=selected_batch_size,
                training_parameters={
                    "model_name": request.model_name,
                    "epochs": request.epochs,
                    "sample_rate": request.sample_rate,
                    "version": request.version,
                },
            )
            outputs.extend(checkpoint.path for checkpoint in checkpoints)
            outputs.append(manifest)
            metadata["valid_checkpoint_count"] = len(checkpoints)
        if request.expected_model is not None:
            model = validate_artifact(
                request.expected_model,
                minimum_bytes=request.minimum_model_bytes,
                allowed_suffixes=(".pth",),
                not_before=stage_started_at,
            )
            metadata["model_validation"] = _json_safe(asdict(model))
            if not model.valid:
                raise RVCCommandError(
                    f"Training process exited successfully, but the expected model "
                    f"is invalid: {model.path} ({model.reason})",
                    result=final_result,
                    command=final_command,
                )
            outputs.append(model.path)
        return StageResult(
            success=True,
            outputs=tuple(outputs),
            metadata=metadata,
            message="RVC training completed with validated process status",
        )

    def build_index(self, request: IndexRequest) -> StageResult:
        """Run and validate a standalone upstream index builder."""

        command = build_index_command(self.inspect_repository(), request)
        started_at = time.time()
        result = self._execute(
            "build_index",
            command,
            timeout_seconds=request.timeout_seconds,
            dry_run=request.dry_run,
            env=request.env,
        )
        if request.dry_run:
            return self._to_stage_result(
                result,
                command,
                outputs=(Path(request.output_path).resolve(),),
                dry_run=True,
            )
        validation = validate_index(
            request.output_path,
            minimum_bytes=request.minimum_index_bytes,
            not_before=started_at,
        )
        if not validation.valid:
            raise RVCCommandError(
                "Index process exited successfully, but its artifact is invalid: "
                f"{validation.path} ({validation.reason})",
                result=result,
                command=command,
            )
        return self._to_stage_result(
            result,
            command,
            outputs=(validation.path,),
            dry_run=False,
            extra_metadata={"index_validation": _json_safe(asdict(validation))},
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run one conversion and verify that a fresh audio file was created."""

        command = build_inference_command(self.inspect_repository(), request)
        started_at = time.time()
        result = self._execute(
            "infer",
            command,
            timeout_seconds=request.timeout_seconds,
            dry_run=request.dry_run,
            env=request.env,
        )
        output = Path(request.output_path).resolve()
        if request.dry_run:
            return InferenceResult(
                success=True,
                output_path=output,
                command=command,
                metadata={"dry_run": True},
                message="Dry run: inference command was not executed",
            )
        assert result is not None
        validation = validate_artifact(
            output,
            minimum_bytes=request.minimum_output_bytes,
            allowed_suffixes=(".wav", ".flac"),
            not_before=started_at,
        )
        if not validation.valid:
            raise RVCCommandError(
                "Inference exited successfully, but its audio artifact is invalid: "
                f"{output} ({validation.reason})",
                result=result,
                command=command,
            )
        return InferenceResult(
            success=True,
            output_path=output,
            command=command,
            return_code=result.return_code,
            stdout_log=result.stdout_log,
            stderr_log=result.stderr_log,
            metadata={"artifact_validation": _json_safe(asdict(validation))},
            message="Inference completed and output file passed basic validation",
        )

    def _execute(
        self,
        stage: str,
        command: Sequence[str],
        *,
        timeout_seconds: float | None,
        dry_run: bool,
        env: Mapping[str, str] | None,
        raise_on_failure: bool = True,
        suffix: str | None = None,
    ) -> ProcessResult | None:
        if dry_run:
            self.logger.info("Dry-run RVC command: %s", _display_command(command))
            return None
        self._log_sequence += 1
        name = f"{self._log_sequence:03d}_{stage}"
        if suffix:
            name += f"_{suffix}"
        stdout_log = self.logs_dir / f"{name}_stdout.log"
        stderr_log = self.logs_dir / f"{name}_stderr.log"
        self._append_command_log(stage, command)
        result = self.process_monitor.run(
            command,
            cwd=self.repository,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            timeout_seconds=timeout_seconds,
            cancel_event=self.cancel_event,
            env=env,
            output_callback=self.output_callback,
        )
        if raise_on_failure and not result.success:
            raise self._command_error(stage, command, result)
        return result

    def _to_stage_result(
        self,
        result: ProcessResult | None,
        command: Sequence[str],
        *,
        outputs: Sequence[Path],
        dry_run: bool,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> StageResult:
        metadata: dict[str, Any] = {
            "command": list(command),
            "dry_run": dry_run,
        }
        if result is not None:
            metadata.update(
                {
                    "return_code": result.return_code,
                    "duration_seconds": result.duration_seconds,
                    "stdout_log": str(result.stdout_log),
                    "stderr_log": str(result.stderr_log),
                }
            )
        if extra_metadata:
            metadata.update(extra_metadata)
        return StageResult(
            success=True,
            outputs=tuple(Path(path).resolve() for path in outputs),
            metadata=metadata,
            message="Dry run; command not executed" if dry_run else "RVC command completed",
        )

    def _command_error(
        self, stage: str, command: Sequence[str], result: ProcessResult
    ) -> RVCCommandError:
        reason = "timed out" if result.timed_out else "was cancelled" if result.cancelled else "failed"
        stderr_tail = "\n".join(result.stderr_tail[-12:]).strip() or "<stderr empty>"
        return RVCCommandError(
            f"RVC stage {stage!r} {reason} with return code {result.return_code}. "
            f"Command: {_display_command(command)}. stderr tail:\n{stderr_tail}\n"
            f"Full logs: {result.stdout_log} and {result.stderr_log}",
            result=result,
            command=command,
        )

    def _append_command_log(self, stage: str, command: Sequence[str]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "command": _redact_command(command),
            "shell": False,
        }
        with (self.logs_dir / "commands.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


_SECRET_OPTION = re.compile(r"(?i)(token|password|secret|api[_-]?key)")


def _redact_command(command: Sequence[str]) -> list[str]:
    rendered: list[str] = []
    redact_next = False
    for argument in command:
        value = str(argument)
        if redact_next:
            rendered.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("-") and _SECRET_OPTION.search(value):
            rendered.append(value)
            redact_next = "=" not in value
            if "=" in value:
                rendered[-1] = value.split("=", 1)[0] + "=<redacted>"
            continue
        rendered.append(value)
    return rendered


def _display_command(command: Sequence[str]) -> str:
    return " ".join(_quote_for_display(value) for value in _redact_command(command))


def _quote_for_display(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
