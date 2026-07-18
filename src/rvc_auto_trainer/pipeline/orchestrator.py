"""Persisted, resumable orchestration with injectable stage handlers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ..state import PipelineStage, StageResult, StageStatus
from .stages import (
    EXECUTABLE_STAGES,
    StageDefinition,
    StageExecutionContext,
    StageHandler,
    StageRunRecord,
    resolved_config_fingerprint,
)


@dataclass(frozen=True)
class PipelineRunResult:
    """Result of one complete/resumed/dry-run orchestrator invocation."""

    success: bool
    records: tuple[StageRunRecord, ...]
    dry_run: bool = False

    @property
    def outputs(self) -> tuple[Path, ...]:
        return tuple(output for record in self.records for output in record.result.outputs)


class PipelineExecutionError(RuntimeError):
    """Raised after an actionable stage failure has been persisted."""

    def __init__(self, stage: PipelineStage, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class PipelineOrchestrator:
    """Execute ordered stages and persist state around each handler call.

    The ``adapter`` and every handler are injectable, so integration tests never
    need a real RVC checkout, FFmpeg, CUDA, or long-running training process.
    """

    def __init__(
        self,
        run_context: object,
        adapter: object,
        handlers: Mapping[PipelineStage | str, StageHandler | StageDefinition],
        *,
        dry_run: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.run_context = run_context
        self.adapter = adapter
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(__name__)
        self.definitions: dict[PipelineStage, StageDefinition] = {}
        for stage, handler in handlers.items():
            normalized = stage if isinstance(stage, PipelineStage) else PipelineStage(stage)
            definition = (
                handler
                if isinstance(handler, StageDefinition)
                else StageDefinition(
                    stage=normalized,
                    handler=handler,
                    fingerprint=resolved_config_fingerprint,
                )
            )
            if definition.stage is not normalized:
                raise ValueError(
                    f"Handler mapping key {normalized.value} does not match "
                    f"definition stage {definition.stage.value}"
                )
            self.definitions[normalized] = definition

    def register(self, definition: StageDefinition) -> None:
        """Register or replace a handler before execution starts."""

        self.definitions[definition.stage] = definition

    def run(
        self,
        stages: Iterable[PipelineStage | str] | None = None,
        *,
        resume: bool = True,
    ) -> PipelineRunResult:
        """Execute requested stages in canonical order."""

        requested = _normalize_stages(stages)
        records: list[StageRunRecord] = []
        if not self.dry_run:
            self._ensure_state()
        for stage in EXECUTABLE_STAGES:
            if stage not in requested:
                continue
            definition = self.definitions.get(stage)
            if definition is None:
                raise PipelineExecutionError(
                    stage,
                    f"No handler is registered for pipeline stage {stage.value}. "
                    "Register a concrete handler or use build_default_stage_handlers().",
                )
            execution_context = StageExecutionContext(
                run_context=self.run_context,  # type: ignore[arg-type]
                adapter=self.adapter,  # type: ignore[arg-type]
                stage=stage,
                dry_run=self.dry_run,
            )
            fingerprint = (
                definition.fingerprint(execution_context)
                if definition.fingerprint is not None
                else None
            )
            enabled = (
                definition.enabled(execution_context)
                if definition.enabled is not None
                else True
            )
            if not enabled:
                result = StageResult(
                    success=True,
                    metadata={"skipped": True, "skip_reason": definition.skip_reason},
                    message=definition.skip_reason,
                )
                if not self.dry_run:
                    self._persist_skip(stage, definition.skip_reason, fingerprint)
                records.append(StageRunRecord(stage, result, skipped=True))
                continue
            required_outputs = (
                tuple(definition.required_outputs(execution_context))
                if definition.required_outputs is not None
                else ()
            )
            if not self.dry_run and resume and self._can_reuse(
                stage, fingerprint, required_outputs
            ):
                persisted = self.run_context.state_store.load().record_for(stage)
                result = StageResult(
                    success=True,
                    outputs=persisted.outputs,
                    metadata={**persisted.metadata, "reused": True},
                    message="Reused completed stage with matching fingerprint",
                )
                records.append(StageRunRecord(stage, result, reused=True))
                continue
            if not self.dry_run:
                self._prepare_stage_for_run(stage, fingerprint)
            try:
                result = definition.handler(execution_context)
                if not isinstance(result, StageResult):
                    result = StageResult.model_validate(result)
                if not result.success:
                    raise PipelineExecutionError(
                        stage, result.message or f"Stage {stage.value} reported failure"
                    )
                if not self.dry_run:
                    self.run_context.state_store.complete_stage(
                        stage,
                        outputs=result.outputs,
                        metadata=result.metadata,
                        fingerprint=fingerprint,
                    )
            except KeyboardInterrupt:
                if hasattr(self.adapter, "cancel_event"):
                    self.adapter.cancel_event.set()
                if not self.dry_run:
                    self._persist_failure(
                        stage,
                        "Interrupted by user; child processes were asked to stop. "
                        f"Resume with: python -m rvc_auto_trainer resume --run-id "
                        f"{self.run_context.run_id}",
                    )
                raise
            except Exception as exc:
                message = (
                    f"Pipeline stage {stage.value} failed: {exc}. Completed outputs "
                    "from earlier stages were kept. Resume after fixing the cause with: "
                    f"python -m rvc_auto_trainer resume --run-id {self.run_context.run_id}"
                )
                if not self.dry_run:
                    self._persist_failure(stage, message)
                if isinstance(exc, PipelineExecutionError):
                    raise
                raise PipelineExecutionError(stage, message) from exc
            records.append(StageRunRecord(stage, result))
        return PipelineRunResult(success=True, records=tuple(records), dry_run=self.dry_run)

    def _ensure_state(self) -> None:
        store = self.run_context.state_store
        if not store.exists:
            store.initialize(self.run_context.run_id)

    def _can_reuse(
        self,
        stage: PipelineStage,
        fingerprint: str | None,
        required_outputs: tuple[Path, ...],
    ) -> bool:
        state = self.run_context.state_store.load()
        record = state.record_for(stage)
        outputs = required_outputs or record.outputs
        if fingerprint is not None:
            return self.run_context.state_store.can_reuse(stage, fingerprint, outputs)
        return record.status is StageStatus.COMPLETED and all(
            _output_exists(output, self.run_context.run_dir) for output in outputs
        )

    def _prepare_stage_for_run(
        self, stage: PipelineStage, fingerprint: str | None
    ) -> None:
        store = self.run_context.state_store
        state = store.load()
        status = state.record_for(stage).status
        if status in {StageStatus.COMPLETED, StageStatus.SKIPPED}:
            store.invalidate_from(stage, reason="Fingerprint or output changed")
        store.start_stage(stage, fingerprint=fingerprint)

    def _persist_skip(
        self, stage: PipelineStage, reason: str, fingerprint: str | None
    ) -> None:
        store = self.run_context.state_store
        state = store.load()
        status = state.record_for(stage).status
        if status in {StageStatus.COMPLETED, StageStatus.SKIPPED}:
            if status is StageStatus.SKIPPED:
                return
            store.invalidate_from(stage, reason=reason)
            state = store.load()
        state.skip_stage(stage, reason=reason, fingerprint=fingerprint)
        store.save(state)

    def _persist_failure(self, stage: PipelineStage, message: str) -> None:
        store = self.run_context.state_store
        state = store.load()
        if state.record_for(stage).status is StageStatus.RUNNING:
            store.fail_stage(stage, message)


def _normalize_stages(
    stages: Iterable[PipelineStage | str] | None,
) -> set[PipelineStage]:
    if stages is None:
        return set(EXECUTABLE_STAGES)
    normalized = {
        stage if isinstance(stage, PipelineStage) else PipelineStage(stage)
        for stage in stages
    }
    normalized.discard(PipelineStage.INITIALIZED)
    return normalized


def _output_exists(path: Path, run_dir: Path) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(run_dir) / candidate
    return candidate.exists()
