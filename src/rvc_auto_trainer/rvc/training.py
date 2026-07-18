"""RVC training command construction, OOM classification and artifact checks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..monitoring.process_monitor import ProcessResult
from .repository_inspector import RVCRepositoryInfo, RVCRepositoryInspectionError


@dataclass(frozen=True)
class TrainingRequest:
    """Parameters for one RVC training stage."""

    experiment_dir: Path
    model_name: str
    sample_rate: str = "40k"
    version: str = "v2"
    use_f0: bool = True
    gpu_ids: tuple[int, ...] = (0,)
    epochs: int = 200
    save_every_epochs: int = 25
    batch_size: int = 12
    automatic_batch_size: bool = True
    batch_size_candidates: tuple[int, ...] = (16, 12, 8, 6, 4)
    maximum_oom_retries: int = 4
    pretrained_generator: Path | None = None
    pretrained_discriminator: Path | None = None
    save_only_latest: bool = False
    cache_dataset_in_gpu: bool = False
    save_every_weights: bool = True
    resume_if_available: bool = True
    expected_model: Path | None = None
    checkpoint_dir: Path | None = None
    minimum_model_bytes: int = 1_024
    minimum_checkpoint_bytes: int = 1_024
    timeout_seconds: float | None = None
    dry_run: bool = False
    command: tuple[str, ...] | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ArtifactValidation:
    """Evidence that a generated model/checkpoint/index is a current file."""

    path: Path
    valid: bool
    size_bytes: int = 0
    sha256: str | None = None
    modified_at: str | None = None
    reason: str = ""


def build_training_command(
    repository: RVCRepositoryInfo,
    request: TrainingRequest,
    *,
    batch_size: int | None = None,
) -> tuple[str, ...]:
    """Build an upstream WebUI training command for the inspected layout."""

    selected_batch_size = request.batch_size if batch_size is None else batch_size
    _validate_training_request(request, selected_batch_size)
    if request.command:
        return _replace_batch_placeholder(request.command, selected_batch_size)
    script = repository.require_script("train")
    if script.contract not in {"webui-train", "legacy-train"}:
        raise RVCRepositoryInspectionError(
            f"Unsupported training contract {script.contract!r}; use command."
        )
    command = [
        str(repository.python_executable),
        str(script.path),
        "-e",
        request.model_name,
        "-sr",
        request.sample_rate,
        "-f0",
        "1" if request.use_f0 else "0",
        "-bs",
        str(selected_batch_size),
        "-g",
        "-".join(str(identifier) for identifier in request.gpu_ids),
        "-te",
        str(request.epochs),
        "-se",
        str(request.save_every_epochs),
        "-l",
        "1" if request.save_only_latest else "0",
        "-c",
        "1" if request.cache_dataset_in_gpu else "0",
        "-sw",
        "1" if request.save_every_weights else "0",
        "-v",
        request.version,
    ]
    if request.pretrained_generator is not None:
        command.extend(("-pg", str(Path(request.pretrained_generator).resolve())))
    if request.pretrained_discriminator is not None:
        command.extend(("-pd", str(Path(request.pretrained_discriminator).resolve())))
    command.extend(request.extra_args)
    return tuple(command)


def batch_size_attempts(request: TrainingRequest) -> tuple[int, ...]:
    """Return deterministic, decreasing batch-size attempts."""

    if not request.automatic_batch_size:
        return (request.batch_size,)
    positive = {value for value in request.batch_size_candidates if value > 0}
    positive.add(request.batch_size)
    ordered = sorted((value for value in positive if value <= request.batch_size), reverse=True)
    if not ordered:
        ordered = [request.batch_size]
    return tuple(ordered[: request.maximum_oom_retries + 1])


_OOM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"CUDA\s+out\s+of\s+memory",
        r"CUDNN_STATUS_ALLOC_FAILED",
        r"CUDA error:\s*out of memory",
        r"torch\.OutOfMemoryError",
    )
)


def detect_cuda_oom(result: ProcessResult, extra_logs: Sequence[Path] = ()) -> bool:
    """Classify OOM only for failed processes with explicit CUDA allocation evidence."""

    if result.return_code == 0 or result.timed_out or result.cancelled:
        return False
    fragments = [*result.stderr_tail, *result.stdout_tail]
    for log_path in extra_logs:
        try:
            with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
                fragments.extend(handle.readlines()[-200:])
        except OSError:
            continue
    evidence = "\n".join(fragments)
    return any(pattern.search(evidence) is not None for pattern in _OOM_PATTERNS)


def validate_artifact(
    path: Path,
    *,
    minimum_bytes: int,
    allowed_suffixes: Sequence[str],
    not_before: float | None = None,
) -> ArtifactValidation:
    """Validate existence, type, size, freshness and complete readability."""

    path = Path(path).resolve()
    suffixes = {suffix.lower() for suffix in allowed_suffixes}
    if path.suffix.lower() not in suffixes:
        return ArtifactValidation(path, False, reason=f"Unexpected suffix {path.suffix!r}")
    if not path.is_file():
        return ArtifactValidation(path, False, reason="File does not exist")
    try:
        stat = path.stat()
        if stat.st_size < minimum_bytes:
            return ArtifactValidation(
                path,
                False,
                size_bytes=stat.st_size,
                reason=f"File is smaller than {minimum_bytes} bytes",
            )
        if not_before is not None and stat.st_mtime + 2.0 < not_before:
            return ArtifactValidation(
                path,
                False,
                size_bytes=stat.st_size,
                reason="File predates this stage and may be a stale artifact",
            )
        digest = _sha256(path)
    except OSError as exc:
        return ArtifactValidation(path, False, reason=f"Could not read artifact: {exc}")
    return ArtifactValidation(
        path=path,
        valid=True,
        size_bytes=stat.st_size,
        sha256=digest,
        modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    )


def discover_valid_checkpoints(
    directory: Path,
    *,
    minimum_bytes: int = 1_024,
    not_before: float | None = None,
) -> tuple[ArtifactValidation, ...]:
    """Return validated checkpoints ordered by modification time, not filename."""

    directory = Path(directory)
    if not directory.is_dir():
        return ()
    validations = [
        validate_artifact(
            candidate,
            minimum_bytes=minimum_bytes,
            allowed_suffixes=(".pth", ".pt", ".ckpt"),
            not_before=not_before,
        )
        for candidate in directory.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".pth", ".pt", ".ckpt"}
    ]
    valid = [validation for validation in validations if validation.valid]
    return tuple(sorted(valid, key=lambda item: item.path.stat().st_mtime))


def write_checkpoint_manifest(
    manifest_path: Path,
    checkpoints: Sequence[ArtifactValidation],
    *,
    batch_size: int,
    training_parameters: Mapping[str, Any],
) -> Path:
    """Atomically write validation evidence for every accepted checkpoint."""

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_size": batch_size,
        "training_parameters": dict(training_parameters),
        "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
    }
    for item in payload["checkpoints"]:
        item["path"] = str(item["path"])
        item["write_completed"] = bool(item["valid"])
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest_path.resolve()


def _validate_training_request(request: TrainingRequest, batch_size: int) -> None:
    if not request.dry_run and not Path(request.experiment_dir).is_dir():
        raise FileNotFoundError(
            f"RVC experiment directory is missing: {request.experiment_dir}"
        )
    if not request.model_name.strip():
        raise ValueError("model_name must not be empty")
    if any(value <= 0 for value in (request.epochs, request.save_every_epochs, batch_size)):
        raise ValueError("epochs, save_every_epochs and batch_size must be positive")
    if not request.gpu_ids or any(identifier < 0 for identifier in request.gpu_ids):
        raise ValueError("gpu_ids must contain non-negative identifiers")
    for pretrained in (request.pretrained_generator, request.pretrained_discriminator):
        if not request.dry_run and pretrained is not None and not Path(pretrained).is_file():
            raise FileNotFoundError(f"Configured pretrained model is missing: {pretrained}")


def _replace_batch_placeholder(command: Sequence[str], batch_size: int) -> tuple[str, ...]:
    return tuple(str(part).replace("{batch_size}", str(batch_size)) for part in command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
