"""F0/content feature extraction request models and command builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .repository_inspector import RVCRepositoryInfo, RVCRepositoryInspectionError


@dataclass(frozen=True)
class FeatureRequest:
    """Inputs shared by F0 and content-feature extraction."""

    experiment_dir: Path
    f0_method: str = "rmvpe"
    process_count: int = 4
    gpu_id: int = 0
    device: str | None = None
    version: str = "v2"
    is_half: bool = True
    hop_length: int = 128
    timeout_seconds: float | None = None
    dry_run: bool = False
    f0_command: tuple[str, ...] | None = None
    feature_command: tuple[str, ...] | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = field(default=None, compare=False, repr=False)


def build_extract_f0_command(
    repository: RVCRepositoryInfo, request: FeatureRequest
) -> tuple[str, ...]:
    """Build the F0 command supported by the inspected script contract."""

    if request.f0_command:
        return tuple(request.f0_command)
    if request.f0_method.lower() in {"rmvpe", "rmvpe_gpu"} and (
        "extract_f0_rmvpe" in repository.scripts
    ):
        script = repository.require_script("extract_f0_rmvpe")
    else:
        script = repository.require_script("extract_f0")
    _validate_request(request)
    python_and_script = (str(repository.python_executable), str(script.path))
    if script.contract in {"webui-f0", "legacy-f0"}:
        return (
            *python_and_script,
            str(Path(request.experiment_dir).resolve()),
            str(request.process_count),
            request.f0_method,
            str(request.hop_length),
            *request.extra_args,
        )
    if script.contract in {"webui-rmvpe-gpu", "legacy-rmvpe-gpu"}:
        if request.f0_method.lower() not in {"rmvpe", "rmvpe_gpu"}:
            raise RVCRepositoryInspectionError(
                f"{script.relative_path} is an RMVPE-only extractor, but "
                f"f0_method={request.f0_method!r}."
            )
        return (
            *python_and_script,
            "1",  # number of worker/GPU partitions
            "0",  # this adapter launches the sole partition
            str(request.gpu_id),
            str(Path(request.experiment_dir).resolve()),
            str(bool(request.is_half)),
            *request.extra_args,
        )
    raise RVCRepositoryInspectionError(
        f"Unsupported F0 script contract {script.contract!r}; use f0_command."
    )


def build_extract_features_command(
    repository: RVCRepositoryInfo, request: FeatureRequest
) -> tuple[str, ...]:
    """Build a one-GPU content feature extraction command."""

    if request.feature_command:
        return tuple(request.feature_command)
    script = repository.require_script("extract_features")
    _validate_request(request)
    if script.contract not in {"webui-features", "legacy-features"}:
        raise RVCRepositoryInspectionError(
            f"Unsupported feature script contract {script.contract!r}; "
            "use feature_command."
        )
    device = request.device or f"cuda:{request.gpu_id}"
    return (
        str(repository.python_executable),
        str(script.path),
        device,
        "1",  # number of partitions
        "0",  # partition index for this process
        str(request.gpu_id),
        str(Path(request.experiment_dir).resolve()),
        request.version,
        str(bool(request.is_half)),
        *request.extra_args,
    )


def _validate_request(request: FeatureRequest) -> None:
    if not request.dry_run and not Path(request.experiment_dir).is_dir():
        raise FileNotFoundError(
            f"RVC experiment directory is missing: {request.experiment_dir}"
        )
    if request.process_count <= 0 or request.hop_length <= 0:
        raise ValueError("process_count and hop_length must be positive")
    if request.gpu_id < 0:
        raise ValueError("gpu_id must be non-negative")
