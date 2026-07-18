"""RVC preprocessing request model and version-aware command construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .repository_inspector import RVCRepositoryInfo, RVCRepositoryInspectionError


@dataclass(frozen=True)
class PreprocessRequest:
    """Inputs for an upstream RVC preprocessing command."""

    input_dir: Path
    experiment_dir: Path
    sample_rate: int = 40_000
    process_count: int = 4
    no_parallel: bool = False
    timeout_seconds: float | None = None
    dry_run: bool = False
    command: tuple[str, ...] | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = field(default=None, compare=False, repr=False)


def build_preprocess_command(
    repository: RVCRepositoryInfo, request: PreprocessRequest
) -> tuple[str, ...]:
    """Build a command only for an inspected, recognized preprocessing script."""

    if request.command:
        return tuple(request.command)
    script = repository.require_script("preprocess")
    if not request.dry_run:
        _validate_directory(request.input_dir, "preprocessing input")
    if request.sample_rate <= 0 or request.process_count <= 0:
        raise ValueError("sample_rate and process_count must be positive")

    # Both recognized WebUI families use these five positional arguments.  A
    # script exposing fewer argv positions is not treated as compatible.
    if script.positional_arity is not None and script.positional_arity < 5:
        raise RVCRepositoryInspectionError(
            f"Discovered preprocessing script {script.relative_path!r}, but static "
            f"inspection found only sys.argv[0..{script.positional_arity}]. Its CLI "
            "does not match the known RVC contract; provide an explicit command."
        )
    return (
        str(repository.python_executable),
        str(script.path),
        str(Path(request.input_dir).resolve()),
        str(request.sample_rate),
        str(request.process_count),
        str(Path(request.experiment_dir).resolve()),
        str(bool(request.no_parallel)),
        *request.extra_args,
    )


def _validate_directory(path: Path, description: str) -> None:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{description.capitalize()} directory is missing: {path}")
