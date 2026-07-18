"""Safe FAISS index CLI adaptation and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .repository_inspector import RVCRepositoryInfo, RVCRepositoryInspectionError
from .training import ArtifactValidation, validate_artifact


@dataclass(frozen=True)
class IndexRequest:
    """Inputs for an upstream standalone index-building command."""

    feature_dir: Path
    output_path: Path
    algorithm: str = "auto"
    experiment_dir: Path | None = None
    minimum_index_bytes: int = 128
    timeout_seconds: float | None = None
    dry_run: bool = False
    command: tuple[str, ...] | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = field(default=None, compare=False, repr=False)


def build_index_command(
    repository: RVCRepositoryInfo, request: IndexRequest
) -> tuple[str, ...]:
    """Build an index command from observed argparse flags.

    RVC WebUI commonly implements index generation only as an in-process UI
    callback. This function refuses to invent a command when no standalone CLI
    contract is visible.
    """

    _validate_index_request(request)
    if request.command:
        replacements = {
            "{feature_dir}": str(Path(request.feature_dir).resolve()),
            "{output_path}": str(Path(request.output_path).resolve()),
            "{experiment_dir}": str(
                Path(request.experiment_dir or request.feature_dir).resolve()
            ),
        }
        return _replace_placeholders(request.command, replacements)

    script = repository.require_script("build_index")
    options = set(script.argparse_options)
    feature_flag = _first_option(options, ("--feature-dir", "--feature_dir", "--input"))
    output_flag = _first_option(options, ("--output", "--output-path", "--output_path"))
    experiment_flag = _first_option(
        options, ("--experiment-dir", "--experiment_dir", "--exp-dir", "--exp_dir")
    )
    if output_flag is None or (feature_flag is None and experiment_flag is None):
        raise RVCRepositoryInspectionError(
            f"Index script {script.relative_path!r} exists, but its statically "
            f"observed options {sorted(options)!r} do not expose a recognized "
            "feature/experiment input and output path. Configure an explicit command."
        )
    command = [str(repository.python_executable), str(script.path)]
    if feature_flag is not None:
        command.extend((feature_flag, str(Path(request.feature_dir).resolve())))
    else:
        command.extend(
            (
                experiment_flag or "",
                str(Path(request.experiment_dir or request.feature_dir).resolve()),
            )
        )
    command.extend((output_flag, str(Path(request.output_path).resolve())))
    algorithm_flag = _first_option(options, ("--algorithm", "--index-algorithm"))
    if algorithm_flag is not None:
        command.extend((algorithm_flag, request.algorithm))
    command.extend(request.extra_args)
    return tuple(command)


def validate_index(
    path: Path, *, minimum_bytes: int = 128, not_before: float | None = None
) -> ArtifactValidation:
    """Validate a generated FAISS index as a fresh, readable binary file."""

    validation = validate_artifact(
        path,
        minimum_bytes=minimum_bytes,
        allowed_suffixes=(".index",),
        not_before=not_before,
    )
    if not validation.valid:
        return validation
    try:
        header = Path(path).read_bytes()[:4]
    except OSError as exc:
        return ArtifactValidation(Path(path).resolve(), False, reason=str(exc))
    # Common FAISS index magics start with Ix. A custom upstream serializer can
    # still be accepted after size/freshness checks, but the evidence is explicit.
    if len(header) < 4 or header in {b"\x00\x00\x00\x00", b"    "}:
        return ArtifactValidation(
            validation.path,
            False,
            size_bytes=validation.size_bytes,
            sha256=validation.sha256,
            modified_at=validation.modified_at,
            reason="Index header is empty or invalid",
        )
    return validation


def _validate_index_request(request: IndexRequest) -> None:
    if not request.dry_run and not Path(request.feature_dir).is_dir():
        raise FileNotFoundError(f"Feature directory is missing: {request.feature_dir}")
    if Path(request.output_path).suffix.lower() != ".index":
        raise ValueError("Index output_path must end in .index")
    if request.minimum_index_bytes <= 0:
        raise ValueError("minimum_index_bytes must be positive")


def _first_option(options: set[str], aliases: Sequence[str]) -> str | None:
    return next((alias for alias in aliases if alias in options), None)


def _replace_placeholders(
    command: Sequence[str], replacements: Mapping[str, str]
) -> tuple[str, ...]:
    result: list[str] = []
    for part in command:
        rendered = str(part)
        for marker, value in replacements.items():
            rendered = rendered.replace(marker, value)
        result.append(rendered)
    return tuple(result)
