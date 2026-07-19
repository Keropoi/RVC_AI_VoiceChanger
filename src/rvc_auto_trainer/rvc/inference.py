"""RVC batch-inference request model and inspected CLI command builder."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .repository_inspector import RVCRepositoryInfo, RVCRepositoryInspectionError


@dataclass(frozen=True)
class InferenceRequest:
    """One RVC voice-conversion request."""

    input_path: Path
    output_path: Path
    model_path: Path
    index_path: Path | None = None
    speaker_id: int = 0
    transpose: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.65
    filter_radius: int = 3
    resample_sample_rate: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    allow_without_index: bool = False
    minimum_output_bytes: int = 44
    timeout_seconds: float | None = None
    dry_run: bool = False
    command: tuple[str, ...] | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class InferenceResult:
    """Inference stage result with the generated audio path."""

    success: bool
    output_path: Path
    command: tuple[str, ...]
    return_code: int | None = None
    stdout_log: Path | None = None
    stderr_log: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    message: str = ""


def build_inference_command(
    repository: RVCRepositoryInfo, request: InferenceRequest
) -> tuple[str, ...]:
    """Build a CLI invocation only when required argparse flags were observed."""

    _validate_inference_request(request)
    replacements = {
        "{input_path}": str(Path(request.input_path).resolve()),
        "{output_path}": str(Path(request.output_path).resolve()),
        "{model_path}": str(Path(request.model_path).resolve()),
        "{index_path}": str(Path(request.index_path).resolve())
        if request.index_path is not None
        else "",
        "{transpose}": str(request.transpose),
        "{index_rate}": str(request.index_rate),
    }
    if request.command:
        return _replace_placeholders(request.command, replacements)

    script = repository.require_script("infer")
    options = set(script.argparse_options)
    semantic_options: tuple[tuple[Sequence[str], str, bool], ...] = (
        (("--input-path", "--input_path", "--input", "-i"), replacements["{input_path}"], True),
        (
            ("--output-path", "--output_path", "--opt_path", "--output", "-o"),
            replacements["{output_path}"],
            True,
        ),
        (
            ("--model-path", "--model_path", "--model_name", "--model", "-m"),
            replacements["{model_path}"],
            True,
        ),
        (("--index-path", "--index_path", "--index"), replacements["{index_path}"], False),
        (("--speaker-id", "--speaker_id", "--sid"), str(request.speaker_id), False),
        (("--transpose", "--f0up-key", "--f0up_key", "--pitch"), str(request.transpose), False),
        (("--f0-method", "--f0_method", "--f0method"), request.f0_method, False),
        (("--index-rate", "--index_rate"), str(request.index_rate), False),
        (("--filter-radius", "--filter_radius"), str(request.filter_radius), False),
        (("--resample-sr", "--resample_sr"), str(request.resample_sample_rate), False),
        (("--rms-mix-rate", "--rms_mix_rate"), str(request.rms_mix_rate), False),
        (("--protect",), str(request.protect), False),
    )
    command = [str(repository.python_executable), str(script.path)]
    missing: list[str] = []
    for aliases, value, required in semantic_options:
        option = _first_option(options, aliases)
        if option is None:
            if required:
                missing.append("/".join(aliases))
            continue
        if option == "--model_name":
            value = _official_model_name(repository, request.model_path)
        if value or required:
            command.extend((option, value))
    if missing:
        raise RVCRepositoryInspectionError(
            f"Inference script {script.relative_path!r} lacks recognized required "
            f"options: {', '.join(missing)}. Observed: {sorted(options)!r}. "
            "Configure an explicit command for this RVC version."
        )
    command.extend(request.extra_args)
    return tuple(command)


def _validate_inference_request(request: InferenceRequest) -> None:
    for path, label in (
        (request.input_path, "Inference input audio"),
        (request.model_path, "RVC model"),
    ):
        if not request.dry_run and not Path(path).is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if request.index_path is None and not request.allow_without_index:
        raise FileNotFoundError(
            "No RVC index was provided and allow_without_index is false"
        )
    if (
        not request.dry_run
        and request.index_path is not None
        and not Path(request.index_path).is_file()
    ):
        raise FileNotFoundError(f"RVC index is missing: {request.index_path}")
    if Path(request.output_path).suffix.lower() not in {".wav", ".flac"}:
        raise ValueError("Inference output must be .wav or .flac")
    for value, name in (
        (request.index_rate, "index_rate"),
        (request.rms_mix_rate, "rms_mix_rate"),
        (request.protect, "protect"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")


def _first_option(options: set[str], aliases: Sequence[str]) -> str | None:
    return next((alias for alias in aliases if alias in options), None)


def _official_model_name(repository: RVCRepositoryInfo, model_path: Path) -> str:
    """Render a model path for the official ``--model_name`` CLI contract.

    Upstream prefixes this argument with ``assets/weights`` rather than accepting
    an absolute path. A relative path addresses the caller's exact artifact
    without copying it into the external checkout.
    """

    weight_root = repository.repository / "assets" / "weights"
    try:
        return os.path.relpath(Path(model_path).resolve(), weight_root.resolve())
    except ValueError as exc:
        raise RVCRepositoryInspectionError(
            "The official RVC --model_name CLI cannot address a model on a "
            "different filesystem volume from its assets/weights directory. "
            "Move the model to the RVC volume or provide an explicit command."
        ) from exc


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
