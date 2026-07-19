"""Prepare and validate the file layout required by official RVC training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class TrainingWorkspaceError(RuntimeError):
    """Raised when upstream preprocessing/feature artifacts are incomplete."""


@dataclass(frozen=True)
class TrainingWorkspaceResult:
    """Validated official-RVC training inputs."""

    sample_names: tuple[str, ...]
    filelist_path: Path
    config_path: Path
    manifest_path: Path


def prepare_training_workspace(
    repository: Path,
    experiment_dir: Path,
    *,
    sample_rate: str,
    version: str,
    use_f0: bool,
    speaker_id: int,
    random_seed: int,
    mixed_precision: bool,
    dry_run: bool = False,
) -> TrainingWorkspaceResult:
    """Create official ``filelist.txt`` and ``config.json`` from real intersections."""

    repository = Path(repository).resolve()
    experiment = Path(experiment_dir).resolve()
    feature_name = "3_feature256" if version == "v1" else "3_feature768"
    directories = {
        "ground_truth": experiment / "0_gt_wavs",
        "features": experiment / feature_name,
    }
    if use_f0:
        directories.update(
            {
                "f0": experiment / "2a_f0",
                "f0nsf": experiment / "2b-f0nsf",
            }
        )
    names = _artifact_intersection(directories, use_f0=use_f0, dry_run=dry_run)
    filelist_path = experiment / "filelist.txt"
    config_path = experiment / "config.json"
    manifest_path = experiment / "training_workspace_manifest.json"
    if dry_run:
        return TrainingWorkspaceResult(names, filelist_path, config_path, manifest_path)

    experiment.mkdir(parents=True, exist_ok=True)
    template = _config_template(repository, sample_rate, version)
    payload = _read_json(template)
    train = payload.get("train")
    model = payload.get("model")
    if not isinstance(train, dict) or not isinstance(model, dict):
        raise TrainingWorkspaceError(f"RVC config template has an invalid shape: {template}")
    train["seed"] = random_seed
    train["fp16_run"] = bool(mixed_precision)
    speaker_capacity = int(model.get("spk_embed_dim", 0))
    if speaker_id < 0 or speaker_id >= speaker_capacity:
        raise TrainingWorkspaceError(
            f"speaker_id={speaker_id} is outside config capacity 0-{speaker_capacity - 1}"
        )

    lines = [
        _sample_line(directories, name, speaker_id, use_f0=use_f0) for name in names
    ]
    mute_lines = _mute_lines(
        repository,
        sample_rate,
        version,
        speaker_id,
        use_f0=use_f0,
    )
    lines.extend(mute_lines)
    random.Random(random_seed).shuffle(lines)
    _write_text_atomic(filelist_path, "\n".join(lines) + "\n")
    _write_json_atomic(config_path, payload)
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "repository": str(repository),
            "experiment_dir": str(experiment),
            "sample_rate": sample_rate,
            "version": version,
            "use_f0": use_f0,
            "speaker_id": speaker_id,
            "random_seed": random_seed,
            "mixed_precision": mixed_precision,
            "config_template": str(template),
            "config_template_sha256": _sha256(template),
            "real_sample_count": len(names),
            "mute_sample_count": len(mute_lines),
            "sample_names": list(names),
            "artifact_counts": {
                label: len(_names_for_directory(path, label))
                for label, path in directories.items()
            },
        },
    )
    return TrainingWorkspaceResult(names, filelist_path, config_path, manifest_path)


def validate_stage_artifacts(
    experiment_dir: Path,
    stage: str,
    *,
    version: str = "v2",
) -> tuple[str, ...]:
    """Ensure a nominally successful upstream stage produced real non-empty files."""

    experiment = Path(experiment_dir).resolve()
    gt = _names_for_directory(experiment / "0_gt_wavs", "ground_truth")
    wav16 = _names_for_directory(experiment / "1_16k_wavs", "wav16")
    if not gt or not wav16:
        raise TrainingWorkspaceError(
            "RVC preprocessing produced no usable 0_gt_wavs/1_16k_wavs artifacts"
        )
    common = gt.intersection(wav16)
    if stage == "preprocess":
        if not common:
            raise TrainingWorkspaceError(
                "RVC preprocessing artifact names do not intersect between output folders"
            )
        return tuple(sorted(common, key=str.casefold))
    if stage == "f0":
        f0 = _names_for_directory(experiment / "2a_f0", "f0")
        f0nsf = _names_for_directory(experiment / "2b-f0nsf", "f0nsf")
        common &= f0
        common &= f0nsf
    elif stage == "features":
        folder = "3_feature256" if version == "v1" else "3_feature768"
        common &= _names_for_directory(experiment / folder, "features")
    else:
        raise ValueError(f"Unsupported artifact validation stage: {stage}")
    if not common:
        raise TrainingWorkspaceError(
            f"RVC {stage} stage exited without a complete real artifact intersection"
        )
    return tuple(sorted(common, key=str.casefold))


def _artifact_intersection(
    directories: Mapping[str, Path], *, use_f0: bool, dry_run: bool
) -> tuple[str, ...]:
    del use_f0
    if dry_run:
        return ()
    sets = [_names_for_directory(path, label) for label, path in directories.items()]
    if any(not names for names in sets):
        missing = [label for (label, _), names in zip(directories.items(), sets) if not names]
        raise TrainingWorkspaceError(
            "Official RVC workspace is missing non-empty artifacts for: " + ", ".join(missing)
        )
    common = set.intersection(*sets)
    if not common:
        raise TrainingWorkspaceError(
            "Official RVC workspace has no same-name real sample across required artifacts"
        )
    return tuple(sorted(common, key=str.casefold))


def _names_for_directory(directory: Path, label: str) -> set[str]:
    if not directory.is_dir():
        return set()
    names: set[str] = set()
    for path in directory.iterdir():
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        if label in {"ground_truth", "wav16"} and path.suffix.lower() == ".wav":
            names.add(path.stem)
        elif label == "features" and path.suffix.lower() == ".npy":
            names.add(path.stem)
        elif label in {"f0", "f0nsf"} and path.name.lower().endswith(".wav.npy"):
            names.add(path.name[: -len(".wav.npy")])
    return names


def _sample_line(
    directories: Mapping[str, Path],
    name: str,
    speaker_id: int,
    *,
    use_f0: bool,
) -> str:
    parts = [
        _portable_path(directories["ground_truth"] / f"{name}.wav"),
        _portable_path(directories["features"] / f"{name}.npy"),
    ]
    if use_f0:
        parts.extend(
            (
                _portable_path(directories["f0"] / f"{name}.wav.npy"),
                _portable_path(directories["f0nsf"] / f"{name}.wav.npy"),
            )
        )
    parts.append(str(speaker_id))
    return "|".join(parts)


def _mute_lines(
    repository: Path,
    sample_rate: str,
    version: str,
    speaker_id: int,
    *,
    use_f0: bool,
) -> list[str]:
    dimension = 256 if version == "v1" else 768
    root = repository / "logs" / "mute"
    parts = [
        root / "0_gt_wavs" / f"mute{sample_rate}.wav",
        root / f"3_feature{dimension}" / "mute.npy",
    ]
    if use_f0:
        parts.extend(
            (
                root / "2a_f0" / "mute.wav.npy",
                root / "2b-f0nsf" / "mute.wav.npy",
            )
        )
    missing = [str(path) for path in parts if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise TrainingWorkspaceError(
            "Official RVC mute assets are missing or empty: " + ", ".join(missing)
        )
    line = "|".join([*(_portable_path(path) for path in parts), str(speaker_id)])
    return [line, line]


def _config_template(repository: Path, sample_rate: str, version: str) -> Path:
    family = "v1" if version == "v1" or sample_rate == "40k" else "v2"
    template = repository / "configs" / family / f"{sample_rate}.json"
    if not template.is_file():
        raise TrainingWorkspaceError(f"Official RVC config template is missing: {template}")
    return template


def _portable_path(path: Path) -> str:
    return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingWorkspaceError(f"Cannot read RVC JSON config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingWorkspaceError(f"RVC JSON config is not an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    return _write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "TrainingWorkspaceError",
    "TrainingWorkspaceResult",
    "prepare_training_workspace",
    "validate_stage_artifacts",
]
