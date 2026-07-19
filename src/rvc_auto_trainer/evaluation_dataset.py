"""Freeze and validate the fixed Mandarin/Japanese RVC evaluation set."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from .audio.decoder import probe_audio
from .audio.discovery import AudioFile, discover_audio_files
from .config import AppConfig
from .exceptions import RVCAutoTrainerError


class EvaluationDatasetError(RVCAutoTrainerError):
    """Raised when the fixed test set is incomplete, changed, or leaked."""


@dataclass(frozen=True)
class FrozenTestSet:
    """Validated immutable test-set evidence."""

    manifest_path: Path
    files: tuple[AudioFile, ...]
    records: tuple[Mapping[str, Any], ...]


_ROLE_RULES: Mapping[str, tuple[str, float, float]] = {
    "mandarin_neutral": ("zh", 10.0, 15.0),
    "mandarin_phoneme_stress": ("zh", 10.0, 15.0),
    "mandarin_expressive": ("zh", 10.0, 20.0),
    "mandarin_paragraph": ("zh", 30.0, 60.0),
    "japanese_control": ("ja", 10.0, 20.0),
}


def freeze_test_manifest(
    config: AppConfig, manifest_path: Optional[Path] = None
) -> FrozenTestSet:
    """Validate exactly five role-based inputs and atomically record hashes."""

    destination = _manifest_path(config, manifest_path)
    payload = _read_manifest(destination)
    records = _validate_role_records(payload.get("tests"))
    discovered = discover_audio_files(config.paths.test_audio_dir)
    by_relative = {item.relative_path.casefold(): item for item in discovered.files}
    used: set[str] = set()
    frozen_records: list[dict[str, Any]] = []
    selected: list[AudioFile] = []
    for record in records:
        relative = _safe_relative_file(record.get("file"))
        item = by_relative.get(relative.casefold())
        if item is None:
            raise EvaluationDatasetError(
                f"Test manifest file is missing or unsupported: {relative}"
            )
        if item.sha256 in used:
            raise EvaluationDatasetError(
                f"Fixed test inputs must have unique audio content: {relative}"
            )
        used.add(item.sha256)
        role = str(record["role"])
        language, minimum, maximum = _ROLE_RULES[role]
        probe = probe_audio(item.path, ffprobe_executable="ffprobe")
        duration = float(probe.duration_seconds or 0.0)
        if not minimum <= duration <= maximum:
            raise EvaluationDatasetError(
                f"{relative} duration {duration:.3f}s is outside {minimum:g}-{maximum:g}s "
                f"for role {role}"
            )
        frozen = dict(record)
        frozen.update(
            {
                "language": language,
                "minimum_duration_seconds": minimum,
                "maximum_duration_seconds": maximum,
                "duration_seconds": round(duration, 6),
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
        )
        frozen_records.append(frozen)
        selected.append(item)
    leakage = _find_leakage(config, used)
    if leakage:
        raise EvaluationDatasetError(
            "Fixed test audio also appears in training/curation input: " + ", ".join(leakage)
        )
    frozen_payload = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Japanese target timbre to Mandarin dialogue fixed evaluation set",
        "tests": frozen_records,
    }
    _write_yaml_atomic(destination, frozen_payload)
    evidence = destination.with_name("test_manifest.lock.json")
    _write_json_atomic(
        evidence,
        {
            "schema_version": 1,
            "manifest": str(destination),
            "frozen_at": frozen_payload["frozen_at"],
            "files": [
                {
                    "file": item.relative_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in selected
            ],
        },
    )
    return FrozenTestSet(destination, tuple(selected), tuple(frozen_records))


def validate_frozen_test_manifest(
    config: AppConfig, manifest_path: Optional[Path] = None
) -> FrozenTestSet:
    """Fail if a frozen input, duration, hash, role, or leakage boundary changed."""

    path = _manifest_path(config, manifest_path)
    payload = _read_manifest(path)
    if payload.get("schema_version") != 1 or not payload.get("frozen_at"):
        raise EvaluationDatasetError(
            f"Test manifest is not frozen: {path}. Run freeze-tests first."
        )
    records = _validate_role_records(payload.get("tests"))
    discovered = discover_audio_files(config.paths.test_audio_dir)
    by_relative = {item.relative_path.casefold(): item for item in discovered.files}
    selected: list[AudioFile] = []
    used: set[str] = set()
    for record in records:
        relative = _safe_relative_file(record.get("file"))
        item = by_relative.get(relative.casefold())
        if item is None:
            raise EvaluationDatasetError(f"Frozen test input is missing: {relative}")
        expected_hash = str(record.get("sha256", "")).strip().lower()
        if not expected_hash or item.sha256 != expected_hash:
            raise EvaluationDatasetError(f"Frozen test input changed: {relative}")
        role = str(record["role"])
        language, minimum, maximum = _ROLE_RULES[role]
        if str(record.get("language", "")).strip().lower() != language:
            raise EvaluationDatasetError(f"Frozen language tag changed for {relative}")
        if float(record.get("minimum_duration_seconds", -1)) != minimum or float(
            record.get("maximum_duration_seconds", -1)
        ) != maximum:
            raise EvaluationDatasetError(f"Frozen duration rule changed for {relative}")
        duration = float(
            probe_audio(item.path, ffprobe_executable="ffprobe").duration_seconds or 0.0
        )
        recorded_duration = float(record.get("duration_seconds", -1.0))
        if abs(duration - recorded_duration) > 0.02 or not minimum <= duration <= maximum:
            raise EvaluationDatasetError(f"Frozen test duration changed for {relative}")
        if item.sha256 in used:
            raise EvaluationDatasetError("Frozen test inputs no longer have unique content")
        used.add(item.sha256)
        selected.append(item)
    leakage = _find_leakage(config, used)
    if leakage:
        raise EvaluationDatasetError(
            "Frozen test audio leaked into training/curation input: " + ", ".join(leakage)
        )
    return FrozenTestSet(path, tuple(selected), tuple(records))


def _validate_role_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvaluationDatasetError("Test manifest 'tests' must be a list")
    records: list[dict[str, Any]] = []
    roles: list[str] = []
    identifiers: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise EvaluationDatasetError(f"Test entry {index} must be a mapping")
        record = dict(item)
        identifier = str(record.get("id", "")).strip()
        role = str(record.get("role", "")).strip()
        if not identifier or identifier in identifiers:
            raise EvaluationDatasetError("Every test entry needs a unique non-empty id")
        if role not in _ROLE_RULES:
            raise EvaluationDatasetError(f"Unsupported or missing fixed test role: {role!r}")
        identifiers.add(identifier)
        roles.append(role)
        records.append(record)
    if len(records) != len(_ROLE_RULES) or set(roles) != set(_ROLE_RULES):
        expected = ", ".join(_ROLE_RULES)
        raise EvaluationDatasetError(
            f"Fixed test manifest must contain each of these five roles exactly once: {expected}"
        )
    if len(roles) != len(set(roles)):
        raise EvaluationDatasetError("Fixed test roles cannot be duplicated")
    return records


def _find_leakage(config: AppConfig, test_hashes: set[str]) -> list[str]:
    roots = (
        config.paths.training_audio_dir,
        config.paths.raw_audio_dir,
        config.paths.speaker_selected_audio_dir,
        config.paths.training_candidates_dir,
    )
    leaked: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for item in discover_audio_files(root).files:
            if item.sha256 in test_hashes:
                leaked.append(str(item.path))
    return sorted(leaked, key=str.casefold)


def _safe_relative_file(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise EvaluationDatasetError(f"Unsafe test manifest path: {text!r}")
    return path.as_posix()


def _manifest_path(config: AppConfig, manifest_path: Optional[Path]) -> Path:
    return (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else (config.paths.test_audio_dir / "test_manifest.yaml").resolve()
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationDatasetError(f"Test manifest is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EvaluationDatasetError(f"Cannot read test manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationDatasetError("Test manifest must contain a YAML mapping")
    return dict(payload)


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))
    return path


__all__ = [
    "EvaluationDatasetError",
    "FrozenTestSet",
    "freeze_test_manifest",
    "validate_frozen_test_manifest",
]
