"""Non-destructive dataset curation for the Japanese-to-Chinese workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .audio.decoder import probe_audio
from .audio.discovery import AudioFile, discover_audio_files, sha256_file
from .audio.quality import AudioQualityResult, QualityStatus, audit_audio_files
from .audio.reports import write_quality_reports
from .audio.slicer import SlicingConfig as AudioSlicingConfig
from .audio.slicer import slice_audio
from .config import AppConfig
from .exceptions import RVCAutoTrainerError


class DatasetCurationError(RVCAutoTrainerError):
    """Raised when a non-destructive curation step cannot be completed safely."""


@dataclass(frozen=True)
class DatasetPreparationResult:
    """Files and counts produced by :func:`prepare_dataset`."""

    raw_manifest_path: Path
    candidate_quality_json_path: Path
    review_queue_path: Path
    summary_path: Path
    raw_file_count: int
    candidate_file_count: int
    review_required_count: int


@dataclass(frozen=True)
class ReviewApplicationResult:
    """Copy-only result from applying human curation decisions."""

    kept_count: int
    reference_count: int
    rejected_count: int
    kept_duration_minutes: float
    summary_path: Path


_REVIEW_FIELDS = (
    "candidate_relative_path",
    "sha256",
    "duration_seconds",
    "status",
    "estimated_snr_db",
    "integrated_lufs",
    "reasons",
    "review_required",
    "recommended_action",
    "decision",
    "category",
    "reviewer_notes",
)
_DECISIONS = frozenset({"KEEP", "REFERENCE", "REJECT"})
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def prepare_dataset(
    config: AppConfig,
    *,
    source_label: str,
    rights_note: str,
    language: str = "ja",
) -> DatasetPreparationResult:
    """Hash raw originals, make immutable coarse candidates, and build review files.

    Raw inputs are never edited or deleted.  Candidate paths include a source hash,
    so rerunning this command is idempotent and changed originals create a new version.
    """

    source_label = source_label.strip()
    rights_note = rights_note.strip()
    language = language.strip()
    if not source_label or not rights_note or not language:
        raise DatasetCurationError(
            "source_label, rights_note, and language must be recorded before curation"
        )
    candidate_root = config.paths.training_candidates_dir
    manifests_root = config.paths.dataset_manifests_dir
    source_collections = (
        ("raw_archive", config.paths.raw_audio_dir),
        ("speaker_selected", config.paths.speaker_selected_audio_dir),
    )
    discoveries = []
    for collection, root in source_collections:
        root.mkdir(parents=True, exist_ok=True)
        discoveries.append((collection, root, discover_audio_files(root)))
    raw_file_count = sum(len(discovery.files) for _, _, discovery in discoveries)
    if raw_file_count == 0:
        roots = ", ".join(str(root) for _, root, _ in discoveries)
        raise DatasetCurationError(
            f"No supported non-empty curation input found in: {roots}"
        )

    candidate_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    source_records: list[dict[str, Any]] = []
    current_candidate_paths: set[str] = set()
    ignored_records: list[dict[str, str]] = []
    for collection, root, discovery in discoveries:
        for source in discovery.files:
            outputs, action, duration = _prepare_source_candidate(
                source, candidate_root, config, namespace=collection
            )
            relative_outputs = [
                output.relative_to(candidate_root.resolve()).as_posix() for output in outputs
            ]
            current_candidate_paths.update(relative_outputs)
            source_records.append(
                {
                    **source.to_manifest_record(),
                    "source_collection": collection,
                    "source_root": str(root),
                    "source_label": source_label,
                    "rights_note": rights_note,
                    "language": language,
                    "duration_seconds": duration,
                    "candidate_action": action,
                    "candidate_files": relative_outputs,
                }
            )
        ignored_records.extend(
            {
                "source_collection": collection,
                "relative_path": item.relative_path,
                "reason": item.reason,
            }
            for item in discovery.ignored
        )

    raw_manifest_path = _write_json_atomic(
        manifests_root / "raw_archive_manifest.json",
        {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "source_roots": {
                collection: str(root) for collection, root, _ in discoveries
            },
            "source_label": source_label,
            "rights_note": rights_note,
            "language": language,
            "candidate_root": str(candidate_root),
            "files": source_records,
            "ignored": ignored_records,
        },
    )

    discovered_candidates = discover_audio_files(candidate_root)
    by_relative = {item.relative_path: item for item in discovered_candidates.files}
    missing = sorted(current_candidate_paths.difference(by_relative))
    if missing:
        raise DatasetCurationError(
            "Prepared candidate files could not be rediscovered: " + ", ".join(missing)
        )
    current_candidates = tuple(by_relative[path] for path in sorted(current_candidate_paths))
    quality_results = audit_audio_files(
        current_candidates,
        config,
        root=candidate_root,
        ffmpeg_executable="ffmpeg",
    )
    quality_dir = manifests_root / "candidate_quality"
    quality_paths = write_quality_reports(quality_results, quality_dir, config)
    review_queue_path, review_stats = write_review_queue(
        quality_results,
        candidate_root,
        manifests_root / "review_queue.csv",
        random_seed=config.project.random_seed,
        pass_fraction=config.curation.review_pass_fraction,
        minimum_pass_samples=config.curation.review_minimum_pass_samples,
    )
    accepted_duration = sum(
        item.duration_seconds or 0.0
        for item in quality_results
        if item.status is not QualityStatus.FAIL
    )
    summary_path = _write_json_atomic(
        manifests_root / "dataset_summary.json",
        {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "raw_file_count": raw_file_count,
            "candidate_file_count": len(current_candidates),
            "candidate_duration_minutes": round(accepted_duration / 60.0, 6),
            "quality_counts": {
                status.value: sum(1 for item in quality_results if item.status is status)
                for status in QualityStatus
            },
            "review": review_stats,
            "raw_manifest": str(raw_manifest_path),
            "candidate_quality": str(quality_paths.json_path),
            "review_queue": str(review_queue_path),
            "next_step": (
                "Fill decision/category/reviewer_notes in review_queue.csv, then run "
                "apply-data-review. No candidate is promoted automatically."
            ),
        },
    )
    return DatasetPreparationResult(
        raw_manifest_path=raw_manifest_path,
        candidate_quality_json_path=quality_paths.json_path,
        review_queue_path=review_queue_path,
        summary_path=summary_path,
        raw_file_count=raw_file_count,
        candidate_file_count=len(current_candidates),
        review_required_count=int(review_stats["review_required_count"]),
    )


def write_review_queue(
    results: Sequence[AudioQualityResult],
    candidate_root: Path,
    destination: Path,
    *,
    random_seed: int,
    pass_fraction: float = 0.10,
    minimum_pass_samples: int = 50,
) -> tuple[Path, dict[str, int]]:
    """Write all candidates and mark every warning plus a stable PASS sample."""

    if not 0.0 < pass_fraction <= 1.0:
        raise ValueError("pass_fraction must be in (0, 1]")
    if minimum_pass_samples < 0:
        raise ValueError("minimum_pass_samples cannot be negative")
    root = candidate_root.resolve()
    passed = [item for item in results if item.status is QualityStatus.PASS]
    requested = max(math.ceil(len(passed) * pass_fraction), minimum_pass_samples)
    sample_count = min(len(passed), requested)
    sampled_paths = {
        item.file_path.resolve()
        for item in sorted(passed, key=lambda item: _review_sort_key(item, random_seed))[
            :sample_count
        ]
    }

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    required_count = 0
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_REVIEW_FIELDS)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.relative_path.casefold()):
            relative = _relative_audio_path(result.file_path, root)
            required = result.status is not QualityStatus.PASS or result.file_path.resolve() in sampled_paths
            required_count += int(required)
            writer.writerow(
                {
                    "candidate_relative_path": relative,
                    "sha256": result.sha256 or sha256_file(result.file_path),
                    "duration_seconds": _optional_number(result.duration_seconds),
                    "status": result.status.value,
                    "estimated_snr_db": _optional_number(result.estimated_snr_db),
                    "integrated_lufs": _optional_number(result.integrated_lufs),
                    "reasons": " | ".join(result.reasons),
                    "review_required": "YES" if required else "NO",
                    "recommended_action": (
                        "REJECT"
                        if result.status is QualityStatus.FAIL
                        else "REVIEW"
                        if result.status is QualityStatus.WARNING
                        else "KEEP"
                    ),
                    "decision": "",
                    "category": "",
                    "reviewer_notes": "",
                }
            )
    os.replace(str(temporary), str(destination))
    return destination, {
        "total_rows": len(results),
        "warning_rows": sum(item.status is QualityStatus.WARNING for item in results),
        "fail_rows": sum(item.status is QualityStatus.FAIL for item in results),
        "sampled_pass_rows": sample_count,
        "review_required_count": required_count,
    }


def apply_review_decisions(
    config: AppConfig,
    review_queue: Optional[Path] = None,
    *,
    accept_unreviewed_pass: bool = False,
) -> ReviewApplicationResult:
    """Copy reviewed candidates into training/reference roots without deleting inputs."""

    queue = (
        Path(review_queue).expanduser().resolve()
        if review_queue is not None
        else config.paths.dataset_manifests_dir / "review_queue.csv"
    )
    if not queue.is_file():
        raise DatasetCurationError(f"Review queue does not exist: {queue}")
    with queue.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DatasetCurationError(f"Review queue contains no candidate rows: {queue}")
    missing_fields = set(_REVIEW_FIELDS).difference(rows[0])
    if missing_fields:
        raise DatasetCurationError(
            "Review queue is missing required columns: " + ", ".join(sorted(missing_fields))
        )

    candidate_root = config.paths.training_candidates_dir.resolve()
    training_root = config.paths.training_audio_dir.resolve()
    reference_root = config.paths.voice_reference_dir.resolve()
    kept = references = rejected = 0
    kept_seconds = 0.0
    applied: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        decision = str(row.get("decision", "")).strip().upper()
        required = str(row.get("review_required", "")).strip().upper() == "YES"
        if not decision and not required and accept_unreviewed_pass:
            decision = "KEEP"
        if not decision:
            raise DatasetCurationError(
                f"Review decision is blank at CSV line {line_number}; fill KEEP, REFERENCE, or REJECT"
            )
        if decision not in _DECISIONS:
            raise DatasetCurationError(
                f"Unsupported review decision {decision!r} at CSV line {line_number}"
            )
        relative = str(row.get("candidate_relative_path", "")).strip()
        source = _safe_join(candidate_root, relative)
        if not source.is_file():
            raise DatasetCurationError(f"Reviewed candidate is missing: {source}")
        expected_hash = str(row.get("sha256", "")).strip().lower()
        actual_hash = sha256_file(source)
        if expected_hash != actual_hash:
            raise DatasetCurationError(
                f"Candidate hash changed after review: {relative} ({actual_hash} != {expected_hash})"
            )
        destination: Path | None = None
        if decision == "KEEP":
            destination = _safe_join(training_root, relative)
            _copy_immutable(source, destination, actual_hash)
            kept += 1
            kept_seconds += float(row.get("duration_seconds") or 0.0)
        elif decision == "REFERENCE":
            destination = _safe_join(reference_root, relative)
            _copy_immutable(source, destination, actual_hash)
            references += 1
        else:
            rejected += 1
        applied.append(
            {
                "candidate_relative_path": relative,
                "sha256": actual_hash,
                "decision": decision,
                "category": str(row.get("category", "")).strip(),
                "destination": str(destination) if destination is not None else None,
            }
        )

    summary_path = _write_json_atomic(
        config.paths.dataset_manifests_dir / "applied_review.json",
        {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "review_queue": str(queue),
            "kept_count": kept,
            "reference_count": references,
            "rejected_count": rejected,
            "kept_duration_minutes": round(kept_seconds / 60.0, 6),
            "target_core_duration_minutes": [15, 20],
            "items": applied,
        },
    )
    return ReviewApplicationResult(
        kept_count=kept,
        reference_count=references,
        rejected_count=rejected,
        kept_duration_minutes=kept_seconds / 60.0,
        summary_path=summary_path,
    )


def _prepare_source_candidate(
    source: AudioFile,
    candidate_root: Path,
    config: AppConfig,
    *,
    namespace: str,
) -> tuple[tuple[Path, ...], str, float]:
    probe = probe_audio(source.path, ffprobe_executable="ffprobe")
    duration = float(probe.duration_seconds or 0.0)
    if duration <= 0.0:
        raise DatasetCurationError(f"Cannot determine a positive duration for {source.path}")
    if config.curation.coarse_split_enabled and duration > config.curation.maximum_chunk_seconds:
        output_dir = candidate_root / _safe_component(namespace) / (
            f"{_safe_component(Path(source.relative_path).stem)}__{source.sha256[:12]}"
        )
        manifest = output_dir / "segments_manifest.csv"
        if output_dir.exists():
            if not manifest.is_file():
                raise DatasetCurationError(
                    f"Existing coarse-candidate directory is incomplete: {output_dir}"
                )
            existing = discover_audio_files(output_dir).files
            if not existing:
                raise DatasetCurationError(
                    f"Existing coarse-candidate directory has no audio: {output_dir}"
                )
            return tuple(item.path for item in existing), "reused_slices", duration
        slicing = AudioSlicingConfig(
            backend="internal",
            minimum_segment_seconds=config.curation.minimum_chunk_seconds,
            preferred_segment_seconds=config.curation.preferred_chunk_seconds,
            maximum_segment_seconds=config.curation.maximum_chunk_seconds,
            silence_threshold_dbfs=config.curation.silence_threshold_dbfs,
            minimum_silence_duration_ms=config.curation.minimum_silence_duration_ms,
            segment_padding_ms=config.curation.chunk_padding_ms,
            output_subtype=config.preprocessing.output_subtype,
        )
        sliced = slice_audio(
            source.path,
            output_dir,
            slicing,
            quality_thresholds=config,
            ffmpeg_executable="ffmpeg",
        )
        return tuple(item.segment_file for item in sliced.segments), "sliced", duration

    source_relative = Path(source.relative_path)
    filename = f"{source_relative.stem}__{source.sha256[:12]}{source.extension}"
    relative = Path(_safe_component(namespace)) / source_relative.parent / filename
    destination = _safe_join(candidate_root.resolve(), relative.as_posix())
    _copy_immutable(source.path, destination, source.sha256)
    return (destination,), "copied", duration


def _copy_immutable(source: Path, destination: Path, expected_hash: str) -> None:
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == expected_hash:
            return
        raise DatasetCurationError(
            f"Refusing to overwrite a different existing file: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        actual_hash = sha256_file(temporary)
        if actual_hash != expected_hash:
            raise DatasetCurationError(
                f"Copied candidate hash mismatch: {actual_hash} != {expected_hash}"
            )
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_join(root: Path, relative: str) -> Path:
    if not relative:
        raise DatasetCurationError("Candidate relative path must not be empty")
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetCurationError(f"Candidate path escapes configured root: {relative}") from exc
    return candidate


def _relative_audio_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise DatasetCurationError(f"Candidate audio is outside candidate root: {resolved}") from exc


def _review_sort_key(result: AudioQualityResult, seed: int) -> str:
    identity = result.sha256 or result.relative_path
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    cleaned = _WINDOWS_INVALID.sub("_", value).strip(" ._")
    return cleaned[:80] or "audio"


def _optional_number(value: Optional[float]) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DatasetCurationError",
    "DatasetPreparationResult",
    "ReviewApplicationResult",
    "apply_review_decisions",
    "prepare_dataset",
    "write_review_queue",
]
