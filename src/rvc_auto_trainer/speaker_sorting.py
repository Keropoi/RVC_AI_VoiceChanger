"""Non-destructive multi-speaker diarization and target-character selection."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .audio.decoder import decode_audio, write_wav
from .audio.discovery import AudioFile, discover_audio_files, sha256_file
from .config import AppConfig
from .exceptions import RVCAutoTrainerError


class SpeakerSortingError(RVCAutoTrainerError):
    """Raised when speaker candidates cannot be produced or promoted safely."""


@dataclass(frozen=True)
class SpeakerTurn:
    """One diarized time interval in seconds."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class SpeakerSortingResult:
    """Artifacts produced by one diarization run."""

    run_id: str
    source_count: int
    speaker_cluster_count: int
    segment_count: int
    review_queue_path: Path
    segment_manifest_path: Path
    summary_path: Path


@dataclass(frozen=True)
class SpeakerReviewResult:
    """Copy-only outcome of a reviewed target-speaker queue."""

    target_cluster_count: int
    copied_segment_count: int
    copied_duration_minutes: float
    summary_path: Path


_REVIEW_FIELDS = (
    "source_relative_path",
    "source_sha256",
    "speaker_label",
    "segment_count",
    "usable_duration_seconds",
    "overlap_segment_count",
    "target_similarity",
    "rank_in_source",
    "score_margin",
    "recommended_action",
    "sample_files",
    "segment_relative_paths",
    "segment_sha256s",
    "segment_durations_seconds",
    "decision",
    "reviewer_notes",
)
_VALID_DECISIONS = frozenset({"TARGET", "OTHER", "REJECT"})
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def sort_speakers(
    config: AppConfig,
    *,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> SpeakerSortingResult:
    """Diarize mixed recordings, cut candidates, and write a human review queue.

    Anonymous speaker labels are local to each source recording.  If clean target
    references exist, speaker embeddings only rank candidates; they never promote
    audio without an explicit ``TARGET`` decision.
    """

    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise SpeakerSortingError(
            "Use either num_speakers or min_speakers/max_speakers, not both"
        )
    _validate_speaker_bounds(num_speakers, min_speakers, max_speakers)
    sources = discover_audio_files(config.paths.mixed_speaker_audio_dir)
    if not sources.files:
        raise SpeakerSortingError(
            "No supported non-empty mixed-speaker recordings found in "
            f"{config.paths.mixed_speaker_audio_dir}"
        )

    pipeline, embedding_inference = _load_pyannote(config)
    references = discover_audio_files(config.paths.voice_reference_dir)
    target_embedding = (
        _mean_file_embeddings(embedding_inference, references.files)
        if references.files
        else None
    )
    run_id = datetime.now().strftime("speakers_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    segment_run_root = (config.paths.speaker_segments_dir / run_id).resolve()
    manifest_run_root = (config.paths.speaker_manifests_dir / run_id).resolve()
    segment_run_root.mkdir(parents=True, exist_ok=False)
    manifest_run_root.mkdir(parents=True, exist_ok=False)

    segment_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    ignored_short_turns = 0
    for source in sources.files:
        kwargs = _diarization_kwargs(config, num_speakers, min_speakers, max_speakers)
        try:
            output = pipeline(str(source.path), **kwargs)
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise SpeakerSortingError(
                f"Speaker diarization failed for {source.path.name}: {exc}"
            ) from exc
        turns = _normalise_turns(
            _iter_diarization_turns(output),
            merge_gap_seconds=config.speaker_sorting.merge_gap_seconds,
        )
        if not turns:
            raise SpeakerSortingError(
                f"Diarization returned no speaker turns for {source.path.name}"
            )
        decoded = decode_audio(source.path, ffmpeg_executable="ffmpeg")
        source_key = (
            f"{_safe_component(Path(source.relative_path).stem)}__{source.sha256[:12]}"
        )
        source_rows: list[dict[str, Any]] = []
        for turn in _split_turns(turns, config.speaker_sorting.maximum_segment_seconds):
            duration = turn.end - turn.start
            if duration < config.speaker_sorting.minimum_segment_seconds:
                ignored_short_turns += 1
                continue
            has_overlap = _has_other_speaker_overlap(turn, turns)
            speaker = _safe_component(turn.speaker)
            speaker_dir = segment_run_root / source_key / speaker
            sequence = 1 + sum(
                row["source_sha256"] == source.sha256 and row["speaker_label"] == turn.speaker
                for row in source_rows
            )
            destination = speaker_dir / f"segment_{sequence:04d}.wav"
            actual_start, actual_end = _write_turn_audio(
                decoded.samples,
                decoded.sample_rate,
                turn,
                destination,
                padding_seconds=config.speaker_sorting.segment_padding_seconds,
            )
            relative = destination.relative_to(config.paths.speaker_segments_dir).as_posix()
            row = {
                "source_relative_path": source.relative_path,
                "source_sha256": source.sha256,
                "speaker_label": turn.speaker,
                "start_seconds": round(actual_start, 6),
                "end_seconds": round(actual_end, 6),
                "duration_seconds": round(actual_end - actual_start, 6),
                "has_other_speaker_overlap": has_overlap,
                "segment_relative_path": relative,
                "segment_sha256": sha256_file(destination),
            }
            source_rows.append(row)
            segment_rows.append(row)
        if not source_rows:
            raise SpeakerSortingError(
                f"All diarized turns were shorter than the configured minimum for {source.path.name}"
            )
        cluster_rows.extend(
            _summarize_source_clusters(
                source,
                source_rows,
                config,
                embedding_inference,
                target_embedding,
            )
        )

    segment_manifest = _write_csv_atomic(
        manifest_run_root / "speaker_segments.csv",
        segment_rows,
        (
            "source_relative_path",
            "source_sha256",
            "speaker_label",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "has_other_speaker_overlap",
            "segment_relative_path",
            "segment_sha256",
        ),
    )
    review_queue = _write_csv_atomic(
        manifest_run_root / "speaker_review.csv", cluster_rows, _REVIEW_FIELDS
    )
    summary = _write_json_atomic(
        manifest_run_root / "speaker_sorting_summary.json",
        {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "run_id": run_id,
            "source_root": str(config.paths.mixed_speaker_audio_dir),
            "segment_root": str(segment_run_root),
            "source_count": len(sources.files),
            "speaker_cluster_count": len(cluster_rows),
            "segment_count": len(segment_rows),
            "ignored_short_turn_count": ignored_short_turns,
            "reference_file_count": len(references.files),
            "review_queue": str(review_queue),
            "segment_manifest": str(segment_manifest),
            "next_step": (
                "Listen to sample_files, fill every decision with TARGET/OTHER/REJECT, "
                "then run apply-speaker-review. Similarity is ranking evidence only."
            ),
        },
    )
    return SpeakerSortingResult(
        run_id=run_id,
        source_count=len(sources.files),
        speaker_cluster_count=len(cluster_rows),
        segment_count=len(segment_rows),
        review_queue_path=review_queue,
        segment_manifest_path=segment_manifest,
        summary_path=summary,
    )


def apply_speaker_review(
    config: AppConfig, review_queue: Path
) -> SpeakerReviewResult:
    """Copy reviewed target-speaker clips to the curation input without deletion."""

    queue = Path(review_queue).expanduser().resolve()
    if not queue.is_file():
        raise SpeakerSortingError(f"Speaker review queue does not exist: {queue}")
    with queue.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SpeakerSortingError(f"Speaker review queue is empty: {queue}")
    missing = set(_REVIEW_FIELDS).difference(rows[0])
    if missing:
        raise SpeakerSortingError(
            "Speaker review queue is missing columns: " + ", ".join(sorted(missing))
        )

    segment_root = config.paths.speaker_segments_dir.resolve()
    selected_root = config.paths.speaker_selected_audio_dir.resolve()
    target_clusters = copied = 0
    copied_seconds = 0.0
    applied: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        decision = str(row.get("decision", "")).strip().upper()
        if decision not in _VALID_DECISIONS:
            raise SpeakerSortingError(
                f"CSV line {line_number} must contain TARGET, OTHER, or REJECT"
            )
        paths = _split_pipe_field(row.get("segment_relative_paths"))
        hashes = _split_pipe_field(row.get("segment_sha256s"))
        durations = _split_pipe_field(row.get("segment_durations_seconds"))
        if len(paths) != len(hashes) or len(paths) != len(durations):
            raise SpeakerSortingError(
                f"CSV line {line_number} has inconsistent segment path/hash/duration values"
            )
        copied_paths: list[str] = []
        if decision == "TARGET":
            if not paths:
                raise SpeakerSortingError(
                    f"CSV line {line_number} has no non-overlapping segment to promote"
                )
            target_clusters += 1
            for relative, expected_hash, duration in zip(paths, hashes, durations):
                source = _safe_join(segment_root, relative)
                if not source.is_file():
                    raise SpeakerSortingError(f"Reviewed speaker segment is missing: {source}")
                actual_hash = sha256_file(source)
                if actual_hash.lower() != expected_hash.lower():
                    raise SpeakerSortingError(
                        f"Speaker segment changed after review: {relative}"
                    )
                destination = _safe_join(selected_root, relative)
                _copy_immutable(source, destination, actual_hash)
                copied += 1
                copied_seconds += float(duration)
                copied_paths.append(str(destination))
        applied.append(
            {
                "source_relative_path": row["source_relative_path"],
                "source_sha256": row["source_sha256"],
                "speaker_label": row["speaker_label"],
                "decision": decision,
                "copied_paths": copied_paths,
            }
        )
    if target_clusters == 0:
        raise SpeakerSortingError(
            "Review contains no TARGET speaker cluster; nothing was copied"
        )
    summary = _write_json_atomic(
        queue.parent / "applied_speaker_review.json",
        {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "review_queue": str(queue),
            "selected_root": str(selected_root),
            "target_cluster_count": target_clusters,
            "copied_segment_count": copied,
            "copied_duration_minutes": round(copied_seconds / 60.0, 6),
            "items": applied,
            "next_step": "Run prepare-data; copied clips remain separate from originals.",
        },
    )
    return SpeakerReviewResult(
        target_cluster_count=target_clusters,
        copied_segment_count=copied,
        copied_duration_minutes=copied_seconds / 60.0,
        summary_path=summary,
    )


def _load_pyannote(config: AppConfig) -> Tuple[Any, Any]:
    try:
        import torch
        from pyannote.audio import Inference, Model, Pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SpeakerSortingError(
            "Speaker sorting needs the optional pyannote environment. Install it with "
            "scripts\\setup_speaker_venv.bat, then run this command with "
            ".speaker_venv\\Scripts\\python.exe."
        ) from exc

    settings = config.speaker_sorting
    token = os.environ.get(settings.token_environment_variable)
    pipeline_kwargs = {"token": token} if token else {}
    try:
        pipeline = Pipeline.from_pretrained(settings.diarization_model, **pipeline_kwargs)
        model = Model.from_pretrained(settings.embedding_model, **pipeline_kwargs)
    except Exception as exc:  # pragma: no cover - network/model runtime
        raise SpeakerSortingError(
            "Unable to load pyannote models. Accept the Community-1 model conditions, "
            f"set {settings.token_environment_variable}, and retry: {exc}"
        ) from exc
    inference = Inference(model, window="whole")
    if settings.use_gpu:
        if not torch.cuda.is_available():
            raise SpeakerSortingError(
                "speaker_sorting.use_gpu is true, but CUDA is unavailable in this environment"
            )
        device = torch.device(f"cuda:{settings.gpu_id}")
        pipeline.to(device)
        inference.to(device)
    return pipeline, inference


def _diarization_kwargs(
    config: AppConfig,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict[str, int]:
    settings = config.speaker_sorting
    if num_speakers is not None:
        return {"num_speakers": num_speakers}
    minimum = min_speakers if min_speakers is not None else settings.minimum_speakers
    maximum = max_speakers if max_speakers is not None else settings.maximum_speakers
    result: dict[str, int] = {}
    if minimum is not None:
        result["min_speakers"] = minimum
    if maximum is not None:
        result["max_speakers"] = maximum
    return result


def _iter_diarization_turns(output: Any) -> Iterable[SpeakerTurn]:
    annotation = getattr(output, "speaker_diarization", output)
    if hasattr(annotation, "itertracks"):
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            yield SpeakerTurn(float(segment.start), float(segment.end), str(speaker))
        return
    for item in annotation:
        if not isinstance(item, Sequence) or len(item) < 2:
            raise SpeakerSortingError("Unsupported pyannote diarization output structure")
        segment, speaker = item[0], item[-1]
        yield SpeakerTurn(float(segment.start), float(segment.end), str(speaker))


def _normalise_turns(
    turns: Iterable[SpeakerTurn], *, merge_gap_seconds: float
) -> Tuple[SpeakerTurn, ...]:
    cleaned = sorted(
        (
            SpeakerTurn(max(0.0, item.start), item.end, item.speaker)
            for item in turns
            if math.isfinite(item.start)
            and math.isfinite(item.end)
            and item.end > max(0.0, item.start)
            and item.speaker.strip()
        ),
        key=lambda item: (item.start, item.end, item.speaker.casefold()),
    )
    merged: list[SpeakerTurn] = []
    for turn in cleaned:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.speaker == turn.speaker
            and turn.start - previous.end <= merge_gap_seconds
        ):
            merged[-1] = SpeakerTurn(previous.start, max(previous.end, turn.end), turn.speaker)
        else:
            merged.append(turn)
    return tuple(merged)


def _split_turns(
    turns: Sequence[SpeakerTurn], maximum_seconds: float
) -> Tuple[SpeakerTurn, ...]:
    result: list[SpeakerTurn] = []
    for turn in turns:
        start = turn.start
        while turn.end - start > maximum_seconds:
            result.append(SpeakerTurn(start, start + maximum_seconds, turn.speaker))
            start += maximum_seconds
        if turn.end > start:
            result.append(SpeakerTurn(start, turn.end, turn.speaker))
    return tuple(result)


def _has_other_speaker_overlap(turn: SpeakerTurn, all_turns: Sequence[SpeakerTurn]) -> bool:
    return any(
        other.speaker != turn.speaker
        and min(turn.end, other.end) - max(turn.start, other.start) > 0.05
        for other in all_turns
    )


def _write_turn_audio(
    samples: np.ndarray,
    sample_rate: int,
    turn: SpeakerTurn,
    destination: Path,
    *,
    padding_seconds: float,
) -> Tuple[float, float]:
    total_seconds = samples.shape[0] / float(sample_rate)
    start = max(0.0, turn.start - padding_seconds)
    end = min(total_seconds, turn.end + padding_seconds)
    first = max(0, int(round(start * sample_rate)))
    last = min(samples.shape[0], int(round(end * sample_rate)))
    if last <= first:
        raise SpeakerSortingError(f"Diarized turn contains no decodable audio: {turn}")
    write_wav(destination, samples[first:last], sample_rate, subtype="PCM_24")
    return first / float(sample_rate), last / float(sample_rate)


def _summarize_source_clusters(
    source: AudioFile,
    rows: Sequence[Mapping[str, Any]],
    config: AppConfig,
    inference: Any,
    target_embedding: Optional[np.ndarray],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["speaker_label"]), []).append(row)
    scores: dict[str, Optional[float]] = {}
    for label, group in groups.items():
        usable = [item for item in group if not bool(item["has_other_speaker_overlap"])]
        scores[label] = (
            _cluster_similarity(
                inference,
                usable,
                config.paths.speaker_segments_dir,
                target_embedding,
                config.speaker_sorting.maximum_embedding_segments_per_speaker,
            )
            if target_embedding is not None and usable
            else None
        )
    ranked = sorted(
        groups,
        key=lambda label: (
            scores[label] is None,
            -(scores[label] if scores[label] is not None else -2.0),
            label.casefold(),
        ),
    )
    best = scores[ranked[0]] if ranked else None
    second = scores[ranked[1]] if len(ranked) > 1 else None
    margin = (
        best - second if best is not None and second is not None else None
    )
    records: list[dict[str, Any]] = []
    for rank, label in enumerate(ranked, start=1):
        group = sorted(groups[label], key=lambda item: float(item["start_seconds"]))
        usable = [item for item in group if not bool(item["has_other_speaker_overlap"])]
        score = scores[label]
        recommendation = _recommend_cluster(score, rank, margin, config)
        records.append(
            {
                "source_relative_path": source.relative_path,
                "source_sha256": source.sha256,
                "speaker_label": label,
                "segment_count": len(usable),
                "usable_duration_seconds": round(
                    sum(float(item["duration_seconds"]) for item in usable), 6
                ),
                "overlap_segment_count": len(group) - len(usable),
                "target_similarity": "" if score is None else round(score, 6),
                "rank_in_source": rank,
                "score_margin": "" if rank != 1 or margin is None else round(margin, 6),
                "recommended_action": recommendation,
                "sample_files": " | ".join(
                    str(item["segment_relative_path"]) for item in usable[:3]
                ),
                "segment_relative_paths": " | ".join(
                    str(item["segment_relative_path"]) for item in usable
                ),
                "segment_sha256s": " | ".join(
                    str(item["segment_sha256"]) for item in usable
                ),
                "segment_durations_seconds": " | ".join(
                    str(item["duration_seconds"]) for item in usable
                ),
                "decision": "",
                "reviewer_notes": "",
            }
        )
    return records


def _recommend_cluster(
    score: Optional[float], rank: int, margin: Optional[float], config: AppConfig
) -> str:
    if score is None:
        return "LISTEN_AND_LABEL"
    settings = config.speaker_sorting
    if (
        rank == 1
        and score >= settings.target_similarity_threshold
        and (margin is None or margin >= settings.target_similarity_margin)
    ):
        return "TARGET_CANDIDATE_REVIEW_REQUIRED"
    if rank == 1:
        return "REVIEW_BEST_MATCH"
    return "OTHER_CANDIDATE"


def _mean_file_embeddings(inference: Any, files: Sequence[AudioFile]) -> np.ndarray:
    vectors = [_normalise_embedding(inference(str(item.path))) for item in files]
    return _normalise_embedding(np.mean(np.stack(vectors), axis=0))


def _cluster_similarity(
    inference: Any,
    rows: Sequence[Mapping[str, Any]],
    segment_root: Path,
    target_embedding: np.ndarray,
    maximum_segments: int,
) -> float:
    selected = sorted(
        rows, key=lambda item: float(item["duration_seconds"]), reverse=True
    )[:maximum_segments]
    vectors = [
        _normalise_embedding(
            inference(str(_safe_join(segment_root, str(item["segment_relative_path"]))))
        )
        for item in selected
    ]
    cluster = _normalise_embedding(np.mean(np.stack(vectors), axis=0))
    return float(np.clip(np.dot(cluster, target_embedding), -1.0, 1.0))


def _normalise_embedding(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise SpeakerSortingError("Speaker embedding is empty or non-finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise SpeakerSortingError("Speaker embedding has zero magnitude")
    return vector / norm


def _validate_speaker_bounds(
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    for name, value in (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ):
        if value is not None and value < 1:
            raise SpeakerSortingError(f"{name} must be at least 1")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise SpeakerSortingError("min_speakers cannot exceed max_speakers")


def _write_csv_atomic(
    destination: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(destination))
    return destination


def _write_json_atomic(destination: Path, payload: Mapping[str, Any]) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(destination))
    return destination


def _copy_immutable(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == expected_hash:
            return
        raise SpeakerSortingError(f"Refusing to overwrite a different file: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_hash:
            raise SpeakerSortingError(f"Copied segment failed hash validation: {source}")
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_join(root: Path, relative: str) -> Path:
    if not relative.strip():
        raise SpeakerSortingError("A required relative path is blank")
    base = root.resolve()
    candidate = (base / Path(relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SpeakerSortingError(f"Path escapes the configured root: {relative}") from exc
    return candidate


def _safe_component(value: str) -> str:
    cleaned = _WINDOWS_INVALID.sub("_", value).strip(" ._")
    return cleaned or "speaker"


def _split_pipe_field(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
