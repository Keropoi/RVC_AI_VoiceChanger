"""Silence-aware internal slicer used when an RVC-native slicer is unavailable."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .decoder import decode_audio, write_wav
from .quality import (
    AudioQualityResult,
    QualityStatus,
    QualityThresholds,
    analyze_samples,
    db_to_amplitude,
)


@dataclass(frozen=True)
class SlicingConfig:
    """Settings for the built-in, silence-aware fallback slicer."""

    backend: str = "internal"
    minimum_segment_seconds: float = 1.0
    preferred_segment_seconds: float = 4.0
    maximum_segment_seconds: float = 12.0
    silence_threshold_dbfs: float = -42.0
    minimum_silence_duration_ms: float = 250.0
    segment_padding_ms: float = 80.0
    output_subtype: str = "PCM_16"

    def __post_init__(self) -> None:
        if self.minimum_segment_seconds <= 0:
            raise ValueError("minimum_segment_seconds must be positive")
        if self.preferred_segment_seconds < self.minimum_segment_seconds:
            raise ValueError("preferred_segment_seconds must be at least the minimum")
        if self.maximum_segment_seconds < self.preferred_segment_seconds:
            raise ValueError("maximum_segment_seconds must be at least the preferred duration")
        if self.minimum_silence_duration_ms < 0 or self.segment_padding_ms < 0:
            raise ValueError("silence duration and padding cannot be negative")

    @classmethod
    def from_config(cls, config: Optional[object]) -> "SlicingConfig":
        """Coerce a mapping/Pydantic/dataclass-like slicing section."""

        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        section = _nested_config(config, "slicing")
        values: Dict[str, Any] = {}
        for field in fields(cls):
            value = _config_value(section, field.name)
            if value is not _MISSING:
                values[field.name] = value
        return cls(**values)


@dataclass(frozen=True)
class SegmentRecord:
    """Traceability and post-slice quality values for one output WAV."""

    segment_file: Path
    source_file: Path
    source_start_seconds: float
    source_end_seconds: float
    duration_seconds: float
    lufs: Optional[float]
    peak_dbfs: Optional[float]
    silence_ratio: Optional[float]
    status: QualityStatus
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        """Return a manifest-safe representation."""

        return {
            "segment_file": self.segment_file.name,
            "source_file": str(self.source_file),
            "source_start_seconds": self.source_start_seconds,
            "source_end_seconds": self.source_end_seconds,
            "duration_seconds": self.duration_seconds,
            "lufs": self.lufs,
            "peak_dbfs": self.peak_dbfs,
            "silence_ratio": self.silence_ratio,
            "status": self.status.value,
            "reasons": " | ".join(self.reasons),
        }


@dataclass(frozen=True)
class SliceResult:
    """Outputs and manifest location produced for one source file."""

    source_path: Path
    segments: Tuple[SegmentRecord, ...]
    manifest_path: Path

    @property
    def accepted_segments(self) -> Tuple[SegmentRecord, ...]:
        """Segments eligible for feature extraction/training."""

        return tuple(item for item in self.segments if item.status is not QualityStatus.FAIL)


class AudioSlicingError(RuntimeError):
    """Raised when an internal slice cannot be produced safely."""


def slice_audio(
    source: Union[str, Path],
    output_directory: Union[str, Path],
    config: Optional[object] = None,
    *,
    quality_thresholds: Optional[object] = None,
    ffmpeg_executable: Union[str, Path] = "ffmpeg",
    overwrite: bool = False,
) -> SliceResult:
    """Slice one decoded file near silence and write a segment manifest.

    The internal implementation may be called regardless of ``backend``; backend
    selection belongs to the pipeline/RVC adapter.  This keeps the fallback useful
    and independently testable.
    """

    settings = SlicingConfig.from_config(config)
    decoded = decode_audio(source, ffmpeg_executable=ffmpeg_executable)
    data = np.nan_to_num(decoded.samples, nan=0.0, posinf=0.0, neginf=0.0)
    if data.shape[0] == 0:
        raise AudioSlicingError("Decoded source contains no audio frames")

    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = find_slice_boundaries(data, decoded.sample_rate, settings)
    if len(boundaries) < 2:
        raise AudioSlicingError("Unable to derive a non-empty segment boundary")

    limits = _segment_quality_thresholds(quality_thresholds, settings)
    padding_frames = int(round(decoded.sample_rate * settings.segment_padding_ms / 1000.0))
    stable_stem = _safe_stem(decoded.source_path.stem)
    records: List[SegmentRecord] = []
    for index, (core_start, core_stop) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        start = max(0, core_start - padding_frames)
        stop = min(data.shape[0], core_stop + padding_frames)
        segment = data[start:stop]
        if segment.shape[0] == 0:
            continue
        filename = "{}__seg_{:04d}.wav".format(stable_stem, index)
        destination = output_dir / filename
        write_wav(
            destination,
            segment,
            decoded.sample_rate,
            subtype=settings.output_subtype,
            overwrite=overwrite,
        )
        quality = analyze_samples(
            segment,
            decoded.sample_rate,
            limits,
            file_path=destination,
            relative_path=filename,
            codec_format="WAV/{}".format(settings.output_subtype),
            pcm_bit_depth=_subtype_depth(settings.output_subtype),
        )
        records.append(
            _record_from_quality(
                decoded.source_path,
                destination,
                start,
                stop,
                decoded.sample_rate,
                quality,
            )
        )

    if not records:
        raise AudioSlicingError("Internal slicer produced no non-empty segments")
    manifest = write_segments_manifest(records, output_dir / "segments_manifest.csv")
    return SliceResult(decoded.source_path, tuple(records), manifest)


def find_slice_boundaries(
    samples: np.ndarray,
    sample_rate: int,
    config: Optional[object] = None,
) -> Tuple[int, ...]:
    """Return stable frame boundaries, preferring sufficiently long silences."""

    settings = SlicingConfig.from_config(config)
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("samples must contain at least one frame")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    total_frames = int(data.shape[0])
    minimum = max(1, int(round(settings.minimum_segment_seconds * sample_rate)))
    preferred = max(minimum, int(round(settings.preferred_segment_seconds * sample_rate)))
    maximum = max(preferred, int(round(settings.maximum_segment_seconds * sample_rate)))
    silence_candidates, frame_energy = _silence_candidates(data, sample_rate, settings)

    boundaries = [0]
    current = 0
    while total_frames - current > maximum:
        low = current + minimum
        high = min(current + maximum, total_frames - minimum)
        target = min(current + preferred, high)
        eligible = [candidate for candidate in silence_candidates if low <= candidate <= high]
        if eligible:
            cut = min(eligible, key=lambda candidate: (abs(candidate - target), candidate))
        else:
            cut = _quiet_fallback_cut(frame_energy, target, low, high, sample_rate)
        if cut <= current:
            cut = min(current + maximum, total_frames)
        boundaries.append(cut)
        current = cut

    if total_frames > boundaries[-1]:
        boundaries.append(total_frames)
    if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < minimum:
        boundaries.pop(-2)
    return tuple(boundaries)


def write_segments_manifest(
    records: Sequence[SegmentRecord], path: Union[str, Path]
) -> Path:
    """Write the required stable CSV manifest for produced segments."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "segment_file",
        "source_file",
        "source_start_seconds",
        "source_end_seconds",
        "duration_seconds",
        "lufs",
        "peak_dbfs",
        "silence_ratio",
        "status",
        "reasons",
    ]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
    return destination


def _silence_candidates(
    samples: np.ndarray, sample_rate: int, settings: SlicingConfig
) -> Tuple[List[int], np.ndarray]:
    mono = np.mean(samples, axis=1)
    frame_length = max(1, int(round(sample_rate * 0.020)))
    frame_rms: List[float] = []
    starts = list(range(0, mono.size, frame_length))
    for start in starts:
        frame = mono[start : start + frame_length]
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        frame_rms.append(rms)
    energy = np.asarray(frame_rms, dtype=np.float64)
    silent = energy <= db_to_amplitude(settings.silence_threshold_dbfs)
    minimum_silent_frames = max(
        1,
        int(math.ceil(settings.minimum_silence_duration_ms / 20.0)),
    )
    candidates: List[int] = []
    run_start: Optional[int] = None
    for index, is_silent in enumerate(np.append(silent, False)):
        if is_silent and run_start is None:
            run_start = index
        elif not is_silent and run_start is not None:
            if index - run_start >= minimum_silent_frames:
                midpoint_frame = (run_start + index) // 2
                candidates.append(min(midpoint_frame * frame_length, samples.shape[0]))
            run_start = None
    return candidates, energy


def _quiet_fallback_cut(
    frame_energy: np.ndarray,
    target: int,
    low: int,
    high: int,
    sample_rate: int,
) -> int:
    if high <= low or frame_energy.size == 0:
        return max(low, min(target, high))
    frame_length = max(1, int(round(sample_rate * 0.020)))
    search_radius = int(round(sample_rate * 0.75))
    search_low = max(low, target - search_radius)
    search_high = min(high, target + search_radius)
    first_index = max(0, search_low // frame_length)
    last_index = min(frame_energy.size, max(first_index + 1, search_high // frame_length + 1))
    local_index = int(np.argmin(frame_energy[first_index:last_index])) + first_index
    return max(low, min(local_index * frame_length, high))


def _segment_quality_thresholds(
    configured: Optional[object], settings: SlicingConfig
) -> QualityThresholds:
    base = QualityThresholds.from_config(configured)
    values = {
        field.name: getattr(base, field.name)
        for field in fields(QualityThresholds)
    }
    values["minimum_file_duration_seconds"] = min(
        base.minimum_file_duration_seconds, settings.minimum_segment_seconds
    )
    values["maximum_file_duration_seconds"] = max(
        base.maximum_file_duration_seconds,
        settings.maximum_segment_seconds + (2.0 * settings.segment_padding_ms / 1000.0),
    )
    values["silence_threshold_dbfs"] = settings.silence_threshold_dbfs
    return QualityThresholds(**values)


def _record_from_quality(
    source: Path,
    destination: Path,
    start: int,
    stop: int,
    sample_rate: int,
    quality: AudioQualityResult,
) -> SegmentRecord:
    return SegmentRecord(
        segment_file=destination,
        source_file=source,
        source_start_seconds=start / float(sample_rate),
        source_end_seconds=stop / float(sample_rate),
        duration_seconds=(stop - start) / float(sample_rate),
        lufs=quality.integrated_lufs,
        peak_dbfs=quality.sample_peak_dbfs,
        silence_ratio=quality.silence_ratio,
        status=quality.status,
        reasons=quality.reasons,
    )


def _safe_stem(stem: str) -> str:
    # Retain Unicode while replacing characters invalid in Windows filenames.
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" ._")
    return cleaned or "audio"


def _subtype_depth(subtype: str) -> Optional[int]:
    match = re.search(r"(8|16|24|32|64)", subtype)
    return int(match.group(1)) if match else None


class _Missing:
    pass


_MISSING = _Missing()


def _nested_config(config: object, name: str) -> object:
    value = _config_value(config, name)
    return config if value is _MISSING else value


def _config_value(config: object, name: str) -> object:
    if isinstance(config, Mapping):
        return config.get(name, _MISSING)
    return getattr(config, name, _MISSING)
