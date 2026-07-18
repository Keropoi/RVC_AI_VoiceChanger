"""Quantitative audio quality analysis and PASS/WARNING/FAIL classification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal

from .decoder import AudioDecodeError, DecodedAudio, decode_audio

try:
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover - only an incomplete environment
    pyln = None  # type: ignore[assignment]


_EPSILON = float(np.finfo(np.float64).tiny)


class QualityStatus(str, Enum):
    """Training eligibility outcome for an audio input."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True)
class QualityIssue:
    """A machine-readable quality finding with a user-facing explanation."""

    code: str
    severity: QualityStatus
    message: str

    def to_dict(self) -> Dict[str, str]:
        """Serialize without leaking enum implementation details."""

        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds used by the quality classifier."""

    minimum_file_duration_seconds: float = 0.5
    maximum_file_duration_seconds: float = 600.0
    minimum_total_accepted_duration_minutes: float = 5.0
    recommended_total_duration_minutes: float = 10.0
    minimum_sample_rate: int = 22050
    target_integrated_lufs: float = -20.0
    acceptable_lufs_min: float = -32.0
    acceptable_lufs_max: float = -12.0
    maximum_sample_peak_dbfs: float = -0.1
    maximum_clipping_ratio: float = 0.0001
    maximum_dc_offset: float = 0.02
    silence_threshold_dbfs: float = -50.0
    maximum_silence_ratio: float = 0.45
    minimum_estimated_snr_db: float = 15.0
    recommended_estimated_snr_db: float = 25.0
    fail_on_corrupt_file: bool = False
    fail_on_insufficient_total_duration: bool = True
    almost_silent_ratio: float = 0.98
    short_segment_seconds: float = 0.30
    analysis_frame_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.minimum_file_duration_seconds < 0:
            raise ValueError("minimum_file_duration_seconds cannot be negative")
        if self.maximum_file_duration_seconds <= self.minimum_file_duration_seconds:
            raise ValueError("maximum_file_duration_seconds must exceed the minimum")
        if self.minimum_sample_rate <= 0:
            raise ValueError("minimum_sample_rate must be positive")
        if self.acceptable_lufs_min >= self.acceptable_lufs_max:
            raise ValueError("acceptable_lufs_min must be below acceptable_lufs_max")
        for name in ("maximum_clipping_ratio", "maximum_silence_ratio", "almost_silent_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must be between zero and one".format(name))
        if self.analysis_frame_ms <= 0:
            raise ValueError("analysis_frame_ms must be positive")

    @classmethod
    def from_config(cls, config: Optional[object]) -> "QualityThresholds":
        """Coerce a mapping/Pydantic/dataclass-like quality config."""

        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        section = _nested_config(config, "quality")
        values: Dict[str, Any] = {}
        for field in fields(cls):
            value = _config_value(section, field.name)
            if value is not _MISSING:
                values[field.name] = value
        return cls(**values)


@dataclass(frozen=True)
class AudioQualityResult:
    """All measured metrics and the resulting per-file quality decision."""

    file_path: Path
    relative_path: str
    sha256: Optional[str]
    decodable: bool
    duration_seconds: Optional[float]
    sample_rate: Optional[int]
    channels: Optional[int]
    codec_format: Optional[str]
    pcm_bit_depth: Optional[int]
    integrated_lufs: Optional[float]
    rms_dbfs: Optional[float]
    sample_peak: Optional[float]
    sample_peak_dbfs: Optional[float]
    true_peak_dbfs: Optional[float]
    crest_factor_db: Optional[float]
    clipping_sample_ratio: Optional[float]
    dc_offset: Optional[float]
    silence_ratio: Optional[float]
    short_segment_ratio: Optional[float]
    dynamic_range_db: Optional[float]
    estimated_noise_floor_dbfs: Optional[float]
    estimated_snr_db: Optional[float]
    non_finite_count: Optional[int]
    channel_difference_db: Optional[float]
    almost_all_silence: Optional[bool]
    status: QualityStatus
    issues: Tuple[QualityIssue, ...]
    decode_error: Optional[str] = None

    @property
    def reasons(self) -> Tuple[str, ...]:
        """Human-readable reasons suitable for logs and rejection CSVs."""

        return tuple(issue.message for issue in self.issues)

    @property
    def accepted_for_training(self) -> bool:
        """FAIL is the sole state excluded from downstream training."""

        return self.status is not QualityStatus.FAIL

    def to_dict(self) -> Dict[str, object]:
        """Convert values to a JSON/CSV-friendly dictionary."""

        payload = asdict(self)
        payload["file_path"] = str(self.file_path)
        payload["status"] = self.status.value
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class QualitySummary:
    """Dataset-level counts and duration checks."""

    total_files: int
    pass_files: int
    warning_files: int
    fail_files: int
    accepted_duration_seconds: float
    accepted_duration_minutes: float
    status: QualityStatus
    issues: Tuple[QualityIssue, ...]

    def to_dict(self) -> Dict[str, object]:
        """Return a serializable summary."""

        payload = asdict(self)
        payload["status"] = self.status.value
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


def analyze_audio(
    audio: object,
    thresholds: Optional[object] = None,
    *,
    root: Optional[Union[str, Path]] = None,
    ffmpeg_executable: Union[str, Path] = "ffmpeg",
    allow_ffmpeg: bool = True,
) -> AudioQualityResult:
    """Decode and audit a path or discovery ``AudioFile``-like object."""

    path, relative_path, digest = _input_identity(audio, root)
    limits = QualityThresholds.from_config(thresholds)
    try:
        decoded = decode_audio(
            path,
            ffmpeg_executable=ffmpeg_executable,
            allow_ffmpeg=allow_ffmpeg,
        )
    except (AudioDecodeError, OSError, RuntimeError, ValueError) as exc:
        issue = QualityIssue("DECODE_FAILED", QualityStatus.FAIL, str(exc))
        return AudioQualityResult(
            file_path=path,
            relative_path=relative_path,
            sha256=digest,
            decodable=False,
            duration_seconds=None,
            sample_rate=None,
            channels=None,
            codec_format=None,
            pcm_bit_depth=None,
            integrated_lufs=None,
            rms_dbfs=None,
            sample_peak=None,
            sample_peak_dbfs=None,
            true_peak_dbfs=None,
            crest_factor_db=None,
            clipping_sample_ratio=None,
            dc_offset=None,
            silence_ratio=None,
            short_segment_ratio=None,
            dynamic_range_db=None,
            estimated_noise_floor_dbfs=None,
            estimated_snr_db=None,
            non_finite_count=None,
            channel_difference_db=None,
            almost_all_silence=None,
            status=QualityStatus.FAIL,
            issues=(issue,),
            decode_error=str(exc),
        )
    return analyze_samples(
        decoded.samples,
        decoded.sample_rate,
        thresholds=limits,
        file_path=path,
        relative_path=relative_path,
        sha256=digest,
        codec_format=_codec_description(decoded),
        pcm_bit_depth=decoded.bit_depth,
    )


def analyze_samples(
    samples: np.ndarray,
    sample_rate: int,
    thresholds: Optional[object] = None,
    *,
    file_path: Union[str, Path] = "<memory>",
    relative_path: Optional[str] = None,
    sha256: Optional[str] = None,
    codec_format: Optional[str] = None,
    pcm_bit_depth: Optional[int] = None,
) -> AudioQualityResult:
    """Measure an already decoded signal and classify it."""

    limits = QualityThresholds.from_config(thresholds)
    path = Path(file_path)
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2:
        raise ValueError("samples must have shape (frames,) or (frames, channels)")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("samples must not be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    original = data.astype(np.float64, copy=False)
    finite = np.isfinite(original)
    non_finite_count = int(original.size - np.count_nonzero(finite))
    clean = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
    mono = np.mean(clean, axis=1)
    frame_dbfs = frame_rms_dbfs(
        mono,
        sample_rate,
        frame_ms=limits.analysis_frame_ms,
    )

    duration = clean.shape[0] / float(sample_rate)
    rms_dbfs = amplitude_to_db(float(np.sqrt(np.mean(np.square(clean), dtype=np.float64))))
    sample_peak = float(np.max(np.abs(clean)))
    sample_peak_dbfs = amplitude_to_db(sample_peak)
    true_peak_dbfs = approximate_true_peak_dbfs(clean, sample_rate)
    crest_factor_db = _finite_difference(sample_peak_dbfs, rms_dbfs)
    clipping_amplitude = db_to_amplitude(limits.maximum_sample_peak_dbfs)
    clipping_ratio = float(np.count_nonzero(np.abs(clean) >= clipping_amplitude) / clean.size)
    dc_offset = float(np.max(np.abs(np.mean(clean, axis=0))))
    silence_ratio = float(np.mean(frame_dbfs <= limits.silence_threshold_dbfs))
    short_ratio = _short_segment_ratio(
        frame_dbfs > limits.silence_threshold_dbfs,
        frame_ms=limits.analysis_frame_ms,
        short_segment_seconds=limits.short_segment_seconds,
    )
    dynamic_range, noise_floor, snr = _level_distribution(frame_dbfs)
    integrated_lufs = measure_integrated_lufs(mono, sample_rate)
    channel_difference = _channel_difference_db(clean)
    almost_silent = bool(
        silence_ratio >= limits.almost_silent_ratio
        or sample_peak_dbfs <= limits.silence_threshold_dbfs
    )

    issues = _classify(
        limits=limits,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=int(clean.shape[1]),
        integrated_lufs=integrated_lufs,
        sample_peak_dbfs=sample_peak_dbfs,
        clipping_ratio=clipping_ratio,
        dc_offset=dc_offset,
        silence_ratio=silence_ratio,
        snr_db=snr,
        non_finite_count=non_finite_count,
        almost_silent=almost_silent,
        channel_difference_db=channel_difference,
    )
    status = _highest_status(issues)
    return AudioQualityResult(
        file_path=path,
        relative_path=relative_path or path.name,
        sha256=sha256,
        decodable=True,
        duration_seconds=duration,
        sample_rate=int(sample_rate),
        channels=int(clean.shape[1]),
        codec_format=codec_format,
        pcm_bit_depth=pcm_bit_depth,
        integrated_lufs=_finite_or_none(integrated_lufs),
        rms_dbfs=_finite_or_none(rms_dbfs),
        sample_peak=sample_peak,
        sample_peak_dbfs=_finite_or_none(sample_peak_dbfs),
        true_peak_dbfs=_finite_or_none(true_peak_dbfs),
        crest_factor_db=_finite_or_none(crest_factor_db),
        clipping_sample_ratio=clipping_ratio,
        dc_offset=dc_offset,
        silence_ratio=silence_ratio,
        short_segment_ratio=short_ratio,
        dynamic_range_db=_finite_or_none(dynamic_range),
        estimated_noise_floor_dbfs=_finite_or_none(noise_floor),
        estimated_snr_db=_finite_or_none(snr),
        non_finite_count=non_finite_count,
        channel_difference_db=_finite_or_none(channel_difference),
        almost_all_silence=almost_silent,
        status=status,
        issues=tuple(issues),
    )


def audit_audio_files(
    files: Iterable[object],
    thresholds: Optional[object] = None,
    *,
    root: Optional[Union[str, Path]] = None,
    ffmpeg_executable: Union[str, Path] = "ffmpeg",
    allow_ffmpeg: bool = True,
) -> Tuple[AudioQualityResult, ...]:
    """Audit all inputs, retaining a FAIL row instead of aborting on corruption."""

    return tuple(
        analyze_audio(
            item,
            thresholds,
            root=root,
            ffmpeg_executable=ffmpeg_executable,
            allow_ffmpeg=allow_ffmpeg,
        )
        for item in files
    )


def check_audio_quality(
    audio: object,
    thresholds: Optional[object] = None,
    **kwargs: object,
) -> AudioQualityResult:
    """Readable alias for :func:`analyze_audio`."""

    return analyze_audio(audio, thresholds, **kwargs)


def summarize_quality(
    results: Sequence[AudioQualityResult], thresholds: Optional[object] = None
) -> QualitySummary:
    """Summarize file states and enforce configured total accepted duration."""

    limits = QualityThresholds.from_config(thresholds)
    counts = {status: 0 for status in QualityStatus}
    accepted_seconds = 0.0
    for result in results:
        counts[result.status] += 1
        if result.accepted_for_training and result.duration_seconds is not None:
            accepted_seconds += result.duration_seconds

    accepted_minutes = accepted_seconds / 60.0
    issues: List[QualityIssue] = []
    if accepted_minutes < limits.minimum_total_accepted_duration_minutes:
        severity = (
            QualityStatus.FAIL
            if limits.fail_on_insufficient_total_duration
            else QualityStatus.WARNING
        )
        issues.append(
            QualityIssue(
                "INSUFFICIENT_TOTAL_DURATION",
                severity,
                "Accepted audio totals {:.2f} min; at least {:.2f} min is required".format(
                    accepted_minutes, limits.minimum_total_accepted_duration_minutes
                ),
            )
        )
    elif accepted_minutes < limits.recommended_total_duration_minutes:
        issues.append(
            QualityIssue(
                "BELOW_RECOMMENDED_TOTAL_DURATION",
                QualityStatus.WARNING,
                "Accepted audio totals {:.2f} min; {:.2f} min is recommended".format(
                    accepted_minutes, limits.recommended_total_duration_minutes
                ),
            )
        )
    if counts[QualityStatus.FAIL] and not issues:
        issues.append(
            QualityIssue(
                "FILES_REJECTED",
                QualityStatus.WARNING,
                "{} input file(s) failed quality checks and will be excluded".format(
                    counts[QualityStatus.FAIL]
                ),
            )
        )
    return QualitySummary(
        total_files=len(results),
        pass_files=counts[QualityStatus.PASS],
        warning_files=counts[QualityStatus.WARNING],
        fail_files=counts[QualityStatus.FAIL],
        accepted_duration_seconds=accepted_seconds,
        accepted_duration_minutes=accepted_minutes,
        status=_highest_status(issues),
        issues=tuple(issues),
    )


def measure_integrated_lufs(samples: np.ndarray, sample_rate: int) -> float:
    """Measure BS.1770 integrated loudness, with a documented RMS fallback."""

    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    if data.size == 0 or not np.any(np.abs(data) > 0.0):
        return float("-inf")
    if pyln is not None and data.size >= int(0.4 * sample_rate):
        try:
            return float(pyln.Meter(sample_rate).integrated_loudness(data))
        except (ValueError, RuntimeError, OverflowError):
            pass
    rms = float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))
    return amplitude_to_db(rms) - 0.691


def approximate_true_peak_dbfs(samples: np.ndarray, sample_rate: int) -> float:
    """Approximate inter-sample true peak with 4x polyphase oversampling."""

    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.size == 0:
        return float("-inf")
    chunk_frames = max(sample_rate * 10, 1)
    overlap = min(max(sample_rate // 100, 16), max(data.shape[0] - 1, 0))
    maximum = 0.0
    start = 0
    while start < data.shape[0]:
        stop = min(data.shape[0], start + chunk_frames)
        chunk_start = max(0, start - overlap)
        chunk_stop = min(data.shape[0], stop + overlap)
        chunk = data[chunk_start:chunk_stop]
        if chunk.shape[0] > 1:
            oversampled = signal.resample_poly(chunk, up=4, down=1, axis=0)
            maximum = max(maximum, float(np.max(np.abs(oversampled))))
        else:
            maximum = max(maximum, float(np.max(np.abs(chunk))))
        start = stop
    return amplitude_to_db(maximum)


def frame_rms_dbfs(samples: np.ndarray, sample_rate: int, *, frame_ms: float = 50.0) -> np.ndarray:
    """Return overlapping frame RMS levels used by silence/noise metrics."""

    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return np.asarray([float("-inf")], dtype=np.float64)
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, frame_length // 2)
    if data.size < frame_length:
        data = np.pad(data, (0, frame_length - data.size))
    levels: List[float] = []
    for start in range(0, max(data.size - frame_length + 1, 1), hop_length):
        frame = data[start : start + frame_length]
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        levels.append(amplitude_to_db(rms))
    final_start = data.size - frame_length
    if final_start > 0 and final_start % hop_length:
        frame = data[final_start:]
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        levels.append(amplitude_to_db(rms))
    return np.asarray(levels, dtype=np.float64)


def amplitude_to_db(amplitude: float) -> float:
    """Convert full-scale linear amplitude to dBFS, preserving digital silence."""

    value = abs(float(amplitude))
    if value <= _EPSILON:
        return float("-inf")
    return 20.0 * math.log10(value)


def db_to_amplitude(dbfs: float) -> float:
    """Convert dBFS to a linear full-scale amplitude."""

    return 10.0 ** (float(dbfs) / 20.0)


def _classify(
    *,
    limits: QualityThresholds,
    duration_seconds: float,
    sample_rate: int,
    channels: int,
    integrated_lufs: float,
    sample_peak_dbfs: float,
    clipping_ratio: float,
    dc_offset: float,
    silence_ratio: float,
    snr_db: float,
    non_finite_count: int,
    almost_silent: bool,
    channel_difference_db: Optional[float],
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []

    def add(code: str, severity: QualityStatus, message: str) -> None:
        issues.append(QualityIssue(code, severity, message))

    if duration_seconds < limits.minimum_file_duration_seconds:
        add(
            "TOO_SHORT",
            QualityStatus.FAIL,
            "Duration {:.3f}s is below the {:.3f}s minimum".format(
                duration_seconds, limits.minimum_file_duration_seconds
            ),
        )
    elif duration_seconds > limits.maximum_file_duration_seconds:
        add(
            "TOO_LONG",
            QualityStatus.WARNING,
            "Duration {:.2f}s exceeds the {:.2f}s per-file recommendation".format(
                duration_seconds, limits.maximum_file_duration_seconds
            ),
        )
    if non_finite_count:
        add(
            "NON_FINITE_SAMPLES",
            QualityStatus.FAIL,
            "Decoded audio contains {} NaN or Infinity sample(s)".format(non_finite_count),
        )
    if almost_silent:
        add(
            "ALMOST_ALL_SILENCE",
            QualityStatus.FAIL,
            "Audio is almost entirely below {:.1f} dBFS".format(
                limits.silence_threshold_dbfs
            ),
        )
    elif silence_ratio > limits.maximum_silence_ratio:
        add(
            "EXCESSIVE_SILENCE",
            QualityStatus.WARNING,
            "Silence ratio {:.2%} exceeds {:.2%}".format(
                silence_ratio, limits.maximum_silence_ratio
            ),
        )
    if clipping_ratio > limits.maximum_clipping_ratio:
        add(
            "SEVERE_CLIPPING",
            QualityStatus.FAIL,
            "Clipping ratio {:.6%} exceeds {:.6%}".format(
                clipping_ratio, limits.maximum_clipping_ratio
            ),
        )
    elif sample_peak_dbfs > limits.maximum_sample_peak_dbfs:
        add(
            "PEAK_TOO_HIGH",
            QualityStatus.WARNING,
            "Sample peak {:.2f} dBFS exceeds {:.2f} dBFS".format(
                sample_peak_dbfs, limits.maximum_sample_peak_dbfs
            ),
        )
    if dc_offset > limits.maximum_dc_offset:
        add(
            "DC_OFFSET",
            QualityStatus.WARNING,
            "DC offset {:.5f} exceeds {:.5f}".format(dc_offset, limits.maximum_dc_offset),
        )
    if sample_rate < limits.minimum_sample_rate:
        add(
            "LOW_SAMPLE_RATE",
            QualityStatus.WARNING,
            "Sample rate {} Hz is below {} Hz and requires resampling".format(
                sample_rate, limits.minimum_sample_rate
            ),
        )
    if channels > 1:
        detail = ""
        if channel_difference_db is not None:
            detail = "; channel level difference is {:.2f} dB".format(channel_difference_db)
        add(
            "MULTICHANNEL_INPUT",
            QualityStatus.WARNING,
            "Input has {} channels and will be converted to mono{}".format(channels, detail),
        )
    if math.isfinite(integrated_lufs):
        if integrated_lufs < limits.acceptable_lufs_min:
            add(
                "LOUDNESS_LOW",
                QualityStatus.WARNING,
                "Integrated loudness {:.2f} LUFS is below {:.2f} LUFS".format(
                    integrated_lufs, limits.acceptable_lufs_min
                ),
            )
        elif integrated_lufs > limits.acceptable_lufs_max:
            add(
                "LOUDNESS_HIGH",
                QualityStatus.WARNING,
                "Integrated loudness {:.2f} LUFS is above {:.2f} LUFS".format(
                    integrated_lufs, limits.acceptable_lufs_max
                ),
            )
    if math.isfinite(snr_db):
        if snr_db < limits.minimum_estimated_snr_db:
            add(
                "LOW_ESTIMATED_SNR",
                QualityStatus.WARNING,
                "Estimated SNR {:.2f} dB is below {:.2f} dB".format(
                    snr_db, limits.minimum_estimated_snr_db
                ),
            )
        elif snr_db < limits.recommended_estimated_snr_db:
            add(
                "SNR_BELOW_RECOMMENDED",
                QualityStatus.WARNING,
                "Estimated SNR {:.2f} dB is below the {:.2f} dB recommendation".format(
                    snr_db, limits.recommended_estimated_snr_db
                ),
            )
    return issues


def _level_distribution(frame_dbfs: np.ndarray) -> Tuple[float, float, float]:
    finite = frame_dbfs[np.isfinite(frame_dbfs)]
    if finite.size == 0:
        return 0.0, float("-inf"), 0.0
    noise_floor = float(np.percentile(finite, 10.0))
    signal_level = float(np.percentile(finite, 90.0))
    lower = float(np.percentile(finite, 20.0))
    upper = float(np.percentile(finite, 95.0))
    return max(0.0, upper - lower), noise_floor, max(0.0, signal_level - noise_floor)


def _short_segment_ratio(
    active_frames: np.ndarray, *, frame_ms: float, short_segment_seconds: float
) -> float:
    active = np.asarray(active_frames, dtype=bool)
    if active.size == 0:
        return 0.0
    minimum_frames = max(1, int(math.ceil(short_segment_seconds * 1000.0 / (frame_ms / 2.0))))
    short_frames = 0
    start: Optional[int] = None
    for index, is_active in enumerate(np.append(active, False)):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            run_length = index - start
            if run_length < minimum_frames:
                short_frames += run_length
            start = None
    return short_frames / float(active.size)


def _channel_difference_db(samples: np.ndarray) -> Optional[float]:
    if samples.shape[1] < 2:
        return None
    channel_rms = np.sqrt(np.mean(np.square(samples), axis=0, dtype=np.float64))
    levels = np.asarray([amplitude_to_db(value) for value in channel_rms], dtype=np.float64)
    finite = levels[np.isfinite(levels)]
    if finite.size == 0:
        return 0.0
    if finite.size != levels.size:
        return float("inf")
    return float(np.max(finite) - np.min(finite))


def _input_identity(
    audio: object, root: Optional[Union[str, Path]]
) -> Tuple[Path, str, Optional[str]]:
    candidate = getattr(audio, "path", audio)
    path = Path(candidate).expanduser().resolve()
    relative = getattr(audio, "relative_path", None)
    if relative is None and root is not None:
        try:
            relative = path.relative_to(Path(root).expanduser().resolve()).as_posix()
        except ValueError:
            relative = path.name
    digest = getattr(audio, "sha256", None)
    return path, str(relative or path.name), str(digest) if digest else None


def _codec_description(decoded: DecodedAudio) -> str:
    details = [decoded.format]
    if decoded.codec and decoded.codec.upper() not in decoded.format.upper():
        details.append(decoded.codec)
    return "/".join(item for item in details if item)


def _highest_status(issues: Sequence[QualityIssue]) -> QualityStatus:
    if any(issue.severity is QualityStatus.FAIL for issue in issues):
        return QualityStatus.FAIL
    if any(issue.severity is QualityStatus.WARNING for issue in issues):
        return QualityStatus.WARNING
    return QualityStatus.PASS


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _finite_difference(first: float, second: float) -> Optional[float]:
    if not math.isfinite(first) or not math.isfinite(second):
        return None
    return first - second


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
