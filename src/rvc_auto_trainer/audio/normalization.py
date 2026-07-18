"""Non-destructive preprocessing for RVC training inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy import signal

from .decoder import DecodedAudio, decode_audio, write_wav
from .quality import (
    amplitude_to_db,
    approximate_true_peak_dbfs,
    db_to_amplitude,
    measure_integrated_lufs,
)


@dataclass(frozen=True)
class LoudnessNormalizationConfig:
    """Gain-only loudness normalization limits (no compressor or limiter)."""

    enabled: bool = True
    target_lufs: float = -20.0
    maximum_gain_db: float = 12.0
    minimum_gain_db: float = -12.0
    true_peak_limit_db: float = -3.0

    def __post_init__(self) -> None:
        if self.minimum_gain_db > self.maximum_gain_db:
            raise ValueError("minimum_gain_db cannot exceed maximum_gain_db")
        if self.true_peak_limit_db > 0:
            raise ValueError("true_peak_limit_db cannot be above 0 dBFS")


@dataclass(frozen=True)
class HighPassFilterConfig:
    """Optional gentle high-pass filter, disabled by default."""

    enabled: bool = False
    cutoff_hz: float = 60.0


@dataclass(frozen=True)
class NoiseReductionConfig:
    """Noise-reduction switch retained for explicit capability validation."""

    enabled: bool = False


@dataclass(frozen=True)
class PreprocessingConfig:
    """Audio preprocessing settings accepted from mappings or config models."""

    target_sample_rate: int = 40000
    output_subtype: str = "PCM_16"
    mono: bool = True
    trim_leading_trailing_silence: bool = True
    trim_threshold_dbfs: float = -48.0
    trim_padding_ms: float = 120.0
    loudness_normalization: LoudnessNormalizationConfig = LoudnessNormalizationConfig()
    high_pass_filter: HighPassFilterConfig = HighPassFilterConfig()
    noise_reduction: NoiseReductionConfig = NoiseReductionConfig()

    def __post_init__(self) -> None:
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.trim_padding_ms < 0:
            raise ValueError("trim_padding_ms cannot be negative")
        if not self.output_subtype:
            raise ValueError("output_subtype cannot be empty")

    @classmethod
    def from_config(cls, config: Optional[object]) -> "PreprocessingConfig":
        """Coerce a mapping/Pydantic/dataclass-like preprocessing section."""

        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        section = _nested_config(config, "preprocessing")
        values: Dict[str, Any] = {}
        for field in fields(cls):
            value = _config_value(section, field.name)
            if value is _MISSING:
                continue
            if field.name == "loudness_normalization":
                value = _coerce_dataclass(LoudnessNormalizationConfig, value)
            elif field.name == "high_pass_filter":
                value = _coerce_dataclass(HighPassFilterConfig, value)
            elif field.name == "noise_reduction":
                value = _coerce_dataclass(NoiseReductionConfig, value)
            values[field.name] = value
        return cls(**values)


@dataclass(frozen=True)
class ProcessingStats:
    """Detailed transformations applied to one in-memory signal."""

    input_sample_rate: int
    output_sample_rate: int
    input_channels: int
    output_channels: int
    input_frames: int
    output_frames: int
    trimmed_leading_frames: int
    trimmed_trailing_frames: int
    lufs_before: Optional[float]
    lufs_after: Optional[float]
    requested_gain_db: Optional[float]
    applied_gain_db: float
    true_peak_dbfs_after: Optional[float]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class PreprocessResult:
    """Filesystem result and measurements for a normalized training file."""

    source_path: Path
    output_path: Path
    stats: ProcessingStats

    @property
    def warnings(self) -> Tuple[str, ...]:
        """Expose processing warnings directly to pipeline callers."""

        return self.stats.warnings


class AudioPreprocessingError(RuntimeError):
    """Raised when requested preprocessing cannot be performed safely."""


def preprocess_audio(
    source: Union[str, Path],
    output: Union[str, Path],
    config: Optional[object] = None,
    *,
    ffmpeg_executable: Union[str, Path] = "ffmpeg",
    overwrite: bool = False,
) -> PreprocessResult:
    """Decode, gently normalize, and save one training input as PCM WAV."""

    settings = PreprocessingConfig.from_config(config)
    decoded = decode_audio(source, ffmpeg_executable=ffmpeg_executable)
    processed, stats = normalize_decoded_audio(decoded, settings)
    output_path = write_wav(
        output,
        processed,
        settings.target_sample_rate,
        subtype=settings.output_subtype,
        overwrite=overwrite,
    )
    return PreprocessResult(decoded.source_path, output_path, stats)


def normalize_decoded_audio(
    decoded: DecodedAudio, config: Optional[object] = None
) -> Tuple[np.ndarray, ProcessingStats]:
    """Apply preprocessing to decoded audio and return float32 frames plus stats."""

    settings = PreprocessingConfig.from_config(config)
    return normalize_samples(decoded.samples, decoded.sample_rate, settings)


def normalize_samples(
    samples: np.ndarray,
    sample_rate: int,
    config: Optional[object] = None,
) -> Tuple[np.ndarray, ProcessingStats]:
    """Apply finite cleanup, mono mix, resampling, trim, filter, and safe gain."""

    settings = PreprocessingConfig.from_config(config)
    if settings.noise_reduction.enabled:
        raise AudioPreprocessingError(
            "Noise reduction was enabled, but the safe first release intentionally does not "
            "alter source timbre with a noise-reduction algorithm"
        )
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise AudioPreprocessingError("Input audio must contain frames and channels")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    warnings: List[str] = []
    non_finite = int(data.size - np.count_nonzero(np.isfinite(data)))
    if non_finite:
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        warnings.append("Replaced {} NaN/Infinity sample(s) with silence".format(non_finite))
    input_frames = int(data.shape[0])
    input_channels = int(data.shape[1])

    if settings.mono and data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)

    if sample_rate != settings.target_sample_rate:
        data = _resample(data, sample_rate, settings.target_sample_rate)

    leading = 0
    trailing = 0
    if settings.trim_leading_trailing_silence:
        data, leading, trailing = trim_edge_silence(
            data,
            settings.target_sample_rate,
            threshold_dbfs=settings.trim_threshold_dbfs,
            padding_ms=settings.trim_padding_ms,
        )

    if settings.high_pass_filter.enabled:
        data = _high_pass(
            data,
            settings.target_sample_rate,
            settings.high_pass_filter.cutoff_hz,
        )

    mono_for_meter = np.mean(data, axis=1)
    lufs_before = measure_integrated_lufs(mono_for_meter, settings.target_sample_rate)
    requested_gain: Optional[float] = None
    applied_gain = 0.0
    loudness = settings.loudness_normalization
    if loudness.enabled:
        if math.isfinite(lufs_before):
            requested_gain = loudness.target_lufs - lufs_before
            applied_gain = min(
                loudness.maximum_gain_db,
                max(loudness.minimum_gain_db, requested_gain),
            )
            if not math.isclose(requested_gain, applied_gain, abs_tol=1e-6):
                warnings.append(
                    "Requested loudness gain {:.2f} dB was limited to {:.2f} dB".format(
                        requested_gain, applied_gain
                    )
                )
            data *= db_to_amplitude(applied_gain)
        else:
            warnings.append("Loudness normalization skipped because the signal is silent")

    peak_after_gain = approximate_true_peak_dbfs(data, settings.target_sample_rate)
    if math.isfinite(peak_after_gain) and peak_after_gain > loudness.true_peak_limit_db:
        peak_attenuation = loudness.true_peak_limit_db - peak_after_gain
        data *= db_to_amplitude(peak_attenuation)
        applied_gain += peak_attenuation
        warnings.append(
            "Applied {:.2f} dB safety attenuation to respect the {:.2f} dBTP limit".format(
                peak_attenuation, loudness.true_peak_limit_db
            )
        )

    # A final numerical guard prevents encoding overs caused by floating-point error.
    absolute_peak = float(np.max(np.abs(data)))
    if absolute_peak > 1.0:
        safety_gain = -amplitude_to_db(absolute_peak)
        data *= db_to_amplitude(safety_gain)
        applied_gain += safety_gain
        warnings.append("Applied {:.3f} dB full-scale safety attenuation".format(safety_gain))

    lufs_after = measure_integrated_lufs(np.mean(data, axis=1), settings.target_sample_rate)
    true_peak_after = approximate_true_peak_dbfs(data, settings.target_sample_rate)
    stats = ProcessingStats(
        input_sample_rate=sample_rate,
        output_sample_rate=settings.target_sample_rate,
        input_channels=input_channels,
        output_channels=int(data.shape[1]),
        input_frames=input_frames,
        output_frames=int(data.shape[0]),
        trimmed_leading_frames=leading,
        trimmed_trailing_frames=trailing,
        lufs_before=_finite_or_none(lufs_before),
        lufs_after=_finite_or_none(lufs_after),
        requested_gain_db=requested_gain,
        applied_gain_db=applied_gain,
        true_peak_dbfs_after=_finite_or_none(true_peak_after),
        warnings=tuple(warnings),
    )
    return data.astype(np.float32), stats


def trim_edge_silence(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float,
    padding_ms: float,
    frame_ms: float = 20.0,
) -> Tuple[np.ndarray, int, int]:
    """Trim only leading/trailing silent frames, preserving configurable padding."""

    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    mono = np.mean(data.astype(np.float64), axis=1)
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    threshold = db_to_amplitude(threshold_dbfs)
    active_frames: List[int] = []
    for start in range(0, mono.size, frame_length):
        frame = mono[start : start + frame_length]
        if frame.size and float(np.sqrt(np.mean(np.square(frame)))) > threshold:
            active_frames.append(start)
    if not active_frames:
        return data, 0, 0

    padding = int(round(sample_rate * padding_ms / 1000.0))
    first = max(0, active_frames[0] - padding)
    last_active_end = min(mono.size, active_frames[-1] + frame_length)
    stop = min(mono.size, last_active_end + padding)
    if stop <= first:
        return data, 0, 0
    return data[first:stop], first, mono.size - stop


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return np.asarray(signal.resample_poly(samples, up, down, axis=0), dtype=np.float64)


def _high_pass(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2.0:
        raise AudioPreprocessingError(
            "High-pass cutoff must be above 0 and below Nyquist ({:.1f} Hz)".format(
                sample_rate / 2.0
            )
        )
    second_order_sections = signal.butter(
        2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos"
    )
    try:
        return np.asarray(
            signal.sosfiltfilt(second_order_sections, samples, axis=0), dtype=np.float64
        )
    except ValueError:
        return np.asarray(signal.sosfilt(second_order_sections, samples, axis=0), dtype=np.float64)


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


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


def _coerce_dataclass(model: object, value: object) -> object:
    if isinstance(value, model):
        return value
    values: Dict[str, Any] = {}
    for field in fields(model):
        field_value = _config_value(value, field.name)
        if field_value is not _MISSING:
            values[field.name] = field_value
    return model(**values)
