"""Tests for gentle preprocessing and the internal silence-aware slicer."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from rvc_auto_trainer.audio.normalization import (  # noqa: E402
    LoudnessNormalizationConfig,
    PreprocessingConfig,
    normalize_samples,
    preprocess_audio,
)
from rvc_auto_trainer.audio.quality import QualityThresholds  # noqa: E402
from rvc_auto_trainer.audio.slicer import (  # noqa: E402
    SlicingConfig,
    find_slice_boundaries,
    slice_audio,
)


def test_normalization_mixes_mono_resamples_trims_and_limits_peak() -> None:
    sample_rate = 24000
    silence = np.zeros((sample_rate // 2, 2), dtype=np.float32)
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    tone = 0.03 * np.sin(2.0 * np.pi * 220.0 * time)
    stereo_tone = np.column_stack((tone, tone * 0.8)).astype(np.float32)
    samples = np.concatenate((silence, stereo_tone, silence), axis=0)
    config = PreprocessingConfig(
        target_sample_rate=40000,
        loudness_normalization=LoudnessNormalizationConfig(
            target_lufs=-14.0,
            maximum_gain_db=12.0,
            minimum_gain_db=-12.0,
            true_peak_limit_db=-3.0,
        ),
    )

    processed, stats = normalize_samples(samples, sample_rate, config)

    assert processed.dtype == np.float32
    assert processed.ndim == 2 and processed.shape[1] == 1
    assert stats.output_sample_rate == 40000
    assert stats.input_channels == 2 and stats.output_channels == 1
    assert stats.trimmed_leading_frames > 0
    assert stats.trimmed_trailing_frames > 0
    assert np.isfinite(processed).all()
    assert float(np.max(np.abs(processed))) <= 10.0 ** (-3.0 / 20.0) + 0.01
    assert any("limited" in warning for warning in stats.warnings)


def test_preprocess_audio_writes_pcm_wav_without_ffmpeg(tmp_path: Path) -> None:
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = tmp_path / "输入 音频.wav"
    sf.write(str(source), 0.1 * np.sin(2.0 * np.pi * 330.0 * time), sample_rate)
    destination = tmp_path / "processed" / "normalized.wav"

    result = preprocess_audio(
        source,
        destination,
        {"target_sample_rate": 40000, "trim_leading_trailing_silence": False},
    )

    assert result.output_path == destination.resolve()
    info = sf.info(str(destination))
    assert info.samplerate == 40000
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_internal_slicer_uses_silence_and_writes_traceable_manifest(tmp_path: Path) -> None:
    sample_rate = 24000
    tone_time = np.arange(int(sample_rate * 2.5), dtype=np.float64) / sample_rate
    tone = 0.1 * np.sin(2.0 * np.pi * 220.0 * tone_time)
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float64)
    samples = np.concatenate((tone, silence, tone, silence, tone, silence, tone[:12000]))
    source = tmp_path / "角色 source.wav"
    sf.write(str(source), samples, sample_rate, subtype="PCM_16")
    slicing = SlicingConfig(
        minimum_segment_seconds=1.0,
        preferred_segment_seconds=3.0,
        maximum_segment_seconds=4.0,
        minimum_silence_duration_ms=250.0,
        segment_padding_ms=40.0,
    )
    quality = QualityThresholds(
        minimum_file_duration_seconds=0.05,
        maximum_file_duration_seconds=30.0,
        minimum_total_accepted_duration_minutes=0.0,
        recommended_total_duration_minutes=0.0,
        minimum_sample_rate=1,
        acceptable_lufs_min=-100.0,
        acceptable_lufs_max=10.0,
        maximum_sample_peak_dbfs=1.0,
        maximum_clipping_ratio=1.0,
        maximum_dc_offset=1.0,
        maximum_silence_ratio=1.0,
        minimum_estimated_snr_db=0.0,
        recommended_estimated_snr_db=0.0,
        almost_silent_ratio=1.0,
    )

    boundaries = find_slice_boundaries(samples, sample_rate, slicing)
    result = slice_audio(source, tmp_path / "segments", slicing, quality_thresholds=quality)

    assert boundaries[0] == 0 and boundaries[-1] == samples.size
    assert len(result.segments) >= 3
    assert all(record.segment_file.is_file() for record in result.segments)
    assert all("__seg_" in record.segment_file.name for record in result.segments)
    assert result.manifest_path.is_file()
    with result.manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.segments)
    assert set(rows[0]) >= {
        "segment_file",
        "source_file",
        "source_start_seconds",
        "source_end_seconds",
        "duration_seconds",
        "lufs",
        "peak_dbfs",
        "status",
    }
