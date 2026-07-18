"""Tests for numeric quality metrics and per-file classification."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from rvc_auto_trainer.audio.quality import (  # noqa: E402
    QualityStatus,
    QualityThresholds,
    analyze_audio,
    analyze_samples,
)
from rvc_auto_trainer.audio.reports import write_quality_reports  # noqa: E402


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 24000) -> Path:
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def _permissive_thresholds(**changes: object) -> QualityThresholds:
    values = {
        "minimum_file_duration_seconds": 0.05,
        "maximum_file_duration_seconds": 30.0,
        "minimum_total_accepted_duration_minutes": 0.0,
        "recommended_total_duration_minutes": 0.0,
        "minimum_sample_rate": 1,
        "acceptable_lufs_min": -100.0,
        "acceptable_lufs_max": 10.0,
        "maximum_sample_peak_dbfs": 1.0,
        "maximum_clipping_ratio": 1.0,
        "maximum_dc_offset": 1.0,
        "maximum_silence_ratio": 1.0,
        "minimum_estimated_snr_db": 0.0,
        "recommended_estimated_snr_db": 0.0,
        "almost_silent_ratio": 1.0,
    }
    values.update(changes)
    return QualityThresholds(**values)


def test_clean_mono_wav_produces_complete_metrics_and_passes(tmp_path: Path) -> None:
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    samples = 0.12 * np.sin(2.0 * np.pi * 440.0 * time)
    source = _write_wav(tmp_path / "clean.wav", samples, sample_rate)

    result = analyze_audio(source, _permissive_thresholds(), allow_ffmpeg=False)

    assert result.status is QualityStatus.PASS
    assert result.decodable is True
    assert result.duration_seconds == pytest.approx(1.0, abs=1e-4)
    assert result.sample_rate == sample_rate
    assert result.channels == 1
    assert result.pcm_bit_depth == 16
    assert result.integrated_lufs is not None and math.isfinite(result.integrated_lufs)
    assert result.rms_dbfs == pytest.approx(-21.43, abs=0.15)
    assert result.sample_peak_dbfs == pytest.approx(-18.42, abs=0.1)
    assert result.true_peak_dbfs is not None
    assert result.crest_factor_db == pytest.approx(3.01, abs=0.15)
    assert result.clipping_sample_ratio == 0.0
    assert result.non_finite_count == 0


def test_silence_is_a_failure(tmp_path: Path) -> None:
    source = _write_wav(tmp_path / "silence.wav", np.zeros(24000, dtype=np.float32))

    result = analyze_audio(source, allow_ffmpeg=False)

    assert result.status is QualityStatus.FAIL
    assert result.almost_all_silence is True
    assert "ALMOST_ALL_SILENCE" in {issue.code for issue in result.issues}
    assert result.integrated_lufs is None


def test_heavy_clipping_is_a_failure(tmp_path: Path) -> None:
    samples = np.ones(24000, dtype=np.float32)
    samples[1::2] = -1.0
    source = _write_wav(tmp_path / "clipped.wav", samples)

    result = analyze_audio(source, allow_ffmpeg=False)

    assert result.status is QualityStatus.FAIL
    assert result.clipping_sample_ratio is not None
    assert result.clipping_sample_ratio > 0.99
    assert "SEVERE_CLIPPING" in {issue.code for issue in result.issues}


def test_dc_offset_is_reported_as_warning(tmp_path: Path) -> None:
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    samples = 0.05 + 0.10 * np.sin(2.0 * np.pi * 220.0 * time)
    source = _write_wav(tmp_path / "dc.wav", samples, sample_rate)
    thresholds = _permissive_thresholds(maximum_dc_offset=0.02)

    result = analyze_audio(source, thresholds, allow_ffmpeg=False)

    assert result.status is QualityStatus.WARNING
    assert result.dc_offset == pytest.approx(0.05, abs=1e-3)
    assert "DC_OFFSET" in {issue.code for issue in result.issues}


def test_corrupt_file_returns_fail_row_without_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "broken.wav"
    source.write_bytes(b"this is not wave data")

    result = analyze_audio(source, allow_ffmpeg=False)

    assert result.status is QualityStatus.FAIL
    assert result.decodable is False
    assert result.decode_error
    assert result.duration_seconds is None
    assert result.issues[0].code == "DECODE_FAILED"


def test_non_finite_samples_are_detected_before_cleanup() -> None:
    samples = np.full(24000, 0.1, dtype=np.float32)
    samples[10] = np.nan
    samples[20] = np.inf

    result = analyze_samples(samples, 24000, _permissive_thresholds())

    assert result.status is QualityStatus.FAIL
    assert result.non_finite_count == 2
    assert "NON_FINITE_SAMPLES" in {issue.code for issue in result.issues}


def test_quality_reports_emit_csv_json_and_escaped_offline_html(tmp_path: Path) -> None:
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = _write_wav(
        tmp_path / "voice & sample.wav",
        0.1 * np.sin(2.0 * np.pi * 330.0 * time),
        sample_rate,
    )
    thresholds = _permissive_thresholds()
    result = analyze_audio(source, thresholds, allow_ffmpeg=False)

    paths = write_quality_reports([result], tmp_path / "quality", thresholds)

    assert paths.csv_path.is_file()
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_files"] == 1
    assert payload["files"][0]["status"] == "PASS"
    html_text = paths.html_path.read_text(encoding="utf-8")
    assert "voice &amp; sample.wav" in html_text
    assert "<script" not in html_text
