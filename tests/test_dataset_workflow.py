"""Tests for non-destructive curation, speaker review, and fixed evaluation data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rvc_auto_trainer.audio.decoder import write_wav
from rvc_auto_trainer.config import AppConfig, PathsConfig
from rvc_auto_trainer.dataset import apply_review_decisions, prepare_dataset
from rvc_auto_trainer.evaluation_dataset import (
    EvaluationDatasetError,
    freeze_test_manifest,
    validate_frozen_test_manifest,
)
from rvc_auto_trainer.speaker_sorting import (
    SpeakerSortingError,
    SpeakerTurn,
    _normalise_turns,
    apply_speaker_review,
)


def _config(tmp_path: Path) -> AppConfig:
    paths = PathsConfig().resolved(tmp_path)
    config = AppConfig(
        project_root=tmp_path.resolve(),
        paths=paths,
        quality={
            "minimum_total_accepted_duration_minutes": 0,
            "fail_on_insufficient_total_duration": False,
        },
        curation={"review_minimum_pass_samples": 1},
        speaker_sorting={"use_gpu": False},
    )
    for value in paths.model_dump().values():
        path = Path(value)
        if path.suffix.lower() != ".exe":
            path.mkdir(parents=True, exist_ok=True)
    return config


def _write_voice(path: Path, duration: float, frequency: float = 220.0) -> Path:
    sample_rate = 24_000
    frames = int(round(duration * sample_rate))
    time = np.arange(frames, dtype=np.float32) / sample_rate
    signal = 0.08 * np.sin(2.0 * np.pi * frequency * time)
    signal += np.random.default_rng(7).normal(0.0, 0.002, frames).astype(np.float32)
    return write_wav(path, signal, sample_rate, subtype="PCM_16")


def test_prepare_and_apply_review_preserve_metadata_and_originals(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = _write_voice(config.paths.raw_audio_dir / "session.wav", 2.0)
    original_bytes = original.read_bytes()

    prepared = prepare_dataset(
        config,
        source_label="licensed session A",
        rights_note="recorded with speaker consent",
        language="ja",
    )

    manifest = json.loads(prepared.raw_manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_label"] == "licensed session A"
    assert manifest["rights_note"] == "recorded with speaker consent"
    assert manifest["files"][0]["language"] == "ja"
    assert original.read_bytes() == original_bytes
    with prepared.review_queue_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = rows[0].keys()
    rows[0]["decision"] = "KEEP"
    with prepared.review_queue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    applied = apply_review_decisions(config, prepared.review_queue_path)

    assert applied.kept_count == 1
    assert len(list(config.paths.training_audio_dir.rglob("*.wav"))) == 1
    assert original.read_bytes() == original_bytes


def test_speaker_turn_merge_and_copy_only_target_review(tmp_path: Path) -> None:
    turns = _normalise_turns(
        (
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.1, 2.0, "SPEAKER_00"),
            SpeakerTurn(2.2, 3.0, "SPEAKER_01"),
        ),
        merge_gap_seconds=0.2,
    )
    assert turns == (
        SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
        SpeakerTurn(2.2, 3.0, "SPEAKER_01"),
    )

    config = _config(tmp_path)
    segment = _write_voice(
        config.paths.speaker_segments_dir / "run" / "source" / "SPEAKER_00" / "one.wav",
        1.5,
    )
    relative = segment.relative_to(config.paths.speaker_segments_dir).as_posix()
    from rvc_auto_trainer.audio.discovery import sha256_file

    review = config.paths.speaker_manifests_dir / "run" / "speaker_review.csv"
    review.parent.mkdir(parents=True, exist_ok=True)
    fields = (
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
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "source_relative_path": "mixed.wav",
                "source_sha256": "source-hash",
                "speaker_label": "SPEAKER_00",
                "segment_count": 1,
                "usable_duration_seconds": 1.5,
                "overlap_segment_count": 0,
                "target_similarity": 0.8,
                "rank_in_source": 1,
                "score_margin": 0.2,
                "recommended_action": "TARGET_CANDIDATE_REVIEW_REQUIRED",
                "sample_files": relative,
                "segment_relative_paths": relative,
                "segment_sha256s": sha256_file(segment),
                "segment_durations_seconds": 1.5,
                "decision": "TARGET",
                "reviewer_notes": "listened",
            }
        )

    result = apply_speaker_review(config, review)

    assert result.copied_segment_count == 1
    copied = config.paths.speaker_selected_audio_dir / relative
    assert copied.read_bytes() == segment.read_bytes()
    with pytest.raises(SpeakerSortingError, match="changed after review"):
        segment.write_bytes(b"changed")
        apply_speaker_review(config, review)


def test_freeze_and_validate_exact_five_role_test_set(tmp_path: Path) -> None:
    config = _config(tmp_path)
    specs = (
        ("zh_neutral", "mandarin_neutral", "neutral.wav", 10.0),
        ("zh_stress", "mandarin_phoneme_stress", "stress.wav", 10.0),
        ("zh_expressive", "mandarin_expressive", "expressive.wav", 10.0),
        ("zh_paragraph", "mandarin_paragraph", "paragraph.wav", 30.0),
        ("ja_control", "japanese_control", "control.wav", 10.0),
    )
    tests = []
    for index, (identifier, role, filename, duration) in enumerate(specs):
        _write_voice(config.paths.test_audio_dir / filename, duration, 180.0 + index * 30)
        tests.append({"id": identifier, "role": role, "file": filename, "transpose": 0})
    manifest = config.paths.test_audio_dir / "test_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"tests": tests}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    frozen = freeze_test_manifest(config)
    validated = validate_frozen_test_manifest(config)

    assert len(frozen.files) == len(validated.files) == 5
    assert all(record.get("sha256") for record in validated.records)
    neutral = config.paths.test_audio_dir / "neutral.wav"
    neutral.write_bytes(neutral.read_bytes() + b"changed")
    with pytest.raises(EvaluationDatasetError, match="changed"):
        validate_frozen_test_manifest(config)
