"""Tests for ordered transitions, cache reuse, and atomic state persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from rvc_auto_trainer.config import load_config
from rvc_auto_trainer.exceptions import InvalidStateTransition, StateFileError
from rvc_auto_trainer.pipeline.run_context import RunContext
from rvc_auto_trainer.state import (
    PipelineStage,
    PipelineState,
    StageStatus,
    StateStore,
)


def test_pipeline_state_enforces_stage_order() -> None:
    """A stage cannot run before its direct predecessor has completed."""
    state = PipelineState.create("run_one")

    with pytest.raises(InvalidStateTransition, match="predecessor"):
        state.start_stage(PipelineStage.QUALITY_CHECKED)

    state.mark_completed(PipelineStage.AUDIO_DISCOVERED)
    assert state.current_stage is PipelineStage.AUDIO_DISCOVERED
    assert state.last_completed_stage is PipelineStage.AUDIO_DISCOVERED
    assert state.next_stage is PipelineStage.QUALITY_CHECKED


def test_state_store_atomically_round_trips_stage_metadata(tmp_path: Path) -> None:
    """Every mutation is immediately represented by valid state JSON."""
    path = tmp_path / "中文 run" / "state.json"
    store = StateStore(path, "中文_run")
    store.initialize()
    store.start_stage(PipelineStage.AUDIO_DISCOVERED, fingerprint="fingerprint-a")
    output = tmp_path / "中文 run" / "input_manifest.json"
    output.write_text("{}", encoding="utf-8")
    store.complete_stage(
        PipelineStage.AUDIO_DISCOVERED,
        outputs=("input_manifest.json",),
        metadata={"files": 3},
    )

    loaded = store.load()
    record = loaded.record_for(PipelineStage.AUDIO_DISCOVERED)
    assert record.status is StageStatus.COMPLETED
    assert record.fingerprint == "fingerprint-a"
    assert record.metadata == {"files": 3}
    assert store.can_reuse(
        PipelineStage.AUDIO_DISCOVERED,
        "fingerprint-a",
        required_outputs=("input_manifest.json",),
    )
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "中文_run"
    assert not list(path.parent.glob("*.tmp"))


def test_failed_stage_can_retry_without_losing_attempt_count(tmp_path: Path) -> None:
    """Resume retries a failed stage and preserves diagnostic attempt history."""
    store = StateStore(tmp_path / "state.json", "retry_run")
    store.initialize()
    store.start_stage(PipelineStage.AUDIO_DISCOVERED)
    store.fail_stage(PipelineStage.AUDIO_DISCOVERED, "decoder returned exit code 1")
    store.start_stage(PipelineStage.AUDIO_DISCOVERED, fingerprint="new")
    retried = store.load().record_for(PipelineStage.AUDIO_DISCOVERED)

    assert retried.status is StageStatus.RUNNING
    assert retried.attempts == 2
    assert retried.error is None


def test_invalidation_cascades_without_deleting_outputs() -> None:
    """Changed inputs invalidate dependent stages while retaining old artifacts."""
    state = PipelineState.create("invalidate_run")
    state.mark_completed(PipelineStage.AUDIO_DISCOVERED)
    state.mark_completed(PipelineStage.QUALITY_CHECKED)
    state.mark_completed(PipelineStage.PREPROCESSED, outputs=("old.wav",))

    state.invalidate_from(PipelineStage.QUALITY_CHECKED, reason="training audio changed")

    assert state.record_for(PipelineStage.AUDIO_DISCOVERED).status is StageStatus.COMPLETED
    assert state.record_for(PipelineStage.QUALITY_CHECKED).status is StageStatus.INVALIDATED
    assert state.record_for(PipelineStage.PREPROCESSED).outputs == (Path("old.wav"),)
    assert state.next_stage is PipelineStage.QUALITY_CHECKED


def test_corrupt_state_file_has_actionable_error(tmp_path: Path) -> None:
    """Invalid JSON never gets mistaken for a fresh pipeline run."""
    path = tmp_path / "state.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(StateFileError, match="Cannot load state file"):
        StateStore(path).load()


def test_run_context_uses_collision_safe_directories_and_initial_state(tmp_path: Path) -> None:
    """Repeated runs never overwrite an earlier model/run directory."""
    config_path = tmp_path / "config" / "run.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "model:\n  name: 中文 角色\npaths:\n  runs_dir: 运行结果\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    now = datetime(2026, 7, 19, 12, 30, 45)

    first = RunContext.create(config, now=now)
    second = RunContext.create(config, now=now)

    assert first.run_id == "20260719_123045_中文_角色"
    assert second.run_id == "20260719_123045_中文_角色_02"
    assert first.config_resolved_path.is_file()
    assert first.logs_dir.is_dir()
    assert first.state.last_completed_stage is PipelineStage.INITIALIZED
