"""Integration tests for resumable orchestration with a mock RVC adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rvc_auto_trainer.audio.discovery import AudioFile
from rvc_auto_trainer.pipeline.default_handlers import (
    _load_test_manifest,
    _select_manifest_tests,
    _test_parameter_variants,
)
from rvc_auto_trainer.pipeline.orchestrator import (
    PipelineExecutionError,
    PipelineOrchestrator,
)
from rvc_auto_trainer.pipeline.stages import EXECUTABLE_STAGES
from rvc_auto_trainer.reporting import HTMLReportGenerator, ReportData
from rvc_auto_trainer.state import PipelineStage, StageResult, StageStatus, StateStore


@dataclass
class FakeContext:
    run_id: str
    run_dir: Path
    state_store: StateStore
    config: Any


class MockAdapter:
    """Marker object proving the pipeline has no real RVC dependency."""


def _context(tmp_path: Path, *, initialize: bool = True) -> FakeContext:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = StateStore(run_dir / "state.json", "run")
    if initialize:
        store.initialize("run")
    return FakeContext("run", run_dir, store, {"model": {"name": "mock"}})


def test_complete_mock_pipeline_and_resume_without_external_tools(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[PipelineStage] = []

    def make_handler(stage: PipelineStage):
        def handler(_execution_context: object) -> StageResult:
            calls.append(stage)
            output = context.run_dir / f"{stage.value}.done"
            output.write_text("ok", encoding="utf-8")
            return StageResult(outputs=(output,), metadata={"mock": True})

        return handler

    handlers = {stage: make_handler(stage) for stage in EXECUTABLE_STAGES}
    first = PipelineOrchestrator(context, MockAdapter(), handlers).run()
    assert first.success
    assert calls == list(EXECUTABLE_STAGES)
    assert context.state_store.load().last_completed_stage is PipelineStage.REPORT_GENERATED

    calls.clear()
    resumed = PipelineOrchestrator(context, MockAdapter(), handlers).run(resume=True)
    assert resumed.success
    assert calls == []
    assert all(record.reused for record in resumed.records)


def test_failure_is_persisted_with_resume_guidance(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def fail(_execution_context: object) -> StageResult:
        raise RuntimeError("synthetic failure")

    orchestrator = PipelineOrchestrator(
        context,
        MockAdapter(),
        {PipelineStage.AUDIO_DISCOVERED: fail},
    )
    with pytest.raises(PipelineExecutionError, match="resume --run-id run"):
        orchestrator.run(stages=(PipelineStage.AUDIO_DISCOVERED,))
    record = context.state_store.load().record_for(PipelineStage.AUDIO_DISCOVERED)
    assert record.status is StageStatus.FAILED
    assert "synthetic failure" in (record.error or "")


def test_dry_run_does_not_create_state_file(tmp_path: Path) -> None:
    context = _context(tmp_path, initialize=False)
    called: list[bool] = []

    def handler(execution_context: object) -> StageResult:
        called.append(execution_context.dry_run)
        return StageResult(metadata={"command": ["mock", "--dry-run"]})

    result = PipelineOrchestrator(
        context,
        MockAdapter(),
        {PipelineStage.AUDIO_DISCOVERED: handler},
        dry_run=True,
    ).run(stages=(PipelineStage.AUDIO_DISCOVERED,))
    assert result.success
    assert called == [True]
    assert not context.state_store.path.exists()


def test_html_report_uses_relative_audio_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    original = run_dir / "test_results" / "originals" / "原音.wav"
    converted = run_dir / "test_results" / "converted" / "结果.wav"
    original.parent.mkdir(parents=True)
    converted.parent.mkdir(parents=True)
    original.write_bytes(b"RIFF")
    converted.write_bytes(b"RIFF")
    report = HTMLReportGenerator().generate(
        run_dir,
        ReportData(
            summary={"run_id": "run"},
            test_results=(
                {
                    "name": "试听",
                    "original_path": original,
                    "converted_path": converted,
                    "status": "PASS",
                },
            ),
        ),
    )
    html = report.read_text(encoding="utf-8")
    assert "../test_results/originals/" in html
    assert str(run_dir) not in html
    assert (run_dir / "report" / "manual_review_template.csv").is_file()


def test_test_manifest_overrides_and_parameter_sweep_are_bounded(tmp_path: Path) -> None:
    manifest = tmp_path / "test_manifest.yaml"
    manifest.write_text(
        "tests:\n"
        "  - file: voice.wav\n"
        "    name: expressive\n"
        "    transpose: 7\n"
        "    index_rate: 0.7\n"
        "    protect: 0.4\n",
        encoding="utf-8",
    )
    entry = _load_test_manifest(tmp_path)[0]
    disabled_sweep = SimpleNamespace(enabled=False)
    testing = SimpleNamespace(
        transpose=0,
        index_rate=0.65,
        protect=0.33,
        rms_mix_rate=0.25,
        f0_method="rmvpe",
        filter_radius=3,
        resample_sample_rate=0,
        parameter_sweep=disabled_sweep,
    )
    variant = _test_parameter_variants(testing, entry)[0]
    assert variant["transpose"] == 7
    assert variant["index_rate"] == 0.7
    assert variant["protect"] == 0.4

    testing.parameter_sweep = SimpleNamespace(
        enabled=True,
        transpose_values=[0, 3],
        index_rate_values=[0.5, 0.8],
        maximum_combinations_per_file=4,
    )
    variants = _test_parameter_variants(testing, entry)
    assert {(item["transpose"], item["index_rate"]) for item in variants} == {
        (0, 0.5),
        (0, 0.8),
        (3, 0.5),
        (3, 0.8),
    }
    testing.parameter_sweep.maximum_combinations_per_file = 3
    with pytest.raises(ValueError, match="exceeding maximum"):
        _test_parameter_variants(testing, entry)


def test_manifest_order_controls_the_same_test_selection_used_for_leakage() -> None:
    first = AudioFile(Path("a.wav"), "a.wav", "a" * 64, 10, ".wav")
    requested = AudioFile(Path("z.wav"), "z.wav", "z" * 64, 10, ".wav")

    selected = _select_manifest_tests(
        (first, requested),
        ({"file": "z.wav"}, {"file": "a.wav"}),
        maximum_files=1,
    )

    assert selected == (requested,)
