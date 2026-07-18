"""Unit tests for runtime RVC inspection and safe process adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rvc_auto_trainer.monitoring.process_monitor import ProcessMonitor, ProcessResult
from rvc_auto_trainer.rvc import (
    PreprocessRequest,
    RepositoryInspector,
    RVCAdapter,
    RVCCommandError,
    TrainingRequest,
    batch_size_attempts,
    detect_cuda_oom,
)
from rvc_auto_trainer.rvc.preprocess import build_preprocess_command


def test_repository_inspection_does_not_claim_missing_checkout(tmp_path: Path) -> None:
    info = RepositoryInspector(tmp_path / "missing", Path(sys.executable)).inspect()
    assert not info.exists
    assert not info.is_supported
    assert "train" in info.missing_purposes


def test_preprocess_command_supports_unicode_and_spaces(tmp_path: Path) -> None:
    repository = tmp_path / "RVC 仓库"
    script = repository / "infer" / "modules" / "train" / "preprocess.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import sys\nprint(sys.argv[1], sys.argv[2], sys.argv[3], "
        "sys.argv[4], sys.argv[5])\n",
        encoding="utf-8",
    )
    input_dir = tmp_path / "训练 音频"
    input_dir.mkdir()
    info = RepositoryInspector(repository, Path(sys.executable)).inspect()
    request = PreprocessRequest(
        input_dir=input_dir,
        experiment_dir=tmp_path / "实验 目录",
        sample_rate=40_000,
        process_count=3,
    )
    command = build_preprocess_command(info, request)
    assert command[0] == str(Path(sys.executable).resolve())
    assert command[2] == str(input_dir.resolve())
    assert command[5] == str((tmp_path / "实验 目录").resolve())
    assert isinstance(command, tuple)


def test_process_monitor_captures_both_streams(tmp_path: Path) -> None:
    monitor = ProcessMonitor(poll_interval_seconds=0.02)
    result = monitor.run(
        [
            sys.executable,
            "-c",
            "import sys; print('progress'); print('warning', file=sys.stderr)",
        ],
        cwd=tmp_path,
        stdout_log=tmp_path / "stdout.log",
        stderr_log=tmp_path / "stderr.log",
    )
    assert result.success
    assert "progress" in (tmp_path / "stdout.log").read_text(encoding="utf-8")
    assert "warning" in (tmp_path / "stderr.log").read_text(encoding="utf-8")


def test_oom_requires_failed_exit_and_cuda_evidence(tmp_path: Path) -> None:
    common = {
        "command": ("python", "train.py"),
        "started_at": 1.0,
        "ended_at": 2.0,
        "stdout_log": tmp_path / "out.log",
        "stderr_log": tmp_path / "err.log",
        "stdout_tail": (),
        "stderr_tail": ("torch.OutOfMemoryError: CUDA out of memory",),
    }
    assert detect_cuda_oom(ProcessResult(return_code=1, **common))
    assert not detect_cuda_oom(ProcessResult(return_code=0, **common))


def test_training_retries_only_oom_with_lower_batch(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    trainer = tmp_path / "fake trainer.py"
    trainer.write_text(
        "import sys\n"
        "batch = int(sys.argv[1])\n"
        "if batch > 4:\n"
        "    print('CUDA out of memory', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print('epoch=1 loss=0.1')\n",
        encoding="utf-8",
    )
    adapter = RVCAdapter(
        repository,
        Path(sys.executable),
        logs_dir=tmp_path / "logs",
        monitor_gpu=False,
    )
    request = TrainingRequest(
        experiment_dir=experiment,
        model_name="mock",
        batch_size=8,
        batch_size_candidates=(8, 4),
        maximum_oom_retries=1,
        command=(sys.executable, str(trainer), "{batch_size}"),
    )
    result = adapter.train(request)
    assert result.success
    assert result.metadata["selected_batch_size"] == 4
    assert result.metadata["oom_retry_count"] == 1
    assert batch_size_attempts(request) == (8, 4)


def test_non_oom_process_failure_is_actionable(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    adapter = RVCAdapter(
        repository,
        Path(sys.executable),
        logs_dir=tmp_path / "logs",
        monitor_gpu=False,
    )
    request = TrainingRequest(
        experiment_dir=experiment,
        model_name="mock",
        command=(sys.executable, "-c", "import sys; sys.stderr.write('bad config'); sys.exit(7)"),
    )
    with pytest.raises(RVCCommandError, match="return code 7") as error:
        adapter.train(request)
    assert error.value.result is not None
    assert error.value.result.stderr_log.is_file()
