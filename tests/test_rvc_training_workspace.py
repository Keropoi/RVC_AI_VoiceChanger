"""Contract tests for official RVC workspace and intentional completion exit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rvc_auto_trainer.rvc import RVCAdapter, RVCCommandError, TrainingRequest
from rvc_auto_trainer.rvc.training_workspace import (
    TrainingWorkspaceError,
    prepare_training_workspace,
    validate_stage_artifacts,
)


def _write_bytes(path: Path, size: int = 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _official_workspace(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "RVC"
    experiment = repository / "logs" / "run"
    config = {
        "train": {"seed": 1234, "fp16_run": True},
        "data": {"sampling_rate": 40000},
        "model": {"spk_embed_dim": 109},
    }
    template = repository / "configs" / "v1" / "40k.json"
    template.parent.mkdir(parents=True)
    template.write_text(json.dumps(config), encoding="utf-8")
    _write_bytes(experiment / "0_gt_wavs" / "voice.wav")
    _write_bytes(experiment / "1_16k_wavs" / "voice.wav")
    for folder in ("2a_f0", "2b-f0nsf"):
        npy = experiment / folder / "voice.wav.npy"
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, np.ones((4,), dtype=np.float32))
    feature = experiment / "3_feature768" / "voice.npy"
    feature.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature, np.ones((3, 768), dtype=np.float32))
    _write_bytes(repository / "logs" / "mute" / "0_gt_wavs" / "mute40k.wav")
    for folder, name in (
        ("3_feature768", "mute.npy"),
        ("2a_f0", "mute.wav.npy"),
        ("2b-f0nsf", "mute.wav.npy"),
    ):
        _write_bytes(repository / "logs" / "mute" / folder / name)
    return repository, experiment


def test_prepare_official_training_workspace_is_deterministic(tmp_path: Path) -> None:
    repository, experiment = _official_workspace(tmp_path)

    result = prepare_training_workspace(
        repository,
        experiment,
        sample_rate="40k",
        version="v2",
        use_f0=True,
        speaker_id=0,
        random_seed=42,
        mixed_precision=False,
    )

    assert result.sample_names == ("voice",)
    lines = result.filelist_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert sum("voice.wav" in line for line in lines) == 1
    saved = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert saved["train"]["seed"] == 42
    assert saved["train"]["fp16_run"] is False
    assert validate_stage_artifacts(experiment, "preprocess") == ("voice",)
    assert validate_stage_artifacts(experiment, "f0") == ("voice",)
    assert validate_stage_artifacts(experiment, "features", version="v2") == ("voice",)


def test_incomplete_official_workspace_cannot_train(tmp_path: Path) -> None:
    repository, experiment = _official_workspace(tmp_path)
    (experiment / "3_feature768" / "voice.npy").unlink()

    with pytest.raises(TrainingWorkspaceError, match="missing non-empty artifacts"):
        prepare_training_workspace(
            repository,
            experiment,
            sample_rate="40k",
            version="v2",
            use_f0=True,
            speaker_id=0,
            random_seed=42,
            mixed_precision=True,
        )


def test_official_intentional_nonzero_exit_requires_completion_evidence(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "RVC"
    repository.mkdir()
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    expected_model = tmp_path / "weights" / "run.pth"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "import os, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_bytes(b'x' * 2048)\n"
        "print('Training is done. The program is closed.', flush=True)\n"
        "os._exit(2333333)\n",
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
        model_name="run",
        expected_model=expected_model,
        command=(sys.executable, str(trainer), str(expected_model)),
    )

    result = adapter.train(request)

    assert result.success
    assert result.metadata["return_code"] in {2333333, 149}
    assert result.metadata["attempts"][0]["accepted_official_exit"] is True


def test_nonzero_exit_without_marker_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "RVC"
    repository.mkdir()
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    expected_model = tmp_path / "weights" / "run.pth"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "import os, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_bytes(b'x' * 2048)\n"
        "os._exit(2333333)\n",
        encoding="utf-8",
    )
    adapter = RVCAdapter(
        repository,
        Path(sys.executable),
        logs_dir=tmp_path / "logs",
        monitor_gpu=False,
    )

    with pytest.raises(RVCCommandError, match="return code"):
        adapter.train(
            TrainingRequest(
                experiment_dir=experiment,
                model_name="run",
                expected_model=expected_model,
                command=(sys.executable, str(trainer), str(expected_model)),
            )
        )
