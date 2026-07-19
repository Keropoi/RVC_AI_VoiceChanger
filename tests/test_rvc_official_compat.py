"""Compatibility tests for the pinned official RVC checkout contracts."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from rvc_auto_trainer.rvc import (
    FeatureRequest,
    InferenceRequest,
    PreprocessRequest,
    RepositoryInspector,
)
from rvc_auto_trainer.rvc.feature_extraction import build_extract_f0_command
from rvc_auto_trainer.rvc.inference import build_inference_command
from rvc_auto_trainer.rvc.preprocess import build_preprocess_command


def _write_script(repository: Path, relative_path: str, source: str) -> None:
    script = repository / relative_path
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(source, encoding="utf-8")


def test_official_preprocess_receives_default_segment_duration(tmp_path: Path) -> None:
    repository = tmp_path / "RVC"
    _write_script(
        repository,
        "infer/modules/train/preprocess.py",
        "import sys\nprint(sys.argv[1], sys.argv[2], sys.argv[3], "
        "sys.argv[4], sys.argv[5], sys.argv[6])\n",
    )
    input_dir = tmp_path / "training audio"
    input_dir.mkdir()
    info = RepositoryInspector(repository, Path(sys.executable)).inspect()

    command = build_preprocess_command(
        info,
        PreprocessRequest(
            input_dir=input_dir,
            experiment_dir=tmp_path / "experiment",
            process_count=2,
        ),
    )

    assert command[6] == "False"
    assert command[7] == "3.7"


def test_legacy_preprocess_keeps_five_argument_contract(tmp_path: Path) -> None:
    repository = tmp_path / "RVC"
    _write_script(
        repository,
        "infer/modules/train/preprocess.py",
        "import sys\nprint(sys.argv[1], sys.argv[2], sys.argv[3], "
        "sys.argv[4], sys.argv[5])\n",
    )
    input_dir = tmp_path / "training"
    input_dir.mkdir()
    info = RepositoryInspector(repository, Path(sys.executable)).inspect()

    command = build_preprocess_command(
        info,
        PreprocessRequest(input_dir=input_dir, experiment_dir=tmp_path / "experiment"),
    )

    assert len(command) == 7
    assert command[-1] == "False"


def test_official_infer_cli_flags_and_model_name_are_adapted(tmp_path: Path) -> None:
    repository = tmp_path / "external" / "RVC"
    _write_script(
        repository,
        "tools/infer_cli.py",
        """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path')
parser.add_argument('--opt_path')
parser.add_argument('--model_name')
parser.add_argument('--index_path')
parser.add_argument('--f0up_key')
parser.add_argument('--f0method')
parser.add_argument('--index_rate')
parser.add_argument('--filter_radius')
parser.add_argument('--resample_sr')
parser.add_argument('--rms_mix_rate')
parser.add_argument('--protect')
""",
    )
    info = RepositoryInspector(repository, Path(sys.executable)).inspect()
    model = tmp_path / "runs" / "artifacts" / "voice.pth"
    request = InferenceRequest(
        input_path=tmp_path / "source.wav",
        output_path=tmp_path / "converted.wav",
        model_path=model,
        index_path=tmp_path / "voice.index",
        f0_method="rmvpe",
        dry_run=True,
    )

    command = build_inference_command(info, request)
    arguments = dict(zip(command[2::2], command[3::2]))

    assert arguments["--input_path"] == str(request.input_path.resolve())
    assert arguments["--opt_path"] == str(request.output_path.resolve())
    assert arguments["--f0method"] == "rmvpe"
    model_name = arguments["--model_name"]
    assert (repository / "assets" / "weights" / model_name).resolve() == model.resolve()


def test_generic_aliases_and_explicit_command_remain_supported(tmp_path: Path) -> None:
    repository = tmp_path / "RVC"
    _write_script(
        repository,
        "tools/infer_cli.py",
        """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input-path')
parser.add_argument('--output-path')
parser.add_argument('--model-path')
parser.add_argument('--f0-method')
""",
    )
    info = RepositoryInspector(repository, Path(sys.executable)).inspect()
    request = InferenceRequest(
        input_path=tmp_path / "source.wav",
        output_path=tmp_path / "converted.wav",
        model_path=tmp_path / "voice.pth",
        allow_without_index=True,
        dry_run=True,
    )

    command = build_inference_command(info, request)
    assert "--input-path" in command
    assert "--output-path" in command
    assert "--model-path" in command
    assert "--f0-method" in command

    explicit = replace(
        request,
        command=("runner", "{input_path}", "{model_path}", "{output_path}"),
    )
    explicit_command = build_inference_command(info, explicit)
    assert explicit_command == (
        "runner",
        str(request.input_path.resolve()),
        str(request.model_path.resolve()),
        str(request.output_path.resolve()),
    )


def test_rmvpe_prefers_the_official_gpu_extractor(tmp_path: Path) -> None:
    repository = tmp_path / "RVC"
    _write_script(
        repository,
        "infer/modules/train/extract/extract_f0_print.py",
        "import sys\nprint(sys.argv[1])\n",
    )
    _write_script(
        repository,
        "infer/modules/train/extract/extract_f0_rmvpe.py",
        "import sys\nprint(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])\n",
    )
    info = RepositoryInspector(repository, Path(sys.executable)).inspect()

    command = build_extract_f0_command(
        info,
        FeatureRequest(
            experiment_dir=tmp_path / "experiment",
            f0_method="rmvpe",
            gpu_id=0,
            dry_run=True,
        ),
    )

    assert Path(command[1]).name == "extract_f0_rmvpe.py"
    assert command[4] == "0"
