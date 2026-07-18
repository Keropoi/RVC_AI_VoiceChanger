"""CLI smoke tests that avoid external RVC, CUDA, and FFmpeg dependencies."""

from pathlib import Path

from typer.testing import CliRunner

from rvc_auto_trainer.cli import app

runner = CliRunner()


def test_help_and_version_are_available() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])
    assert help_result.exit_code == 0
    assert "doctor" in help_result.stdout
    assert "resume" in help_result.stdout
    assert version_result.exit_code == 0
    assert "0.1.0" in version_result.stdout


def test_init_is_idempotent_and_does_not_overwrite(tmp_path: Path) -> None:
    first = runner.invoke(app, ["init", "--project-root", str(tmp_path)])
    assert first.exit_code == 0, first.stdout
    config = tmp_path / "config" / "default.yaml"
    assert config.is_file()
    config.write_text("sentinel: true\n", encoding="utf-8")

    second = runner.invoke(app, ["init", "--project-root", str(tmp_path)])
    assert second.exit_code == 0, second.stdout
    assert config.read_text(encoding="utf-8") == "sentinel: true\n"
    assert (tmp_path / "data" / "training_audio" / "README.txt").is_file()
