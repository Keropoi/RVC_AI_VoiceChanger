"""Tests for merged Pydantic configuration and Windows-safe path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rvc_auto_trainer.config import load_config, save_resolved_config
from rvc_auto_trainer.exceptions import ConfigurationError


def _write_yaml(path: Path, payload: object) -> None:
    """Write a UTF-8 YAML fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_user_yaml_recursively_merges_project_default(tmp_path: Path) -> None:
    """A partial user file retains nested values from config/default.yaml."""
    config_dir = tmp_path / "config"
    _write_yaml(
        config_dir / "default.yaml",
        {
            "project": {"name": "默认项目", "random_seed": 17},
            "training": {"epochs": 321, "batch_size": 12},
            "paths": {"runs_dir": "输出/runs"},
        },
    )
    user_path = config_dir / "用户设置.yaml"
    _write_yaml(user_path, {"training": {"batch_size": 4}})

    config = load_config(user_path)

    assert config.project.name == "默认项目"
    assert config.project.random_seed == 17
    assert config.training.epochs == 321
    assert config.training.batch_size == 4
    assert config.paths.runs_dir == (tmp_path / "输出" / "runs").resolve()


def test_all_paths_resolve_from_project_root_with_chinese_names(tmp_path: Path) -> None:
    """Relative paths do not accidentally resolve from the config subdirectory."""
    config_path = tmp_path / "config" / "中文 配置.yaml"
    _write_yaml(
        config_path,
        {
            "paths": {
                "training_audio_dir": "素材/训练 音频",
                "test_audio_dir": "素材/测试音频",
                "runs_dir": "运行结果",
            }
        },
    )

    config = load_config(config_path)

    assert config.project_root == tmp_path.resolve()
    assert config.paths.training_audio_dir == (tmp_path / "素材/训练 音频").resolve()
    assert config.paths.test_audio_dir == (tmp_path / "素材/测试音频").resolve()
    assert config.resolve_path("额外/文件.txt") == (tmp_path / "额外/文件.txt").resolve()
    assert config.paths.rvc_repository.is_absolute()


def test_invalid_cross_field_thresholds_raise_configuration_error(tmp_path: Path) -> None:
    """Pydantic validation is surfaced as one domain-specific error."""
    config_path = tmp_path / "config" / "invalid.yaml"
    _write_yaml(
        config_path,
        {
            "quality": {
                "minimum_file_duration_seconds": 20,
                "maximum_file_duration_seconds": 10,
            }
        },
    )

    with pytest.raises(ConfigurationError, match="minimum file duration"):
        load_config(config_path)


def test_enabled_parameter_sweep_is_bounded(tmp_path: Path) -> None:
    """An unsafe inference sweep is rejected before work begins."""
    config_path = tmp_path / "config" / "sweep.yaml"
    _write_yaml(
        config_path,
        {
            "testing": {
                "parameter_sweep": {
                    "enabled": True,
                    "transpose_values": [0, 1, 2],
                    "index_rate_values": [0.5, 0.6],
                    "maximum_combinations_per_file": 5,
                }
            }
        },
    )

    with pytest.raises(ConfigurationError, match="6 combinations"):
        load_config(config_path)


def test_save_resolved_config_contains_defaults_and_absolute_paths(tmp_path: Path) -> None:
    """Run snapshots include the complete effective config, not only overrides."""
    source = tmp_path / "config" / "partial.yaml"
    _write_yaml(source, {"model": {"name": "测试角色"}})
    config = load_config(source)
    destination = tmp_path / "runs" / "one" / "config_resolved.yaml"

    save_resolved_config(config, destination)
    saved = yaml.safe_load(destination.read_text(encoding="utf-8"))

    assert saved["model"]["name"] == "测试角色"
    assert saved["training"]["epochs"] == 200
    assert Path(saved["paths"]["runs_dir"]).is_absolute()
    assert "config_path" not in saved
