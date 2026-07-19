"""Typed YAML configuration loading and project-relative path resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .exceptions import ConfigurationError


class ConfigModel(BaseModel):
    """Base model shared by all strict, assignment-validated config sections."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectConfig(ConfigModel):
    """Human-facing project metadata."""

    name: str = "rvc_voice_training"
    random_seed: int = 42

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject empty project names that would produce ambiguous reports."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("project.name must not be empty")
        return cleaned


class PathsConfig(ConfigModel):
    """Filesystem locations used by the orchestrator and external RVC runtime."""

    training_audio_dir: Path = Path("data/training_audio")
    test_audio_dir: Path = Path("data/test_audio")
    rejected_audio_dir: Path = Path("data/rejected_audio")
    mixed_speaker_audio_dir: Path = Path("data/mixed_speaker_audio")
    speaker_segments_dir: Path = Path("data/speaker_segments")
    speaker_manifests_dir: Path = Path("data/speaker_manifests")
    speaker_selected_audio_dir: Path = Path("data/speaker_selected_audio")
    raw_audio_dir: Path = Path("data/raw_archive")
    training_candidates_dir: Path = Path("data/training_candidates")
    voice_reference_dir: Path = Path("data/voice_references")
    dataset_manifests_dir: Path = Path("data/dataset_manifests")
    runs_dir: Path = Path("runs")
    rvc_repository: Path = Path("external/RVC")
    orchestration_python: Path = Path(".venv/Scripts/python.exe")
    rvc_python: Path = Path("external/RVC/.venv/Scripts/python.exe")
    pretrained_models_dir: Path = Path("models/pretrained")

    def resolved(self, project_root: Path) -> "PathsConfig":
        """Return a copy with every path expanded relative to ``project_root``."""
        updates = {
            field_name: resolve_config_path(path, project_root)
            for field_name, path in self.model_dump().items()
        }
        return self.model_copy(update=updates)


class EnvironmentConfig(ConfigModel):
    """Requirements evaluated by the ``doctor`` command."""

    require_cuda: bool = True
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    minimum_free_disk_gb: float = Field(default=20.0, ge=0.0)
    allow_official_asset_download: bool = False

    @field_validator("gpu_ids")
    @classmethod
    def unique_gpu_ids(cls, value: list[int]) -> list[int]:
        """Require non-negative, unique GPU identifiers in stable order."""
        if any(gpu_id < 0 for gpu_id in value):
            raise ValueError("environment.gpu_ids cannot contain negative values")
        if len(value) != len(set(value)):
            raise ValueError("environment.gpu_ids cannot contain duplicates")
        return value


class ModelConfig(ConfigModel):
    """RVC model identity and feature-extraction settings."""

    name: str = "character_voice_v001"
    version: Literal["v1", "v2"] = "v2"
    sample_rate: Literal["32k", "40k", "48k"] = "40k"
    use_f0: bool = True
    f0_method: str = "rmvpe"
    speaker_id: int = Field(default=0, ge=0)

    @field_validator("name", "f0_method")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Normalize values that are used in filenames or RVC arguments."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned


class QualityConfig(ConfigModel):
    """Thresholds for numerical audio-quality classification."""

    minimum_file_duration_seconds: float = Field(default=0.5, gt=0.0)
    maximum_file_duration_seconds: float = Field(default=600.0, gt=0.0)
    minimum_total_accepted_duration_minutes: float = Field(default=5.0, ge=0.0)
    recommended_total_duration_minutes: float = Field(default=10.0, ge=0.0)
    minimum_sample_rate: int = Field(default=22_050, gt=0)
    target_integrated_lufs: float = -20.0
    acceptable_lufs_min: float = -32.0
    acceptable_lufs_max: float = -12.0
    maximum_sample_peak_dbfs: float = -0.1
    maximum_clipping_ratio: float = Field(default=0.0001, ge=0.0, le=1.0)
    maximum_dc_offset: float = Field(default=0.02, ge=0.0, le=1.0)
    silence_threshold_dbfs: float = -50.0
    maximum_silence_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_estimated_snr_db: float = 15.0
    recommended_estimated_snr_db: float = 25.0
    fail_on_corrupt_file: bool = False
    fail_on_insufficient_total_duration: bool = True

    @model_validator(mode="after")
    def validate_ranges(self) -> "QualityConfig":
        """Validate relationships between lower, target, and upper thresholds."""
        if self.minimum_file_duration_seconds >= self.maximum_file_duration_seconds:
            raise ValueError("minimum file duration must be below maximum file duration")
        if self.minimum_total_accepted_duration_minutes > self.recommended_total_duration_minutes:
            raise ValueError("minimum accepted duration cannot exceed recommended duration")
        if not self.acceptable_lufs_min <= self.target_integrated_lufs <= self.acceptable_lufs_max:
            raise ValueError("target LUFS must be inside the acceptable LUFS range")
        if self.minimum_estimated_snr_db > self.recommended_estimated_snr_db:
            raise ValueError("minimum SNR cannot exceed recommended SNR")
        return self


class LoudnessNormalizationConfig(ConfigModel):
    """Bounded gain-based loudness normalization settings."""

    enabled: bool = True
    target_lufs: float = -20.0
    maximum_gain_db: float = 12.0
    minimum_gain_db: float = -12.0
    true_peak_limit_db: float = -3.0

    @model_validator(mode="after")
    def validate_gain_bounds(self) -> "LoudnessNormalizationConfig":
        """Ensure the configured gain interval is ordered."""
        if self.minimum_gain_db > self.maximum_gain_db:
            raise ValueError("minimum_gain_db cannot exceed maximum_gain_db")
        return self


class HighPassFilterConfig(ConfigModel):
    """Optional conservative high-pass filter settings."""

    enabled: bool = False
    cutoff_hz: float = Field(default=60.0, gt=0.0)


class NoiseReductionConfig(ConfigModel):
    """Optional noise-reduction switch, deliberately disabled by default."""

    enabled: bool = False


class PreprocessingConfig(ConfigModel):
    """Non-destructive normalization settings for accepted training audio."""

    target_sample_rate: int = Field(default=40_000, gt=0)
    output_subtype: str = "PCM_16"
    mono: bool = True
    trim_leading_trailing_silence: bool = True
    trim_threshold_dbfs: float = -48.0
    trim_padding_ms: int = Field(default=120, ge=0)
    loudness_normalization: LoudnessNormalizationConfig = Field(
        default_factory=LoudnessNormalizationConfig
    )
    high_pass_filter: HighPassFilterConfig = Field(default_factory=HighPassFilterConfig)
    noise_reduction: NoiseReductionConfig = Field(default_factory=NoiseReductionConfig)


class SlicingConfig(ConfigModel):
    """RVC-native or internal silence-aware slicing options."""

    backend: Literal["rvc", "internal"] = "rvc"
    minimum_segment_seconds: float = Field(default=1.0, gt=0.0)
    preferred_segment_seconds: float = Field(default=4.0, gt=0.0)
    maximum_segment_seconds: float = Field(default=12.0, gt=0.0)
    silence_threshold_dbfs: float = -42.0
    minimum_silence_duration_ms: int = Field(default=250, ge=0)
    segment_padding_ms: int = Field(default=80, ge=0)

    @model_validator(mode="after")
    def validate_segment_order(self) -> "SlicingConfig":
        """Ensure minimum, preferred, and maximum durations are ordered."""
        if not (
            self.minimum_segment_seconds
            <= self.preferred_segment_seconds
            <= self.maximum_segment_seconds
        ):
            raise ValueError("segment durations must satisfy minimum <= preferred <= maximum")
        return self


class DatasetCurationConfig(ConfigModel):
    """Non-destructive raw-audio curation and human-review settings."""

    coarse_split_enabled: bool = True
    minimum_chunk_seconds: float = Field(default=120.0, gt=0.0)
    preferred_chunk_seconds: float = Field(default=210.0, gt=0.0)
    maximum_chunk_seconds: float = Field(default=300.0, gt=0.0)
    silence_threshold_dbfs: float = -42.0
    minimum_silence_duration_ms: int = Field(default=400, ge=0)
    chunk_padding_ms: int = Field(default=120, ge=0)
    review_pass_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    review_minimum_pass_samples: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def validate_chunk_order(self) -> "DatasetCurationConfig":
        """Keep coarse chunks in the configured minimum/preferred/maximum order."""

        if not (
            self.minimum_chunk_seconds
            <= self.preferred_chunk_seconds
            <= self.maximum_chunk_seconds
        ):
            raise ValueError(
                "curation chunk durations must satisfy minimum <= preferred <= maximum"
            )
        return self


class SpeakerSortingConfig(ConfigModel):
    """Optional local diarization and target-speaker review settings."""

    diarization_model: str = "pyannote/speaker-diarization-community-1"
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    token_environment_variable: str = "HF_TOKEN"
    use_gpu: bool = True
    gpu_id: int = Field(default=0, ge=0)
    minimum_speakers: Optional[int] = Field(default=None, ge=1)
    maximum_speakers: Optional[int] = Field(default=None, ge=1)
    minimum_segment_seconds: float = Field(default=1.0, gt=0.0)
    maximum_segment_seconds: float = Field(default=30.0, gt=0.0)
    merge_gap_seconds: float = Field(default=0.20, ge=0.0)
    segment_padding_seconds: float = Field(default=0.05, ge=0.0)
    target_similarity_threshold: float = Field(default=0.65, ge=-1.0, le=1.0)
    target_similarity_margin: float = Field(default=0.05, ge=0.0, le=2.0)
    maximum_embedding_segments_per_speaker: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_speaker_sorting(self) -> "SpeakerSortingConfig":
        """Reject ambiguous speaker bounds and invalid segment duration order."""

        if (
            self.minimum_speakers is not None
            and self.maximum_speakers is not None
            and self.minimum_speakers > self.maximum_speakers
        ):
            raise ValueError("minimum_speakers cannot exceed maximum_speakers")
        if self.minimum_segment_seconds > self.maximum_segment_seconds:
            raise ValueError(
                "speaker sorting segment durations must satisfy minimum <= maximum"
            )
        if not self.token_environment_variable.strip():
            raise ValueError("token_environment_variable must not be empty")
        return self


class TrainingConfig(ConfigModel):
    """RVC training, checkpoint, and OOM-retry settings."""

    epochs: int = Field(default=200, gt=0)
    save_every_epochs: int = Field(default=25, gt=0)
    save_only_latest: bool = False
    save_every_weights: bool = True
    batch_size: int = Field(default=12, gt=0)
    automatic_batch_size: bool = True
    batch_size_candidates: list[int] = Field(default_factory=lambda: [16, 12, 8, 6, 4])
    use_pretrained_model: bool = True
    use_gpu: bool = True
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    mixed_precision: bool = True
    cache_dataset_in_gpu: bool = False
    resume_if_available: bool = True
    stop_on_nan_loss: bool = True
    maximum_oom_retries: int = Field(default=4, ge=0)

    @field_validator("batch_size_candidates")
    @classmethod
    def validate_batch_candidates(cls, value: list[int]) -> list[int]:
        """Normalize batch candidates to unique descending positive values."""
        if not value or any(candidate <= 0 for candidate in value):
            raise ValueError("batch_size_candidates must contain positive integers")
        if len(value) != len(set(value)):
            raise ValueError("batch_size_candidates cannot contain duplicates")
        if value != sorted(value, reverse=True):
            raise ValueError("batch_size_candidates must be in descending order")
        return value

    @field_validator("gpu_ids")
    @classmethod
    def validate_training_gpu_ids(cls, value: list[int]) -> list[int]:
        """Reject negative or duplicate GPU identifiers."""
        if any(gpu_id < 0 for gpu_id in value) or len(value) != len(set(value)):
            raise ValueError("training.gpu_ids must be unique non-negative integers")
        return value


class IndexConfig(ConfigModel):
    """FAISS index generation settings."""

    enabled: bool = True
    algorithm: str = "auto"


class ParameterSweepConfig(ConfigModel):
    """Bounded optional inference parameter sweep."""

    enabled: bool = False
    transpose_values: list[int] = Field(default_factory=lambda: [0, 3, 6])
    index_rate_values: list[float] = Field(default_factory=lambda: [0.5, 0.65, 0.8])
    maximum_combinations_per_file: int = Field(default=9, gt=0)

    @field_validator("index_rate_values")
    @classmethod
    def validate_index_rates(cls, value: list[float]) -> list[float]:
        """Require every FAISS blend ratio to be within the supported interval."""
        if not value or any(rate < 0.0 or rate > 1.0 for rate in value):
            raise ValueError("index_rate_values must be non-empty and between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_combination_limit(self) -> "ParameterSweepConfig":
        """Reject enabled sweeps that exceed the explicit safety limit."""
        combinations = len(self.transpose_values) * len(self.index_rate_values)
        if self.enabled and combinations > self.maximum_combinations_per_file:
            raise ValueError(
                f"parameter sweep creates {combinations} combinations, exceeding "
                f"maximum_combinations_per_file={self.maximum_combinations_per_file}"
            )
        return self


class TestingConfig(ConfigModel):
    """Batch inference and converted-audio verification settings."""

    enabled: bool = True
    maximum_test_files: int = Field(default=5, gt=0)
    require_frozen_manifest: bool = False
    allow_inference_without_index: bool = False
    f0_method: str = "rmvpe"
    transpose: int = 0
    index_rate: float = Field(default=0.65, ge=0.0, le=1.0)
    filter_radius: int = Field(default=3, ge=0)
    resample_sample_rate: int = Field(default=0, ge=0)
    rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    output_format: Literal["wav", "flac"] = "wav"
    output_sample_rate: int = Field(default=48_000, gt=0)
    maximum_duration_difference_ratio: float = Field(default=0.05, ge=0.0)
    parameter_sweep: ParameterSweepConfig = Field(default_factory=ParameterSweepConfig)


class MonitoringConfig(ConfigModel):
    """Resource sampling settings used while RVC training is active."""

    enabled: bool = True
    interval_seconds: float = Field(default=30.0, gt=0.0)
    record_gpu: bool = True
    record_cpu: bool = True
    record_memory: bool = True
    record_disk: bool = True


class ReportConfig(ConfigModel):
    """Local report output switches."""

    generate_html: bool = True
    generate_csv: bool = True
    include_audio_players: bool = True


class AppConfig(ConfigModel):
    """Fully merged and validated application configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    slicing: SlicingConfig = Field(default_factory=SlicingConfig)
    curation: DatasetCurationConfig = Field(default_factory=DatasetCurationConfig)
    speaker_sorting: SpeakerSortingConfig = Field(default_factory=SpeakerSortingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    config_path: Optional[Path] = Field(default=None, exclude=True, repr=False)
    project_root: Path = Field(default_factory=lambda: Path.cwd().resolve(), exclude=True)

    @property
    def root_dir(self) -> Path:
        """Return the absolute project root used for all relative paths."""
        return self.project_root

    def resolve_path(self, value: str | Path) -> Path:
        """Expand ``value`` against this configuration's project root."""
        return resolve_config_path(Path(value), self.project_root)

    def serializable_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe resolved configuration mapping."""
        return self.model_dump(
            mode="json",
            exclude={"config_path", "project_root"},
            round_trip=True,
        )


RVCConfig = AppConfig
Config = AppConfig


def infer_project_root(config_path: Path) -> Path:
    """Infer the project root that owns ``config_path``.

    A nearest ``pyproject.toml`` or ``.git`` marker wins.  For the conventional
    ``<root>/config/name.yaml`` layout, ``<root>`` is used even before project
    scaffolding exists.  Arbitrary standalone files resolve against their own
    parent directory, which keeps tests and portable example configs intuitive.
    """
    path = config_path.expanduser().resolve(strict=False)
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    if path.parent.name.casefold() == "config":
        return path.parent.parent
    return path.parent


def resolve_config_path(value: Path, project_root: Path) -> Path:
    """Expand environment/user markers and make a path project-root relative."""
    expanded_text = os.path.expandvars(os.path.expanduser(str(value)))
    expanded = Path(expanded_text)
    if not expanded.is_absolute():
        expanded = project_root / expanded
    return expanded.resolve(strict=False)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read one UTF-8 YAML mapping with consistent configuration errors."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"cannot read valid UTF-8 YAML: {exc}", config_path=path
        ) from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigurationError("top-level YAML value must be a mapping", config_path=path)
    return dict(raw)


def merge_config_mappings(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge YAML mappings while replacing scalar/list values.

    A fresh dictionary is always returned, so loading a user configuration
    never mutates parsed defaults or leaks state between calls.
    """
    merged: dict[str, Any] = {
        str(key): dict(value) if isinstance(value, Mapping) else value
        for key, value in defaults.items()
    }
    for key, override in overrides.items():
        existing = merged.get(str(key))
        if isinstance(existing, Mapping) and isinstance(override, Mapping):
            merged[str(key)] = merge_config_mappings(existing, override)
        else:
            merged[str(key)] = override
    return merged


def load_config(
    config_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    default_config_path: Optional[Union[str, Path]] = None,
) -> AppConfig:
    """Load a UTF-8 YAML file, apply defaults, validate it, and resolve paths.

    Args:
        config_path: YAML configuration file to read.
        project_root: Optional explicit base directory for relative paths.

    Raises:
        ConfigurationError: If the file cannot be read, parsed, or validated.
    """
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise ConfigurationError("file does not exist", config_path=path)

    root = (
        Path(project_root).expanduser().resolve(strict=False)
        if project_root is not None
        else infer_project_root(path)
    )
    raw = _read_yaml_mapping(path)
    if default_config_path is not None:
        default_path = Path(default_config_path).expanduser().resolve(strict=False)
        if not default_path.is_file():
            raise ConfigurationError("default file does not exist", config_path=default_path)
    else:
        default_path = root / "config" / "default.yaml"

    if default_path.is_file() and default_path.resolve(strict=False) != path:
        defaults = _read_yaml_mapping(default_path)
        raw = merge_config_mappings(defaults, raw)
    try:
        parsed = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(str(exc), config_path=path) from exc

    return parsed.model_copy(
        update={
            "config_path": path,
            "project_root": root,
            "paths": parsed.paths.resolved(root),
        }
    )


def save_resolved_config(config: AppConfig, destination: Union[str, Path]) -> Path:
    """Write the complete resolved configuration as portable UTF-8 YAML."""
    target = Path(destination)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(
            config.serializable_dict(),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"cannot write resolved configuration: {exc}", config_path=target
        ) from exc
    return target


def load_yaml_config(
    config_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    default_config_path: Optional[Union[str, Path]] = None,
) -> AppConfig:
    """Compatibility alias for :func:`load_config`."""
    return load_config(
        config_path,
        project_root=project_root,
        default_config_path=default_config_path,
    )


__all__ = [
    "AppConfig",
    "Config",
    "DatasetCurationConfig",
    "EnvironmentConfig",
    "HighPassFilterConfig",
    "IndexConfig",
    "LoudnessNormalizationConfig",
    "ModelConfig",
    "MonitoringConfig",
    "NoiseReductionConfig",
    "ParameterSweepConfig",
    "PathsConfig",
    "PreprocessingConfig",
    "ProjectConfig",
    "QualityConfig",
    "RVCConfig",
    "ReportConfig",
    "SlicingConfig",
    "SpeakerSortingConfig",
    "TestingConfig",
    "TrainingConfig",
    "infer_project_root",
    "load_config",
    "load_yaml_config",
    "merge_config_mappings",
    "resolve_config_path",
    "save_resolved_config",
]
