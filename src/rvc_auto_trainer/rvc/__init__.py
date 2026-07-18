"""Runtime-inspected adapter for external RVC repositories."""

from .adapter import RVCAdapter, RVCAdapterError, RVCCommandError
from .feature_extraction import FeatureRequest
from .index_builder import IndexRequest, validate_index
from .inference import InferenceRequest, InferenceResult
from .preprocess import PreprocessRequest
from .repository_inspector import (
    RepositoryInspector,
    RVCRepositoryInfo,
    RVCRepositoryInspectionError,
    ScriptInfo,
    inspect_repository,
)
from .training import (
    ArtifactValidation,
    TrainingRequest,
    batch_size_attempts,
    detect_cuda_oom,
    discover_valid_checkpoints,
    validate_artifact,
)

__all__ = [
    "ArtifactValidation",
    "FeatureRequest",
    "IndexRequest",
    "InferenceRequest",
    "InferenceResult",
    "PreprocessRequest",
    "RVCAdapter",
    "RVCAdapterError",
    "RVCCommandError",
    "RVCRepositoryInfo",
    "RVCRepositoryInspectionError",
    "RepositoryInspector",
    "ScriptInfo",
    "TrainingRequest",
    "batch_size_attempts",
    "detect_cuda_oom",
    "discover_valid_checkpoints",
    "inspect_repository",
    "validate_artifact",
    "validate_index",
]
