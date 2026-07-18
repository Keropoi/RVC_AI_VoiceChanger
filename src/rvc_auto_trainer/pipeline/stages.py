"""Typed pipeline stage definitions and execution records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

from ..state import PipelineStage, StageResult

if TYPE_CHECKING:
    from ..rvc.adapter import RVCAdapter
    from .run_context import RunContext


@dataclass(frozen=True)
class StageExecutionContext:
    """Objects available to one injected stage handler."""

    run_context: RunContext
    adapter: RVCAdapter
    stage: PipelineStage
    dry_run: bool = False


class StageHandler(Protocol):
    """Callable contract used by the orchestrator and mock integration tests."""

    def __call__(self, context: StageExecutionContext) -> StageResult:
        ...


FingerprintProvider = Callable[[StageExecutionContext], str]
EnabledPredicate = Callable[[StageExecutionContext], bool]
RequiredOutputsProvider = Callable[[StageExecutionContext], Sequence[Path]]


@dataclass(frozen=True)
class StageDefinition:
    """Execution and reuse policy for one persisted pipeline stage."""

    stage: PipelineStage
    handler: StageHandler
    description: str = ""
    fingerprint: FingerprintProvider | None = None
    enabled: EnabledPredicate | None = None
    required_outputs: RequiredOutputsProvider | None = None
    skip_reason: str = "Disabled by resolved configuration"


@dataclass(frozen=True)
class StageRunRecord:
    """In-memory account of a stage during one orchestrator invocation."""

    stage: PipelineStage
    result: StageResult
    reused: bool = False
    skipped: bool = False


EXECUTABLE_STAGES: tuple[PipelineStage, ...] = tuple(
    stage for stage in PipelineStage if stage is not PipelineStage.INITIALIZED
)


def stable_stage_fingerprint(stage: PipelineStage, payload: Any) -> str:
    """Hash JSON-compatible stage inputs for cache/reuse decisions."""

    serialized = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(stage.value.encode("utf-8"))
    digest.update(b"\0")
    digest.update(serialized)
    return digest.hexdigest()


def resolved_config_fingerprint(context: StageExecutionContext) -> str:
    """Conservative default fingerprint of resolved config for a stage."""

    config = context.run_context.config
    if hasattr(config, "serializable_dict"):
        payload = config.serializable_dict()
    elif hasattr(config, "model_dump"):
        payload = config.model_dump(mode="json")
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = repr(config)
    return stable_stage_fingerprint(context.stage, payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    return value
