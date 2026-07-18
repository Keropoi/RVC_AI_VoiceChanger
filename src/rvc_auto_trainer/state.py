"""Atomic persisted pipeline state and ordered stage-transition rules."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .exceptions import InvalidStateTransition, StateFileError


def utc_now() -> datetime:
    """Return an aware UTC timestamp for state and manifest records."""
    return datetime.now(timezone.utc)


class PipelineStage(str, Enum):
    """Ordered stages in one complete RVC automation run."""

    INITIALIZED = "INITIALIZED"
    AUDIO_DISCOVERED = "AUDIO_DISCOVERED"
    QUALITY_CHECKED = "QUALITY_CHECKED"
    PREPROCESSED = "PREPROCESSED"
    FEATURES_EXTRACTED = "FEATURES_EXTRACTED"
    MODEL_TRAINED = "MODEL_TRAINED"
    INDEX_BUILT = "INDEX_BUILT"
    TEST_INFERENCE_COMPLETED = "TEST_INFERENCE_COMPLETED"
    REPORT_GENERATED = "REPORT_GENERATED"


STAGE_ORDER: tuple[PipelineStage, ...] = tuple(PipelineStage)


class StageStatus(str, Enum):
    """Execution status for an individual pipeline stage."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    INVALIDATED = "INVALIDATED"


class StageRecord(BaseModel):
    """Persistent execution details for one pipeline stage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: StageStatus = StageStatus.PENDING
    fingerprint: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    outputs: tuple[Path, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    attempts: int = Field(default=0, ge=0)


class StageResult(BaseModel):
    """Small, transportable result shared by adapters and pipeline stages."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    success: bool = True
    outputs: tuple[Path, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


def _empty_stage_records() -> dict[PipelineStage, StageRecord]:
    """Build fresh records without sharing mutable state between runs."""
    return {stage: StageRecord() for stage in STAGE_ORDER}


class PipelineState(BaseModel):
    """Serializable state machine for an interruptible, resumable run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: int = 1
    run_id: str
    current_stage: PipelineStage = PipelineStage.INITIALIZED
    stages: dict[PipelineStage, StageRecord] = Field(default_factory=_empty_stage_records)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        """Reject empty run identifiers in persisted state."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("run_id must not be empty")
        return cleaned

    @field_validator("stages", mode="after")
    @classmethod
    def include_all_stages(
        cls, value: dict[PipelineStage, StageRecord]
    ) -> dict[PipelineStage, StageRecord]:
        """Backfill new stage records when loading an older compatible state."""
        return {stage: value.get(stage, StageRecord()) for stage in STAGE_ORDER}

    @classmethod
    def create(cls, run_id: str) -> "PipelineState":
        """Create a state whose run-directory initialization already succeeded."""
        now = utc_now()
        state = cls(run_id=run_id, created_at=now, updated_at=now)
        state.stages[PipelineStage.INITIALIZED] = StageRecord(
            status=StageStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            attempts=1,
        )
        return state

    def record_for(self, stage: Union[PipelineStage, str]) -> StageRecord:
        """Return the record for a stage name or enum value."""
        normalized = _coerce_stage(stage)
        return self.stages[normalized]

    @property
    def completed_stages(self) -> tuple[PipelineStage, ...]:
        """Return completed or deliberately skipped stages in canonical order."""
        reusable = {StageStatus.COMPLETED, StageStatus.SKIPPED}
        return tuple(stage for stage in STAGE_ORDER if self.stages[stage].status in reusable)

    @property
    def last_completed_stage(self) -> Optional[PipelineStage]:
        """Return the furthest contiguous completed stage, if any."""
        last: Optional[PipelineStage] = None
        for stage in STAGE_ORDER:
            if self.stages[stage].status not in {StageStatus.COMPLETED, StageStatus.SKIPPED}:
                break
            last = stage
        return last

    @property
    def next_stage(self) -> Optional[PipelineStage]:
        """Return the first stage that is not complete, or ``None`` at the end."""
        for stage in STAGE_ORDER:
            if self.stages[stage].status not in {StageStatus.COMPLETED, StageStatus.SKIPPED}:
                return stage
        return None

    def start_stage(
        self,
        stage: Union[PipelineStage, str],
        fingerprint: Optional[str] = None,
    ) -> StageRecord:
        """Move an eligible stage to ``RUNNING`` after validating predecessors."""
        normalized = _coerce_stage(stage)
        self._require_predecessor(normalized)
        record = self.stages[normalized]
        if record.status is StageStatus.RUNNING:
            raise InvalidStateTransition(f"Stage {normalized.value} is already RUNNING")
        if record.status in {StageStatus.COMPLETED, StageStatus.SKIPPED}:
            raise InvalidStateTransition(
                f"Stage {normalized.value} is already {record.status.value}; invalidate it before rerun"
            )

        now = utc_now()
        record.status = StageStatus.RUNNING
        record.fingerprint = fingerprint
        record.started_at = now
        record.completed_at = None
        record.error = None
        record.outputs = ()
        record.metadata = {}
        record.attempts += 1
        self.current_stage = normalized
        self.updated_at = now
        return record

    def complete_stage(
        self,
        stage: Union[PipelineStage, str],
        outputs: Iterable[Union[str, Path]] = (),
        metadata: Optional[Mapping[str, Any]] = None,
        fingerprint: Optional[str] = None,
    ) -> StageRecord:
        """Mark a running stage complete and retain its reusable fingerprint."""
        normalized = _coerce_stage(stage)
        record = self.stages[normalized]
        if record.status is not StageStatus.RUNNING:
            raise InvalidStateTransition(
                f"Stage {normalized.value} cannot complete from {record.status.value}; start it first"
            )
        now = utc_now()
        record.status = StageStatus.COMPLETED
        record.completed_at = now
        record.error = None
        record.outputs = tuple(Path(output) for output in outputs)
        record.metadata = dict(metadata or {})
        if fingerprint is not None:
            record.fingerprint = fingerprint
        self.current_stage = normalized
        self.updated_at = now
        return record

    def fail_stage(
        self,
        stage: Union[PipelineStage, str],
        error: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StageRecord:
        """Persist an actionable failure for a running stage."""
        normalized = _coerce_stage(stage)
        record = self.stages[normalized]
        if record.status is not StageStatus.RUNNING:
            raise InvalidStateTransition(
                f"Stage {normalized.value} cannot fail from {record.status.value}; start it first"
            )
        message = error.strip()
        if not message:
            raise ValueError("failure error message must not be empty")
        now = utc_now()
        record.status = StageStatus.FAILED
        record.completed_at = now
        record.error = message
        record.metadata = dict(metadata or {})
        self.current_stage = normalized
        self.updated_at = now
        return record

    def skip_stage(
        self,
        stage: Union[PipelineStage, str],
        *,
        reason: str,
        fingerprint: Optional[str] = None,
    ) -> StageRecord:
        """Deliberately skip an eligible stage while retaining the reason."""
        normalized = _coerce_stage(stage)
        self._require_predecessor(normalized)
        record = self.stages[normalized]
        if record.status not in {StageStatus.PENDING, StageStatus.INVALIDATED, StageStatus.FAILED}:
            raise InvalidStateTransition(
                f"Stage {normalized.value} cannot be skipped from {record.status.value}"
            )
        now = utc_now()
        record.status = StageStatus.SKIPPED
        record.started_at = record.started_at or now
        record.completed_at = now
        record.fingerprint = fingerprint
        record.error = None
        record.metadata = {"skip_reason": reason}
        self.current_stage = normalized
        self.updated_at = now
        return record

    def mark_completed(
        self,
        stage: Union[PipelineStage, str],
        *,
        fingerprint: Optional[str] = None,
        outputs: Iterable[Union[str, Path]] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StageRecord:
        """Convenience transition that starts and immediately completes a stage."""
        self.start_stage(stage, fingerprint=fingerprint)
        return self.complete_stage(
            stage,
            outputs=outputs,
            metadata=metadata,
            fingerprint=fingerprint,
        )

    def transition_to(self, stage: Union[PipelineStage, str]) -> StageRecord:
        """Compatibility shorthand for completing the next ordered stage."""
        return self.mark_completed(stage)

    transition = transition_to

    def can_reuse(
        self,
        stage: Union[PipelineStage, str],
        fingerprint: str,
        required_outputs: Iterable[Union[str, Path]] = (),
        *,
        output_root: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Return whether a completed fingerprint and all required files remain valid."""
        record = self.record_for(stage)
        if record.status is not StageStatus.COMPLETED or record.fingerprint != fingerprint:
            return False
        root = Path(output_root) if output_root is not None else None
        for raw_output in required_outputs:
            output = Path(raw_output)
            if root is not None and not output.is_absolute():
                output = root / output
            if not output.exists():
                return False
        return True

    def invalidate_from(self, stage: Union[PipelineStage, str], *, reason: str = "") -> None:
        """Invalidate ``stage`` and every dependent later stage without deleting outputs."""
        normalized = _coerce_stage(stage)
        start_index = STAGE_ORDER.index(normalized)
        now = utc_now()
        for affected in STAGE_ORDER[start_index:]:
            record = self.stages[affected]
            record.status = StageStatus.INVALIDATED
            record.completed_at = None
            record.error = None
            record.metadata = {"invalidation_reason": reason} if reason else {}
        previous_index = max(0, start_index - 1)
        self.current_stage = STAGE_ORDER[previous_index]
        self.updated_at = now

    def _require_predecessor(self, stage: PipelineStage) -> None:
        """Require every direct predecessor to have completed or been skipped."""
        index = STAGE_ORDER.index(stage)
        if index == 0:
            return
        predecessor = STAGE_ORDER[index - 1]
        status = self.stages[predecessor].status
        if status not in {StageStatus.COMPLETED, StageStatus.SKIPPED}:
            raise InvalidStateTransition(
                f"Cannot start {stage.value}: predecessor {predecessor.value} is {status.value}"
            )


class StateStore:
    """Load, mutate, and atomically save one run's ``state.json``."""

    def __init__(self, path: Union[str, Path], run_id: Optional[str] = None) -> None:
        self.path = Path(path)
        self.run_id = run_id

    @property
    def exists(self) -> bool:
        """Return whether the final state file exists."""
        return self.path.is_file()

    def initialize(self, run_id: Optional[str] = None) -> PipelineState:
        """Create initial state without overwriting an existing run."""
        effective_run_id = run_id or self.run_id or self.path.parent.name
        if self.exists:
            raise StateFileError(f"Refusing to overwrite existing state file '{self.path}'")
        state = PipelineState.create(effective_run_id)
        self.save(state)
        self.run_id = state.run_id
        return state

    def load(self) -> PipelineState:
        """Read and validate the state JSON file."""
        if not self.exists:
            raise StateFileError(f"State file does not exist: '{self.path}'")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            state = PipelineState.model_validate(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise StateFileError(f"Cannot load state file '{self.path}': {exc}") from exc
        self.run_id = state.run_id
        return state

    def save(self, state: PipelineState) -> Path:
        """Atomically replace state JSON after serializing a validated model."""
        if self.run_id is not None and state.run_id != self.run_id:
            raise StateFileError(
                f"State run_id '{state.run_id}' does not match store run_id '{self.run_id}'"
            )
        payload = state.model_dump(mode="json", round_trip=True)
        _atomic_write_state(self.path, payload)
        self.run_id = state.run_id
        return self.path

    def start_stage(
        self,
        stage: Union[PipelineStage, str],
        fingerprint: Optional[str] = None,
    ) -> PipelineState:
        """Start and atomically persist a stage."""
        state = self.load()
        state.start_stage(stage, fingerprint=fingerprint)
        self.save(state)
        return state

    def complete_stage(
        self,
        stage: Union[PipelineStage, str],
        outputs: Iterable[Union[str, Path]] = (),
        metadata: Optional[Mapping[str, Any]] = None,
        fingerprint: Optional[str] = None,
    ) -> PipelineState:
        """Complete and atomically persist a stage."""
        state = self.load()
        state.complete_stage(
            stage,
            outputs=outputs,
            metadata=metadata,
            fingerprint=fingerprint,
        )
        self.save(state)
        return state

    def fail_stage(
        self,
        stage: Union[PipelineStage, str],
        error: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PipelineState:
        """Fail and atomically persist a stage."""
        state = self.load()
        state.fail_stage(stage, error=error, metadata=metadata)
        self.save(state)
        return state

    def can_reuse(
        self,
        stage: Union[PipelineStage, str],
        fingerprint: str,
        required_outputs: Iterable[Union[str, Path]] = (),
    ) -> bool:
        """Evaluate persisted stage reuse relative to the run directory."""
        return self.load().can_reuse(
            stage,
            fingerprint,
            required_outputs,
            output_root=self.path.parent,
        )

    def invalidate_from(
        self,
        stage: Union[PipelineStage, str],
        reason: str = "",
    ) -> PipelineState:
        """Invalidate a persisted stage and all later dependencies."""
        state = self.load()
        state.invalidate_from(stage, reason=reason)
        self.save(state)
        return state


def _coerce_stage(stage: Union[PipelineStage, str]) -> PipelineStage:
    """Convert a stage string to an enum with a domain-specific error."""
    if isinstance(stage, PipelineStage):
        return stage
    try:
        return PipelineStage(stage)
    except ValueError as exc:
        raise InvalidStateTransition(f"Unknown pipeline stage: {stage!r}") from exc


def _atomic_write_state(path: Path, payload: Mapping[str, Any]) -> None:
    """Flush state to a sibling temporary file before an atomic replacement."""
    temporary_path: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise StateFileError(f"Cannot atomically write state file '{path}': {exc}") from exc


StateManager = StateStore


__all__ = [
    "PipelineStage",
    "PipelineState",
    "STAGE_ORDER",
    "StageRecord",
    "StageResult",
    "StageStatus",
    "StateManager",
    "StateStore",
    "utc_now",
]
