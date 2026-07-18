"""Safe creation and resumption of isolated per-run directory trees."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..config import AppConfig, load_config, save_resolved_config
from ..exceptions import RunContextError
from ..state import PipelineState, StateStore

RUN_SUBDIRECTORIES: tuple[str, ...] = (
    "quality",
    "preprocessed",
    "rvc_workspace",
    "checkpoints",
    "artifacts",
    "test_results",
    "logs",
    "report",
)

_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def sanitize_run_component(value: str, *, fallback: str = "model") -> str:
    """Create a readable Windows-safe run-ID component while preserving Unicode."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = _WINDOWS_INVALID.sub("_", normalized)
    normalized = _WHITESPACE.sub("_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip(" ._")
    if not normalized:
        normalized = fallback
    if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        normalized = f"_{normalized}"
    return normalized[:80].rstrip(" .") or fallback


def generate_run_id(model_name: str, *, now: datetime | None = None) -> str:
    """Generate the timestamp/model run identifier specified by the project."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{sanitize_run_component(model_name)}"


@dataclass
class RunContext:
    """Resolved configuration, paths, and state store for exactly one run."""

    config: AppConfig
    run_id: str
    run_dir: Path
    state_store: StateStore

    @classmethod
    def create(
        cls,
        config: AppConfig,
        run_id: str | None = None,
        *,
        now: datetime | None = None,
    ) -> "RunContext":
        """Allocate a unique run directory, write resolved config, and initialize state."""
        base_id = sanitize_run_component(run_id) if run_id else generate_run_id(config.model.name, now=now)
        runs_dir = config.paths.runs_dir
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            actual_id, directory = _allocate_unique_directory(runs_dir, base_id)
            context = cls(
                config=config,
                run_id=actual_id,
                run_dir=directory,
                state_store=StateStore(directory / "state.json", actual_id),
            )
            context.ensure_layout()
            save_resolved_config(config, context.config_resolved_path)
            context.state_store.initialize(actual_id)
        except (OSError, RunContextError) as exc:
            if isinstance(exc, RunContextError):
                raise
            raise RunContextError(f"Cannot create run directory under '{runs_dir}': {exc}") from exc
        return context

    @classmethod
    def load(
        cls,
        run_dir: str | Path,
        config: AppConfig | None = None,
    ) -> "RunContext":
        """Validate and load an existing run without overwriting its artifacts."""
        directory = Path(run_dir).expanduser().resolve(strict=False)
        if not directory.is_dir():
            raise RunContextError(f"Run directory does not exist: '{directory}'")
        resolved_config_path = directory / "config_resolved.yaml"
        effective_config = config
        if effective_config is None:
            if not resolved_config_path.is_file():
                raise RunContextError(
                    f"Run has no resolved configuration: '{resolved_config_path}'"
                )
            effective_config = load_config(
                resolved_config_path,
                project_root=directory.parent.parent,
            )
        store = StateStore(directory / "state.json")
        state = store.load()
        if state.run_id != directory.name:
            raise RunContextError(
                f"State run_id '{state.run_id}' does not match directory '{directory.name}'"
            )
        context = cls(
            config=effective_config,
            run_id=state.run_id,
            run_dir=directory,
            state_store=store,
        )
        context.ensure_layout()
        return context

    @classmethod
    def from_run_id(cls, config: AppConfig, run_id: str) -> "RunContext":
        """Load ``run_id`` from the configured runs directory."""
        return cls.load(config.paths.runs_dir / run_id, config=config)

    @property
    def root_dir(self) -> Path:
        """Compatibility alias for the root of this run."""
        return self.run_dir

    @property
    def state(self) -> PipelineState:
        """Load the latest atomic state snapshot."""
        return self.state_store.load()

    @property
    def state_path(self) -> Path:
        """Return the state JSON path."""
        return self.run_dir / "state.json"

    @property
    def config_resolved_path(self) -> Path:
        """Return the final merged configuration snapshot path."""
        return self.run_dir / "config_resolved.yaml"

    @property
    def input_manifest_path(self) -> Path:
        """Return the input audio manifest path."""
        return self.run_dir / "input_manifest.json"

    @property
    def environment_report_path(self) -> Path:
        """Return the JSON environment report path."""
        return self.run_dir / "environment_report.json"

    @property
    def environment_report_text_path(self) -> Path:
        """Return the text environment report path."""
        return self.run_dir / "environment_report.txt"

    @property
    def quality_dir(self) -> Path:
        """Return quality report output directory."""
        return self.run_dir / "quality"

    @property
    def preprocessed_dir(self) -> Path:
        """Return normalized/sliced training audio directory."""
        return self.run_dir / "preprocessed"

    @property
    def rvc_workspace_dir(self) -> Path:
        """Return run-isolated external RVC workspace directory."""
        return self.run_dir / "rvc_workspace"

    @property
    def checkpoints_dir(self) -> Path:
        """Return checkpoint and checkpoint-manifest directory."""
        return self.run_dir / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        """Return verified final model/index artifact directory."""
        return self.run_dir / "artifacts"

    @property
    def test_results_dir(self) -> Path:
        """Return batch test inference result directory."""
        return self.run_dir / "test_results"

    @property
    def logs_dir(self) -> Path:
        """Return pipeline/RVC log directory."""
        return self.run_dir / "logs"

    @property
    def report_dir(self) -> Path:
        """Return local HTML and manual-review output directory."""
        return self.run_dir / "report"

    def ensure_layout(self) -> None:
        """Create required subdirectories idempotently without touching files."""
        try:
            for name in RUN_SUBDIRECTORIES:
                (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RunContextError(
                f"Cannot create required subdirectories in '{self.run_dir}': {exc}"
            ) from exc

    def path(self, relative_path: str | Path) -> Path:
        """Resolve a path inside this run and reject directory traversal."""
        candidate = (self.run_dir / relative_path).resolve(strict=False)
        try:
            candidate.relative_to(self.run_dir.resolve(strict=False))
        except ValueError as exc:
            raise RunContextError(
                f"Path '{relative_path}' escapes run directory '{self.run_dir}'"
            ) from exc
        return candidate

    def write_json(self, relative_path: str | Path, payload: Mapping[str, Any]) -> Path:
        """Atomically write a run-local JSON manifest."""
        target = self.path(relative_path)
        _atomic_write_json(target, payload)
        return target


def _allocate_unique_directory(runs_dir: Path, base_id: str) -> tuple[str, Path]:
    """Atomically allocate ``base_id`` or a numbered collision-safe variant."""
    for suffix in range(1, 10_000):
        run_id = base_id if suffix == 1 else f"{base_id}_{suffix:02d}"
        candidate = runs_dir / run_id
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return run_id, candidate.resolve(strict=False)
        except FileExistsError:
            continue
        except OSError as exc:
            raise RunContextError(f"Cannot allocate run directory '{candidate}': {exc}") from exc
    raise RunContextError(f"Too many run directory collisions for base ID '{base_id}'")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically flush a UTF-8 JSON manifest."""
    temporary_path: Path | None = None
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
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RunContextError(f"Cannot atomically write JSON '{path}': {exc}") from exc


def create_run_context(
    config: AppConfig,
    run_id: str | None = None,
    *,
    now: datetime | None = None,
) -> RunContext:
    """Compatibility helper for :meth:`RunContext.create`."""
    return RunContext.create(config, run_id=run_id, now=now)


__all__ = [
    "RUN_SUBDIRECTORIES",
    "RunContext",
    "create_run_context",
    "generate_run_id",
    "sanitize_run_component",
]
