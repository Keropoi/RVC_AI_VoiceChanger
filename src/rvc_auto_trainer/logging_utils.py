"""Consistent run/stage-aware logging for pipeline, command, and training logs."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator, Mapping

from .exceptions import RVCAutoTrainerError

LOGGER_NAME = "rvc_auto_trainer"
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | run=%(run_id)s | stage=%(pipeline_stage)s | "
    "%(name)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_run_id_context: ContextVar[str] = ContextVar("rvc_run_id", default="-")
_stage_context: ContextVar[str] = ContextVar("rvc_pipeline_stage", default="-")


class LoggingSetupError(RVCAutoTrainerError):
    """Raised when required log files cannot be initialized."""


class ContextFilter(logging.Filter):
    """Inject context-variable values expected by the shared formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Populate missing contextual fields and retain every record."""
        if not hasattr(record, "run_id"):
            record.run_id = _run_id_context.get()
        if not hasattr(record, "pipeline_stage"):
            record.pipeline_stage = _stage_context.get()
        return True


class PipelineLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that can override run and stage for a single component."""

    def process(
        self,
        msg: object,
        kwargs: dict[str, Any],
    ) -> tuple[object, dict[str, Any]]:
        """Merge contextual ``extra`` values without discarding caller fields."""
        caller_extra = dict(kwargs.get("extra") or {})
        caller_extra.update(self.extra or {})
        kwargs["extra"] = caller_extra
        return msg, kwargs


def _formatter() -> logging.Formatter:
    """Create the formatter used by every output destination."""
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)


def _mark_handler(handler: logging.Handler) -> logging.Handler:
    """Tag application-owned handlers so reconfiguration does not affect others."""
    setattr(handler, "_rvc_auto_trainer_handler", True)
    handler.addFilter(ContextFilter())
    handler.setFormatter(_formatter())
    return handler


def _file_handler(path: Path, *, level: int) -> logging.FileHandler:
    """Build a UTF-8 file handler suitable for Windows and multilingual paths."""
    handler = logging.FileHandler(path, encoding="utf-8", delay=False)
    handler.setLevel(level)
    return _mark_handler(handler)  # type: ignore[return-value]


def _remove_owned_handlers(logger: logging.Logger) -> None:
    """Close handlers installed by this module while retaining third-party ones."""
    for handler in tuple(logger.handlers):
        if getattr(handler, "_rvc_auto_trainer_handler", False):
            logger.removeHandler(handler)
            handler.flush()
            handler.close()


def setup_logging(
    log_dir: str | Path,
    *,
    run_id: str = "-",
    stage: str = "-",
    level: int | str = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure the package logger and the four required UTF-8 log files.

    Repeated calls safely close only handlers previously installed by this
    module, preventing duplicate lines during tests and resume operations.
    """
    directory = Path(log_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LoggingSetupError(f"Cannot create log directory '{directory}': {exc}") from exc

    numeric_level = logging._nameToLevel.get(level.upper(), 0) if isinstance(level, str) else level
    if not isinstance(numeric_level, int) or numeric_level <= 0:
        raise LoggingSetupError(f"Invalid logging level: {level!r}")

    _run_id_context.set(run_id)
    _stage_context.set(stage)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False
    _remove_owned_handlers(logger)

    try:
        logger.addHandler(_file_handler(directory / "pipeline.log", level=numeric_level))
        logger.addHandler(_file_handler(directory / "errors.log", level=logging.ERROR))
        if console:
            stream = logging.StreamHandler()
            stream.setLevel(numeric_level)
            logger.addHandler(_mark_handler(stream))

        _configure_dedicated_logger("commands", directory / "commands.log", numeric_level)
        _configure_dedicated_logger("training", directory / "training.log", numeric_level)
    except OSError as exc:
        close_logging(logger)
        raise LoggingSetupError(f"Cannot initialize log files in '{directory}': {exc}") from exc
    return logger


def _configure_dedicated_logger(name: str, path: Path, level: int) -> logging.Logger:
    """Configure a non-propagating child logger for verbose specialized output."""
    logger = logging.getLogger(f"{LOGGER_NAME}.{name}")
    logger.setLevel(level)
    logger.propagate = False
    _remove_owned_handlers(logger)
    logger.addHandler(_file_handler(path, level=level))
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or a consistently named child logger."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def get_command_logger() -> logging.Logger:
    """Return the logger dedicated to sanitized external command records."""
    return logging.getLogger(f"{LOGGER_NAME}.commands")


def get_training_logger() -> logging.Logger:
    """Return the logger dedicated to detailed RVC training output."""
    return logging.getLogger(f"{LOGGER_NAME}.training")


def logger_adapter(
    logger: logging.Logger | None = None,
    *,
    run_id: str | None = None,
    stage: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> PipelineLoggerAdapter:
    """Create a logger adapter with explicit per-component context."""
    merged = dict(extra or {})
    if run_id is not None:
        merged["run_id"] = run_id
    if stage is not None:
        merged["pipeline_stage"] = stage
    return PipelineLoggerAdapter(logger or get_logger(), merged)


@contextmanager
def logging_context(
    *,
    run_id: str | None = None,
    stage: str | None = None,
) -> Iterator[None]:
    """Temporarily set run/stage values for every log emitted in this context."""
    run_token: Token[str] | None = None
    stage_token: Token[str] | None = None
    if run_id is not None:
        run_token = _run_id_context.set(run_id)
    if stage is not None:
        stage_token = _stage_context.set(stage)
    try:
        yield
    finally:
        if stage_token is not None:
            _stage_context.reset(stage_token)
        if run_token is not None:
            _run_id_context.reset(run_token)


def close_logging(logger: logging.Logger | None = None) -> None:
    """Flush and close all application-owned handlers."""
    target = logger or logging.getLogger(LOGGER_NAME)
    _remove_owned_handlers(target)
    for child_name in ("commands", "training"):
        _remove_owned_handlers(logging.getLogger(f"{LOGGER_NAME}.{child_name}"))


configure_logging = setup_logging
log_context = logging_context


__all__ = [
    "ContextFilter",
    "LOGGER_NAME",
    "LoggingSetupError",
    "PipelineLoggerAdapter",
    "close_logging",
    "configure_logging",
    "get_command_logger",
    "get_logger",
    "get_training_logger",
    "log_context",
    "logger_adapter",
    "logging_context",
    "setup_logging",
]
