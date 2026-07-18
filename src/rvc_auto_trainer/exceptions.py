"""Domain-specific exceptions for :mod:`rvc_auto_trainer`.

The application deliberately raises descriptive exception types at module
boundaries.  CLI code can therefore render a useful recovery message without
having to guess whether an error came from configuration, persisted state, or
an external program.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class RVCAutoTrainerError(Exception):
    """Base class for all expected application errors."""


class ConfigurationError(RVCAutoTrainerError):
    """Raised when a configuration file is missing, malformed, or invalid."""

    def __init__(self, message: str, *, config_path: Path | None = None) -> None:
        self.config_path = config_path
        prefix = f"Configuration {config_path}: " if config_path else "Configuration: "
        super().__init__(f"{prefix}{message}")


class HashingError(RVCAutoTrainerError):
    """Raised when an input cannot be read or a hash cache cannot be decoded."""


class StateError(RVCAutoTrainerError):
    """Base class for persisted pipeline-state errors."""


class StateFileError(StateError):
    """Raised when ``state.json`` cannot be loaded or atomically written."""


class InvalidStateTransition(StateError):
    """Raised when a pipeline stage violates the state-machine ordering."""


class RunContextError(RVCAutoTrainerError):
    """Raised when a run directory cannot be created or resumed safely."""


class EnvironmentCheckError(RVCAutoTrainerError):
    """Raised when an environment report itself cannot be generated or saved."""


class AudioError(RVCAutoTrainerError):
    """Base class for audio discovery, decoding, and analysis errors."""


class AudioDecodeError(AudioError):
    """Raised when an audio file cannot be decoded."""


class RVCRepositoryError(RVCAutoTrainerError):
    """Raised when the configured RVC repository is incomplete or unsupported."""


class ExternalCommandError(RVCAutoTrainerError):
    """Raised when a safely invoked external command returns unsuccessfully."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str] = (),
        return_code: int | None = None,
        stderr_tail: Sequence[str] = (),
        stage: str | None = None,
    ) -> None:
        self.command = tuple(command)
        self.return_code = return_code
        self.stderr_tail = tuple(stderr_tail)
        self.stage = stage

        details: list[str] = [message]
        if stage:
            details.append(f"stage={stage}")
        if return_code is not None:
            details.append(f"return_code={return_code}")
        if self.stderr_tail:
            details.append("stderr_tail=" + " | ".join(self.stderr_tail))
        super().__init__("; ".join(details))


# Backwards-friendly aliases for callers that prefer shorter names.
ConfigError = ConfigurationError
PipelineStateError = StateError
CommandExecutionError = ExternalCommandError
