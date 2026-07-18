"""Recursive, deterministic discovery of user supplied audio files.

Discovery deliberately does not execute or decode anything in an input tree.  It
only applies conservative filename/filesystem filters and computes a content
hash.  Decodability is handled by :mod:`rvc_auto_trainer.audio.decoder` and the
quality audit.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

SUPPORTED_AUDIO_EXTENSIONS: FrozenSet[str] = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
)
"""Extensions accepted by the orchestration layer (case-insensitive)."""

_TEMPORARY_SUFFIXES = (
    ".tmp",
    ".temp",
    ".part",
    ".partial",
    ".crdownload",
    ".download",
    ".bak",
    ".swp",
)
_WINDOWS_HIDDEN_ATTRIBUTE = 0x2


@dataclass(frozen=True)
class AudioFile:
    """A discovered audio input with a stable path and content identity."""

    path: Path
    relative_path: str
    sha256: str
    size_bytes: int
    extension: str

    def to_manifest_record(self) -> Dict[str, object]:
        """Return the discovery fields used in an input manifest."""

        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "extension": self.extension,
            "status": "discovered",
        }


@dataclass(frozen=True)
class IgnoredFile:
    """A regular file omitted from discovery, together with the reason."""

    path: Path
    relative_path: str
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    """Complete result of scanning one input root."""

    root: Path
    files: Tuple[AudioFile, ...]
    ignored: Tuple[IgnoredFile, ...]

    @property
    def accepted(self) -> Tuple[AudioFile, ...]:
        """Alias that reads naturally in pipeline code."""

        return self.files

    @property
    def total_size_bytes(self) -> int:
        """Total byte size of all accepted inputs."""

        return sum(item.size_bytes for item in self.files)


class AudioDiscoveryError(RuntimeError):
    """Raised when an input tree cannot be safely enumerated or hashed."""


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without loading the entire audio file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_audio_files(
    root: Union[str, Path],
    supported_extensions: Optional[Iterable[str]] = None,
) -> DiscoveryResult:
    """Recursively discover supported audio beneath *root*.

    Results are sorted by relative path using case-insensitive ordering so that
    test-audio selection and manifests stay stable on Windows.  Dot-hidden and
    Windows-hidden files/directories, temporary files, symlinks, zero-byte files,
    and unsupported extensions are reported in ``ignored``.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise AudioDiscoveryError("Audio input directory does not exist: {}".format(root_path))
    if not root_path.is_dir():
        raise AudioDiscoveryError("Audio input path is not a directory: {}".format(root_path))

    extensions = _normalise_extensions(supported_extensions)
    accepted: List[AudioFile] = []
    ignored: List[IgnoredFile] = []
    try:
        candidates = sorted(
            root_path.rglob("*"),
            key=lambda item: _relative_posix(item, root_path).casefold(),
        )
    except OSError as exc:
        raise AudioDiscoveryError("Unable to enumerate audio directory: {}".format(root_path)) from exc

    for candidate in candidates:
        relative_path = _relative_posix(candidate, root_path)
        try:
            if candidate.is_dir():
                continue
            reason = _ignore_reason(candidate, root_path, extensions)
            if reason is not None:
                ignored.append(IgnoredFile(candidate, relative_path, reason))
                continue
            stat = candidate.stat()
            digest = sha256_file(candidate)
        except OSError as exc:
            ignored.append(
                IgnoredFile(candidate, relative_path, "unreadable: {}".format(exc))
            )
            continue

        accepted.append(
            AudioFile(
                path=candidate,
                relative_path=relative_path,
                sha256=digest,
                size_bytes=stat.st_size,
                extension=candidate.suffix.lower(),
            )
        )

    return DiscoveryResult(root_path, tuple(accepted), tuple(ignored))


def find_audio_files(
    root: Union[str, Path],
    supported_extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Compatibility helper returning only accepted paths."""

    return [item.path for item in discover_audio_files(root, supported_extensions).files]


def select_test_audio(files: Sequence[AudioFile], maximum_files: int) -> Tuple[AudioFile, ...]:
    """Select the first deterministic set of test inputs by relative filename."""

    if maximum_files < 0:
        raise ValueError("maximum_files cannot be negative")
    ordered = sorted(files, key=lambda item: item.relative_path.casefold())
    return tuple(ordered[:maximum_files])


def duplicate_hashes(
    training_files: Sequence[AudioFile], test_files: Sequence[AudioFile]
) -> Dict[str, Tuple[Tuple[AudioFile, ...], Tuple[AudioFile, ...]]]:
    """Return content hashes that appear in both training and test sets."""

    training_by_hash: Dict[str, List[AudioFile]] = {}
    test_by_hash: Dict[str, List[AudioFile]] = {}
    for item in training_files:
        training_by_hash.setdefault(item.sha256, []).append(item)
    for item in test_files:
        test_by_hash.setdefault(item.sha256, []).append(item)

    return {
        digest: (tuple(training_by_hash[digest]), tuple(test_by_hash[digest]))
        for digest in sorted(set(training_by_hash).intersection(test_by_hash))
    }


def _normalise_extensions(extensions: Optional[Iterable[str]]) -> FrozenSet[str]:
    values = SUPPORTED_AUDIO_EXTENSIONS if extensions is None else extensions
    normalised = set()
    for extension in values:
        value = str(extension).strip().lower()
        if not value:
            continue
        normalised.add(value if value.startswith(".") else ".{}".format(value))
    if not normalised:
        raise ValueError("At least one supported audio extension is required")
    return frozenset(normalised)


def _ignore_reason(path: Path, root: Path, extensions: FrozenSet[str]) -> Optional[str]:
    if path.is_symlink():
        return "symbolic_link"
    if _is_hidden(path, root):
        return "hidden"
    if _is_temporary(path.name):
        return "temporary"
    if path.suffix.lower() not in extensions:
        return "unsupported_extension"
    if path.stat().st_size == 0:
        return "zero_bytes"
    if not path.is_file():
        return "not_a_regular_file"
    return None


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True

    current = root
    for part in relative.parts:
        if part.startswith("."):
            return True
        current = current / part
        try:
            attributes = getattr(current.stat(), "st_file_attributes", 0)
        except OSError:
            continue
        if attributes & _WINDOWS_HIDDEN_ATTRIBUTE:
            return True
    return False


def _is_temporary(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered.startswith("~")
        or lowered.endswith("~")
        or lowered.endswith(_TEMPORARY_SUFFIXES)
        or ".tmp." in lowered
        or ".temp." in lowered
    )


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return os.fspath(path)
