"""SHA-256 helpers, deterministic stage fingerprints, and a file hash cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .exceptions import HashingError

DEFAULT_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest for ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Stream a file and return its lowercase SHA-256 digest.

    Streaming avoids loading training audio or model checkpoints into memory.
    ``Path`` and Python file APIs preserve spaces and non-ASCII Windows paths.
    """
    source = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise HashingError(f"Cannot hash file '{source}': {exc}") from exc
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> bool:
    """Return whether ``path`` matches an expected SHA-256 digest."""
    normalized = expected.strip().casefold()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected SHA-256 must be exactly 64 hexadecimal characters")
    return sha256_file(path) == normalized


def _canonicalize(value: Any) -> Any:
    """Convert common domain values into deterministic JSON-compatible data."""
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="json", round_trip=True))
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=_canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__bytes_sha256__": sha256_bytes(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported fingerprint value of type {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    """Serialize a canonicalized value without locale- or whitespace-variance."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_json_hash(value: Any) -> str:
    """Hash arbitrary supported data using canonical UTF-8 JSON."""
    try:
        payload = _canonical_json(_canonicalize(value)).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HashingError(f"Cannot create deterministic fingerprint: {exc}") from exc
    return sha256_bytes(payload)


def stage_fingerprint(*components: Any, **named_components: Any) -> str:
    """Return a stable fingerprint for ordered and named stage inputs.

    Named component order is irrelevant, while positional order remains
    significant.  This lets a caller label inputs for readability without
    changing cache behavior.
    """
    payload = {
        "components": list(components),
        "named_components": named_components,
    }
    return stable_json_hash(payload)


def manifest_fingerprint(records: Iterable[Mapping[str, Any]]) -> str:
    """Fingerprint manifest records independent of their incoming order."""
    canonical_records = [_canonicalize(record) for record in records]
    ordered = sorted(canonical_records, key=_canonical_json)
    return stable_json_hash(ordered)


def fingerprint_files(paths: Iterable[str | Path], *, root: str | Path | None = None) -> str:
    """Fingerprint file content and stable relative names in deterministic order."""
    root_path = Path(root).resolve(strict=False) if root is not None else None
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=False)
        try:
            name = path.relative_to(root_path).as_posix() if root_path else path.as_posix()
        except ValueError as exc:
            raise HashingError(f"File '{path}' is outside fingerprint root '{root_path}'") from exc
        records.append({"path": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    return manifest_fingerprint(records)


@dataclass(frozen=True)
class FileHashEntry:
    """Cached content hash tied to inexpensive filesystem identity fields."""

    size: int
    mtime_ns: int
    sha256: str

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "FileHashEntry":
        """Validate a serialized cache entry."""
        try:
            size = int(value["size"])
            mtime_ns = int(value["mtime_ns"])
            digest = str(value["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HashingError(f"Invalid SHA-256 cache entry: {value!r}") from exc
        if size < 0 or mtime_ns < 0 or len(digest) != 64:
            raise HashingError(f"Invalid SHA-256 cache entry: {value!r}")
        return cls(size=size, mtime_ns=mtime_ns, sha256=digest)

    def to_json(self) -> dict[str, int | str]:
        """Return a JSON-compatible cache record."""
        return {"size": self.size, "mtime_ns": self.mtime_ns, "sha256": self.sha256}


class FileHashCache:
    """Persistent SHA-256 cache invalidated by file size or nanosecond mtime."""

    VERSION = 1

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._entries: dict[str, FileHashEntry] = {}
        if self.cache_path is not None and self.cache_path.exists():
            self.load()

    @property
    def entries(self) -> Mapping[str, FileHashEntry]:
        """Return a read-only-style snapshot of cached entries."""
        return dict(self._entries)

    @staticmethod
    def _key(path: Path) -> str:
        """Use a normalized absolute string as a Windows-safe cache key."""
        return os.path.normcase(str(path.expanduser().resolve(strict=False)))

    def hash_file(self, path: str | Path, *, persist: bool = True) -> str:
        """Return a cached digest when filesystem identity remains unchanged."""
        source = Path(path).expanduser().resolve(strict=False)
        try:
            stat = source.stat()
        except OSError as exc:
            raise HashingError(f"Cannot stat file '{source}': {exc}") from exc
        if not source.is_file():
            raise HashingError(f"Cannot hash non-file path '{source}'")

        key = self._key(source)
        cached = self._entries.get(key)
        if cached and cached.size == stat.st_size and cached.mtime_ns == stat.st_mtime_ns:
            return cached.sha256

        digest = sha256_file(source)
        self._entries[key] = FileHashEntry(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=digest,
        )
        if persist and self.cache_path is not None:
            self.save()
        return digest

    def invalidate(self, path: str | Path) -> bool:
        """Remove one cache entry and return whether it existed."""
        removed = self._entries.pop(self._key(Path(path)), None) is not None
        if removed and self.cache_path is not None:
            self.save()
        return removed

    def prune_missing(self) -> int:
        """Remove entries whose files no longer exist and return the count."""
        stale = [key for key in self._entries if not Path(key).is_file()]
        for key in stale:
            del self._entries[key]
        if stale and self.cache_path is not None:
            self.save()
        return len(stale)

    def load(self) -> None:
        """Load and validate the configured JSON cache."""
        if self.cache_path is None:
            raise HashingError("Cannot load a hash cache without cache_path")
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HashingError(f"Cannot load SHA-256 cache '{self.cache_path}': {exc}") from exc
        if not isinstance(raw, Mapping) or raw.get("version") != self.VERSION:
            raise HashingError(f"Unsupported SHA-256 cache format in '{self.cache_path}'")
        entries = raw.get("entries")
        if not isinstance(entries, Mapping):
            raise HashingError(f"Invalid SHA-256 cache entries in '{self.cache_path}'")
        self._entries = {
            str(key): FileHashEntry.from_json(value)
            for key, value in entries.items()
            if isinstance(value, Mapping)
        }
        if len(self._entries) != len(entries):
            raise HashingError(f"Invalid SHA-256 cache entry in '{self.cache_path}'")

    def save(self) -> Path:
        """Atomically persist the cache beside its final destination."""
        if self.cache_path is None:
            raise HashingError("Cannot save a hash cache without cache_path")
        payload = {
            "version": self.VERSION,
            "entries": {
                key: entry.to_json() for key, entry in sorted(self._entries.items())
            },
        }
        _atomic_write_json(self.cache_path, payload)
        return self.cache_path

    # Natural shorthand for callers that treat the cache as a hasher.
    hash = hash_file


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON to a sibling temporary file, flush it, then replace atomically."""
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
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise HashingError(f"Cannot atomically write SHA-256 cache '{path}': {exc}") from exc


# Compatibility names used by callers and older manifests.
hash_file = sha256_file
hash_bytes = sha256_bytes
calculate_sha256 = sha256_file
compute_stage_fingerprint = stage_fingerprint
HashCache = FileHashCache


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "FileHashCache",
    "FileHashEntry",
    "HashCache",
    "calculate_sha256",
    "compute_stage_fingerprint",
    "fingerprint_files",
    "hash_bytes",
    "hash_file",
    "manifest_fingerprint",
    "sha256_bytes",
    "sha256_file",
    "stable_json_hash",
    "stage_fingerprint",
    "verify_sha256",
]
