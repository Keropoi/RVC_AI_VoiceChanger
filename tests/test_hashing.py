"""Tests for streaming SHA-256 and deterministic incremental fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rvc_auto_trainer.hashing import (
    FileHashCache,
    manifest_fingerprint,
    sha256_file,
    stable_json_hash,
    stage_fingerprint,
    verify_sha256,
)


def test_sha256_file_supports_chinese_paths(tmp_path: Path) -> None:
    """Hashing uses filesystem APIs that preserve non-ASCII Windows paths."""
    path = tmp_path / "角色 音频" / "你好.wav"
    path.parent.mkdir()
    payload = b"small deterministic audio fixture"
    path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(path, chunk_size=3) == expected
    assert verify_sha256(path, expected.upper())


def test_stable_hash_ignores_mapping_order_but_not_sequence_order() -> None:
    """Canonical JSON avoids false invalidation from dictionary ordering."""
    assert stable_json_hash({"b": 2, "a": 1}) == stable_json_hash({"a": 1, "b": 2})
    assert stable_json_hash([1, 2]) != stable_json_hash([2, 1])
    assert stage_fingerprint(version="1", config={"x": 1}) == stage_fingerprint(
        config={"x": 1}, version="1"
    )


def test_manifest_fingerprint_is_record_order_independent() -> None:
    """Discovery order cannot change a manifest cache key."""
    first = [{"path": "b.wav", "sha256": "2"}, {"path": "a.wav", "sha256": "1"}]
    second = list(reversed(first))
    assert manifest_fingerprint(first) == manifest_fingerprint(second)


def test_file_hash_cache_persists_and_invalidates_changed_content(tmp_path: Path) -> None:
    """Size/mtime identity reuses hashes and content changes refresh the entry."""
    source = tmp_path / "输入.wav"
    cache_path = tmp_path / "cache" / "sha256.json"
    source.write_bytes(b"first")

    cache = FileHashCache(cache_path)
    first = cache.hash_file(source)
    loaded = FileHashCache(cache_path)
    assert loaded.hash_file(source) == first

    source.write_bytes(b"second-longer")
    second = loaded.hash_file(source)
    assert second != first
    assert FileHashCache(cache_path).hash_file(source) == second
