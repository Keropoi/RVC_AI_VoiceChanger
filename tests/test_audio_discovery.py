"""Tests for deterministic and conservative audio discovery."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rvc_auto_trainer.audio.discovery import (
    AudioDiscoveryError,
    discover_audio_files,
    duplicate_hashes,
    select_test_audio,
)


def _write(path: Path, content: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_discovery_recurses_hashes_and_supports_unicode_paths(tmp_path: Path) -> None:
    root = tmp_path / "训练 音频"
    first = _write(root / "角色" / "甲.WAV", b"first-wave")
    second = _write(root / "normal" / "voice.flac", b"second-wave")

    result = discover_audio_files(root)

    assert [item.relative_path for item in result.files] == ["normal/voice.flac", "角色/甲.WAV"]
    by_path = {item.path: item for item in result.files}
    assert by_path[first.resolve()].sha256 == hashlib.sha256(b"first-wave").hexdigest()
    assert by_path[second.resolve()].size_bytes == len(b"second-wave")
    assert result.total_size_bytes == len(b"first-wave") + len(b"second-wave")


def test_discovery_ignores_hidden_temporary_zero_and_unsupported_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "audio"
    _write(root / "kept.ogg")
    _write(root / ".hidden.wav")
    _write(root / ".hidden-dir" / "nested.wav")
    _write(root / "take.tmp.wav")
    _write(root / "notes.txt")
    _write(root / "empty.wav", b"")

    result = discover_audio_files(root)

    assert [item.relative_path for item in result.files] == ["kept.ogg"]
    ignored = {item.relative_path: item.reason for item in result.ignored}
    assert ignored[".hidden.wav"] == "hidden"
    assert ignored[".hidden-dir/nested.wav"] == "hidden"
    assert ignored["take.tmp.wav"] == "temporary"
    assert ignored["notes.txt"] == "unsupported_extension"
    assert ignored["empty.wav"] == "zero_bytes"


def test_discovery_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(AudioDiscoveryError, match="does not exist"):
        discover_audio_files(tmp_path / "missing")


def test_test_selection_and_cross_set_duplicate_detection(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    test_root = tmp_path / "test"
    _write(training_root / "source.wav", b"same-content")
    _write(test_root / "z.wav", b"same-content")
    _write(test_root / "a.wav", b"different")

    training = discover_audio_files(training_root).files
    tests = discover_audio_files(test_root).files

    assert select_test_audio(tests, 1)[0].relative_path == "a.wav"
    duplicates = duplicate_hashes(training, tests)
    expected_hash = hashlib.sha256(b"same-content").hexdigest()
    assert set(duplicates) == {expected_hash}
    assert duplicates[expected_hash][0][0].relative_path == "source.wav"
    assert duplicates[expected_hash][1][0].relative_path == "z.wav"
