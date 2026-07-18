"""Safe audio decoding built on soundfile with an FFmpeg compatibility path."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - exercised only in an incomplete environment
    sf = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AudioProbe:
    """Container/codec metadata available without retaining decoded samples."""

    format: str
    subtype: Optional[str]
    codec: Optional[str]
    bit_depth: Optional[int]
    sample_rate: Optional[int]
    channels: Optional[int]
    duration_seconds: Optional[float]


@dataclass(frozen=True)
class DecodedAudio:
    """Float32 audio with an invariant ``(frames, channels)`` shape."""

    samples: np.ndarray
    sample_rate: int
    source_path: Path
    format: str
    subtype: Optional[str]
    codec: Optional[str]
    bit_depth: Optional[int]
    decoded_with_ffmpeg: bool = False

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError("DecodedAudio.samples must have shape (frames, channels)")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.samples.dtype != np.float32:
            object.__setattr__(self, "samples", self.samples.astype(np.float32, copy=False))

    @property
    def frames(self) -> int:
        """Number of PCM frames."""

        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        """Number of channels."""

        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        """Decoded duration in seconds."""

        return self.frames / float(self.sample_rate)


class AudioDecodeError(RuntimeError):
    """Raised when neither soundfile nor the configured FFmpeg can decode input."""

    def __init__(self, path: Path, message: str, stderr: Optional[str] = None) -> None:
        super().__init__(message)
        self.path = path
        self.stderr = stderr


class AudioEncodeError(RuntimeError):
    """Raised when a processed WAV cannot be written safely."""


def decode_audio(
    path: Union[str, Path],
    *,
    ffmpeg_executable: Union[str, Path] = "ffmpeg",
    allow_ffmpeg: bool = True,
    timeout_seconds: float = 120.0,
) -> DecodedAudio:
    """Decode an audio file to float32, preserving all source channels.

    ``soundfile`` is attempted first.  Formats unsupported by the local libsndfile
    build are transcoded by FFmpeg to a temporary float WAV.  FFmpeg receives an
    argument list with ``shell=False`` and never writes beside user input.
    """

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise AudioDecodeError(source, "Audio file does not exist: {}".format(source))
    if not source.is_file():
        raise AudioDecodeError(source, "Audio path is not a regular file: {}".format(source))
    if source.stat().st_size == 0:
        raise AudioDecodeError(source, "Audio file is empty: {}".format(source))
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    soundfile_error: Optional[Exception] = None
    try:
        return _decode_with_soundfile(source, source, decoded_with_ffmpeg=False)
    except (OSError, RuntimeError, ValueError) as exc:
        soundfile_error = exc

    if not allow_ffmpeg:
        raise AudioDecodeError(
            source,
            "soundfile could not decode {}: {}".format(source.name, soundfile_error),
        ) from soundfile_error

    return _decode_with_ffmpeg(
        source,
        ffmpeg_executable=ffmpeg_executable,
        timeout_seconds=timeout_seconds,
        soundfile_error=soundfile_error,
    )


def probe_audio(
    path: Union[str, Path],
    *,
    ffprobe_executable: Union[str, Path] = "ffprobe",
    timeout_seconds: float = 30.0,
) -> AudioProbe:
    """Inspect audio metadata, using FFprobe only when soundfile cannot inspect it."""

    source = Path(path).expanduser().resolve()
    _require_soundfile()
    try:
        info = sf.info(str(source))
    except (OSError, RuntimeError, ValueError):
        return _probe_with_ffprobe(source, ffprobe_executable, timeout_seconds)
    return AudioProbe(
        format=str(info.format or source.suffix.lstrip(".")).upper(),
        subtype=info.subtype or None,
        codec=info.subtype or info.format or None,
        bit_depth=_bit_depth(info.subtype),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        duration_seconds=float(info.duration),
    )


def write_wav(
    path: Union[str, Path],
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "PCM_16",
    overwrite: bool = False,
) -> Path:
    """Atomically write finite mono/stereo PCM as a WAV file."""

    _require_soundfile()
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise AudioEncodeError("Refusing to overwrite existing audio: {}".format(destination))
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise AudioEncodeError("Audio output must contain at least one frame and channel")
    if not np.isfinite(data).all():
        raise AudioEncodeError("Refusing to write audio containing NaN or Infinity")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.{}.tmp.wav".format(destination.name, uuid.uuid4().hex)
    )
    try:
        sf.write(str(temporary), data, sample_rate, format="WAV", subtype=subtype)
        os.replace(str(temporary), str(destination))
    except (OSError, RuntimeError, ValueError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except TypeError:  # Python 3.7/3.8 compatibility for unusual hosts
            if temporary.exists():
                temporary.unlink()
        raise AudioEncodeError("Unable to write WAV {}: {}".format(destination, exc)) from exc
    return destination


def _decode_with_soundfile(
    decode_path: Path, source_path: Path, *, decoded_with_ffmpeg: bool
) -> DecodedAudio:
    _require_soundfile()
    info = sf.info(str(decode_path))
    samples, sample_rate = sf.read(
        str(decode_path), dtype="float32", always_2d=True, fill_value=0.0
    )
    samples = np.asarray(samples, dtype=np.float32)
    if samples.shape[0] == 0:
        raise ValueError("decoded stream contains no audio frames")
    if int(sample_rate) <= 0:
        raise ValueError("decoded stream has an invalid sample rate")
    return DecodedAudio(
        samples=samples,
        sample_rate=int(sample_rate),
        source_path=source_path,
        format=str(info.format or source_path.suffix.lstrip(".")).upper(),
        subtype=info.subtype or None,
        codec=info.subtype or info.format or None,
        bit_depth=_bit_depth(info.subtype),
        decoded_with_ffmpeg=decoded_with_ffmpeg,
    )


def _decode_with_ffmpeg(
    source: Path,
    *,
    ffmpeg_executable: Union[str, Path],
    timeout_seconds: float,
    soundfile_error: Optional[Exception],
) -> DecodedAudio:
    with tempfile.TemporaryDirectory(prefix="rvc-audio-decode-") as temporary_directory:
        output = Path(temporary_directory) / "decoded.wav"
        command: Sequence[str] = (
            str(ffmpeg_executable),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map_metadata",
            "-1",
            "-vn",
            "-acodec",
            "pcm_f32le",
            "-f",
            "wav",
            "-y",
            str(output),
        )
        try:
            completed = subprocess.run(
                list(command),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise AudioDecodeError(
                source,
                "FFmpeg executable was not found: {} (soundfile error: {})".format(
                    ffmpeg_executable, soundfile_error
                ),
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioDecodeError(
                source,
                "FFmpeg timed out after {:.1f}s while decoding {}".format(
                    timeout_seconds, source.name
                ),
                _tail(exc.stderr),
            ) from exc

        stderr = _tail(completed.stderr)
        if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            raise AudioDecodeError(
                source,
                "FFmpeg failed to decode {} (exit code {}): {}".format(
                    source.name, completed.returncode, stderr or "no diagnostic output"
                ),
                stderr,
            )
        try:
            decoded = _decode_with_soundfile(output, source, decoded_with_ffmpeg=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AudioDecodeError(
                source,
                "FFmpeg produced an unreadable intermediate WAV for {}: {}".format(
                    source.name, exc
                ),
                stderr,
            ) from exc

    original_probe = _try_ffprobe_for_source(source, ffmpeg_executable, timeout_seconds)
    if original_probe is None:
        return decoded
    return DecodedAudio(
        samples=decoded.samples,
        sample_rate=decoded.sample_rate,
        source_path=decoded.source_path,
        format=original_probe.format,
        subtype=original_probe.subtype,
        codec=original_probe.codec,
        bit_depth=original_probe.bit_depth,
        decoded_with_ffmpeg=True,
    )


def _try_ffprobe_for_source(
    source: Path, ffmpeg_executable: Union[str, Path], timeout_seconds: float
) -> Optional[AudioProbe]:
    executable = Path(str(ffmpeg_executable))
    name = executable.name.lower()
    ffprobe_name = "ffprobe.exe" if name.endswith(".exe") else "ffprobe"
    ffprobe = executable.with_name(ffprobe_name) if executable.parent != Path(".") else Path(ffprobe_name)
    try:
        return _probe_with_ffprobe(source, ffprobe, min(timeout_seconds, 30.0))
    except AudioDecodeError:
        return None


def _probe_with_ffprobe(
    source: Path, ffprobe_executable: Union[str, Path], timeout_seconds: float
) -> AudioProbe:
    command = [
        str(ffprobe_executable),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_sample,duration:format=format_name,duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise AudioDecodeError(source, "FFprobe could not inspect {}: {}".format(source.name, exc)) from exc
    if completed.returncode != 0:
        raise AudioDecodeError(
            source,
            "FFprobe failed for {}: {}".format(source.name, _tail(completed.stderr)),
            _tail(completed.stderr),
        )
    try:
        payload: Dict[str, Any] = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        stream = streams[0]
        format_data = payload.get("format") or {}
        duration = stream.get("duration") or format_data.get("duration")
        sample_rate = stream.get("sample_rate")
        channels = stream.get("channels")
        bits = stream.get("bits_per_sample")
        codec = stream.get("codec_name")
        format_name = str(format_data.get("format_name") or source.suffix.lstrip("."))
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioDecodeError(source, "FFprobe returned malformed metadata for {}".format(source.name)) from exc
    return AudioProbe(
        format=format_name.upper(),
        subtype=str(codec).upper() if codec else None,
        codec=str(codec) if codec else None,
        bit_depth=int(bits) if bits not in (None, "", 0, "0") else None,
        sample_rate=int(sample_rate) if sample_rate else None,
        channels=int(channels) if channels else None,
        duration_seconds=float(duration) if duration else None,
    )


def _bit_depth(subtype: Optional[str]) -> Optional[int]:
    if not subtype:
        return None
    upper = subtype.upper()
    for token, depth in (
        ("PCM_U8", 8),
        ("PCM_S8", 8),
        ("PCM_16", 16),
        ("PCM_24", 24),
        ("PCM_32", 32),
        ("FLOAT", 32),
        ("DOUBLE", 64),
    ):
        if token in upper:
            return depth
    return None


def _require_soundfile() -> None:
    if sf is None:
        raise RuntimeError(
            "The soundfile package is required for audio decoding; install project dependencies"
        )


def _tail(value: object, maximum_lines: int = 12) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    return "\n".join(text.strip().splitlines()[-maximum_lines:])
