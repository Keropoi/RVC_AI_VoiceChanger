"""Safe external-process execution with live, persistent output capture."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TextIO

OutputCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class ProcessResult:
    """Result of a monitored subprocess."""

    command: tuple[str, ...]
    return_code: int
    started_at: float
    ended_at: float
    stdout_log: Path
    stderr_log: Path
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]
    timed_out: bool = False
    cancelled: bool = False

    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out and not self.cancelled

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


class ProcessExecutionError(RuntimeError):
    """Raised when a monitored command cannot be started or completed."""


class ProcessMonitor:
    """Execute argument-vector commands with ``shell=False``.

    Both streams are drained concurrently to avoid pipe deadlocks. On Windows,
    the child gets a new process group and cancellation uses ``taskkill /T`` to
    include descendants created by RVC training scripts.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        tail_lines: int = 80,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.tail_lines = max(1, tail_lines)
        self.poll_interval_seconds = max(0.02, poll_interval_seconds)

    def run(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
        env: Mapping[str, str] | None = None,
        output_callback: OutputCallback | None = None,
    ) -> ProcessResult:
        """Run a command and persist complete stdout/stderr logs."""

        argv = tuple(os.fspath(part) for part in command)
        if not argv or not argv[0]:
            raise ValueError("command must contain a non-empty executable")
        cwd = Path(cwd).resolve()
        if not cwd.is_dir():
            raise ProcessExecutionError(f"Process working directory is missing: {cwd}")

        stdout_log = Path(stdout_log)
        stderr_log = Path(stderr_log)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        started_at = time.time()
        stdout_tail: deque[str] = deque(maxlen=self.tail_lines)
        stderr_tail: deque[str] = deque(maxlen=self.tail_lines)
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=({**os.environ, **dict(env)} if env is not None else None),
                shell=False,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
        except OSError as exc:
            raise ProcessExecutionError(
                f"Could not start executable {argv[0]!r} in {cwd}: {exc}"
            ) from exc

        timed_out = False
        cancelled = False
        with ExitStack() as stack:
            stdout_file = stack.enter_context(
                stdout_log.open("w", encoding="utf-8", newline="")
            )
            stderr_file = stack.enter_context(
                stderr_log.open("w", encoding="utf-8", newline="")
            )
            stdout_thread = _start_reader(
                process.stdout,
                stdout_file,
                stdout_tail,
                "stdout",
                output_callback,
                self.logger,
            )
            stderr_thread = _start_reader(
                process.stderr,
                stderr_file,
                stderr_tail,
                "stderr",
                output_callback,
                self.logger,
            )
            try:
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        terminate_process_tree(process, logger=self.logger)
                        break
                    if (
                        timeout_seconds is not None
                        and time.time() - started_at > timeout_seconds
                    ):
                        timed_out = True
                        terminate_process_tree(process, logger=self.logger)
                        break
                    time.sleep(self.poll_interval_seconds)
            except KeyboardInterrupt:
                cancelled = True
                terminate_process_tree(process, logger=self.logger)
                raise
            finally:
                if process.poll() is None:
                    terminate_process_tree(process, logger=self.logger)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)

        return ProcessResult(
            command=argv,
            return_code=int(process.returncode if process.returncode is not None else -1),
            started_at=started_at,
            ended_at=time.time(),
            stdout_log=stdout_log.resolve(),
            stderr_log=stderr_log.resolve(),
            stdout_tail=tuple(stdout_tail),
            stderr_tail=tuple(stderr_tail),
            timed_out=timed_out,
            cancelled=cancelled,
        )


def terminate_process_tree(
    process: subprocess.Popen[str], *, logger: logging.Logger | None = None
) -> None:
    """Best-effort termination of a subprocess and all descendants."""

    if process.poll() is not None:
        return
    logger = logger or logging.getLogger(__name__)
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                shell=False,
            )
            if completed.returncode == 0:
                return
            logger.warning(
                "taskkill failed for PID %s: %s",
                process.pid,
                completed.stderr.strip(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Could not run taskkill for PID %s: %s", process.pid, exc)
        process.terminate()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()


def _start_reader(
    pipe: TextIO | None,
    destination: TextIO,
    tail: deque[str],
    stream_name: str,
    callback: OutputCallback | None,
    logger: logging.Logger,
) -> threading.Thread:
    def reader() -> None:
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                destination.write(line)
                destination.flush()
                clean_line = line.rstrip("\r\n")
                tail.append(clean_line)
                if callback is not None:
                    try:
                        callback(stream_name, clean_line)
                    except Exception as exc:  # callback is an isolation boundary
                        logger.warning("Output callback failed: %s", exc)
        finally:
            pipe.close()

    thread = threading.Thread(
        target=reader,
        name=f"rvc-{stream_name}-reader",
        daemon=True,
    )
    thread.start()
    return thread
