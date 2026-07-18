"""Best-effort GPU and host resource metrics collection."""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUSample:
    """One GPU/host resource observation."""

    timestamp: str
    gpu_index: int
    gpu_name: str | None = None
    gpu_utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    cpu_percent: float | None = None
    system_memory_used_mb: float | None = None
    disk_free_gb: float | None = None
    backend: str = "unavailable"
    error: str | None = None


def sample_gpu(gpu_index: int = 0, *, disk_path: Path | None = None) -> GPUSample:
    """Sample resources using NVML first and ``nvidia-smi`` second."""

    timestamp = datetime.now(timezone.utc).isoformat()
    host = _sample_host(disk_path)
    try:
        values = _sample_with_nvml(gpu_index)
        return GPUSample(timestamp=timestamp, gpu_index=gpu_index, **values, **host)
    except (ImportError, OSError, RuntimeError) as nvml_error:
        try:
            values = _sample_with_nvidia_smi(gpu_index)
            return GPUSample(timestamp=timestamp, gpu_index=gpu_index, **values, **host)
        except (OSError, RuntimeError, subprocess.SubprocessError) as smi_error:
            return GPUSample(
                timestamp=timestamp,
                gpu_index=gpu_index,
                error=f"NVML: {nvml_error}; nvidia-smi: {smi_error}",
                **host,
            )


class GPUMetricsMonitor:
    """Periodically append non-fatal GPU metrics to a CSV file."""

    def __init__(
        self,
        output_csv: Path,
        *,
        interval_seconds: float = 30.0,
        gpu_index: int = 0,
        disk_path: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_csv = Path(output_csv)
        self.interval_seconds = max(0.2, interval_seconds)
        self.gpu_index = gpu_index
        self.disk_path = Path(disk_path) if disk_path is not None else None
        self.logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the monitor once; repeated calls are harmless."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="rvc-gpu-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        """Request a prompt stop and wait briefly for CSV flush."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout_seconds))

    def __enter__(self) -> GPUMetricsMonitor:
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            fields = tuple(asdict(sample_gpu(self.gpu_index)).keys())
            needs_header = not self.output_csv.exists() or self.output_csv.stat().st_size == 0
            with self.output_csv.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if needs_header:
                    writer.writeheader()
                while not self._stop_event.is_set():
                    writer.writerow(asdict(sample_gpu(self.gpu_index, disk_path=self.disk_path)))
                    handle.flush()
                    self._stop_event.wait(self.interval_seconds)
        except (OSError, csv.Error) as exc:
            self.logger.warning("GPU metrics monitoring stopped: %s", exc)


def _sample_with_nvml(gpu_index: int) -> dict[str, Any]:
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        raise
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            power_w = None
        return {
            "gpu_name": str(name),
            "gpu_utilization_percent": float(utilization.gpu),
            "memory_used_mb": memory.used / (1024**2),
            "memory_total_mb": memory.total / (1024**2),
            "temperature_c": float(temperature),
            "power_w": power_w,
            "backend": "pynvml",
        }
    except pynvml.NVMLError as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def _sample_with_nvidia_smi(gpu_index: int) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise OSError("nvidia-smi is not on PATH")
    fields = (
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    )
    completed = subprocess.run(
        [
            executable,
            f"--id={gpu_index}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(parts) != len(fields):
        raise RuntimeError(f"Unexpected nvidia-smi output: {completed.stdout!r}")
    return {
        "gpu_name": parts[0],
        "gpu_utilization_percent": _float_or_none(parts[1]),
        "memory_used_mb": _float_or_none(parts[2]),
        "memory_total_mb": _float_or_none(parts[3]),
        "temperature_c": _float_or_none(parts[4]),
        "power_w": _float_or_none(parts[5]),
        "backend": "nvidia-smi",
    }


def _sample_host(disk_path: Path | None) -> dict[str, float | None]:
    cpu_percent: float | None = None
    memory_used_mb: float | None = None
    try:
        import psutil  # type: ignore[import-not-found]

        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory_used_mb = psutil.virtual_memory().used / (1024**2)
    except (ImportError, OSError, RuntimeError):
        pass
    disk_free_gb: float | None = None
    try:
        target = disk_path or Path.cwd()
        disk_free_gb = shutil.disk_usage(target).free / (1024**3)
    except OSError:
        pass
    return {
        "cpu_percent": cpu_percent,
        "system_memory_used_mb": memory_used_mb,
        "disk_free_gb": disk_free_gb,
    }


def _float_or_none(value: str) -> float | None:
    if value.lower() in {"n/a", "[not supported]", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None
