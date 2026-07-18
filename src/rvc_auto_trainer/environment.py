"""Read-only environment diagnostics and JSON/text doctor reports."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import AppConfig
from .exceptions import EnvironmentCheckError


class CheckStatus(str, Enum):
    """Outcome of one doctor diagnostic."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class EnvironmentCheck(BaseModel):
    """One actionable fact in an environment doctor report."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: CheckStatus
    message: str
    required: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    expected_paths: tuple[Path, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether this check is fully satisfied."""
        return self.status is CheckStatus.PASS


class EnvironmentReport(BaseModel):
    """Complete machine, dependency, repository, and input readiness report."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    project_root: Path
    checks: dict[str, EnvironmentCheck]
    facts: dict[str, Any] = Field(default_factory=dict)
    healthy: bool

    @property
    def failures(self) -> tuple[EnvironmentCheck, ...]:
        """Return failed checks in insertion order."""
        return tuple(check for check in self.checks.values() if check.status is CheckStatus.FAIL)

    @property
    def warnings(self) -> tuple[EnvironmentCheck, ...]:
        """Return warning checks in insertion order."""
        return tuple(
            check for check in self.checks.values() if check.status is CheckStatus.WARNING
        )

    @property
    def is_healthy(self) -> bool:
        """Compatibility property mirroring the serialized ``healthy`` field."""
        return self.healthy

    @property
    def ok(self) -> bool:
        """Short compatibility alias for command-line exit-code decisions."""
        return self.healthy


class EnvironmentDoctor:
    """Inspect configured local resources without downloading or modifying RVC."""

    def __init__(self, config: AppConfig, *, command_timeout: float = 10.0) -> None:
        if command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        self.config = config
        self.command_timeout = command_timeout

    def run(self) -> EnvironmentReport:
        """Execute all diagnostics, preserving failures as report data."""
        checks: dict[str, EnvironmentCheck] = {}
        facts: dict[str, Any] = {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "operating_system": platform.system(),
            "machine": platform.machine(),
        }

        checks["python"] = _check_python()
        checks["operating_system"] = _check_operating_system()

        ffmpeg_check, ffmpeg_version = self._check_executable("ffmpeg", ("-version",))
        checks["ffmpeg"] = ffmpeg_check
        facts["ffmpeg_version"] = ffmpeg_version
        ffprobe_check, ffprobe_version = self._check_executable("ffprobe", ("-version",))
        checks["ffprobe"] = ffprobe_check
        facts["ffprobe_version"] = ffprobe_version

        torch_check, torch_facts = _check_torch_runtime(
            self.config.paths.rvc_python,
            require_cuda=self.config.environment.require_cuda,
            gpu_ids=self.config.environment.gpu_ids,
            timeout=self.command_timeout,
        )
        checks["pytorch_cuda"] = torch_check
        facts.update(torch_facts)

        nvidia_check, nvidia_facts = self._check_nvidia_driver()
        checks["nvidia_driver"] = nvidia_check
        facts.update(nvidia_facts)

        repository = self.config.paths.rvc_repository
        checks["rvc_repository"] = _check_directory(
            "RVC repository",
            repository,
            required=True,
            missing_message="Configured RVC repository does not exist",
        )
        script_check, script_facts = _check_rvc_scripts(repository)
        checks["rvc_scripts"] = script_check
        facts.update(script_facts)

        commit_check, commit = self._check_git_commit(repository)
        checks["rvc_git_commit"] = commit_check
        facts["rvc_git_commit"] = commit

        checks["rvc_python"] = self._check_python_executable(self.config.paths.rvc_python)
        checks.update(_check_rvc_assets(repository, self.config))

        writable_check, disk_facts = _check_output_and_disk(
            self.config.paths.runs_dir,
            self.config.environment.minimum_free_disk_gb,
        )
        checks["output_directory"] = writable_check
        facts.update(disk_facts)

        checks["training_audio"] = _check_input_directory(
            "Training audio", self.config.paths.training_audio_dir
        )
        checks["test_audio"] = _check_input_directory(
            "Test audio", self.config.paths.test_audio_dir
        )

        healthy = not any(
            check.status is CheckStatus.FAIL and check.required for check in checks.values()
        )
        return EnvironmentReport(
            generated_at=datetime.now(timezone.utc),
            project_root=self.config.project_root,
            checks=checks,
            facts=facts,
            healthy=healthy,
        )

    def _check_executable(
        self,
        executable_name: str,
        arguments: tuple[str, ...],
    ) -> tuple[EnvironmentCheck, str | None]:
        """Locate and safely query an executable with an argument list."""
        executable = shutil.which(executable_name)
        if not executable:
            return (
                EnvironmentCheck(
                    name=executable_name,
                    status=CheckStatus.FAIL,
                    message=f"{executable_name} was not found on PATH",
                    required=True,
                ),
                None,
            )
        result = _run_command(
            (executable, *arguments),
            timeout=self.command_timeout,
        )
        if result.returncode != 0:
            return (
                EnvironmentCheck(
                    name=executable_name,
                    status=CheckStatus.FAIL,
                    message=f"{executable_name} returned exit code {result.returncode}",
                    required=True,
                    details={"executable": executable, "stderr": _tail(result.stderr)},
                ),
                None,
            )
        first_line = next(
            (line.strip() for line in result.stdout.splitlines() if line.strip()),
            "version output was empty",
        )
        return (
            EnvironmentCheck(
                name=executable_name,
                status=CheckStatus.PASS,
                message=first_line,
                required=True,
                details={"executable": executable},
            ),
            first_line,
        )

    def _check_nvidia_driver(self) -> tuple[EnvironmentCheck, dict[str, Any]]:
        """Query GPU/driver facts via ``nvidia-smi`` without a shell."""
        executable = shutil.which("nvidia-smi")
        if not executable:
            status = CheckStatus.FAIL if self.config.environment.require_cuda else CheckStatus.WARNING
            return (
                EnvironmentCheck(
                    name="NVIDIA driver",
                    status=status,
                    message="nvidia-smi was not found on PATH",
                    required=self.config.environment.require_cuda,
                ),
                {},
            )
        result = _run_command(
            (
                executable,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ),
            timeout=self.command_timeout,
        )
        if result.returncode != 0:
            status = CheckStatus.FAIL if self.config.environment.require_cuda else CheckStatus.WARNING
            return (
                EnvironmentCheck(
                    name="NVIDIA driver",
                    status=status,
                    message=f"nvidia-smi returned exit code {result.returncode}",
                    required=self.config.environment.require_cuda,
                    details={"stderr": _tail(result.stderr)},
                ),
                {},
            )
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return (
            EnvironmentCheck(
                name="NVIDIA driver",
                status=CheckStatus.PASS,
                message=f"Detected {len(devices)} NVIDIA GPU(s)",
                required=self.config.environment.require_cuda,
                details={"devices": devices},
            ),
            {"nvidia_smi_devices": devices},
        )

    def _check_git_commit(self, repository: Path) -> tuple[EnvironmentCheck, str | None]:
        """Read the checked-out RVC commit without changing repository state."""
        git = shutil.which("git")
        if not repository.is_dir():
            return (
                EnvironmentCheck(
                    name="RVC Git commit",
                    status=CheckStatus.WARNING,
                    message="Cannot read commit because the RVC repository is missing",
                ),
                None,
            )
        if not git:
            return (
                EnvironmentCheck(
                    name="RVC Git commit",
                    status=CheckStatus.WARNING,
                    message="Git was not found on PATH; commit could not be recorded",
                ),
                None,
            )
        result = _run_command(
            (git, "-C", str(repository), "rev-parse", "HEAD"),
            timeout=self.command_timeout,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
        if commit:
            return (
                EnvironmentCheck(
                    name="RVC Git commit",
                    status=CheckStatus.PASS,
                    message=commit,
                ),
                commit,
            )
        return (
            EnvironmentCheck(
                name="RVC Git commit",
                status=CheckStatus.WARNING,
                message="Configured RVC directory is not a readable Git checkout",
                details={"stderr": _tail(result.stderr)},
            ),
            None,
        )

    def _check_python_executable(self, executable: Path) -> EnvironmentCheck:
        """Verify the separate RVC interpreter exists and starts successfully."""
        if not executable.is_file():
            return EnvironmentCheck(
                name="RVC Python",
                status=CheckStatus.FAIL,
                message="Configured RVC Python interpreter does not exist",
                required=True,
                expected_paths=(executable,),
            )
        result = _run_command((str(executable), "--version"), timeout=self.command_timeout)
        output = (result.stdout or result.stderr).strip()
        if result.returncode != 0:
            return EnvironmentCheck(
                name="RVC Python",
                status=CheckStatus.FAIL,
                message=f"RVC Python returned exit code {result.returncode}",
                required=True,
                details={"stderr": _tail(result.stderr)},
                expected_paths=(executable,),
            )
        return EnvironmentCheck(
            name="RVC Python",
            status=CheckStatus.PASS,
            message=output or "Interpreter started successfully",
            required=True,
            expected_paths=(executable,),
        )


def _check_python() -> EnvironmentCheck:
    """Check the automation interpreter against the supported version floor."""
    supported = sys.version_info >= (3, 9)
    return EnvironmentCheck(
        name="Python",
        status=CheckStatus.PASS if supported else CheckStatus.FAIL,
        message=f"Python {platform.python_version()} at {sys.executable}",
        required=True,
        details={"minimum_version": "3.9"},
    )


def _check_operating_system() -> EnvironmentCheck:
    """Report the OS, warning when outside the supported Windows target."""
    is_windows = platform.system().casefold() == "windows"
    return EnvironmentCheck(
        name="Operating system",
        status=CheckStatus.PASS if is_windows else CheckStatus.WARNING,
        message=platform.platform(),
        required=False,
        details={"target": "Windows 10 or Windows 11"},
    )


def _check_torch(
    *,
    require_cuda: bool,
    gpu_ids: Iterable[int],
) -> tuple[EnvironmentCheck, dict[str, Any]]:
    """Inspect PyTorch/CUDA lazily so doctor still works without PyTorch."""
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        status = CheckStatus.FAIL if require_cuda else CheckStatus.WARNING
        return (
            EnvironmentCheck(
                name="PyTorch CUDA",
                status=status,
                message=f"PyTorch could not be imported: {exc}",
                required=require_cuda,
            ),
            {},
        )

    facts: dict[str, Any] = {
        "pytorch_version": str(getattr(torch, "__version__", "unknown")),
        "pytorch_cuda_version": str(getattr(getattr(torch, "version", None), "cuda", None)),
    }
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        devices = []
        for device_index in range(device_count):
            properties = torch.cuda.get_device_properties(device_index)
            devices.append(
                {
                    "index": device_index,
                    "name": str(properties.name),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        status = CheckStatus.FAIL if require_cuda else CheckStatus.WARNING
        return (
            EnvironmentCheck(
                name="PyTorch CUDA",
                status=status,
                message=f"PyTorch CUDA inspection failed: {exc}",
                required=require_cuda,
                details=facts,
            ),
            facts,
        )

    facts["cuda_available"] = cuda_available
    facts["torch_gpu_devices"] = devices
    requested = tuple(gpu_ids)
    missing = [gpu_id for gpu_id in requested if gpu_id >= device_count]
    if require_cuda and not cuda_available:
        return (
            EnvironmentCheck(
                name="PyTorch CUDA",
                status=CheckStatus.FAIL,
                message="CUDA is required but PyTorch reports it unavailable",
                required=True,
                details=facts,
            ),
            facts,
        )
    if missing:
        return (
            EnvironmentCheck(
                name="PyTorch CUDA",
                status=CheckStatus.FAIL,
                message=f"Configured GPU IDs are unavailable: {missing}",
                required=True,
                details=facts,
            ),
            facts,
        )
    status = CheckStatus.PASS if cuda_available else CheckStatus.WARNING
    message = f"PyTorch detected {device_count} CUDA device(s)" if cuda_available else "CPU-only mode"
    return (
        EnvironmentCheck(
            name="PyTorch CUDA",
            status=status,
            message=message,
            required=require_cuda,
            details=facts,
        ),
        facts,
    )


def _check_torch_runtime(
    python_executable: Path,
    *,
    require_cuda: bool,
    gpu_ids: Iterable[int],
    timeout: float,
) -> tuple[EnvironmentCheck, dict[str, Any]]:
    """Inspect PyTorch inside the configured, independent RVC environment."""

    runtime = Path(python_executable)
    if not runtime.is_file():
        status = CheckStatus.FAIL if require_cuda else CheckStatus.WARNING
        return (
            EnvironmentCheck(
                name="RVC PyTorch CUDA",
                status=status,
                message="RVC Python is missing, so its PyTorch/CUDA runtime cannot be checked",
                required=require_cuda,
                expected_paths=(runtime,),
            ),
            {"pytorch_runtime": str(runtime)},
        )

    probe = (
        "import json\n"
        "import torch\n"
        "available = bool(torch.cuda.is_available())\n"
        "count = int(torch.cuda.device_count()) if available else 0\n"
        "devices = []\n"
        "for index in range(count):\n"
        "    props = torch.cuda.get_device_properties(index)\n"
        "    devices.append({'index': index, 'name': str(props.name), "
        "'total_memory_bytes': int(props.total_memory)})\n"
        "print(json.dumps({'pytorch_version': str(torch.__version__), "
        "'pytorch_cuda_version': str(getattr(torch.version, 'cuda', None)), "
        "'cuda_available': available, 'device_count': count, 'devices': devices}))\n"
    )
    result = _run_command((str(runtime), "-c", probe), timeout=timeout)
    if result.returncode != 0:
        status = CheckStatus.FAIL if require_cuda else CheckStatus.WARNING
        return (
            EnvironmentCheck(
                name="RVC PyTorch CUDA",
                status=status,
                message=(
                    "PyTorch/CUDA probe failed inside the configured RVC environment "
                    f"with exit code {result.returncode}"
                ),
                required=require_cuda,
                details={"stderr": _tail(result.stderr), "runtime": str(runtime)},
            ),
            {"pytorch_runtime": str(runtime)},
        )

    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError, TypeError) as exc:
        status = CheckStatus.FAIL if require_cuda else CheckStatus.WARNING
        return (
            EnvironmentCheck(
                name="RVC PyTorch CUDA",
                status=status,
                message=f"RVC PyTorch probe returned invalid JSON: {exc}",
                required=require_cuda,
                details={"stdout": _tail(result.stdout), "runtime": str(runtime)},
            ),
            {"pytorch_runtime": str(runtime)},
        )

    facts: dict[str, Any] = {
        "pytorch_runtime": str(runtime),
        "pytorch_version": payload.get("pytorch_version"),
        "pytorch_cuda_version": payload.get("pytorch_cuda_version"),
        "cuda_available": bool(payload.get("cuda_available")),
        "torch_gpu_devices": payload.get("devices", []),
    }
    device_count = int(payload.get("device_count", 0))
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
    if require_cuda and not facts["cuda_available"]:
        return (
            EnvironmentCheck(
                name="RVC PyTorch CUDA",
                status=CheckStatus.FAIL,
                message="CUDA is required but PyTorch in the RVC environment reports it unavailable",
                required=True,
                details=facts,
            ),
            facts,
        )
    if missing:
        return (
            EnvironmentCheck(
                name="RVC PyTorch CUDA",
                status=CheckStatus.FAIL,
                message=f"Configured GPU IDs are unavailable in RVC PyTorch: {missing}",
                required=True,
                details=facts,
            ),
            facts,
        )
    status = CheckStatus.PASS if facts["cuda_available"] else CheckStatus.WARNING
    message = (
        f"RVC PyTorch detected {device_count} CUDA device(s)"
        if facts["cuda_available"]
        else "RVC PyTorch is available in CPU-only mode"
    )
    return (
        EnvironmentCheck(
            name="RVC PyTorch CUDA",
            status=status,
            message=message,
            required=require_cuda,
            details=facts,
        ),
        facts,
    )


def _check_directory(
    name: str,
    path: Path,
    *,
    required: bool,
    missing_message: str,
) -> EnvironmentCheck:
    """Build an explicit directory existence check."""
    exists = path.is_dir()
    return EnvironmentCheck(
        name=name,
        status=CheckStatus.PASS if exists else (CheckStatus.FAIL if required else CheckStatus.WARNING),
        message=f"Found {path}" if exists else missing_message,
        required=required,
        expected_paths=(path,),
    )


_RVC_SCRIPT_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "preprocess": (
        "infer/modules/train/preprocess.py",
        "trainset_preprocess_pipeline_print.py",
    ),
    "extract_f0": (
        "infer/modules/train/extract/extract_f0_print.py",
        "infer/modules/train/extract/extract_f0_rmvpe.py",
        "extract_f0_print.py",
    ),
    "extract_features": (
        "infer/modules/train/extract_feature_print.py",
        "extract_feature_print.py",
    ),
    "train": ("infer/modules/train/train.py", "train_nsf_sim_cache_sid_load_pretrain.py"),
    "infer": ("infer-web.py", "tools/infer_cli.py"),
}


def _check_rvc_scripts(repository: Path) -> tuple[EnvironmentCheck, dict[str, Any]]:
    """Probe known upstream layouts and report every missing purpose explicitly."""
    found: dict[str, str] = {}
    expected: list[Path] = []
    for purpose, relative_candidates in _RVC_SCRIPT_CANDIDATES.items():
        candidates = tuple(repository / relative for relative in relative_candidates)
        expected.extend(candidates)
        match = next((candidate for candidate in candidates if candidate.is_file()), None)
        if match is not None:
            found[purpose] = str(match)
    required_purposes = {"preprocess", "extract_features", "train", "infer"}
    missing = sorted(required_purposes.difference(found))
    status = CheckStatus.PASS if not missing else CheckStatus.FAIL
    message = "Required RVC scripts were discovered" if not missing else f"Missing RVC script purposes: {', '.join(missing)}"
    return (
        EnvironmentCheck(
            name="RVC scripts",
            status=status,
            message=message,
            required=True,
            details={"found": found, "missing_purposes": missing},
            expected_paths=tuple(expected) if missing else (),
        ),
        {"rvc_scripts": found, "rvc_missing_script_purposes": missing},
    )


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    """Return the first file from an ordered candidate sequence."""
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _asset_check(
    name: str,
    candidates: tuple[Path, ...],
    *,
    required: bool,
) -> EnvironmentCheck:
    """Report an asset match or every expected location without downloading it."""
    match = _first_existing(candidates)
    if match is not None:
        return EnvironmentCheck(
            name=name,
            status=CheckStatus.PASS,
            message=f"Found {match}",
            required=required,
            expected_paths=(match,),
        )
    return EnvironmentCheck(
        name=name,
        status=CheckStatus.FAIL if required else CheckStatus.WARNING,
        message="Required model asset was not found; no download was attempted",
        required=required,
        expected_paths=candidates,
    )


def _check_rvc_assets(repository: Path, config: AppConfig) -> dict[str, EnvironmentCheck]:
    """Check HuBERT, RMVPE, and paired RVC v2 pretrained assets."""
    local = config.paths.pretrained_models_dir
    hubert_candidates = (
        repository / "assets/hubert/hubert_base.pt",
        repository / "hubert_base.pt",
        local / "hubert_base.pt",
        local / "contentvec_base.pt",
    )
    rmvpe_candidates = (
        repository / "assets/rmvpe/rmvpe.pt",
        repository / "rmvpe.pt",
        local / "rmvpe.pt",
    )
    sample_rate = config.model.sample_rate
    prefix = "f0" if config.model.use_f0 else ""
    generator_candidates = (
        repository / f"assets/pretrained_v2/{prefix}G{sample_rate}.pth",
        repository / f"pretrained_v2/{prefix}G{sample_rate}.pth",
        local / f"{prefix}G{sample_rate}.pth",
    )
    discriminator_candidates = (
        repository / f"assets/pretrained_v2/{prefix}D{sample_rate}.pth",
        repository / f"pretrained_v2/{prefix}D{sample_rate}.pth",
        local / f"{prefix}D{sample_rate}.pth",
    )
    pretrained_required = config.training.use_pretrained_model
    return {
        "hubert_contentvec": _asset_check(
            "HuBERT/ContentVec model", hubert_candidates, required=True
        ),
        "rmvpe": _asset_check(
            "RMVPE model",
            rmvpe_candidates,
            required=config.model.use_f0 and config.model.f0_method.casefold() == "rmvpe",
        ),
        "pretrained_generator": _asset_check(
            "RVC pretrained generator", generator_candidates, required=pretrained_required
        ),
        "pretrained_discriminator": _asset_check(
            "RVC pretrained discriminator", discriminator_candidates, required=pretrained_required
        ),
    }


def _check_output_and_disk(
    output_dir: Path,
    minimum_free_disk_gb: float,
) -> tuple[EnvironmentCheck, dict[str, Any]]:
    """Verify output writability and configured minimum free disk space."""
    temporary_path: Path | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".doctor-", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(b"doctor")
        temporary_path.unlink()
        temporary_path = None
        usage = shutil.disk_usage(output_dir)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        return (
            EnvironmentCheck(
                name="Output directory",
                status=CheckStatus.FAIL,
                message=f"Output directory is not writable: {exc}",
                required=True,
                expected_paths=(output_dir,),
            ),
            {},
        )
    free_gb = usage.free / (1024**3)
    enough = free_gb >= minimum_free_disk_gb
    return (
        EnvironmentCheck(
            name="Output directory",
            status=CheckStatus.PASS if enough else CheckStatus.FAIL,
            message=(
                f"Writable with {free_gb:.2f} GiB free"
                if enough
                else f"Only {free_gb:.2f} GiB free; {minimum_free_disk_gb:.2f} GiB required"
            ),
            required=True,
            details={
                "free_disk_gb": round(free_gb, 3),
                "minimum_free_disk_gb": minimum_free_disk_gb,
            },
            expected_paths=(output_dir,),
        ),
        {"output_free_disk_gb": free_gb},
    )


_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"})


def _check_input_directory(name: str, path: Path) -> EnvironmentCheck:
    """Report whether an input directory exists and contains supported audio."""
    if not path.is_dir():
        return EnvironmentCheck(
            name=name,
            status=CheckStatus.WARNING,
            message=f"Input directory does not exist yet: {path}",
            expected_paths=(path,),
        )
    try:
        count = sum(
            1
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.casefold() in _AUDIO_EXTENSIONS
            and candidate.stat().st_size > 0
        )
    except OSError as exc:
        return EnvironmentCheck(
            name=name,
            status=CheckStatus.WARNING,
            message=f"Input directory could not be fully scanned: {exc}",
            expected_paths=(path,),
        )
    return EnvironmentCheck(
        name=name,
        status=CheckStatus.PASS if count else CheckStatus.WARNING,
        message=f"Found {count} supported audio file(s)",
        details={"audio_file_count": count},
        expected_paths=(path,),
    )


def _run_command(command: tuple[str, ...], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a diagnostic command with no shell and bounded capture."""
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(list(command), 127, stdout="", stderr=str(exc))


def _tail(value: str, line_count: int = 8) -> list[str]:
    """Return the last non-empty diagnostic lines."""
    return [line for line in value.splitlines() if line.strip()][-line_count:]


def inspect_environment(config: AppConfig) -> EnvironmentReport:
    """Convenience entry point used by CLI and pipeline code."""
    return EnvironmentDoctor(config).run()


def format_environment_report(report: EnvironmentReport) -> str:
    """Render a compact, human-readable offline doctor report."""
    lines = [
        "RVC Auto Trainer Environment Report",
        f"Generated: {report.generated_at.isoformat()}",
        f"Project root: {report.project_root}",
        f"Overall: {'PASS' if report.healthy else 'FAIL'}",
        "",
    ]
    for key, check in report.checks.items():
        required = " required" if check.required else ""
        lines.append(f"[{check.status.value}] {key} ({check.name},{required.strip() or 'optional'}): {check.message}")
        if check.expected_paths and check.status is not CheckStatus.PASS:
            lines.extend(f"    expected: {path}" for path in check.expected_paths)
    if report.facts:
        lines.extend(("", "Recorded facts:"))
        for key, value in report.facts.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def write_environment_report(
    report: EnvironmentReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Atomically write ``environment_report.json`` and ``.txt``."""
    directory = Path(output_dir)
    json_path = directory / "environment_report.json"
    text_path = directory / "environment_report.txt"
    try:
        json_text = json.dumps(
            report.model_dump(mode="json", round_trip=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        _atomic_write_text(json_path, json_text)
        _atomic_write_text(text_path, format_environment_report(report))
    except (OSError, TypeError, ValueError) as exc:
        raise EnvironmentCheckError(
            f"Cannot write environment reports in '{directory}': {exc}"
        ) from exc
    return json_path, text_path


def doctor(
    config: AppConfig,
    output_dir: str | Path | None = None,
) -> EnvironmentReport:
    """Inspect the environment and optionally persist both report formats."""
    report = inspect_environment(config)
    if output_dir is not None:
        write_environment_report(report, output_dir)
    return report


def _atomic_write_text(path: Path, content: str) -> None:
    """Flush text to a sibling temporary file before atomic replacement."""
    temporary_path: Path | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


run_doctor = inspect_environment


__all__ = [
    "CheckStatus",
    "EnvironmentCheck",
    "EnvironmentDoctor",
    "EnvironmentReport",
    "doctor",
    "format_environment_report",
    "inspect_environment",
    "run_doctor",
    "write_environment_report",
]
