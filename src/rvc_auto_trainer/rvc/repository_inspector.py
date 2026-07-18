"""Runtime inspection of an external RVC checkout.

The upstream RVC ecosystem has several incompatible repository layouts.  This
module deliberately reports what is present instead of claiming compatibility
with a repository that was not inspected.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable


@dataclass(frozen=True)
class ScriptInfo:
    """A discovered RVC command-line script and its statically observed CLI."""

    purpose: str
    path: Path
    relative_path: str
    contract: str
    argparse_options: tuple[str, ...] = ()
    positional_arity: int | None = None
    inspection_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RVCRepositoryInfo:
    """Facts discovered from an RVC repository at runtime."""

    repository: Path
    python_executable: Path
    exists: bool
    git_commit: str | None = None
    layout: str = "unknown"
    scripts: dict[str, ScriptInfo] = field(default_factory=dict)
    missing_purposes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_supported(self) -> bool:
        """Return whether the minimum preprocessing/training entry points exist."""

        return self.exists and all(
            purpose in self.scripts
            for purpose in ("preprocess", "extract_features", "train")
        )

    def require_script(self, purpose: str) -> ScriptInfo:
        """Return a script or raise an actionable repository error."""

        try:
            return self.scripts[purpose]
        except KeyError as exc:
            searched = ", ".join(_candidate_paths(purpose)) or "no known candidates"
            raise RVCRepositoryInspectionError(
                f"RVC repository '{self.repository}' has no safely recognized "
                f"{purpose!r} CLI script. Searched: {searched}. Inspect this RVC "
                "version and configure an explicit reviewed command for the stage."
            ) from exc


class RVCRepositoryInspectionError(RuntimeError):
    """Raised when an external RVC checkout cannot be safely adapted."""


_CANDIDATES: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "preprocess": (
        ("infer/modules/train/preprocess.py", "webui-preprocess"),
        ("trainset_preprocess_pipeline_print.py", "legacy-preprocess"),
    ),
    "extract_f0": (
        ("infer/modules/train/extract/extract_f0_print.py", "webui-f0"),
        ("infer/modules/train/extract/extract_f0_rmvpe.py", "webui-rmvpe-gpu"),
        ("extract_f0_print.py", "legacy-f0"),
        ("extract_f0_rmvpe.py", "legacy-rmvpe-gpu"),
    ),
    "extract_features": (
        ("infer/modules/train/extract_feature_print.py", "webui-features"),
        ("extract_feature_print.py", "legacy-features"),
    ),
    "train": (
        ("infer/modules/train/train.py", "webui-train"),
        ("train_nsf_sim_cache_sid_load_pretrain.py", "legacy-train"),
    ),
    "build_index": (
        ("infer/modules/train/train_index.py", "argparse-index"),
        ("infer/modules/train/index.py", "argparse-index"),
        ("tools/train_index.py", "argparse-index"),
        ("train_index.py", "argparse-index"),
    ),
    "infer": (
        ("tools/infer_cli.py", "argparse-infer"),
        ("infer_cli.py", "argparse-infer"),
        ("rvc_cli.py", "argparse-infer"),
    ),
}


def _candidate_paths(purpose: str) -> tuple[str, ...]:
    return tuple(path for path, _contract in _CANDIDATES.get(purpose, ()))


class RepositoryInspector:
    """Inspect an external checkout without importing or executing its Python code."""

    def __init__(self, repository: Path, python_executable: Path) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.python_executable = Path(python_executable).expanduser().resolve()

    def inspect(self) -> RVCRepositoryInfo:
        """Inspect repository layout, script syntax and Git revision."""

        if not self.repository.is_dir():
            return RVCRepositoryInfo(
                repository=self.repository,
                python_executable=self.python_executable,
                exists=False,
                missing_purposes=tuple(_CANDIDATES),
                warnings=(
                    f"RVC repository directory does not exist: {self.repository}",
                ),
            )

        scripts: dict[str, ScriptInfo] = {}
        warnings: list[str] = []
        for purpose, candidates in _CANDIDATES.items():
            for relative_path, contract in candidates:
                candidate = (self.repository / Path(relative_path)).resolve()
                if not _is_within(candidate, self.repository) or not candidate.is_file():
                    continue
                scripts[purpose] = _inspect_script(
                    purpose=purpose,
                    path=candidate,
                    repository=self.repository,
                    contract=contract,
                )
                break

        if not self.python_executable.is_file():
            warnings.append(
                "Configured RVC Python interpreter does not exist: "
                f"{self.python_executable}"
            )

        missing = tuple(purpose for purpose in _CANDIDATES if purpose not in scripts)
        if "build_index" in missing:
            warnings.append(
                "No standalone index CLI was recognized. Some RVC versions only "
                "build indexes inside their web UI; this adapter will not emulate it."
            )
        layout = _classify_layout(scripts.values())
        return RVCRepositoryInfo(
            repository=self.repository,
            python_executable=self.python_executable,
            exists=True,
            git_commit=_read_git_commit(self.repository),
            layout=layout,
            scripts=scripts,
            missing_purposes=missing,
            warnings=tuple(warnings),
        )


def inspect_repository(repository: Path, python_executable: Path) -> RVCRepositoryInfo:
    """Convenience wrapper around :class:`RepositoryInspector`."""

    return RepositoryInspector(repository, python_executable).inspect()


def _inspect_script(
    *, purpose: str, path: Path, repository: Path, contract: str
) -> ScriptInfo:
    warnings: list[str] = []
    options: set[str] = set()
    max_argv_index: int | None = None
    try:
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(source, filename=str(path))
        options = _argparse_options(tree)
        max_argv_index = _maximum_sys_argv_index(tree)
    except (OSError, SyntaxError, UnicodeError) as exc:
        warnings.append(f"Static CLI inspection failed: {exc}")
    return ScriptInfo(
        purpose=purpose,
        path=path,
        relative_path=path.relative_to(repository).as_posix(),
        contract=contract,
        argparse_options=tuple(sorted(options)),
        positional_arity=max_argv_index,
        inspection_warnings=tuple(warnings),
    )


def _argparse_options(tree: ast.AST) -> set[str]:
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute) and function.attr == "add_argument"):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("-"):
                    options.add(argument.value)
    return options


def _maximum_sys_argv_index(tree: ast.AST) -> int | None:
    maximum: int | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "argv"
            and isinstance(value.value, ast.Name)
            and value.value.id == "sys"
        ):
            continue
        index_node = node.slice
        if isinstance(index_node, ast.Constant) and isinstance(index_node.value, int):
            maximum = max(maximum or 0, index_node.value)
    return maximum


def _classify_layout(scripts: Iterable[ScriptInfo]) -> str:
    contracts = {script.contract for script in scripts}
    if any(contract.startswith("webui-") for contract in contracts):
        return "retrieval-based-voice-conversion-webui"
    if any(contract.startswith("legacy-") for contract in contracts):
        return "legacy-rvc-webui"
    return "unknown"


def _read_git_commit(repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and len(commit) >= 7 else None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
