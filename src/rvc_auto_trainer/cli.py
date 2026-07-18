"""Typer command-line interface for local RVC automation workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import AppConfig, load_config
from .environment import format_environment_report, inspect_environment, write_environment_report
from .exceptions import RVCAutoTrainerError
from .logging_utils import close_logging, setup_logging
from .pipeline.default_handlers import build_default_pipeline
from .pipeline.orchestrator import PipelineExecutionError, PipelineRunResult
from .pipeline.run_context import RunContext, generate_run_id
from .reporting.html_report import generate_html_report
from .rvc.adapter import RVCAdapter
from .state import PipelineStage, StateStore

app = typer.Typer(
    name="rvc-auto-trainer",
    help="Audit audio and orchestrate a resumable external RVC training workflow.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
DEFAULT_CONFIG = Path("config/example_windows_3090.yaml")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"rvc-auto-trainer {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """RVC automation control layer; upstream RVC remains a separate checkout."""

    del version


@app.command("init")
def initialize_project(
    project_root: Path = typer.Option(
        Path("."),
        "--project-root",
        file_okay=False,
        resolve_path=True,
        help="Directory to initialize without overwriting user files.",
    ),
) -> None:
    """Create the standard input/output tree and starter configuration."""

    root = project_root.resolve()
    directories = (
        "config",
        "data/training_audio",
        "data/test_audio",
        "data/rejected_audio",
        "external/RVC",
        "models/pretrained",
        "runs",
    )
    for relative in directories:
        (root / relative).mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    default_path = root / "config" / "default.yaml"
    example_path = root / "config" / "example_windows_3090.yaml"
    template = AppConfig(project_root=root).model_dump(
        mode="json", exclude={"config_path", "project_root"}
    )
    rendered = yaml.safe_dump(template, allow_unicode=True, sort_keys=False)
    for destination in (default_path, example_path):
        if not destination.exists():
            destination.write_text(rendered, encoding="utf-8")
            created.append(destination)

    hints = {
        root / "data" / "training_audio" / "README.txt": (
            "Place legally owned or authorized training audio here. Subdirectories are supported.\n"
        ),
        root / "data" / "test_audio" / "README.txt": (
            "Place 3 to 5 legally owned or authorized test recordings here.\n"
        ),
        root / "external" / "RVC" / "README.md": (
            "Place the selected official RVC checkout and its separate virtual environment here.\n"
        ),
    }
    for destination, content in hints.items():
        if not destination.exists():
            destination.write_text(content, encoding="utf-8")
            created.append(destination)

    console.print(f"Initialized project layout at [bold]{root}[/bold]")
    if created:
        console.print("Created: " + ", ".join(str(path.relative_to(root)) for path in created))
    else:
        console.print("All starter files already existed; nothing was overwritten.")


@app.command()
def doctor(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        help="Optional report directory; defaults to a new doctor run.",
    ),
) -> None:
    """Inspect Python, FFmpeg, CUDA, GPU, disk, RVC, assets, and inputs."""

    def action() -> int:
        resolved = load_config(config)
        if output_dir is None:
            run_id = "doctor_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            context = RunContext.create(resolved, run_id=run_id)
            destination = context.run_dir
        else:
            destination = output_dir.resolve()
            destination.mkdir(parents=True, exist_ok=True)
        report = inspect_environment(resolved)
        json_path, text_path = write_environment_report(report, destination)
        console.print(format_environment_report(report), markup=False)
        console.print(f"Reports: {json_path} | {text_path}")
        return 0 if report.healthy else 1

    _execute(action)


@app.command()
def audit(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
) -> None:
    """Discover, hash, measure, classify, and report training audio only."""

    def action() -> int:
        context = RunContext.create(load_config(config))
        logger = setup_logging(context.logs_dir, run_id=context.run_id)
        try:
            pipeline = _pipeline(context, dry_run=False, logger=logger)
            result = pipeline.run(
                stages=(PipelineStage.AUDIO_DISCOVERED, PipelineStage.QUALITY_CHECKED)
            )
            _print_pipeline_result(result, context)
        finally:
            close_logging(logger)
        return 0

    _execute(action)


@app.command("run")
def run_pipeline(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan without writing or training."),
) -> None:
    """Run the complete quality, RVC training, index, inference, and report pipeline."""

    def action() -> int:
        resolved = load_config(config)
        if dry_run:
            run_id = generate_run_id(resolved.model.name)
            run_dir = resolved.paths.runs_dir / run_id
            context = RunContext(
                resolved,
                run_id,
                run_dir,
                StateStore(run_dir / "state.json", run_id),
            )
            console.print(f"Planned run directory (not created): [bold]{run_dir}[/bold]")
            result = _pipeline(context, dry_run=True).run(resume=False)
            _print_pipeline_result(result, context)
            return 0

        context = RunContext.create(resolved)
        logger = setup_logging(context.logs_dir, run_id=context.run_id)
        try:
            result = _pipeline(context, dry_run=False, logger=logger).run(resume=True)
            _print_pipeline_result(result, context)
        finally:
            close_logging(logger)
        return 0

    _execute(action)


@app.command()
def resume(
    run_id: str = typer.Option(..., "--run-id", help="Existing run directory name."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", file_okay=False),
) -> None:
    """Resume the first failed, invalidated, or incomplete stage of an existing run."""

    def action() -> int:
        context = RunContext.load(runs_dir.resolve() / run_id)
        logger = setup_logging(context.logs_dir, run_id=context.run_id)
        try:
            result = _pipeline(context, dry_run=False, logger=logger).run(resume=True)
            _print_pipeline_result(result, context)
        finally:
            close_logging(logger)
        return 0

    _execute(action)


@app.command("test")
def test_model(
    run_id: str = typer.Option(..., "--run-id", help="Existing trained run."),
    test_dir: Optional[Path] = typer.Option(
        None, "--test-dir", exists=True, file_okay=False, help="Optional test-audio override."
    ),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", file_okay=False),
) -> None:
    """Rerun test inference and regenerate the report without retraining."""

    def action() -> int:
        context = RunContext.load(runs_dir.resolve() / run_id)
        if test_dir is not None:
            updated_paths = context.config.paths.model_copy(
                update={"test_audio_dir": test_dir.resolve()}
            )
            context.config = context.config.model_copy(update={"paths": updated_paths})
        logger = setup_logging(context.logs_dir, run_id=context.run_id)
        try:
            result = _pipeline(context, dry_run=False, logger=logger).run(
                stages=(
                    PipelineStage.TEST_INFERENCE_COMPLETED,
                    PipelineStage.REPORT_GENERATED,
                ),
                resume=False,
            )
            _print_pipeline_result(result, context)
        finally:
            close_logging(logger)
        return 0

    _execute(action)


@app.command("report")
def report_command(
    run_id: str = typer.Option(..., "--run-id", help="Existing run directory name."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", file_okay=False),
) -> None:
    """Regenerate the portable offline HTML report from saved manifests."""

    def action() -> int:
        run_dir = (runs_dir.resolve() / run_id).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        report_path = generate_html_report(run_dir)
        console.print(f"Report generated: [bold]{report_path}[/bold]")
        return 0

    _execute(action)


def _pipeline(
    context: RunContext,
    *,
    dry_run: bool,
    logger: Any = None,
) -> Any:
    adapter = RVCAdapter(
        context.config.paths.rvc_repository,
        context.config.paths.rvc_python,
        logs_dir=context.logs_dir,
        logger=logger,
        monitoring_interval_seconds=context.config.monitoring.interval_seconds,
        monitor_gpu=(
            context.config.monitoring.enabled and context.config.monitoring.record_gpu
        ),
    )
    pipeline = build_default_pipeline(context, adapter, dry_run=dry_run)
    if logger is not None:
        pipeline.logger = logger
    return pipeline


def _print_pipeline_result(result: PipelineRunResult, context: RunContext) -> None:
    table = Table(title=f"Run {context.run_id}")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Message")
    for record in result.records:
        status = "REUSED" if record.reused else "SKIPPED" if record.skipped else "PLANNED" if result.dry_run else "DONE"
        table.add_row(record.stage.value, status, record.result.message)
    console.print(table)
    console.print(f"Run directory: [bold]{context.run_dir}[/bold]")


def _execute(action: Callable[[], int]) -> None:
    try:
        code = action()
    except KeyboardInterrupt:
        console.print("[red]Interrupted. Completed stages and logs were preserved.[/red]")
        raise typer.Exit(code=130)
    except (RVCAutoTrainerError, PipelineExecutionError, OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    if code:
        raise typer.Exit(code=code)


if __name__ == "__main__":
    app()
