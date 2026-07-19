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
from .dataset import apply_review_decisions, prepare_dataset
from .environment import format_environment_report, inspect_environment, write_environment_report
from .evaluation_dataset import freeze_test_manifest, validate_frozen_test_manifest
from .exceptions import RVCAutoTrainerError
from .logging_utils import close_logging, setup_logging
from .pipeline.default_handlers import build_default_pipeline
from .pipeline.orchestrator import PipelineExecutionError, PipelineRunResult
from .pipeline.run_context import RunContext, generate_run_id
from .reporting.html_report import generate_html_report
from .rvc.adapter import RVCAdapter
from .speaker_sorting import apply_speaker_review, sort_speakers
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
        "data/mixed_speaker_audio",
        "data/speaker_segments",
        "data/speaker_manifests",
        "data/speaker_selected_audio",
        "data/raw_archive",
        "data/training_candidates",
        "data/training_audio",
        "data/voice_references",
        "data/dataset_manifests",
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
        root / "data" / "raw_archive" / "README.txt": (
            "Place untouched, legally owned or authorized source recordings here. "
            "prepare-data never edits or deletes them.\n"
        ),
        root / "data" / "mixed_speaker_audio" / "README.txt": (
            "Place untouched authorized recordings containing multiple clear speakers here.\n"
        ),
        root / "data" / "speaker_selected_audio" / "README.txt": (
            "Reviewed target-speaker clips are copied here; do not edit generated clips.\n"
        ),
        root / "data" / "training_candidates" / "README.txt": (
            "Generated immutable coarse candidates appear here. Review them before promotion.\n"
        ),
        root / "data" / "training_audio" / "README.txt": (
            "Place legally owned or authorized training audio here. Subdirectories are supported.\n"
        ),
        root / "data" / "voice_references" / "README.txt": (
            "Keep 1 to 2 minutes of target-voice references excluded from training here.\n"
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


@app.command("prepare-data")
def prepare_data(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
    source_label: str = typer.Option(
        ...,
        "--source-label",
        help="Human-readable source/batch identifier recorded in the immutable manifest.",
    ),
    rights_note: str = typer.Option(
        ...,
        "--rights-note",
        help="Short authorization or license note recorded before processing.",
    ),
    language: str = typer.Option(
        "ja",
        "--language",
        help="Primary source language tag; informational and not used by RVC training.",
    ),
) -> None:
    """Hash raw originals, make 2-5 minute candidates, and create a review queue."""

    def action() -> int:
        result = prepare_dataset(
            load_config(config),
            source_label=source_label,
            rights_note=rights_note,
            language=language,
        )
        console.print(
            f"Prepared {result.candidate_file_count} candidate file(s) from "
            f"{result.raw_file_count} raw original(s)."
        )
        console.print(f"Raw manifest: [bold]{result.raw_manifest_path}[/bold]")
        console.print(f"Candidate quality: [bold]{result.candidate_quality_json_path}[/bold]")
        console.print(
            f"Review queue ({result.review_required_count} required): "
            f"[bold]{result.review_queue_path}[/bold]"
        )
        console.print(f"Summary: [bold]{result.summary_path}[/bold]")
        console.print(
            "No audio was promoted automatically. Fill decision/category/reviewer_notes "
            "before running apply-data-review."
        )
        return 0

    _execute(action)


@app.command("sort-speakers")
def sort_speaker_audio(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
    num_speakers: Optional[int] = typer.Option(
        None, "--num-speakers", min=1, help="Exact known speaker count per recording."
    ),
    min_speakers: Optional[int] = typer.Option(
        None, "--min-speakers", min=1, help="Optional lower speaker-count bound."
    ),
    max_speakers: Optional[int] = typer.Option(
        None, "--max-speakers", min=1, help="Optional upper speaker-count bound."
    ),
) -> None:
    """Diarize mixed recordings and create a target-character review queue."""

    def action() -> int:
        result = sort_speakers(
            load_config(config),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        console.print(
            f"Created {result.segment_count} non-destructive segment(s) in "
            f"{result.speaker_cluster_count} anonymous speaker cluster(s)."
        )
        console.print(f"Review queue: [bold]{result.review_queue_path}[/bold]")
        console.print(
            "Listen to sample_files and fill every decision with TARGET, OTHER, or REJECT. "
            "Similarity scores never promote audio automatically."
        )
        return 0

    _execute(action)


@app.command("apply-speaker-review")
def apply_reviewed_speakers(
    review_queue: Path = typer.Option(
        ...,
        "--review-queue",
        exists=True,
        dir_okay=False,
        help="speaker_review.csv produced by sort-speakers.",
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
) -> None:
    """Copy explicitly reviewed target-character clips into the curation input."""

    def action() -> int:
        result = apply_speaker_review(load_config(config), review_queue)
        console.print(
            f"Copied {result.copied_segment_count} segment(s) from "
            f"{result.target_cluster_count} target cluster(s), "
            f"{result.copied_duration_minutes:.2f} minute(s) total."
        )
        console.print(f"Applied review: [bold]{result.summary_path}[/bold]")
        console.print("Next: run prepare-data with source and rights metadata.")
        return 0

    _execute(action)


@app.command("apply-data-review")
def apply_data_review(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
    review_queue: Optional[Path] = typer.Option(
        None,
        "--review-queue",
        exists=True,
        dir_okay=False,
        help="Review CSV; defaults to data/dataset_manifests/review_queue.csv.",
    ),
    accept_unreviewed_pass: bool = typer.Option(
        False,
        "--accept-unreviewed-pass",
        help="Promote non-sampled PASS rows only after required reviews are complete.",
    ),
) -> None:
    """Apply explicit KEEP/REFERENCE/REJECT decisions using copy-only promotion."""

    def action() -> int:
        result = apply_review_decisions(
            load_config(config),
            review_queue,
            accept_unreviewed_pass=accept_unreviewed_pass,
        )
        console.print(
            f"Promoted {result.kept_count} training file(s), retained "
            f"{result.reference_count} reference file(s), and recorded "
            f"{result.rejected_count} rejection(s)."
        )
        console.print(f"Selected training duration: {result.kept_duration_minutes:.2f} min")
        if result.kept_duration_minutes < 15.0:
            console.print("[yellow]Core set is below the 15-minute first-version target.[/yellow]")
        elif result.kept_duration_minutes > 20.0:
            console.print(
                "[yellow]Core set exceeds 20 minutes; confirm every addition improves quality.[/yellow]"
            )
        console.print(f"Applied-review manifest: [bold]{result.summary_path}[/bold]")
        return 0

    _execute(action)


@app.command("freeze-tests")
def freeze_tests(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
) -> None:
    """Freeze the five role-based Mandarin/Japanese test inputs by SHA-256."""

    def action() -> int:
        result = freeze_test_manifest(load_config(config))
        console.print(
            f"Frozen and leakage-checked {len(result.files)} test file(s): "
            f"[bold]{result.manifest_path}[/bold]"
        )
        return 0

    _execute(action)


@app.command("validate-tests")
def validate_tests(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", exists=True, dir_okay=False),
) -> None:
    """Verify frozen test hashes, durations, roles, and training-set isolation."""

    def action() -> int:
        result = validate_frozen_test_manifest(load_config(config))
        console.print(
            f"Validated {len(result.files)} unchanged fixed test file(s): "
            f"[bold]{result.manifest_path}[/bold]"
        )
        return 0

    _execute(action)


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
