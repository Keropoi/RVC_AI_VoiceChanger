"""Generate a self-contained, offline HTML listening report."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass(frozen=True)
class ReportData:
    """Serializable sections consumed by the report template."""

    summary: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    checkpoints: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    test_results: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    logs: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = field(default_factory=tuple)


class HTMLReportGenerator:
    """Render ``report/index.html`` and a manual review CSV template."""

    def __init__(self, template_dir: Path | None = None) -> None:
        directory = (
            Path(template_dir)
            if template_dir is not None
            else Path(__file__).with_name("templates")
        )
        self.environment = Environment(
            loader=FileSystemLoader(str(directory)),
            autoescape=select_autoescape(enabled_extensions=("html", "jinja2")),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(
        self,
        run_dir: Path,
        data: ReportData | Mapping[str, Any] | None = None,
    ) -> Path:
        """Generate a report, discovering standard manifests when data is omitted."""

        run_dir = Path(run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory is missing: {run_dir}")
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_data = (
            discover_report_data(run_dir)
            if data is None
            else _coerce_report_data(data)
        )
        template_context = _prepare_context(report_data, run_dir, report_dir)
        template = self.environment.get_template("report.html.jinja2")
        output_path = report_dir / "index.html"
        output_path.write_text(template.render(**template_context), encoding="utf-8")
        _write_manual_review_csv(report_dir / "manual_review_template.csv", report_data)
        return output_path.resolve()


def generate_html_report(
    run_dir: Path, data: ReportData | Mapping[str, Any] | None = None
) -> Path:
    """Convenience wrapper used by CLI and pipeline handlers."""

    return HTMLReportGenerator().generate(run_dir, data)


def discover_report_data(run_dir: Path) -> ReportData:
    """Read known run manifests without failing on optional/missing sections."""

    run_dir = Path(run_dir).resolve()
    environment = _read_json(run_dir / "environment_report.json", {})
    state = _read_json(run_dir / "state.json", {})
    quality_payload = _read_json(run_dir / "quality" / "audio_quality.json", {})
    model_manifest = _read_json(run_dir / "artifacts" / "model_manifest.json", {})
    checkpoint_payload = _read_json(
        run_dir / "checkpoints" / "checkpoint_manifest.json", {}
    )
    test_payload = _read_json(run_dir / "test_results" / "test_results.json", [])
    summary: dict[str, Any] = {
        "run_id": state.get("run_id", run_dir.name)
        if isinstance(state, dict)
        else run_dir.name,
    }
    if isinstance(state, dict):
        summary.update(
            {
                key: state[key]
                for key in ("created_at", "updated_at")
                if key in state
            }
        )
    if isinstance(environment, dict):
        summary.update(
            {
                key: environment[key]
                for key in (
                    "rvc_commit",
                    "gpu_name",
                    "pytorch_version",
                    "cuda_version",
                )
                if key in environment
            }
        )
    checkpoints = _as_record_sequence(
        checkpoint_payload.get("checkpoints", [])
        if isinstance(checkpoint_payload, dict)
        else []
    )
    if isinstance(test_payload, dict):
        tests = _as_record_sequence(
            test_payload.get("results", test_payload.get("tests", []))
        )
    else:
        tests = _as_record_sequence(test_payload)
    logs = {
        path.name: str(path)
        for path in sorted((run_dir / "logs").glob("*.log"))
        if path.is_file()
    }
    quality_section = (
        quality_payload.get("summary", quality_payload)
        if isinstance(quality_payload, dict)
        else {}
    )
    artifacts = model_manifest if isinstance(model_manifest, dict) else {}
    if isinstance(state, dict):
        stages = state.get("stages", {})
        training_record = stages.get("MODEL_TRAINED", {}) if isinstance(stages, dict) else {}
        training_metadata = (
            training_record.get("metadata", {})
            if isinstance(training_record, dict)
            else {}
        )
        if isinstance(training_metadata, dict):
            artifacts = {
                **artifacts,
                "selected_batch_size": training_metadata.get("selected_batch_size"),
                "oom_retry_count": training_metadata.get("oom_retry_count", 0),
            }
    return ReportData(
        summary=summary,
        quality=quality_section,
        artifacts=artifacts,
        checkpoints=checkpoints,
        test_results=tests,
        logs=logs,
    )


def make_relative_media_path(path: Path | str, run_dir: Path, report_dir: Path) -> str | None:
    """Return a URL-escaped relative path only for files inside the run directory."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    relative = os.path.relpath(candidate, report_dir.resolve()).replace("\\", "/")
    return quote(relative, safe="/._~+-")


def _prepare_context(data: ReportData, run_dir: Path, report_dir: Path) -> dict[str, Any]:
    warnings = list(data.warnings)
    tests: list[dict[str, Any]] = []
    for item in data.test_results:
        record = dict(item)
        original = _first_value(record, ("original_path", "input_path", "original"))
        converted = _first_value(record, ("converted_path", "output_path", "converted"))
        record["original_audio_url"] = (
            make_relative_media_path(original, run_dir, report_dir) if original else None
        )
        record["converted_audio_url"] = (
            make_relative_media_path(converted, run_dir, report_dir) if converted else None
        )
        if original and record["original_audio_url"] is None:
            warnings.append(f"Skipped non-portable original audio path for {record.get('name', 'test')}")
        if converted and record["converted_audio_url"] is None:
            warnings.append(f"Skipped non-portable converted audio path for {record.get('name', 'test')}")
        tests.append(record)
    artifacts = _relativize_path_values(dict(data.artifacts), run_dir, report_dir)
    logs: dict[str, Any] = {}
    for name, value in data.logs.items():
        relative = (
            make_relative_media_path(value, run_dir, report_dir)
            if isinstance(value, (str, Path))
            else None
        )
        logs[str(name)] = relative or "(outside run directory; omitted)"
    return {
        "summary": dict(data.summary),
        "quality": dict(data.quality),
        "artifacts": artifacts,
        "checkpoints": [
            _relativize_path_values(dict(item), run_dir, report_dir)
            for item in data.checkpoints
        ],
        "tests": tests,
        "logs": logs,
        "warnings": warnings,
    }


def _relativize_path_values(
    record: dict[str, Any], run_dir: Path, report_dir: Path
) -> dict[str, Any]:
    for key, value in tuple(record.items()):
        if not isinstance(value, (str, Path)):
            continue
        if "path" not in key.lower() and key.lower() not in {
            "model",
            "index",
            "training_log",
        }:
            continue
        relative = make_relative_media_path(value, run_dir, report_dir)
        if relative is not None:
            record[key] = relative
        elif Path(value).is_absolute():
            record[key] = "(outside run directory; omitted)"
    return record


def _write_manual_review_csv(path: Path, data: ReportData) -> None:
    fields = (
        "test_name",
        "original_path",
        "converted_path",
        "timbre_similarity_1_5",
        "clarity_1_5",
        "naturalness_1_5",
        "emotion_retention_1_5",
        "metallic_artifacts_1_5",
        "notes",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for test in data.test_results:
            writer.writerow(
                {
                    "test_name": test.get("name", test.get("test_name", "")),
                    "original_path": _first_value(
                        test, ("original_path", "input_path", "original")
                    ) or "",
                    "converted_path": _first_value(
                        test, ("converted_path", "output_path", "converted")
                    ) or "",
                }
            )


def _coerce_report_data(data: ReportData | Mapping[str, Any]) -> ReportData:
    if isinstance(data, ReportData):
        return data
    return ReportData(
        summary=_as_mapping(data.get("summary", {})),
        quality=_as_mapping(data.get("quality", {})),
        artifacts=_as_mapping(data.get("artifacts", {})),
        checkpoints=_as_record_sequence(data.get("checkpoints", [])),
        test_results=_as_record_sequence(
            data.get("test_results", data.get("tests", []))
        ),
        logs=_as_mapping(data.get("logs", {})),
        warnings=tuple(str(item) for item in data.get("warnings", [])),
    )


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_record_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    return next((record[key] for key in keys if record.get(key) is not None), None)
