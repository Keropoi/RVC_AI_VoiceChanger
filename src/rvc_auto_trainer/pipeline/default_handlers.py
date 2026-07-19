"""Concrete end-to-end stage handlers used by the CLI."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..audio.discovery import (
    AudioFile,
    discover_audio_files,
    duplicate_hashes,
    select_test_audio,
)
from ..audio.normalization import preprocess_audio
from ..audio.quality import (
    QualityStatus,
    analyze_audio,
    audit_audio_files,
    summarize_quality,
)
from ..audio.reports import copy_rejected_audio, write_quality_reports
from ..audio.slicer import slice_audio
from ..evaluation_dataset import validate_frozen_test_manifest
from ..reporting.html_report import generate_html_report
from ..rvc.adapter import RVCAdapter
from ..rvc.feature_extraction import FeatureRequest
from ..rvc.index_builder import IndexRequest
from ..rvc.inference import InferenceRequest, InferenceResult
from ..rvc.preprocess import PreprocessRequest
from ..rvc.training import (
    TrainingRequest,
    discover_valid_checkpoints,
    validate_artifact,
)
from ..rvc.training_workspace import prepare_training_workspace
from ..state import PipelineStage, StageResult
from .orchestrator import PipelineOrchestrator
from .stages import (
    StageDefinition,
    StageExecutionContext,
    stable_stage_fingerprint,
)


def build_default_stage_handlers(
    run_context: object,
    adapter: object,
) -> dict[PipelineStage, StageDefinition]:
    """Build concrete handlers for all eight executable stages."""

    del run_context, adapter  # handlers consume these through StageExecutionContext
    return {
        PipelineStage.AUDIO_DISCOVERED: StageDefinition(
            PipelineStage.AUDIO_DISCOVERED,
            _discover_audio,
            "Discover and hash training/test audio",
            fingerprint=_input_fingerprint,
            required_outputs=lambda ctx: (ctx.run_context.input_manifest_path,),
        ),
        PipelineStage.QUALITY_CHECKED: StageDefinition(
            PipelineStage.QUALITY_CHECKED,
            _quality_check,
            "Measure quality and exclude failed input",
            fingerprint=_run_fingerprint,
            required_outputs=lambda ctx: (
                ctx.run_context.quality_dir / "audio_quality.json",
            ),
        ),
        PipelineStage.PREPROCESSED: StageDefinition(
            PipelineStage.PREPROCESSED,
            _preprocess,
            "Normalize/slice accepted audio and invoke RVC preprocessing",
            fingerprint=_run_fingerprint,
        ),
        PipelineStage.FEATURES_EXTRACTED: StageDefinition(
            PipelineStage.FEATURES_EXTRACTED,
            _extract_features,
            "Extract F0 and content features through RVC",
            fingerprint=_run_fingerprint,
        ),
        PipelineStage.MODEL_TRAINED: StageDefinition(
            PipelineStage.MODEL_TRAINED,
            _train,
            "Train and validate the final RVC model",
            fingerprint=_run_fingerprint,
            required_outputs=lambda ctx: (
                ctx.run_context.artifacts_dir / f"{ctx.run_context.config.model.name}.pth",
            ),
        ),
        PipelineStage.INDEX_BUILT: StageDefinition(
            PipelineStage.INDEX_BUILT,
            _build_index,
            "Build and validate the FAISS index",
            fingerprint=_run_fingerprint,
            enabled=lambda ctx: bool(ctx.run_context.config.index.enabled),
            required_outputs=lambda ctx: (
                ctx.run_context.artifacts_dir / f"{ctx.run_context.config.model.name}.index",
            ),
            skip_reason="Index generation is disabled in resolved configuration",
        ),
        PipelineStage.TEST_INFERENCE_COMPLETED: StageDefinition(
            PipelineStage.TEST_INFERENCE_COMPLETED,
            _test_inference,
            "Convert configured test audio and inspect outputs",
            fingerprint=_run_fingerprint,
            enabled=lambda ctx: bool(ctx.run_context.config.testing.enabled),
            required_outputs=lambda ctx: (
                ctx.run_context.test_results_dir / "test_results.json",
            ),
            skip_reason="Test inference is disabled in resolved configuration",
        ),
        PipelineStage.REPORT_GENERATED: StageDefinition(
            PipelineStage.REPORT_GENERATED,
            _report,
            "Generate offline HTML listening report",
            fingerprint=_run_fingerprint,
            enabled=lambda ctx: bool(ctx.run_context.config.report.generate_html),
            required_outputs=lambda ctx: (ctx.run_context.report_dir / "index.html",),
            skip_reason="HTML report generation is disabled in resolved configuration",
        ),
    }


def build_default_pipeline(
    run_context: object,
    adapter: RVCAdapter | None = None,
    *,
    dry_run: bool = False,
) -> PipelineOrchestrator:
    """Construct the CLI-ready orchestrator, creating an adapter when omitted."""

    config = run_context.config
    actual_adapter = adapter or RVCAdapter(
        config.paths.rvc_repository,
        config.paths.rvc_python,
        logs_dir=run_context.logs_dir,
        monitoring_interval_seconds=config.monitoring.interval_seconds,
        monitor_gpu=config.monitoring.enabled and config.monitoring.record_gpu,
    )
    return PipelineOrchestrator(
        run_context,
        actual_adapter,
        build_default_stage_handlers(run_context, actual_adapter),
        dry_run=dry_run,
    )


def _discover_audio(context: StageExecutionContext) -> StageResult:
    config = context.run_context.config
    training = discover_audio_files(config.paths.training_audio_dir)
    test = discover_audio_files(config.paths.test_audio_dir)
    manifest_entries = _load_test_manifest(config.paths.test_audio_dir)
    if config.testing.require_frozen_manifest:
        frozen = validate_frozen_test_manifest(config)
        selected_tests = frozen.files
        if len(selected_tests) > config.testing.maximum_test_files:
            raise ValueError(
                "Frozen test set exceeds testing.maximum_test_files; do not silently truncate it"
            )
    else:
        selected_tests = _select_manifest_tests(
            test.files,
            manifest_entries,
            config.testing.maximum_test_files,
        )
        _validate_optional_manifest_hashes(selected_tests, manifest_entries)
    duplicates = duplicate_hashes(training.files, selected_tests)
    if duplicates:
        names = ", ".join(sorted(duplicates)[:5])
        raise ValueError(
            "Training/test leakage detected: identical SHA-256 values appear in "
            f"both sets ({names})"
        )
    payload = {
        "schema_version": 1,
        "training_root": str(training.root),
        "test_root": str(test.root),
        "training_files": [item.to_manifest_record() for item in training.files],
        "test_files": [item.to_manifest_record() for item in selected_tests],
        "ignored_training_files": [
            {
                "relative_path": item.relative_path,
                "reason": item.reason,
            }
            for item in training.ignored
        ],
        "ignored_test_files": [
            {
                "relative_path": item.relative_path,
                "reason": item.reason,
            }
            for item in test.ignored
        ],
    }
    if not context.dry_run:
        context.run_context.write_json("input_manifest.json", payload)
    return StageResult(
        outputs=(context.run_context.input_manifest_path,),
        metadata={
            "training_file_count": len(training.files),
            "selected_test_file_count": len(selected_tests),
            "planned_files": [str(item.path) for item in training.files],
            "dry_run": context.dry_run,
        },
        message="Audio inputs discovered and hashed",
    )


def _quality_check(context: StageExecutionContext) -> StageResult:
    run_context = context.run_context
    config = run_context.config
    predicted = (
        run_context.quality_dir / "audio_quality.csv",
        run_context.quality_dir / "audio_quality.json",
        run_context.quality_dir / "audio_quality.html",
    )
    if context.dry_run:
        return StageResult(
            outputs=predicted,
            metadata={"dry_run": True, "planned_outputs": [str(path) for path in predicted]},
            message="Dry run: quality audit was not executed",
        )
    discovered = discover_audio_files(config.paths.training_audio_dir)
    if not discovered.files:
        raise ValueError(
            f"No supported non-empty training audio found in {config.paths.training_audio_dir}"
        )
    results = audit_audio_files(
        discovered.files,
        config,
        root=config.paths.training_audio_dir,
        ffmpeg_executable="ffmpeg",
    )
    reports = write_quality_reports(results, run_context.quality_dir, config)
    rejected_dir = config.paths.rejected_audio_dir / run_context.run_id
    rejected = copy_rejected_audio(
        results,
        rejected_dir,
        source_root=config.paths.training_audio_dir,
    )
    summary = summarize_quality(results, config)
    accepted = [result for result in results if result.accepted_for_training]
    if not accepted:
        raise ValueError("Every training file failed the quality audit")
    if (
        config.quality.fail_on_insufficient_total_duration
        and summary.accepted_duration_minutes
        < config.quality.minimum_total_accepted_duration_minutes
    ):
        raise ValueError(
            f"Accepted duration is {summary.accepted_duration_minutes:.2f} minutes; "
            "minimum required is "
            f"{config.quality.minimum_total_accepted_duration_minutes:.2f} minutes"
        )
    return StageResult(
        outputs=(reports.csv_path, reports.json_path, reports.html_path),
        metadata={
            "summary": summary.to_dict(),
            "accepted_file_count": len(accepted),
            "rejected_file_count": len(rejected),
            "rejected_directory": str(rejected_dir),
        },
        message="Audio quality audit completed",
    )


def _preprocess(context: StageExecutionContext) -> StageResult:
    run_context = context.run_context
    config = run_context.config
    normalized_dir = run_context.preprocessed_dir / "normalized"
    rvc_input_dir = normalized_dir
    local_outputs: list[Path] = []
    if not context.dry_run:
        quality = _read_json(run_context.quality_dir / "audio_quality.json")
        accepted = [
            item
            for item in quality.get("files", [])
            if item.get("status") in {QualityStatus.PASS.value, QualityStatus.WARNING.value}
        ]
        if not accepted:
            raise ValueError("Quality report contains no accepted files to preprocess")
        for record in accepted:
            source = Path(str(record["file_path"]))
            destination = normalized_dir / _stable_audio_name(
                str(record.get("relative_path", source.name)), record.get("sha256")
            )
            processed = preprocess_audio(source, destination, config, ffmpeg_executable="ffmpeg")
            local_outputs.append(processed.output_path)
        if config.slicing.backend == "internal":
            segment_root = run_context.preprocessed_dir / "segments"
            rvc_flat_input = run_context.preprocessed_dir / "rvc_input"
            segment_outputs: list[Path] = []
            for normalized in local_outputs:
                result = slice_audio(
                    normalized,
                    segment_root / normalized.stem,
                    config,
                    quality_thresholds=config,
                )
                for item in result.accepted_segments:
                    segment_outputs.append(item.segment_file)
                    flat_name = f"{normalized.stem}__{item.segment_file.name}"
                    flat_destination = rvc_flat_input / flat_name
                    flat_destination.parent.mkdir(parents=True, exist_ok=True)
                    if flat_destination.exists():
                        raise FileExistsError(
                            f"Refusing to overwrite RVC slice input: {flat_destination}"
                        )
                    shutil.copy2(item.segment_file, flat_destination)
                segment_outputs.append(result.manifest_path)
            local_outputs.extend(segment_outputs)
            rvc_input_dir = rvc_flat_input
    request = PreprocessRequest(
        input_dir=rvc_input_dir,
        experiment_dir=_experiment_dir(context),
        sample_rate=config.preprocessing.target_sample_rate,
        process_count=max(1, min(8, os.cpu_count() or 1)),
        dry_run=context.dry_run,
    )
    rvc_result = context.adapter.preprocess(request)
    return StageResult(
        outputs=tuple(local_outputs) + tuple(rvc_result.outputs),
        metadata={
            "rvc": rvc_result.metadata,
            "normalized_file_count": len(local_outputs),
            "rvc_input_dir": str(rvc_input_dir),
        },
        message="Audio preprocessing and RVC workspace preparation completed",
    )


def _extract_features(context: StageExecutionContext) -> StageResult:
    config = context.run_context.config
    request = FeatureRequest(
        experiment_dir=_experiment_dir(context),
        f0_method=config.model.f0_method,
        gpu_id=config.training.gpu_ids[0],
        version=config.model.version,
        is_half=config.training.mixed_precision,
        dry_run=context.dry_run,
    )
    results: list[StageResult] = []
    if config.model.use_f0:
        results.append(context.adapter.extract_f0(request))
    results.append(context.adapter.extract_features(request))
    return _merge_results(results, "RVC F0/content features extracted")


def _train(context: StageExecutionContext) -> StageResult:
    run_context = context.run_context
    config = run_context.config
    experiment = _experiment_dir(context)
    started_at = time.time()
    workspace = prepare_training_workspace(
        config.paths.rvc_repository,
        experiment,
        sample_rate=config.model.sample_rate,
        version=config.model.version,
        use_f0=config.model.use_f0,
        speaker_id=config.model.speaker_id,
        random_seed=config.project.random_seed,
        mixed_precision=config.training.mixed_precision,
        dry_run=context.dry_run,
    )
    pretrained_generator, pretrained_discriminator = _resolve_pretrained_pair(config)
    expected_model = (
        config.paths.rvc_repository / "assets" / "weights" / f"{run_context.run_id}.pth"
    )
    request = TrainingRequest(
        experiment_dir=experiment,
        model_name=run_context.run_id,
        sample_rate=config.model.sample_rate,
        version=config.model.version,
        use_f0=config.model.use_f0,
        gpu_ids=tuple(config.training.gpu_ids),
        epochs=config.training.epochs,
        save_every_epochs=config.training.save_every_epochs,
        batch_size=config.training.batch_size,
        automatic_batch_size=config.training.automatic_batch_size,
        batch_size_candidates=tuple(config.training.batch_size_candidates),
        maximum_oom_retries=config.training.maximum_oom_retries,
        pretrained_generator=pretrained_generator,
        pretrained_discriminator=pretrained_discriminator,
        save_only_latest=config.training.save_only_latest,
        cache_dataset_in_gpu=config.training.cache_dataset_in_gpu,
        save_every_weights=config.training.save_every_weights,
        resume_if_available=config.training.resume_if_available,
        expected_model=expected_model,
        model_validator_script=config.project_root / "scripts" / "validate_rvc_model.py",
        checkpoint_dir=experiment,
        dry_run=context.dry_run,
    )
    result = context.adapter.train(request)
    final_model = run_context.artifacts_dir / f"{config.model.name}.pth"
    if context.dry_run:
        return StageResult(
            outputs=(final_model,),
            metadata={
                **result.metadata,
                "planned_model": str(final_model),
                "training_workspace": str(workspace.manifest_path),
            },
            message="Dry run: training command planned",
        )
    source_model = expected_model.resolve()
    final_model.parent.mkdir(parents=True, exist_ok=True)
    if final_model.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {final_model}")
    shutil.copy2(source_model, final_model)
    validation = validate_artifact(
        final_model,
        minimum_bytes=1_024,
        allowed_suffixes=(".pth",),
        not_before=started_at,
    )
    if not validation.valid:
        raise ValueError(f"Copied model failed validation: {validation.reason}")
    checkpoints = discover_valid_checkpoints(experiment, minimum_bytes=1_024)
    copied_checkpoints: list[Path] = []
    for checkpoint in checkpoints:
        destination = run_context.checkpoints_dir / checkpoint.path.name
        if not destination.exists():
            shutil.copy2(checkpoint.path, destination)
        copied_checkpoints.append(destination)
    external_checkpoint_manifest = experiment / "checkpoint_manifest.json"
    local_checkpoint_manifest = run_context.checkpoints_dir / "checkpoint_manifest.json"
    if external_checkpoint_manifest.is_file():
        shutil.copy2(external_checkpoint_manifest, local_checkpoint_manifest)
        copied_checkpoints.append(local_checkpoint_manifest)
    manifest = {
        "model_name": config.model.name,
        "model_path": str(final_model),
        "model_sha256": validation.sha256,
        "model_size_bytes": validation.size_bytes,
        "source_model": str(source_model),
        "rvc_commit": _adapter_commit(context.adapter),
        "disclaimer": "AI voice conversion model; not an official voice performance.",
    }
    run_context.write_json("artifacts/model_manifest.json", manifest)
    return StageResult(
        outputs=(
            final_model,
            run_context.artifacts_dir / "model_manifest.json",
            workspace.filelist_path,
            workspace.config_path,
            workspace.manifest_path,
            *copied_checkpoints,
        ),
        metadata={
            **result.metadata,
            "model_validation": manifest,
            "training_workspace": str(workspace.manifest_path),
        },
        message="RVC model trained, copied and validated",
    )


def _build_index(context: StageExecutionContext) -> StageResult:
    run_context = context.run_context
    config = run_context.config
    output = run_context.artifacts_dir / f"{config.model.name}.index"
    request = IndexRequest(
        feature_dir=_feature_dir(context),
        experiment_dir=_experiment_dir(context),
        output_path=output,
        algorithm=config.index.algorithm,
        command=(
            str(config.paths.rvc_python),
            str(config.project_root / "scripts" / "build_rvc_index.py"),
            "--feature-dir",
            "{feature_dir}",
            "--output",
            "{output_path}",
            "--algorithm",
            config.index.algorithm,
            "--seed",
            str(config.project.random_seed),
        ),
        dry_run=context.dry_run,
    )
    result = context.adapter.build_index(request)
    outputs = list(result.outputs)
    metadata = dict(result.metadata)
    if not context.dry_run:
        validation = validate_artifact(
            output,
            minimum_bytes=128,
            allowed_suffixes=(".index",),
        )
        if not validation.valid:
            raise ValueError(f"Generated index failed validation: {validation.reason}")
        model_manifest_path = run_context.artifacts_dir / "model_manifest.json"
        model_manifest = dict(_read_json(model_manifest_path))
        model_manifest.update(
            {
                "index_path": str(output),
                "index_sha256": validation.sha256,
                "index_size_bytes": validation.size_bytes,
            }
        )
        run_context.write_json("artifacts/model_manifest.json", model_manifest)
        outputs.append(model_manifest_path)
        metadata["index_manifest"] = {
            "path": str(output),
            "sha256": validation.sha256,
            "size_bytes": validation.size_bytes,
        }
    return StageResult(
        outputs=tuple(outputs),
        metadata=metadata,
        message="FAISS index built and validated",
    )


def _test_inference(context: StageExecutionContext) -> StageResult:
    run_context = context.run_context
    config = run_context.config
    discovered = discover_audio_files(config.paths.test_audio_dir)
    manifest_entries = _load_test_manifest(config.paths.test_audio_dir)
    selected = _select_manifest_tests(
        discovered.files,
        manifest_entries,
        config.testing.maximum_test_files,
    )
    if not selected:
        raise ValueError(f"No test audio found in {config.paths.test_audio_dir}")
    model = run_context.artifacts_dir / f"{config.model.name}.pth"
    index = run_context.artifacts_dir / f"{config.model.name}.index"
    originals_dir = run_context.test_results_dir / "originals"
    converted_dir = run_context.test_results_dir / "converted"
    records: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for item in selected:
        case = _manifest_entry_for(item, manifest_entries)
        variants = _test_parameter_variants(config.testing, case)
        original = originals_dir / _stable_original_name(item)
        if not context.dry_run:
            original.parent.mkdir(parents=True, exist_ok=True)
            if not original.exists():
                shutil.copy2(item.path, original)
        source = item.path if context.dry_run else original
        for variant_number, parameters in enumerate(variants, start=1):
            converted = converted_dir / (
                f"{Path(original).stem}__{config.model.name}__e{config.training.epochs}"
                f"__p{parameters['transpose']:+03d}__i{parameters['index_rate']:.2f}"
                f"__v{variant_number:02d}.{config.testing.output_format}"
            )
            if not context.dry_run:
                converted.parent.mkdir(parents=True, exist_ok=True)
            request = InferenceRequest(
                input_path=source,
                output_path=converted,
                model_path=model,
                index_path=index if config.index.enabled else None,
                speaker_id=config.model.speaker_id,
                transpose=int(parameters["transpose"]),
                f0_method=str(parameters["f0_method"]),
                index_rate=float(parameters["index_rate"]),
                filter_radius=int(parameters["filter_radius"]),
                resample_sample_rate=int(parameters["resample_sample_rate"]),
                rms_mix_rate=float(parameters["rms_mix_rate"]),
                protect=float(parameters["protect"]),
                allow_without_index=config.testing.allow_inference_without_index,
                dry_run=context.dry_run,
            )
            inference: InferenceResult = context.adapter.infer(request)
            record: dict[str, Any] = {
                "name": str(case.get("name") or Path(item.relative_path).stem),
                "original_path": str(original),
                "converted_path": str(converted),
                "transpose": parameters["transpose"],
                "index_rate": parameters["index_rate"],
                "protect": parameters["protect"],
                "rms_mix_rate": parameters["rms_mix_rate"],
                "status": "PLANNED" if context.dry_run else "PASS",
                "warnings": [],
            }
            if not context.dry_run:
                input_quality = analyze_audio(original, config, ffmpeg_executable="ffmpeg")
                quality = analyze_audio(converted, config, ffmpeg_executable="ffmpeg")
                warnings: list[str] = []
                duration_ratio = _duration_difference_ratio(
                    input_quality.duration_seconds,
                    quality.duration_seconds,
                )
                if (
                    duration_ratio is not None
                    and duration_ratio > config.testing.maximum_duration_difference_ratio
                ):
                    warnings.append(
                        f"Input/output duration difference ratio {duration_ratio:.4f} "
                        "exceeds configured maximum "
                        f"{config.testing.maximum_duration_difference_ratio:.4f}"
                    )
                status = quality.status.value
                if warnings and quality.status is QualityStatus.PASS:
                    status = QualityStatus.WARNING.value
                record.update(
                    {
                        "status": status,
                        "output_lufs": quality.integrated_lufs,
                        "output_peak_dbfs": quality.sample_peak_dbfs,
                        "duration_seconds": quality.duration_seconds,
                        "duration_difference_ratio": duration_ratio,
                        "warnings": warnings,
                    }
                )
                outputs.extend((original, inference.output_path))
            records.append(record)
    manifest = run_context.test_results_dir / "test_results.json"
    csv_path = run_context.test_results_dir / "test_results.csv"
    if not context.dry_run:
        run_context.write_json("test_results/test_results.json", {"results": records})
        _write_test_csv(csv_path, records)
        outputs.extend((manifest, csv_path))
    else:
        outputs.extend((manifest, csv_path))
    return StageResult(
        outputs=tuple(outputs),
        metadata={"test_count": len(records), "results": records},
        message="Test inference completed" if not context.dry_run else "Dry run: test inference planned",
    )


def _report(context: StageExecutionContext) -> StageResult:
    output = context.run_context.report_dir / "index.html"
    review = context.run_context.report_dir / "manual_review_template.csv"
    if context.dry_run:
        return StageResult(
            outputs=(output, review),
            metadata={"dry_run": True},
            message="Dry run: HTML report generation planned",
        )
    generated = generate_html_report(context.run_context.run_dir)
    return StageResult(
        outputs=(generated, review),
        metadata={"report_path": str(generated)},
        message="Offline HTML listening report generated",
    )


def _input_fingerprint(context: StageExecutionContext) -> str:
    config = context.run_context.config
    training = discover_audio_files(config.paths.training_audio_dir)
    testing = discover_audio_files(config.paths.test_audio_dir)
    payload = {
        "training": [(item.relative_path, item.sha256) for item in training.files],
        "testing": [(item.relative_path, item.sha256) for item in testing.files],
        "maximum_test_files": config.testing.maximum_test_files,
        "test_manifest_sha256": _file_sha256(
            config.paths.test_audio_dir / "test_manifest.yaml"
        ),
    }
    return stable_stage_fingerprint(context.stage, payload)


def _run_fingerprint(context: StageExecutionContext) -> str:
    config = context.run_context.config
    payload: dict[str, Any] = {
        "config": config.serializable_dict(),
        "input_manifest_sha256": _file_sha256(context.run_context.input_manifest_path),
    }
    try:
        info = context.adapter.inspect_repository()
        payload["rvc_commit"] = info.git_commit
    except (AttributeError, OSError, RuntimeError):
        payload["rvc_commit"] = None
    return stable_stage_fingerprint(context.stage, payload)


def _experiment_dir(context: StageExecutionContext) -> Path:
    repository = getattr(context.adapter, "repository", None)
    if repository is None:
        return context.run_context.rvc_workspace_dir
    return Path(repository).resolve() / "logs" / context.run_context.run_id


def _feature_dir(context: StageExecutionContext) -> Path:
    version = context.run_context.config.model.version
    name = "3_feature768" if version == "v2" else "3_feature256"
    return _experiment_dir(context) / name


def _resolve_pretrained_pair(config: object) -> tuple[Path | None, Path | None]:
    """Resolve the exact G/D pair used by the configured official RVC model."""

    if not config.training.use_pretrained_model:
        return None, None
    family = "pretrained_v2" if config.model.version == "v2" else "pretrained"
    prefix = "f0" if config.model.use_f0 else ""
    names = (
        f"{prefix}G{config.model.sample_rate}.pth",
        f"{prefix}D{config.model.sample_rate}.pth",
    )
    roots = (
        config.paths.rvc_repository / "assets" / family,
        config.paths.rvc_repository / family,
        config.paths.pretrained_models_dir,
    )
    resolved: list[Path] = []
    for name in names:
        match = next((root / name for root in roots if (root / name).is_file()), None)
        if match is None:
            searched = ", ".join(str(root / name) for root in roots)
            raise FileNotFoundError(
                f"Configured pretrained RVC weight is missing ({name}). Searched: {searched}"
            )
        resolved.append(match.resolve())
    return resolved[0], resolved[1]


def _find_fresh_model(
    context: StageExecutionContext,
    result_outputs: Sequence[Path],
    started_at: float,
) -> Path:
    candidates = [Path(path) for path in result_outputs if Path(path).suffix.lower() == ".pth"]
    repository = getattr(context.adapter, "repository", None)
    if repository is not None:
        for directory in (
            Path(repository) / "assets" / "weights",
            Path(repository) / "weights",
            _experiment_dir(context),
        ):
            if directory.is_dir():
                candidates.extend(directory.glob(f"*{context.run_context.run_id}*.pth"))
    validations = [
        validate_artifact(
            candidate,
            minimum_bytes=1_024,
            allowed_suffixes=(".pth",),
            not_before=started_at,
        )
        for candidate in candidates
    ]
    valid = [item for item in validations if item.valid]
    if not valid:
        checked = ", ".join(str(path) for path in candidates) or "no candidate files"
        raise FileNotFoundError(
            "Training exited successfully but no fresh, validated RVC .pth model "
            f"was found. Checked: {checked}. This RVC layout may use a different "
            "export path; inspect its training script and configure an adapter override."
        )
    valid.sort(key=lambda item: item.path.stat().st_mtime)
    return valid[-1].path


def _merge_results(results: Sequence[StageResult], message: str) -> StageResult:
    return StageResult(
        success=all(result.success for result in results),
        outputs=tuple(output for result in results for output in result.outputs),
        metadata={"commands": [result.metadata for result in results]},
        message=message,
    )


def _stable_audio_name(relative_path: str, digest: Any) -> str:
    source = Path(relative_path)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", source.stem).strip(" ._") or "audio"
    short_hash = str(digest)[:10] if digest else hashlib.sha256(relative_path.encode()).hexdigest()[:10]
    return f"{stem}__{short_hash}.wav"


def _stable_original_name(item: AudioFile) -> str:
    source = Path(item.relative_path)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", source.stem).strip(" ._") or "audio"
    return f"{stem}__{item.sha256[:10]}{source.suffix.lower()}"


def _load_test_manifest(test_dir: Path) -> tuple[Mapping[str, Any], ...]:
    path = Path(test_dir) / "test_manifest.yaml"
    if not path.is_file():
        return ()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read test manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tests"), list):
        raise ValueError(f"Test manifest must contain a 'tests' list: {path}")
    entries: list[Mapping[str, Any]] = []
    for position, entry in enumerate(payload["tests"], start=1):
        if not isinstance(entry, Mapping) or not str(entry.get("file", "")).strip():
            raise ValueError(f"Test manifest entry {position} needs a non-empty file")
        entries.append(dict(entry))
    return tuple(entries)


def _select_manifest_tests(
    files: Sequence[AudioFile],
    entries: Sequence[Mapping[str, Any]],
    maximum_files: int,
) -> tuple[AudioFile, ...]:
    if not entries:
        return select_test_audio(files, maximum_files)
    by_path = {item.relative_path.replace("\\", "/").casefold(): item for item in files}
    selected: list[AudioFile] = []
    selected_hashes: set[str] = set()
    for entry in entries[:maximum_files]:
        configured = str(entry["file"]).replace("\\", "/")
        configured_path = Path(configured)
        if configured_path.is_absolute() or ".." in configured_path.parts:
            raise ValueError(f"Unsafe test manifest path: {configured}")
        item = by_path.get(configured.casefold())
        if item is None:
            raise FileNotFoundError(
                f"Test manifest references a missing/unsupported audio file: {configured}"
            )
        if item.sha256 in selected_hashes:
            raise ValueError(f"Test manifest selects duplicate audio content: {configured}")
        selected_hashes.add(item.sha256)
        selected.append(item)
    return tuple(selected)


def _validate_optional_manifest_hashes(
    files: Sequence[AudioFile], entries: Sequence[Mapping[str, Any]]
) -> None:
    """Honor hashes in a general manifest without requiring the fixed-five schema."""

    for item in files:
        entry = _manifest_entry_for(item, entries)
        expected = str(entry.get("sha256", "")).strip().lower()
        if expected and expected != item.sha256:
            raise ValueError(f"Test manifest hash does not match {item.relative_path}")


def _manifest_entry_for(
    item: AudioFile, entries: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    relative = item.relative_path.replace("\\", "/").casefold()
    name = Path(item.relative_path).name.casefold()
    for entry in entries:
        configured = str(entry["file"]).replace("\\", "/")
        if configured.casefold() == relative or Path(configured).name.casefold() == name:
            return entry
    return {}


def _test_parameter_variants(
    testing: object, entry: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    base = {
        "transpose": entry.get("transpose", testing.transpose),
        "index_rate": entry.get("index_rate", testing.index_rate),
        "protect": entry.get("protect", testing.protect),
        "rms_mix_rate": entry.get("rms_mix_rate", testing.rms_mix_rate),
        "f0_method": entry.get("f0_method", testing.f0_method),
        "filter_radius": entry.get("filter_radius", testing.filter_radius),
        "resample_sample_rate": entry.get(
            "resample_sample_rate",
            testing.resample_sample_rate or getattr(testing, "output_sample_rate", 0),
        ),
    }
    sweep = testing.parameter_sweep
    if not sweep.enabled:
        return (base,)
    combinations = tuple(
        itertools.product(sweep.transpose_values, sweep.index_rate_values)
    )
    if len(combinations) > sweep.maximum_combinations_per_file:
        raise ValueError(
            f"Parameter sweep creates {len(combinations)} combinations per file, "
            "exceeding maximum_combinations_per_file="
            f"{sweep.maximum_combinations_per_file}"
        )
    return tuple(
        {**base, "transpose": transpose, "index_rate": index_rate}
        for transpose, index_rate in combinations
    )


def _duration_difference_ratio(
    input_seconds: float | None, output_seconds: float | None
) -> float | None:
    if input_seconds is None or output_seconds is None or input_seconds <= 0:
        return None
    return abs(output_seconds - input_seconds) / input_seconds


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read required manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_commit(adapter: object) -> str | None:
    try:
        return getattr(adapter.inspect_repository(), "git_commit", None)
    except (AttributeError, OSError, RuntimeError):
        return None


def _write_test_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "name",
        "original_path",
        "converted_path",
        "transpose",
        "index_rate",
        "protect",
        "output_lufs",
        "output_peak_dbfs",
        "duration_seconds",
        "status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})
