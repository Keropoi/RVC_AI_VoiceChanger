"""Focused tests for offline environment report serialization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rvc_auto_trainer.environment import (
    CheckStatus,
    EnvironmentCheck,
    EnvironmentReport,
    write_environment_report,
)


def test_environment_report_writes_json_and_text_with_expected_paths(tmp_path: Path) -> None:
    """Doctor results remain machine-readable and actionable offline."""
    expected = tmp_path / "模型" / "rmvpe.pt"
    report = EnvironmentReport(
        generated_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
        project_root=tmp_path,
        checks={
            "rmvpe": EnvironmentCheck(
                name="RMVPE model",
                status=CheckStatus.FAIL,
                message="Required model asset was not found; no download was attempted",
                required=True,
                expected_paths=(expected,),
            )
        },
        facts={"python_version": "3.10.0"},
        healthy=False,
    )

    json_path, text_path = write_environment_report(report, tmp_path / "run")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["checks"]["rmvpe"]["status"] == "FAIL"
    assert str(expected) in text_path.read_text(encoding="utf-8")
