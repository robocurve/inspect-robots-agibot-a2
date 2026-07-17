from __future__ import annotations

import json

import pytest
from inspect_robots.compat import CompatibilityReport, CompatIssue

from inspect_robots_agibot_a2 import preflight


def _report(severity: str | None = None) -> CompatibilityReport:
    issues = [] if severity is None else [CompatIssue(severity, "code", "detail")]
    return CompatibilityReport(issues=issues)


def test_build_and_default_preflight_are_inert_and_compatible() -> None:
    policy, embodiment = preflight.build()
    assert policy.info.name == "go1"
    assert embodiment.info.name == "a2_arms"
    report = preflight.run_preflight()
    assert report.errors == report.warnings == []


def test_preflight_resolves_task_and_uses_injected_components() -> None:
    policy, embodiment = preflight.build()
    assert preflight.run_preflight("cubepick-reach", policy=policy, embodiment=embodiment).ok
    sentinel = _report("warning")
    assert preflight.run_preflight(check=lambda *_args, **_kwargs: sentinel) is sentinel


@pytest.mark.parametrize(
    ("report", "args", "code", "text"),
    [
        (_report(), [], 0, "OK:"),
        (_report("warning"), [], 0, "WARNING"),
        (_report("error"), [], 1, "INCOMPATIBLE"),
        (_report(), ["--dry-run"], 0, "dry-run"),
    ],
)
def test_human_cli(
    report: CompatibilityReport,
    args: list[str],
    code: int,
    text: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert preflight.main(args, run=lambda *_args, **_kwargs: report) == code
    assert text in capsys.readouterr().out


def test_json_cli_includes_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        preflight.main(["--json", "--dry-run"], run=lambda *_args, **_kwargs: _report("error")) == 1
    )
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "dry_run": True,
        "errors": [{"code": "code", "message": "detail"}],
        "warnings": [],
    }


def test_main_default_run_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert preflight.main([]) == 0
    assert "OK:" in capsys.readouterr().out
