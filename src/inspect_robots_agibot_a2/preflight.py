"""Hardware-free compatibility preflight for the A2 and GO-1 pair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from inspect_robots.compat import CompatibilityReport, check_compatibility
from inspect_robots.registry import resolve
from inspect_robots.task import Task

from inspect_robots_agibot_a2.config import A2Config, Go1Config
from inspect_robots_agibot_a2.embodiment import A2Embodiment
from inspect_robots_agibot_a2.policy import Go1Policy

CheckFn = Callable[..., CompatibilityReport]


def build(
    a2_cfg: A2Config | None = None,
    go1_cfg: Go1Config | None = None,
) -> tuple[Go1Policy, A2Embodiment]:
    """Construct the default policy and embodiment without I/O."""
    return Go1Policy(go1_cfg), A2Embodiment(a2_cfg)


def run_preflight(
    task_name: str | None = None,
    *,
    policy: Go1Policy | None = None,
    embodiment: A2Embodiment | None = None,
    check: CheckFn = check_compatibility,
) -> CompatibilityReport:
    """Return compatibility findings, optionally including a task."""
    pol = policy if policy is not None else Go1Policy()
    emb = embodiment if embodiment is not None else A2Embodiment()
    task: Task | None = resolve("task", task_name) if task_name else None
    return check(pol, emb, task)


def _format_human(report: CompatibilityReport, *, dry_run: bool) -> str:
    lines = ["OK: policy and embodiment are compatible." if report.ok else "INCOMPATIBLE:"]
    for issue in report.errors:
        lines.append(f"  ERROR   [{issue.code}] {issue.message}")
    for issue in report.warnings:
        lines.append(f"  WARNING [{issue.code}] {issue.message}")
    if dry_run:
        lines.append("(dry-run) No motion will be commanded.")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, run: CheckFn | None = None) -> int:
    """Print a compatibility report and return nonzero only for errors."""
    parser = argparse.ArgumentParser(prog="inspect-robots-agibot-a2-preflight")
    parser.add_argument(
        "--task", default=None, help="optional task name to check scene realizability"
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--dry-run", action="store_true", help="affirm no motion is commanded")
    args = parser.parse_args(argv)
    run_fn: Callable[..., CompatibilityReport] = run if run is not None else run_preflight
    report = run_fn(args.task)
    if args.json:
        payload = {
            "ok": report.ok,
            "dry_run": args.dry_run,
            "errors": [{"code": item.code, "message": item.message} for item in report.errors],
            "warnings": [{"code": item.code, "message": item.message} for item in report.warnings],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_human(report, dry_run=args.dry_run))
    return 1 if report.errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
