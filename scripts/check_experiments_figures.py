#!/usr/bin/env python3
"""Maintenance check: every experiment plan with `Status: run` must have its
referenced figure file present under `figures/`.

Parses `experiment_plans/experiment_*.md` for two lines:
    Figure: `figures/<filename>` (or with backticks/asterisks)
    Status: <designed|implemented|run>

Exits 0 if all run experiments have their figure on disk, else 1.

Run manually:    python scripts/check_experiments_figures.py
Used by:         .claude/settings.json Stop hook (runs at end of each session)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANS = ROOT / "experiment_plans"
FIGURES = ROOT / "figures"

STATUS_RE = re.compile(r"^\s*`?Status:\s*`?\s*(designed|implemented|run)\b", re.IGNORECASE | re.MULTILINE)
FIGURE_RE = re.compile(r"\*\*Figure\*\*:\s*(.+?)$", re.MULTILINE)
PATH_RE = re.compile(r"`(figures/[^`]+)`")


def parse_plan(path: Path) -> tuple[str | None, list[str]]:
    text = path.read_text()
    status_match = STATUS_RE.search(text)
    status = status_match.group(1).lower() if status_match else None

    fig_line_match = FIGURE_RE.search(text)
    figures = PATH_RE.findall(fig_line_match.group(1)) if fig_line_match else []
    return status, figures


def main() -> int:
    if not PLANS.is_dir():
        print(f"check: no experiment_plans/ directory at {PLANS}", file=sys.stderr)
        return 1

    plans = sorted(p for p in PLANS.glob("experiment_*.md") if p.name != "experiment_template.md")
    if not plans:
        print("check: no experiment plans found (skipping)")
        return 0

    failures: list[str] = []
    for plan in plans:
        status, figures = parse_plan(plan)
        if status != "run":
            continue
        if not figures:
            failures.append(f"{plan.name}: Status=run but no `figures/...` path found in the Figure line")
            continue
        for fig_rel in figures:
            fig_path = ROOT / fig_rel
            if not fig_path.exists():
                failures.append(f"{plan.name}: Status=run but missing {fig_rel}")

    if failures:
        print("check: experiment-plan / figure mismatch:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nFix by either (a) running the experiment and saving the figure to the listed path, "
            "or (b) downgrading Status to `designed` or `implemented`.",
            file=sys.stderr,
        )
        return 1

    print(f"check: ok ({sum(1 for p in plans if (parse_plan(p)[0] == 'run')) } run, "
          f"{len(plans)} total experiment plans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
