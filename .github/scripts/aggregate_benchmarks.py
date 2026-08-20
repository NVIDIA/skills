#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate per-skill BENCHMARK.md evaluation reports into benchmarks.json.

Walks skills/*/BENCHMARK.md, extracts the evaluation summary, agents, and
per-dimension results (skill-assisted score plus uplift vs. the no-skill
baseline), and writes a single machine-readable benchmarks.json at the repo
root for downstream dashboards and tooling.

Usage:
    python3 .github/scripts/aggregate_benchmarks.py [--repo-root PATH]
    python3 .github/scripts/aggregate_benchmarks.py --check   # fail on drift
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Two report layouts are in circulation and both must parse:
#   v1 — "Evaluation Report" with an Evaluation Summary list, an "Agents Used"
#        list, and a "## Results" table keyed on Dimension.
#   v2 — SkillEvaluator 0.9.x, with an "Evaluation Metadata" list, agents on a
#        single line, and a "## Results at a Glance" table keyed on Measure.
# Each field lists its patterns most-specific first; the first match wins.
SUMMARY_FIELDS = {
    "skill": [re.compile(r"^- Skill: `?([^`\n]+)`?\s*$")],
    "evaluation_date": [re.compile(r"^- Evaluation date: (.+)$")],
    # v1/v2 only. v3 dropped this line along with the NVSkills-Eval name; it
    # stays None there rather than being inferred. See NO_FABRICATION below.
    "profile": [re.compile(r"^- NVSkills-Eval profile: `?([^`\n]+)`?\s*$")],
    "environment": [re.compile(r"^- Environment: `?([^`\n]+)`?\s*$")],
    # v3 provenance. Absent from v1/v2, which leave these None.
    "evaluator_version": [re.compile(r"^- Evaluator version: `?([^`\n]+?)`?\s*$")],
    # The digest line carries a trailing snapshot name in parentheses, so this
    # cannot use the trailing-anchored form the other fields use.
    "dataset_digest": [re.compile(r"^- Dataset digest: `?([^`\s]+)`?")],
    "validation_status": [re.compile(r"^- Validation status: `?([^`\n]+?)`?\s*$")],
    "tasks": [
        re.compile(r"^- Dataset: (\d+) evaluation tasks?"),          # v1
        re.compile(r"^- Tasks: (\d+) evaluation tasks?"),            # v2
    ],
    "attempts_per_task": [re.compile(r"^- Attempts per task: (\d+)")],
    "pass_threshold_pct": [re.compile(r"^- Pass threshold: (\d+(?:\.\d+)?)%")],
    "verdict": [
        # v2 states the verdict in a callout above the report body.
        re.compile(r"^>?\s*[^\w\s]*\s*\*\*Overall verdict: (\w+)"),
        # v1 states it as a summary bullet. Anchored to end-of-line so it
        # cannot match the v2 methodology bullet, which begins
        # "- Overall verdict: PASS only when every configured dimension ..."
        # and would otherwise report PASS for every v2 report.
        re.compile(r"^- Overall verdict: (\w+)\s*$"),
    ],
}
AGENT_LINE = re.compile(r"^- `([^`]+)`\s*$")
# v2 lists agents inline, e.g. "- Agents: Claude Code (`model/id`), Codex (`model/id`)"
AGENTS_INLINE = re.compile(r"^- Agents: (.+)$")
# v1 cell, e.g. "100% (+70%)" or "97%" — score with optional uplift vs. baseline
CELL = re.compile(r"(\d+(?:\.\d+)?)%(?:\s*\(([+-±]?\d+(?:\.\d+)?)%\))?")
# v2 cell, e.g. "45% → 98% (+53 points)" — baseline, skill score, then uplift.
# An unchanged dimension is written "±0 points", so ± must be accepted here
# alongside + and -; see parse_uplift for why dropping it is not harmless.
CELL_V2 = re.compile(
    r"(\d+(?:\.\d+)?)%\s*(?:→|->)\s*(\d+(?:\.\d+)?)%"
    r"(?:\s*\(([+-±]?\d+(?:\.\d+)?)\s*points?\))?"
)
RESULTS_SECTIONS = {"results", "results at a glance"}
# v2 leads its table with an aggregate row; the per-dimension rows below it are
# what v1 reports, so skipping it keeps average_uplift comparable across both.
SUMMARY_ROWS = {"overall"}
INT_FIELDS = {"tasks", "attempts_per_task"}
FLOAT_FIELDS = {"pass_threshold_pct"}

# NO_FABRICATION: v3 mentions "50%" once, in a static glossary bullet
# ("- The 50% attempt pass threshold is a separate per-task gate; ..."). That
# sentence is byte-identical across skills with 4, 5 and 18 tasks, so it is
# template prose, not a per-skill measurement. Scraping it would stamp 50.0 on
# every v3 skill regardless of what it was evaluated against — a fabricated
# provenance claim in the file whose purpose is recording provenance. Leave
# pass_threshold_pct None on v3 until SkillEvaluator emits it as a real field.


def parse_uplift(raw):
    """Convert an uplift cell capture to a float, or None when absent.

    Reports write an unchanged dimension as "±0 points". ± carries no sign,
    so strip it and keep the magnitude — the only value emitted in practice
    is ±0, which must land as 0.0 rather than None: average_uplift() skips
    None entries, so a dropped ±0 leaves the row out of the denominator and
    inflates the average for any skill that scored the same with and without
    the skill applied.
    """
    if raw is None:
        return None
    return float(raw.replace("±", ""))


def normalize_agent(name: str) -> str:
    """v1 writes `claude-code`; v2 writes "Claude Code (`model/id`)"."""
    name = re.sub(r"\s*\(.*?\)\s*", "", name).strip().strip("`")
    name = re.sub(r"\s*\(Baseline\s*(?:→|->)\s*Skill Uplift\)\s*", "", name).strip()
    return name.lower().replace(" ", "-")


def parse_benchmark(path: Path) -> dict:
    entry = {k: None for k in SUMMARY_FIELDS}
    entry["agents"] = []
    entry["results"] = []
    section = None
    agent_cols = {}  # results-table column index -> agent name, from the header row

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            section = line.lstrip("# ").lower()
            continue

        for key, patterns in SUMMARY_FIELDS.items():
            if entry[key] is not None:
                continue
            for rx in patterns:
                m = rx.match(line)
                if not m:
                    continue
                val = m.group(1).strip()
                if key in INT_FIELDS:
                    val = int(val)
                elif key in FLOAT_FIELDS:
                    val = float(val)
                entry[key] = val
                break

        if section == "agents used":
            m = AGENT_LINE.match(line)
            if m:
                entry["agents"].append(normalize_agent(m.group(1)))
        m = AGENTS_INLINE.match(line)
        if m and not entry["agents"]:
            # Each agent reads "Name (`provider/model`)"; keep the name only.
            names = re.findall(r"([^,(]+?)\s*\(`[^`]*`\)", m.group(1))
            if not names:
                names = m.group(1).split(",")
            entry["agents"] = [normalize_agent(a) for a in names if a.strip()]

        if section in RESULTS_SECTIONS and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells or set(cells[0]) <= {"-", ":", " "}:
                continue
            if cells[0] in ("Dimension", "Measure"):
                # Key agent columns off the header row rather than assuming
                # they match Agents Used order (columns may vary per report).
                agent_cols = {
                    i: normalize_agent(c)
                    for i, c in enumerate(cells)
                    if i > 0 and c.strip("`") and c != "Num"
                }
                continue
            dimension = cells[0]
            if dimension.lower() in SUMMARY_ROWS:
                continue
            num = None
            scores = {}
            for i, cell in enumerate(cells[1:], start=1):
                if i not in agent_cols:
                    if num is None and cell.isdigit():
                        num = int(cell)
                    continue
                m2 = CELL_V2.search(cell)
                if m2:
                    # Record the skill-assisted score, matching v1 semantics.
                    scores[agent_cols[i]] = {
                        "score_pct": float(m2.group(2)),
                        "uplift_pct": parse_uplift(m2.group(3)),
                    }
                    continue
                m = CELL.search(cell)
                if not m:
                    continue
                scores[agent_cols[i]] = {
                    "score_pct": float(m.group(1)),
                    "uplift_pct": parse_uplift(m.group(2)),
                }
            if scores:
                entry["results"].append({
                    "dimension": dimension,
                    "num": num,
                    "agents": scores,
                })
    return entry


def null_rate_regressions(old: dict, new: dict) -> dict:
    """Fields that lost values between two benchmarks.json generations.

    Returns {field: (old_null_count, new_null_count)} for every field whose
    null count rose, counted only over skills present in BOTH files so that
    newly added skills cannot register as a regression.

    This is the generic guard against silent degradation: a regeneration can
    succeed, keep a valid schema, pass --check, and still quietly empty a
    column when an upstream report format changes. Both the v3 profile /
    pass_threshold drift and the 2026-08-03 disappearance of
    cuopt-multi-objective-exploration are that shape.
    """
    old_by_skill = {s["skill"]: s for s in old.get("skills", [])}
    new_by_skill = {s["skill"]: s for s in new.get("skills", [])}
    common = old_by_skill.keys() & new_by_skill.keys()

    fields = {
        k
        for skill in common
        for k in (*old_by_skill[skill], *new_by_skill[skill])
        if k != "skill"
    }

    regressions = {}
    for field in sorted(fields):
        was = sum(1 for s in common if old_by_skill[s].get(field) is None)
        now = sum(1 for s in common if new_by_skill[s].get(field) is None)
        if now > was:
            regressions[field] = (was, now)
    return regressions


def average_uplift(results: list):
    uplifts = [
        s["uplift_pct"]
        for r in results
        for s in r["agents"].values()
        if s["uplift_pct"] is not None
    ]
    return round(sum(uplifts) / len(uplifts), 2) if uplifts else None


def load_component_map(repo_root: Path) -> dict:
    """Map catalog skill dir -> component (product) name.

    Primary source is components.d/ registrations. Skills that intentionally
    exist without a registration (catalog-exceptions.yml) may declare a
    display component there so downstream consumers can still group them.
    """
    mapping = {}
    for yml in sorted((repo_root / "components.d").glob("*.yml")):
        name = None
        for line in yml.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^name:\s*(.+)$", line.strip())
            if m and name is None:
                name = m.group(1).strip()
            m = re.match(r"^-?\s*catalog_dir:\s*(.+)$", line.strip())
            if m:
                mapping[m.group(1).strip()] = name
    manual = repo_root / ".github" / "scripts" / "manual-components.yml"
    if manual.exists():
        name = None
        for line in manual.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^-?\s*name:\s*(.+)$", line.strip())
            if m:
                name = m.group(1).strip()
                continue
            m = re.match(r"^-\s*(\S+)\s*$", line.strip())
            if m and name and m.group(1) not in mapping:
                mapping[m.group(1)] = name
    exceptions = repo_root / "catalog-exceptions.yml"
    if exceptions.exists():
        current_dir = None
        for line in exceptions.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^-?\s*dir:\s*(.+)$", line.strip())
            if m:
                current_dir = m.group(1).strip()
                continue
            m = re.match(r"^component:\s*(.+)$", line.strip())
            if m and current_dir and current_dir not in mapping:
                mapping[current_dir] = m.group(1).strip()
    return mapping


def generate(root: Path) -> str:
    components = load_component_map(root)
    skills = []
    rows = []
    skipped = []
    for bm in sorted(root.glob("skills/*/BENCHMARK.md")):
        catalog_dir = bm.parent.name
        entry = parse_benchmark(bm)
        results = entry.pop("results")
        if not results:
            skipped.append(catalog_dir)
        component = components.get(catalog_dir)

        # Flat measurement rows: one per skill x dimension x agent, so BI
        # tools can load them as a table and join on catalog_dir.
        for r in results:
            for agent, s in r["agents"].items():
                rows.append({
                    "catalog_dir": catalog_dir,
                    "component": component,
                    "dimension": r["dimension"],
                    "num": r["num"],
                    "agent": agent,
                    "score_pct": s["score_pct"],
                    "uplift_pct": s["uplift_pct"],
                })

        entry["catalog_dir"] = catalog_dir
        entry["component"] = component
        entry["has_results"] = bool(results)
        entry["average_uplift_pct"] = average_uplift(results)
        skills.append(entry)

    out = {
        "schema_version": 2,
        "source": "skills/*/BENCHMARK.md",
        "skill_count": len(skills),
        "result_row_count": len(rows),
        "skills_without_results": sorted(skipped),
        "skills": skills,
        "results": rows,
    }
    return json.dumps(out, indent=2) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Fail (exit 1) if the checked-in benchmarks.json does not match "
        "what the BENCHMARK.md sources would generate.",
    )
    ap.add_argument(
        "--allow-null-regressions",
        action="store_true",
        help="Write even when a field lost values against the existing "
        "benchmarks.json. Use when an upstream report format change has "
        "genuinely retired a field, so the new state can be landed.",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()
    target = root / "benchmarks.json"

    payload = generate(root)
    count = json.loads(payload)["skill_count"]

    if args.check:
        existing = target.read_text(encoding="utf-8") if target.exists() else None
        if existing != payload:
            print(
                "benchmarks.json is out of date with skills/*/BENCHMARK.md.\n"
                "Regenerate it with: python3 .github/scripts/aggregate_benchmarks.py",
                file=sys.stderr,
            )
            return 1
        print(f"benchmarks.json is up to date ({count} skills)")
        return 0

    if target.exists() and not args.allow_null_regressions:
        previous = json.loads(target.read_text(encoding="utf-8"))
        lost = null_rate_regressions(previous, json.loads(payload))
        if lost:
            print(
                "Refusing to write benchmarks.json: fields lost values.\n"
                "Skills carried these before and would not after:",
                file=sys.stderr,
            )
            for field, (was, now) in lost.items():
                print(f"  {field}: {was} -> {now} nulls", file=sys.stderr)
            print(
                "\nThis usually means an upstream BENCHMARK.md format change "
                "stopped emitting a field the parser reads.\n"
                "Either teach the parser the new format, or re-run with "
                "--allow-null-regressions if the field is genuinely retired.",
                file=sys.stderr,
            )
            return 1

    target.write_text(payload, encoding="utf-8")
    print(f"Wrote benchmarks.json: {count} skills")
    return 0


if __name__ == "__main__":
    sys.exit(main())
