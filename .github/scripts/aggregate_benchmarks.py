#!/usr/bin/env python3
"""Aggregate per-skill BENCHMARK.md evaluation reports into benchmarks.json.

Walks skills/*/BENCHMARK.md, extracts the evaluation summary, agents, and
per-dimension results (skill-assisted score plus uplift vs. the no-skill
baseline), and writes a single machine-readable benchmarks.json at the repo
root for downstream dashboards and tooling.

Usage: python3 .github/scripts/aggregate_benchmarks.py [--repo-root PATH]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SUMMARY_FIELDS = {
    "skill": re.compile(r"^- Skill: `?([^`\n]+)`?\s*$"),
    "evaluation_date": re.compile(r"^- Evaluation date: (.+)$"),
    "profile": re.compile(r"^- NVSkills-Eval profile: `?([^`\n]+)`?\s*$"),
    "environment": re.compile(r"^- Environment: `?([^`\n]+)`?\s*$"),
    "tasks": re.compile(r"^- Dataset: (\d+) evaluation tasks?"),
    "attempts_per_task": re.compile(r"^- Attempts per task: (\d+)"),
    "pass_threshold_pct": re.compile(r"^- Pass threshold: (\d+(?:\.\d+)?)%"),
    "verdict": re.compile(r"^- Overall verdict: (\w+)"),
}
AGENT_LINE = re.compile(r"^- `([^`]+)`\s*$")
# e.g. "100% (+70%)" or "97%" — score with optional uplift vs. baseline
CELL = re.compile(r"(\d+(?:\.\d+)?)%(?:\s*\(([+-]?\d+(?:\.\d+)?)%\))?")
INT_FIELDS = {"tasks", "attempts_per_task"}
FLOAT_FIELDS = {"pass_threshold_pct"}


def parse_benchmark(path: Path) -> dict:
    entry = {k: None for k in SUMMARY_FIELDS}
    entry["agents"] = []
    entry["results"] = []
    section = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            section = line.lstrip("# ").lower()
            continue

        for key, rx in SUMMARY_FIELDS.items():
            m = rx.match(line)
            if m and entry[key] is None:
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
                entry["agents"].append(m.group(1))

        if section == "results" and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells or cells[0] in ("Dimension", "") or set(cells[0]) <= {"-", ":", " "}:
                continue
            dimension, num = cells[0], cells[1] if len(cells) > 1 else None
            scores = {}
            for agent, cell in zip(entry["agents"], cells[2:]):
                m = CELL.search(cell)
                if not m:
                    continue
                scores[agent] = {
                    "score_pct": float(m.group(1)),
                    "uplift_pct": float(m.group(2)) if m.group(2) is not None else None,
                }
            if scores:
                entry["results"].append({
                    "dimension": dimension,
                    "num": int(num) if num and num.isdigit() else None,
                    "agents": scores,
                })
    return entry


def average_uplift(results: list):
    uplifts = [
        s["uplift_pct"]
        for r in results
        for s in r["agents"].values()
        if s["uplift_pct"] is not None
    ]
    return round(sum(uplifts) / len(uplifts), 2) if uplifts else None


def load_component_map(repo_root: Path) -> dict:
    """Map catalog skill dir -> component (product) name from components.d/."""
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
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", type=Path)
    args = ap.parse_args()
    root = args.repo_root.resolve()

    components = load_component_map(root)
    skills = []
    skipped = []
    for bm in sorted(root.glob("skills/*/BENCHMARK.md")):
        catalog_dir = bm.parent.name
        entry = parse_benchmark(bm)
        if not entry["results"]:
            skipped.append(catalog_dir)
        entry["catalog_dir"] = catalog_dir
        entry["component"] = components.get(catalog_dir)
        entry["average_uplift_pct"] = average_uplift(entry["results"])
        skills.append(entry)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip() or None

    out = {
        "schema_version": 1,
        "source": "skills/*/BENCHMARK.md",
        "commit": commit,
        "skill_count": len(skills),
        "skills_without_results": sorted(skipped),
        "skills": skills,
    }
    (root / "benchmarks.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote benchmarks.json: {len(skills)} skills, {len(skipped)} without parseable results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
