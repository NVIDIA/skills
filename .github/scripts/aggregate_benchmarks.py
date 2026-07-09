#!/usr/bin/env python3
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
    agent_cols = {}  # results-table column index -> agent name, from the header row

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
            if not cells or set(cells[0]) <= {"-", ":", " "}:
                continue
            if cells[0] == "Dimension":
                # Key agent columns off the header row rather than assuming
                # they match Agents Used order (columns may vary per report).
                agent_cols = {
                    i: c.strip("`")
                    for i, c in enumerate(cells)
                    if i > 0 and c.strip("`") and c != "Num"
                }
                continue
            dimension = cells[0]
            num = None
            scores = {}
            for i, cell in enumerate(cells[1:], start=1):
                if i not in agent_cols:
                    if num is None and cell.isdigit():
                        num = int(cell)
                    continue
                m = CELL.search(cell)
                if not m:
                    continue
                scores[agent_cols[i]] = {
                    "score_pct": float(m.group(1)),
                    "uplift_pct": float(m.group(2)) if m.group(2) is not None else None,
                }
            if scores:
                entry["results"].append({
                    "dimension": dimension,
                    "num": num,
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

    target.write_text(payload, encoding="utf-8")
    print(f"Wrote benchmarks.json: {count} skills")
    return 0


if __name__ == "__main__":
    sys.exit(main())
