#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nb_compare: compare a run folder with a baseline run folder, and write the report.

A baseline is used only when its manifest matches: tool versions and image digests, mode,
lengths, loads, repeats, seeds, test sets, items hash, temperature and token limit. A mismatch
is refused with the names of the fields that differ. Speed is compared per load with the
nim-tune rules (three repeats, under 2% is noise, overlapping repeat ranges are noise).
Accuracy uses the paired difference on the same items and the three-way verdict
pass, fail or inconclusive. The report holds numbers only: no prompts, no answers.
Run it to print the key metrics: python3 scripts/nb_compare.py --metrics.
Exit codes: 0 ok, 2 bad input. Never prompts. Stdlib only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_accuracy
import nb_common
import nb_perf
import nb_stats
from nb_common import ContractParser, emit, error

# the speed numbers that decide a load's verdict
KEY_METRICS = ("ttft_p99", "itl_p99", "otps_avg")
LOWER_IS_BETTER = {"ttft": True, "itl": True, "e2e": True, "otps": False, "rps": False}
NO_VERDICT = "no verdict"
MODES = ("perf", "accuracy")


def manifest_diff(a: dict[str, Any], b: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    """The names of the fields that differ between two manifest sections."""
    return [f for f in fields if a.get(f) != b.get(f)]


def lower_is_better(metric_key: str) -> bool:
    return LOWER_IS_BETTER[metric_key.split("_")[0]]


def compare_perf(
    cand_manifest: dict[str, Any], base_manifest: dict[str, Any], cand_dir: Path, base_dir: Path
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Speed verdicts per load, or a refusal when the setups differ."""
    diff = manifest_diff(cand_manifest["perf"], base_manifest["perf"], nb_perf.MATCH_FIELDS)
    if diff:
        return None, [mismatch("perf", diff)]
    cand = nb_common.read_json(cand_dir / "perf" / "summary.json")["loads"]
    base = nb_common.read_json(base_dir / "perf" / "summary.json")["loads"]
    loads: dict[str, Any] = {}
    for load in sorted(set(cand) & set(base), key=int):
        metrics = {
            key: nb_stats.speed_verdict(
                base[load]["metrics"][key]["values"],
                cand[load]["metrics"][key]["values"],
                lower_is_better(key),
            )
            for key in cand[load]["metrics"]
            if key in base[load]["metrics"]
        }
        verdicts = [metrics[k]["verdict"] for k in KEY_METRICS if k in metrics]
        if "worse" in verdicts:
            overall = "worse"
        elif "better" in verdicts:
            overall = "better"
        else:
            overall = "same" if all(v == "same" for v in verdicts) else "within_noise"
        loads[load] = {"verdict": overall, "metrics": metrics}
    return {"loads": loads}, []


def current_scores(folder: Path, manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    """A test set's scores from the folder's last accuracy plan, or None.

    A set that failed (`answers_truncated`, `tool_failed`) writes no scores.json, and a file
    left by an earlier plan in the same folder is not this run's score.
    """
    try:
        summary = nb_common.read_json(folder / "accuracy" / "summary.json")
        if not isinstance(summary, dict) or not isinstance(summary.get("sets"), dict):
            return None
        entry = summary["sets"].get(name)
        if not isinstance(entry, dict):
            return None
        scores = nb_common.read_json(folder / "accuracy" / name / "scores.json")
    except (OSError, ValueError):
        return None
    if not isinstance(scores, dict):
        return None
    items_hash = scores.get("items_hash")
    rewards = scores.get("rewards")
    if not isinstance(items_hash, str) or not items_hash.strip():
        return None
    if not isinstance(rewards, list) or not rewards:
        return None
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in rewards):
        return None
    if entry.get("plan_id") and entry["plan_id"] != scores.get("plan_id"):
        return None
    plans = manifest.get("plan_ids", {})
    if not isinstance(plans, dict):
        return None
    plan = plans.get("accuracy")
    if plan and scores.get("plan_id") != plan:
        return None
    return scores


def compare_accuracy(
    cand_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    cand_dir: Path,
    base_dir: Path,
    margin: float,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Accuracy verdicts per test set, paired on the same items."""
    diff = manifest_diff(
        cand_manifest["accuracy"], base_manifest["accuracy"], nb_accuracy.MATCH_FIELDS
    )
    if diff:
        return None, [mismatch("accuracy", diff)]
    sets: dict[str, Any] = {}
    problems: list[dict[str, str]] = []
    for name in cand_manifest["accuracy"]["sets"]:
        cand = current_scores(cand_dir, cand_manifest, name)
        base = current_scores(base_dir, base_manifest, name)
        if cand is None or base is None:
            sides = [side for side, found in (("run", cand), ("baseline", base)) if found is None]
            sets[name] = {"verdict": NO_VERDICT, "why": f"no scores in the {' and '.join(sides)}"}
            continue
        if cand["items_hash"] != base["items_hash"]:
            problems.append(mismatch("accuracy", [f"items_hash of {name}"]))
            continue
        paired = nb_stats.paired_difference(base["rewards"], cand["rewards"])
        verdict = nb_stats.accuracy_verdict(
            paired["low"], paired["high"], margin, items=paired["items"]
        )
        sets[name] = {**paired, "margin": margin, "verdict": verdict}
        if paired["items"] < nb_stats.MIN_VERDICT_ITEMS:
            sets[name]["why"] = f"under {nb_stats.MIN_VERDICT_ITEMS} items"
    return {"sets": sets}, problems


def mismatch(mode: str, fields: list[str]) -> dict[str, str]:
    return error(
        "baseline_mismatch",
        f"baseline differs in: {', '.join(fields)} ({mode})",
        "Run the baseline again with this setup, or report without a verdict",
    )


def compare_dirs(
    cand_dir: Path, base_dir: Path, margin: float, modes: tuple[str, ...] = MODES
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Compare each of `modes` both folders have. Returns the verdicts and the refusals."""
    try:
        cand_m = nb_common.read_json(cand_dir / "manifest.json")
        base_m = nb_common.read_json(base_dir / "manifest.json")
    except (OSError, ValueError) as exc:
        return {}, [error("bad_input", f"cannot read a manifest: {exc}", "Give a run folder")]
    verdicts: dict[str, Any] = {}
    problems: list[dict[str, str]] = []
    if "perf" in modes and "perf" in cand_m and "perf" in base_m:
        result, errors = compare_perf(cand_m, base_m, cand_dir, base_dir)
        problems += errors
        if result:
            verdicts["perf"] = result
    if "accuracy" in modes and "accuracy" in cand_m and "accuracy" in base_m:
        result, errors = compare_accuracy(cand_m, base_m, cand_dir, base_dir, margin)
        problems += errors
        if result:
            verdicts["accuracy"] = result
    if not verdicts and not problems:
        problems.append(
            error(
                "bad_input",
                "the two folders share no mode",
                "Compare folders that ran the same mode",
            )
        )
    return verdicts, problems


def report_md(manifest: dict[str, Any], out: Path, verdicts: dict[str, Any]) -> str:
    """The report: numbers only. It names no prompt and no answer."""
    lines = [f"# nim-benchmark report: {manifest.get('model', '?')}", ""]
    lines.append(
        f"Endpoint: {manifest.get('endpoint', '?')}   Plan: {manifest.get('plan_ids', {})}"
    )
    if (out / "perf" / "summary.json").exists():
        summary = nb_common.read_json(out / "perf" / "summary.json")
        lines += [
            "",
            "## Speed",
            "",
            "| load | TTFT avg | TTFT p99 | ITL avg | ITL p99 | tokens/s | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for load, entry in sorted(summary["loads"].items(), key=lambda kv: int(kv[0])):
            m = entry["metrics"]
            verdict = (
                verdicts.get("perf", {})
                .get("loads", {})
                .get(load, {})
                .get("verdict", "no baseline")
            )
            lines.append(
                f"| {load} | {m['ttft_avg']['mean']:.1f} | {m['ttft_p99']['mean']:.1f} | "
                f"{m['itl_avg']['mean']:.2f} | {m['itl_p99']['mean']:.2f} | "
                f"{m['otps_avg']['mean']:.1f} | {verdict} |"
            )
        for item in summary.get("incomplete", []):
            lines.append(
                f"\nLoad {item['load']} is incomplete: {item['counted']} of "
                f"{item['needed']} repeats counted. It has no verdict."
            )
        lines.append(f"\nTimes are in ms. Cache salt: {summary.get('cache_salt', '?')}.")
    if (out / "accuracy" / "summary.json").exists():
        summary = nb_common.read_json(out / "accuracy" / "summary.json")
        lines += [
            "",
            "## Accuracy",
            "",
            "| test set | items | score % | 95% range | verdict |",
            "|---|---|---|---|---|",
        ]
        compared = verdicts.get("accuracy", {}).get("sets", {})
        for name, entry in summary["sets"].items():
            found = compared.get(name)
            verdict = "no baseline" if not found else found["verdict"]
            if found and found["verdict"] == "inconclusive":
                verdict = "inconclusive (not proven)"
            if found and "why" in found:
                verdict += f": {found['why']}"
            lines.append(
                f"| {name} | {int(entry['items'])} | {entry['score']} | "
                f"{entry['low']} to {entry['high']} | {verdict} |"
            )
        for name in compared:
            if name not in summary["sets"]:
                lines.append(f"\nTest set {name} has no score in this run. It has no verdict.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = ContractParser(
        description="Comparison helpers for nim-benchmark. Run it to print the key metrics.",
        epilog="Example: python3 scripts/nb_compare.py --metrics",
    )
    parser.add_argument(
        "--metrics", action="store_true", help="print the metrics that decide a speed verdict"
    )
    parser.parse_args(argv)
    emit(True, "metrics", "done", {"key_metrics": list(KEY_METRICS)})


if __name__ == "__main__":
    main()
