#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""compare_aiperf: compare NIM candidates with the baseline across AIPerf repeats.

Reads ROOT/CASE/cCONC-rREPEAT/profile_export_aiperf.csv, or that run's AIPerf console log
(aiperf.log) when the CSV is missing, empty or header-only. For each case and concurrency it
reports the mean, min and max across repeats of TTFT, ITL and request latency (average and p99)
and output and request throughput. For each candidate it compares every metric with the baseline at the same
concurrency: `better` or `worse` only when the candidate's min-max range does not overlap the
baseline's, otherwise `within_noise`. It then picks, per load, the best case for each metric and,
with --sla, the case with the most output tokens/s per GPU whose worst repeat meets the target.
With --sla it also reports each case's highest measured load that meets the target; a load
passes only when every counted repeat measured the target metric, and throughput needs every
counted repeat too; any other metric measured in fewer than --min-repeats repeats is left out
(listed in facts.partly_measured). ITL p99 settles a throughput tie only when every tied case
measured it in every counted repeat; otherwise the recommendation stays provisional. At an
SLA-search load (one not in --loads, so only with --loads and --sla) a repeat with failed requests
is that load's measured miss, as the runner stops there (another repeat there without an error
rate is listed, never a gap); otherwise every load needs --min-repeats failed runs (run_campaign
passes workload.repeats) to be a measured overload. A run with any request error,
or a missing or unreadable error rate, is rejected, and cases whose manifest.json records a
different measurement setup (workload, AIPerf version, node or client) are refused. When the
comparison is incomplete (too few repeats, a missing load, a run without an error rate), the
recommendation carries `provisional` true and its reasons, and is never final. --report writes
the before-and-after table as Markdown.

Prints one JSON object: {"ok", "state", "next", "facts", "errors": [{"code", "message", "fix"}]}.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool failed, 5 refused.
Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, NoReturn

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL, EXIT_REFUSED = 0, 2, 3, 4, 5
CSV_NAME = "profile_export_aiperf.csv"
LOG_NAME = "aiperf.log"
RUN_DIR = re.compile(r"^c(\d+)-r(\d+)$")
# AIPerf metric row -> (key, lower is better)
METRICS = {
    "Time to First Token (ms)": ("ttft", True),
    "Inter Token Latency (ms)": ("itl", True),
    "Request Latency (ms)": ("e2e", True),
    "Output Token Throughput (tokens/sec)": ("otps", False),
    "Request Throughput (requests/sec)": ("rps", False),
}
ERROR_ROW = "Request Error Rate (%)"
STATS = ("avg", "p99")
# manifest.json keys that must match across cases for a like-for-like comparison
MEASUREMENT_KEYS = ("workload", "aiperf_version", "node", "client")
LATENCY = ("ttft_avg", "ttft_p99", "itl_avg", "itl_p99")
SLA = re.compile(r"^(ttft|itl|e2e)_(avg|p99)<=(\d+(?:\.\d+)?)$")


def required_metrics(sla_metric: str | None) -> tuple[str, ...]:
    """The metrics every counted repeat must measure: throughput, which ranks every case, and
    with a target, its metric. run_campaign counts a run only with all of them, and the
    comparison is complete only when each counted repeat has them, so the two never disagree.
    Any other metric only describes a case; ITL p99 settles a throughput tie only when every
    tied case measured it in every counted repeat (see rank)."""
    return ("otps_avg",) if sla_metric is None else ("otps_avg", sla_metric)


class ContractParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                {
                    "code": "bad_usage",
                    "message": message,
                    "fix": "Run with --help and fix the arguments.",
                }
            ],
            code=EXIT_INPUT,
        )


def emit(
    ok: bool,
    state: str,
    next_: str,
    facts: dict[str, Any] | None = None,
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    payload = {
        "ok": ok,
        "state": state,
        "next": next_,
        "facts": facts or {},
        "errors": errors or [],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


def error_rate(cell: str) -> float:
    """The request error rate of a present error row; NaN when the cell is not a number, so a
    malformed row is never read as zero errors."""
    try:
        return float(cell)
    except ValueError:
        return float("nan")


def read_csv(path: Path) -> dict[str, float]:
    """Return {'ttft_avg': .., 'ttft_p99': .., ...} from one AIPerf CSV export."""
    out: dict[str, float] = {}
    header: list[str] | None = None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] in ("Metric", "Endpoint"):
                header = row
                continue
            if row and header is not None and row[0] == ERROR_ROW:
                out["error_rate"] = error_rate(row[1] if len(row) > 1 else "")
                continue
            if not row or header is None or row[0] not in METRICS:
                continue
            key = METRICS[row[0]][0]
            cells = dict(zip(header, row, strict=False))
            if header[:2] == ["Metric", "Value"]:
                # a truncated row has no value: the metric is unmeasured in this run, not a crash
                cells = {"avg": row[1] if len(row) > 1 else ""}
            for stat in STATS:
                with contextlib.suppress(ValueError):
                    out[f"{key}_{stat}"] = float(cells.get(stat) or "nan")
    return out


def read_log(path: Path) -> dict[str, float]:
    """Return the same keys from the final 'LLM Metrics' table of an AIPerf console log."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "NVIDIA AIPerf | LLM Metrics"]
    out: dict[str, float] = {}
    if not starts:
        return out
    for line in lines[starts[-1] :]:
        if line.startswith("\u2514"):
            break
        cells = [c.strip() for c in line.split("\u2502")[1:-1]]
        if len(cells) == 8 and cells[0] == ERROR_ROW:
            out["error_rate"] = error_rate(cells[1].replace(",", ""))
        # console columns: Metric, avg, min, max, p99, p90, p50, std
        if len(cells) == 8 and cells[0] in METRICS:
            key = METRICS[cells[0]][0]
            for stat, cell in (("avg", cells[1]), ("p99", cells[4])):
                with contextlib.suppress(ValueError):
                    out[f"{key}_{stat}"] = float(cell.replace(",", ""))
    return out


def read_run(run_dir: Path) -> dict[str, float] | None:
    """The values of one run, by the one rule run_campaign counts runs with too: the CSV export
    when it holds anything, else the console log (CSV missing, empty or header-only); None when
    the run has neither file. Raises OSError, UnicodeDecodeError or csv.Error when unreadable."""
    export, log = run_dir / CSV_NAME, run_dir / LOG_NAME
    if not export.is_file() and not log.is_file():
        return None
    got = read_csv(export) if export.is_file() else {}
    if not got and log.is_file():
        got = read_log(log)
    return got


def collect(
    root: Path,
    rejected: list[str] | None = None,
    no_error_rate: list[str] | None = None,
    unfinished: list[str] | None = None,
) -> dict[str, dict[int, list[dict[str, float]]]]:
    """Group runs by case and load. Runs with failed requests (a positive error rate) go to
    `rejected`; runs with metrics but a missing or unreadable error rate go to `no_error_rate`,
    because that is missing evidence, not a measured overload; run folders with no export or an
    empty one go to `unfinished`. None of them is counted."""
    runs: dict[str, dict[int, list[dict[str, float]]]] = {}
    for run_dir in sorted(p for p in root.glob("*/c*-r*") if p.is_dir()):
        m = RUN_DIR.match(run_dir.name)
        if not m:
            continue
        values: dict[str, float] | None
        try:
            values = read_run(run_dir)
        except (OSError, UnicodeDecodeError, csv.Error):
            values = None  # an export that cannot be read is an unfinished attempt, not a crash
        if values is None:
            if unfinished is not None:
                unfinished.append(str(run_dir.relative_to(root)))
            continue
        # AIPerf writes the error row on every run, 0.00 when all requests succeeded: a run counts
        # only when it reads exactly 0 (NaN or inf is not clean, and no row is no evidence)
        rate = values.pop("error_rate", None)
        if rate is None or not math.isfinite(rate) or not 0 <= rate <= 100:
            # an export with metrics, or with only an unreadable error row, is missing evidence;
            # an export with nothing in it at all is an unfinished attempt
            if (values or rate is not None) and no_error_rate is not None:
                no_error_rate.append(str(run_dir.relative_to(root)))
            elif not values and rate is None and unfinished is not None:
                unfinished.append(str(run_dir.relative_to(root)))
            continue
        if rate != 0:
            if rejected is not None:
                rejected.append(str(run_dir.relative_to(root)))
            continue
        if values:
            case = run_dir.parent.name
            runs.setdefault(case, {}).setdefault(int(m[1]), []).append(values)
        elif unfinished is not None:
            unfinished.append(str(run_dir.relative_to(root)))  # a clean error row, nothing measured
    return runs


def run_loads(root: Path) -> dict[str, set[int]]:
    """{case: the loads it has run folders for}, counted runs or not."""
    out: dict[str, set[int]] = {}
    for p in root.glob("*/c*-r*"):
        if p.is_dir() and (m := RUN_DIR.match(p.name)):
            out.setdefault(p.parent.name, set()).add(int(m[1]))
    return out


def run_attempts(root: Path) -> dict[tuple[str, int], int]:
    """{(case, load): how many run folders it has}, counted runs or not."""
    out: dict[tuple[str, int], int] = {}
    for p in root.glob("*/c*-r*"):
        if p.is_dir() and (m := RUN_DIR.match(p.name)):
            key = (p.parent.name, int(m[1]))
            out[key] = out.get(key, 0) + 1
    return out


def run_cases(root: Path) -> set[str]:
    """Every case with a run folder (CASE/cCONC-rREPEAT), counted runs or not: a case whose runs
    are all empty or unreadable is missing from collect() but still a gap to report."""
    return {p.parent.name for p in root.glob("*/c*-r*") if p.is_dir() and RUN_DIR.match(p.name)}


def summarize(reps: list[dict[str, float]], gpus: int) -> dict[str, Any]:
    out: dict[str, Any] = {"repeats": len(reps), "gpus": gpus}
    keys = sorted({k for r in reps for k in r})
    for k in keys:
        vals = [r[k] for r in reps if k in r and math.isfinite(r[k])]
        if vals:
            # n: how many repeats really measured this metric (a missing, NaN or inf value is not)
            out[k] = {
                "mean": round(statistics.mean(vals), 3),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    if "otps_avg" in out:
        out["otps_per_gpu"] = round(out["otps_avg"]["mean"] / gpus, 1)
    return out


def change_pct(base: float, cand: float) -> float | None:
    """The change from BASE in percent; None when there is none to state (a zero baseline, or
    finite means whose ratio overflows), so the JSON never carries Infinity or NaN."""
    if not base:
        return None
    pct = round((cand / base - 1) * 100, 1)
    return pct if math.isfinite(pct) else None


MIN_EFFECT = 0.02  # a change smaller than 2% of the baseline is noise, however tight the ranges


def verdict(base: dict[str, float], cand: dict[str, float], lower_better: bool) -> str:
    """better or worse only when the repeat ranges do not overlap and the means differ by 2% or
    more; anything else is within_noise."""
    if base["mean"] and abs(cand["mean"] / base["mean"] - 1) < MIN_EFFECT:
        return "within_noise"
    if cand["min"] > base["max"]:
        return "worse" if lower_better else "better"
    if cand["max"] < base["min"]:
        return "better" if lower_better else "worse"
    return "within_noise"


PLATEAU = 0.03  # throughput within 3% of the best is a tie
LOAD_TIE = 0.10  # highest loads within 10% of the best are a tie (the search's own resolution)


def plateau(stats: dict[str, dict[str, Any]], throughput_key: str) -> list[str]:
    """The cases within 3% of the most tokens/s per GPU: a throughput tie when more than one."""
    if not stats:
        return []
    best = max(float(s.get(throughput_key) or 0.0) for s in stats.values())
    return sorted(
        c for c, s in stats.items() if float(s.get(throughput_key) or 0.0) >= best * (1 - PLATEAU)
    )


def unsettled_tie(stats: dict[str, dict[str, Any]], throughput_key: str) -> list[str]:
    """The tied cases when ITL p99 cannot settle their throughput tie (some tied case lacks it
    in a counted repeat: a partial mean could pick another winner); [] otherwise."""
    tied = plateau(stats, throughput_key)
    if len(tied) > 1 and not all(measured(stats[c], "itl_p99") for c in tied):
        return tied
    return []


def rank(
    stats: dict[str, dict[str, Any]], simplicity: dict[str, int], throughput_key: str
) -> str | None:
    """Most tokens/s per GPU, treating anything within 3% of the best as a tie; ties go to the
    lower ITL p99 (only when every tied case measured it in every counted repeat), then the
    candidate with fewer changed settings, then the name."""
    tied = plateau(stats, throughput_key)
    if len(tied) <= 1:
        return tied[0] if tied else None
    if unsettled_tie(stats, throughput_key):
        # a partial ITL p99 never decides: the caller marks the pick provisional
        return min(tied, key=lambda c: (simplicity.get(c, 0), c))
    return min(
        tied, key=lambda c: (float(stats[c]["itl_p99"]["mean"]), simplicity.get(c, 0), c)
    )


def pareto(here: dict[str, dict[str, Any]]) -> list[str]:
    """Cases no other case beats on both tokens/s per GPU (higher) and TTFT p99 (lower)."""
    pts = {
        c: (float(s.get("otps_per_gpu") or 0.0), float(s["ttft_p99"]["mean"]))
        for c, s in here.items()
        if "ttft_p99" in s
    }
    return sorted(
        c
        for c, (tput, lat) in pts.items()
        if not any(
            (t2 >= tput and l2 <= lat) and (t2 > tput or l2 < lat)
            for o, (t2, l2) in pts.items()
            if o != c
        )
    )


def measured(s: dict[str, Any], metric: str) -> bool:
    """Every counted repeat at this load measured METRIC: one value is not the worst of three."""
    v = s.get(metric)
    return isinstance(v, dict) and v["n"] >= s["repeats"]


def meets(s: dict[str, Any], metric: str, limit: float) -> bool:
    return measured(s, metric) and s[metric]["max"] <= limit


def pick(
    cases: dict[str, dict[int, dict[str, Any]]],
    baseline: str,
    sla: tuple[str, float] | None,
    simplicity: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Per load: the best case for each latency metric and for tokens/s per GPU, plus the SLA pick.

    `tie_with` lists cases whose min-max range overlaps the winner's, so they are not separable.
    """
    picks: list[dict[str, Any]] = []
    loads = sorted({conc for by in cases.values() for conc in by})
    for conc in loads:
        here = {c: by[conc] for c, by in cases.items() if conc in by}
        entry: dict[str, Any] = {"concurrency": conc, "best": {}}
        per_gpu = {
            c: {x: s["otps_avg"][x] / s["gpus"] for x in ("mean", "min", "max")}
            for c, s in here.items()
            if "otps_avg" in s
        }
        # otps_avg is raw tokens/s; otps_per_gpu ranks layouts with different GPU counts fairly
        for k in (*LATENCY, "otps_avg", "otps_per_gpu"):
            have = per_gpu if k == "otps_per_gpu" else {c: s[k] for c, s in here.items() if k in s}
            if not have:
                continue
            lower = k not in ("otps_avg", "otps_per_gpu")
            win = (min if lower else max)(have, key=lambda c: have[c]["mean"])
            w = have[win]
            ties = sorted(
                c
                for c, v in have.items()
                if c != win and not (v["min"] > w["max"] or v["max"] < w["min"])
            )
            entry["best"][k] = {"case": win, "mean": round(w["mean"], 3), "tie_with": ties}
        if sla:
            metric, limit = sla
            ok = {c: s for c, s in here.items() if meets(s, metric, limit)}
            entry["sla"] = {
                "metric": metric,
                "max": limit,
                "meets": sorted(ok),
                "pick": rank(ok, simplicity or {}, "otps_per_gpu"),
            }
        entry["pareto"] = pareto(here)
        entry["baseline_present"] = baseline in here
        picks.append(entry)
    return picks


def sla_boundary(
    cases: dict[str, dict[int, dict[str, Any]]],
    sla: tuple[str, float],
    overloaded: set[tuple[str, int]] | None = None,
    unresolved: set[tuple[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per case: the highest load whose worst repeat meets the target, the lowest load above it
    measured to miss it (a load whose runs all failed requests misses it too), and the loads
    above it that lack the metric in some repeat or have run folders but no counted run and no
    measured overload (`unresolved`) (unmeasured: neither passing nor failing, measure them
    again)."""
    metric, limit = sla
    out: dict[str, dict[str, Any]] = {}
    # a case whose every load failed requests has no counted run (not in `cases`) but a result
    every = set(cases) | {k for k, _ in overloaded or ()} | {k for k, _ in unresolved or ()}
    for case in sorted(every):
        by = cases.get(case, {})
        misses = [c for c, s in by.items() if measured(s, metric) and not meets(s, metric, limit)]
        misses += [c for k, c in overloaded or () if k == case and c not in by]
        # the boundary is below the first measured miss: a load that passes above it again
        # (8 and 32 pass, 16 misses) never lifts the boundary past that miss
        cap = min(misses) if misses else None
        passing = [
            c for c, s in by.items() if meets(s, metric, limit) and (cap is None or c < cap)
        ]
        best = max(passing) if passing else None
        above = [c for c in by if (best is None or c > best) and not measured(by[c], metric)]
        failing = [c for c in misses if best is None or c > best]
        out[case] = {
            "max_concurrency": best,
            "otps_per_gpu": by[best].get("otps_per_gpu") if best is not None else None,
            "first_failing": min(failing) if failing else None,
            "unmeasured": sorted(
                above
                + [
                    c
                    for k, c in unresolved or ()
                    if k == case and c not in by and (best is None or c > best)
                ]
            ),
        }
    return out


def recommend(
    boundary: dict[str, dict[str, Any]],
    cases: dict[str, dict[int, dict[str, Any]]],
    simplicity: dict[str, int],
) -> dict[str, Any] | None:
    """Objective: the highest load within the target wins; at equal load, the
    3% throughput plateau, then ITL p99, then the simpler configuration."""
    passing = {c: b for c, b in boundary.items() if b["max_concurrency"]}
    if not passing:
        return None
    top = max(b["max_concurrency"] for b in passing.values())
    # loads within the search resolution are not separable; let throughput and latency decide
    at_top = {
        c: cases[c][b["max_concurrency"]]
        for c, b in passing.items()
        if b["max_concurrency"] >= top * (1 - LOAD_TIE)
    }
    win = str(rank(at_top, simplicity, "otps_per_gpu"))
    out: dict[str, Any] = {
        "case": win,
        "max_concurrency": boundary[win]["max_concurrency"],
        "otps_per_gpu": boundary[win]["otps_per_gpu"],
    }
    tied = unsettled_tie(at_top, "otps_per_gpu")
    if tied:
        out["unsettled_tie"] = tied  # ITL p99 could not decide it: never a final pick
    return out


def check_manifests(root: Path, cases: list[str]) -> list[dict[str, str]]:
    """Refuse a comparison across different measurement setups (a measurement fingerprint)."""
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    for case in cases:
        path = root / case / "manifest.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "code": "manifest_unreadable",
                    "message": f"{path}: {exc}",
                    "fix": "Fix or remove the manifest.",
                }
            )
            continue
        for key in MEASUREMENT_KEYS:
            if key not in data:
                continue
            first = seen.setdefault(key, (case, data[key]))
            if first[1] != data[key]:
                errors.append(
                    {
                        "code": "manifest_mismatch",
                        "message": f"{key} differs between {first[0]} and {case}",
                        "fix": "Re-measure both cases with the same workload, AIPerf version, "
                        "node and client in one series.",
                    }
                )
    return errors


def report_md(
    cases: dict[str, dict[int, dict[str, Any]]],
    baseline: str,
    comparisons: list[dict[str, Any]],
    picks: list[dict[str, Any]],
    boundary: dict[str, dict[str, Any]] | None = None,
    recommendation: dict[str, Any] | None = None,
) -> str:
    verdicts = {(c["case"], c["concurrency"], c["metric"]): c for c in comparisons}

    def cell(case: str, conc: int, k: str, digits: int) -> str:
        s = cases[case][conc].get(k)
        if s is None:
            return "-"
        text = f"{s['mean']:.{digits}f}"
        v = verdicts.get((case, conc, k))
        if v and v["change_pct"] is not None:
            text += f" ({v['change_pct']:+.0f}%, {v['verdict'].replace('_', ' ')})"
        return text

    out = [
        "# Before and after",
        "",
        f"Baseline: `{baseline}`. Each cell is the mean of the repeats; tuned rows show the change "
        "from the baseline at the same load and the verdict (better or worse only when the "
        "min-max ranges across repeats do not overlap).",
        "",
        "| Concurrent | Case | Repeats | TTFT avg ms | TTFT p99 ms | ITL avg ms | ITL p99 ms "
        "| Tokens/s per GPU |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = [baseline, *sorted(c for c in cases if c != baseline)]
    for p in picks:
        conc = p["concurrency"]
        for case in order:
            if conc not in cases.get(case, {}):
                continue
            s = cases[case][conc]
            out.append(
                f"| {conc} | {case} | {s['repeats']} | {cell(case, conc, 'ttft_avg', 0)} "
                f"| {cell(case, conc, 'ttft_p99', 0)} | {cell(case, conc, 'itl_avg', 2)} "
                f"| {cell(case, conc, 'itl_p99', 2)} | {s.get('otps_per_gpu', '-')} |"
            )
    out += ["", "## Best case per load", ""]
    for p in picks:
        parts = []
        for k, b in p["best"].items():
            tie = f" (tie with {', '.join(b['tie_with'])})" if b["tie_with"] else ""
            parts.append(f"{k}: `{b['case']}`{tie}")
        line = f"- {p['concurrency']} concurrent: " + "; ".join(parts)
        if p.get("pareto"):
            line += f". Pareto set (tokens/s per GPU vs TTFT p99): {', '.join(p['pareto'])}"
        if "sla" in p:
            sla = p["sla"]
            line += f". Within {sla['metric']} <= {sla['max']:g} ms: " + (
                f"`{sla['pick']}`" if sla["pick"] else "no case"
            )
        out.append(line)
    if boundary:
        out += [
            "",
            "## Highest load within the target",
            "",
            "| Case | Max concurrent within target | Tokens/s per GPU there | First failing load |",
            "|---|---|---|---|",
        ]
        for case in order + sorted(set(boundary) - set(order)):
            b = boundary.get(case)
            if b:
                unknown = b.get("unmeasured")
                failing = str(b["first_failing"] or "-") + (
                    f" (unmeasured: {', '.join(map(str, unknown))})" if unknown else ""
                )
                out.append(
                    f"| {case} | {b['max_concurrency'] or 'none'} | {b['otps_per_gpu'] or '-'} "
                    f"| {failing} |"
                )
        if recommendation and recommendation.get("provisional"):
            out += [
                "",
                f"**Provisional, not a recommendation: `{recommendation['case']}`.** The "
                "comparison is incomplete: "
                + "; ".join(recommendation.get("provisional_reasons", []))
                + ". Measure the missing runs, then compare again.",
            ]
        elif recommendation:
            out += [
                "",
                f"**Recommendation: `{recommendation['case']}`**, the highest load within the "
                f"target ({recommendation['max_concurrency']} concurrent) at "
                f"{recommendation['otps_per_gpu']} tokens/s per GPU. Ties: throughput within 3%, "
                "then lower ITL p99, then fewer changed settings.",
            ]
        out += [
            "",
            "A case that misses the target is never the recommendation, even when it is "
            "fastest on another metric.",
        ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = ContractParser(
        description="Compare NIM candidates with the baseline on AIPerf repeats (read-only).",
        epilog=(
            "Example: python3 scripts/compare_aiperf.py runs/qwen7b --baseline before "
            "--gpus before=4 --gpus tp2x2=4 --gpus tp1x1=1"
        ),
    )
    parser.add_argument(
        "root", type=Path, help="folder holding CASE/cCONC-rREPEAT/profile_export_aiperf.csv"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="case name of the baseline (the untuned or current best NIM)",
    )
    parser.add_argument(
        "--gpus", action="append", default=[], metavar="CASE=N", help="GPUs a case uses in total"
    )
    parser.add_argument(
        "--default-gpus", type=int, default=1, help="GPUs for cases without --gpus (default 1)"
    )
    parser.add_argument(
        "--min-repeats", type=int, default=3, help="repeats required per case and load (default 3)"
    )
    parser.add_argument(
        "--sla",
        metavar="METRIC<=MS",
        help="latency target, e.g. ttft_p99<=1000; picks the most tokens/s per GPU that meets it",
    )
    parser.add_argument("--report", type=Path, help="write the before-and-after Markdown here")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="CASE",
        help="leave a case out (run_campaign excludes failed, stopped or partly measured ones)",
    )
    parser.add_argument(
        "--loads",
        help="comma-separated loads to compare and pick at, e.g. 8,32,128 (default: all); "
        "the --sla boundary always uses every measured load",
    )
    args = parser.parse_args(argv)
    loads: set[int] | None = None
    if args.loads:
        try:
            loads = {int(x) for x in args.loads.split(",") if x.strip()}
        except ValueError:
            emit(
                False,
                "error",
                "fix_errors",
                errors=[
                    {
                        "code": "bad_loads",
                        "message": f"--loads {args.loads!r}",
                        "fix": "Pass integers separated by commas, e.g. 8,32,128.",
                    }
                ],
                code=EXIT_INPUT,
            )
    sla: tuple[str, float] | None = None
    if args.sla:
        m = SLA.match(args.sla.replace(" ", ""))
        if not m:
            emit(
                False,
                "error",
                "fix_errors",
                errors=[
                    {
                        "code": "bad_sla",
                        "message": f"--sla {args.sla!r}",
                        "fix": "Use METRIC<=MS with METRIC one of ttft, itl or e2e plus _avg "
                        "or _p99, e.g. ttft_p99<=1000.",
                    }
                ],
                code=EXIT_INPUT,
            )
        sla = (f"{m[1]}_{m[2]}", float(m[3]))

    if not args.root.is_dir():
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                {
                    "code": "root_missing",
                    "message": f"{args.root} is not a folder",
                    "fix": "Pass the folder that holds one subfolder per case.",
                }
            ],
            code=EXIT_INPUT,
        )
    if args.default_gpus < 1:
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                {
                    "code": "bad_gpus",
                    "message": f"--default-gpus {args.default_gpus}",
                    "fix": "Pass --default-gpus with a positive integer (default 1).",
                }
            ],
            code=EXIT_INPUT,
        )
    gpus: dict[str, int] = {}
    for item in args.gpus:
        case, _, n = item.partition("=")
        if not n.isdigit() or int(n) < 1:
            emit(
                False,
                "error",
                "fix_errors",
                errors=[
                    {
                        "code": "bad_gpus",
                        "message": f"--gpus {item!r}",
                        "fix": "Pass --gpus CASE=N with a positive integer.",
                    }
                ],
                code=EXIT_INPUT,
            )
        gpus[case] = int(n)

    rejected: list[str] = []
    no_error_rate: list[str] = []
    unfinished: list[str] = []
    runs = collect(args.root, rejected, no_error_rate, unfinished)
    for case in args.exclude:
        runs.pop(case, None)
    # an excluded case's failed requests are not this comparison's problem
    rejected = [r for r in rejected if r.split("/")[0] not in args.exclude]
    no_error_rate = [r for r in no_error_rate if r.split("/")[0] not in args.exclude]
    unfinished = [r for r in unfinished if r.split("/")[0] not in args.exclude]
    # a baseline whose runs all failed requests has overload evidence, not nothing: judged with
    # the other overloads below (a measured overload at every load is a result, not a gap)
    if args.baseline not in runs and not any(r.split("/")[0] == args.baseline for r in rejected):
        # say why its repeats did not count: exports without a readable error rate need a rerun
        unread = [r for r in no_error_rate if r.split("/")[0] == args.baseline]
        emit(
            False,
            "error",
            "fix_errors",
            {"cases": sorted(runs), "runs_without_error_rate": unread},
            [
                {
                    "code": "baseline_missing",
                    "message": f"no AIPerf runs for baseline {args.baseline!r} under {args.root}"
                    + (
                        f"; {', '.join(unread)} had no readable {ERROR_ROW} row, so they were "
                        "not counted"
                        if unread
                        else ""
                    ),
                    "fix": f"Repeat {', '.join(unread)} so AIPerf writes a complete export."
                    if unread
                    else f"Measure it into ROOT/{args.baseline}/cCONC-rN/ or name another case.",
                }
            ],
            EXIT_PRECONDITION,
        )

    all_cases = {
        c: {
            conc: summarize(reps, int(gpus.get(c, args.default_gpus)))
            for conc, reps in sorted(by.items())
        }
        for c, by in sorted(runs.items())
    }
    # an SLA-search point (a load not in --loads; a search needs --sla) stops at its first repeat
    # with failed requests, by design: that load is a measured miss, whatever repeats before it
    # counted
    # (never a stray failure to repeat: the runner would not measure it again)
    search_missed = {
        (r.split("/")[0], int(m[1]))
        for r in rejected
        if sla is not None
        and loads is not None
        and (m := RUN_DIR.match(r.split("/")[-1]))
        and int(m[1]) not in loads
    }
    for c, conc in search_missed:
        all_cases.get(c, {}).pop(conc, None)
    # comparisons and picks use the fixed loads; SLA-search points only feed the boundary
    cases = {
        c: {conc: s for conc, s in by.items() if loads is None or conc in loads}
        for c, by in all_cases.items()
    }
    # every case with a run folder, also one whose only evidence is an overload (not in `cases`)
    evidence = (set(cases) | run_cases(args.root)) - set(args.exclude)
    errors = check_manifests(args.root, sorted(evidence))
    if any(e["code"] == "manifest_mismatch" for e in errors):
        emit(False, "refused", "fix_errors", {"cases": sorted(cases)}, errors, EXIT_REFUSED)
    for run in rejected:
        errors.append(
            {
                "code": "run_had_request_errors",
                "message": f"{run} had failed requests and was not counted",
                "fix": "Find the cause in the NIM log, then repeat that run.",
            }
        )
    # diagnostics only: at an SLA-search point already measured to miss (search_missed), a
    # repeat without an error rate changes nothing, and the runner never measures it again
    settled: list[dict[str, str]] = []
    for run in no_error_rate:
        errors.append(
            {
                "code": "run_error_rate_missing",
                "message": f"{run} has no readable {ERROR_ROW} row, so its requests are not known "
                "to have succeeded, and was not counted",
                "fix": "Repeat that run so AIPerf writes a complete export.",
            }
        )
        if (m := RUN_DIR.match(run.split("/")[-1])) and (
            run.split("/")[0],
            int(m[1]),
        ) in search_missed:
            settled.append(errors[-1])
    # a load with no counted run is reported as load_missing below; an unfinished attempt next
    # to counted runs would otherwise pass unseen
    for run in unfinished:
        if (m := RUN_DIR.match(run.split("/")[-1])) and int(m[1]) in all_cases.get(
            run.split("/")[0], {}
        ):
            errors.append(
                {
                    "code": "run_unfinished",
                    "message": f"{run} has no readable export and was not counted",
                    "fix": "Repeat that run, or remove the folder if the attempt was abandoned.",
                }
            )
    required = required_metrics(sla[0] if sla else None)
    partly_measured: list[dict[str, Any]] = []
    for c, by in all_cases.items():
        for conc, s in by.items():
            # throughput (it ranks every case) and the target's metric (meets() passes a load
            # only when every counted repeat measured it) need every counted repeat: the same
            # metrics run_campaign needs before it counts a run. Absent counts as 0
            n = {k: s[k]["n"] if isinstance(s.get(k), dict) else 0 for k in required}
            short = sorted(k for k in required if n[k] < max(args.min_repeats, s["repeats"]))
            # any other latency metric only describes: measured in fewer than --min-repeats
            # repeats (a 1-token output has no ITL), it gets no verdict, best or tie-break and
            # never blocks; the runner could not measure it again anyway
            for k in LATENCY:
                v = s.get(k)
                if k not in required and isinstance(v, dict) and v["n"] < args.min_repeats:
                    partly_measured.append(
                        {"case": c, "concurrency": conc, "metric": k, "n": v["n"],
                         "repeats": s["repeats"]}  # fmt: skip
                    )
                    del s[k]
            if s["repeats"] < args.min_repeats or short:
                have = s["repeats"] if not short else min(n[k] for k in short)
                errors.append(
                    {
                        "code": "too_few_repeats",
                        "message": f"{c} at {conc} concurrent has {have} repeats"
                        + (f" measuring {', '.join(short)}" if short else ""),
                        "fix": f"Run {args.min_repeats} repeats with different seeds first.",
                    }
                )

    # (case, load) pairs where every attempt (run folder) was rejected for failed requests: a
    # measured overload, never a missing load. An attempt with no usable result there (empty,
    # or no error rate) keeps it unresolved, so it is still a gap
    rejected_at: dict[tuple[str, int], int] = {}
    for r in rejected:
        if m := RUN_DIR.match(r.split("/")[-1]):
            key = (r.split("/")[0], int(m[1]))
            rejected_at[key] = rejected_at.get(key, 0) + 1
    attempts = run_attempts(args.root)
    # a load is overloaded only once --min-repeats attempts all failed requests (a repeat never
    # started is a gap). Only an SLA-search point (search_missed: a load outside --loads, so
    # only with --loads and --sla) stops at its first failed repeat by design; otherwise
    # nothing marks a load as one, so one failed attempt there is never a final result
    overloaded = {
        k for k, n in rejected_at.items() if n == attempts.get(k, 0) and n >= args.min_repeats
    } | search_missed
    comparisons = []
    for c, by in cases.items():
        if c == args.baseline:
            continue
        for conc, s in by.items():
            b = cases.get(args.baseline, {}).get(conc)
            if b is None and (args.baseline, conc) in overloaded:
                continue  # the untuned NIM failed requests there: nothing to compare, no gap
            if b is None:
                errors.append(
                    {
                        "code": "load_missing",
                        "message": f"baseline has no runs at {conc} concurrent",
                        "fix": "Measure the baseline at every load the candidates use.",
                    }
                )
                continue
            for key, lower in METRICS.values():
                for stat in STATS:
                    k = f"{key}_{stat}"
                    if k in s and k in b:
                        comparisons.append(
                            {
                                "case": c,
                                "concurrency": conc,
                                "metric": k,
                                "baseline": b[k]["mean"],
                                "candidate": s[k]["mean"],
                                "change_pct": change_pct(b[k]["mean"], s[k]["mean"]),
                                "verdict": verdict(b[k], s[k], lower),
                            }
                        )
    # every load a case has run folders for (fixed loads and SLA-search points alike, whether or
    # not --loads lists it) needs a counted run there, or runs that all failed requests (a
    # measured overload): an unfinished attempt is a gap
    candidate_loads = {conc for c, by in cases.items() if c != args.baseline for conc in by}
    reported: set[tuple[str, int]] = set()
    for c, seen in sorted(run_loads(args.root).items()):
        if c in args.exclude:
            continue
        for conc in sorted(seen - set(all_cases.get(c, {}))):
            # the baseline's gap at a compared load is already reported above
            if (c, conc) in overloaded or (c == args.baseline and conc in candidate_loads):
                continue
            reported.add((c, conc))
            errors.append(
                {
                    "code": "load_missing",
                    "message": f"{c} has no counted runs at {conc} concurrent",
                    "fix": "Measure that load again, or pass --exclude for the case.",
                }
            )
    if loads is not None:
        # every requested load needs a counted entry in every case. A load whose runs were all
        # rejected for failed requests is a measured result (overload), not a gap
        # every case with a run folder, also one with no counted run at all (collect leaves it
        # out, so it is not in `cases`), minus the excluded ones
        every_case = sorted((set(cases) | run_cases(args.root)) - set(args.exclude))
        for c in every_case:
            by = cases.get(c, {})
            for conc in sorted(loads - set(by)):
                # the baseline's gap at a load a candidate has is already reported above
                if (
                    (c, conc) in overloaded
                    or (c, conc) in reported
                    or (c == args.baseline and conc in candidate_loads)
                ):
                    continue
                errors.append(
                    {
                        "code": "load_missing",
                        "message": f"{c} has no counted runs at {conc} concurrent",
                        "fix": "Measure every case at every load in --loads.",
                    }
                )
    simplicity: dict[str, int] = {}
    for case in cases:
        with contextlib.suppress(OSError, json.JSONDecodeError, AttributeError):
            cand = json.loads((args.root / case / "manifest.json").read_text()).get("candidate", {})
            simplicity[case] = len(cand.get("args", [])) + len(cand.get("env", {}))
    picks = pick(cases, args.baseline, sla, simplicity)
    # loads with run folders but no counted run that are not a measured overload: unknown
    unresolved = {
        (c, conc)
        for c, seen in run_loads(args.root).items()
        if c not in args.exclude
        for conc in seen
        if conc not in all_cases.get(c, {}) and (c, conc) not in overloaded
    }
    boundary = sla_boundary(all_cases, sla, overloaded, unresolved) if sla else None
    recommendation = recommend(boundary, all_cases, simplicity) if boundary else None
    if recommendation and recommendation.get("unsettled_tie"):
        tied = recommendation["unsettled_tie"]
        errors.append(
            {
                "code": "tie_unsettled",
                "message": f"{', '.join(tied)} tie on tokens/s per GPU at the highest load within "
                "the target, and ITL p99, which settles a tie, was not measured in every counted "
                f"repeat of each; {recommendation['case']} is only the simpler setting",
                "fix": "Choose between the tied cases yourself, or measure them again with "
                "outputs long enough for AIPerf to report ITL p99 (more than one token).",
            }
        )
    # an incomplete comparison has no final winner: a run rejected for failed requests is a
    # measured result, anything else (too few repeats, a missing load, a run without an error
    # rate) leaves the pick provisional
    # a load where every run failed requests is a measured overload; a failed run next to
    # counted ones at the same load is a stray failure to repeat before deciding, so a gap
    stray = [
        r
        for r in rejected
        if (m := RUN_DIR.match(r.split("/")[-1])) and int(m[1]) in all_cases.get(r.split("/")[0], {})
    ]
    gaps = [
        e["message"]
        for e in errors
        if e["code"] != "run_had_request_errors" and not any(e is x for x in settled)
    ]
    gaps += [f"{r} had failed requests next to counted runs: repeat it" for r in stray]
    if recommendation:
        recommendation["provisional"] = bool(gaps)
        if gaps:
            recommendation["provisional_reasons"] = gaps
    facts: dict[str, Any] = {
        "baseline": args.baseline,
        "cases": cases,
        "comparisons": comparisons,
        "picks": picks,
        "sla_boundary": boundary,
        "recommendation": recommendation,
        "rejected_runs": rejected,
        # (case, load) where every repeat failed requests: a measured overload, not a gap
        "overloaded": sorted([c, conc] for c, conc in overloaded),
        "partly_measured": partly_measured,
        "runs_without_error_rate": no_error_rate,
        "rule": "better/worse only when the min-max ranges across repeats do not overlap"
        " and the means differ by 2% or more",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            report_md(cases, args.baseline, comparisons, picks, boundary, recommendation)
        )
        facts["report"] = str(args.report)
    if gaps:
        emit(False, "incomplete", "measure_more", facts, errors, EXIT_PRECONDITION)
    # only whole-load overloads: the comparison is complete (so the result can be written); the
    # overloads stay listed in errors and rejected_runs
    emit(True, "compared", "decide", facts, errors or None)


if __name__ == "__main__":
    main()
