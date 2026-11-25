# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test for scripts/compare_aiperf.py: one JSON object and a documented exit code."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_aiperf.py"
HEADER = "Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std\n"


def write_run(
    root: Path,
    case: str,
    conc: int,
    rep: int,
    ttft: float,
    itl: float,
    otps: float,
    err: str | None = "0.00",
) -> None:
    """An AIPerf CSV export. AIPerf 0.13 writes the error-rate row on every run, 0.00 when all
    requests succeeded; err=None leaves it out."""
    d = root / case / f"c{conc}-r{rep}"
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.csv").write_text(
        HEADER
        + f"Time to First Token (ms),{ttft},,,,,,,,,,,,{ttft * 2},\n"
        + f"Inter Token Latency (ms),{itl},,,,,,,,,,,,{itl * 1.1},\n"
        + "HTTP Blocked (ms),0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
        + "\nMetric,Value\n"
        + f"Output Token Throughput (tokens/sec),{otps}\n"
        + ("" if err is None else f"Request Error Rate (%),{err}\n")
    )


def run(*args: str) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    out = json.loads(proc.stdout)
    assert set(out) == {"ok", "state", "next", "facts", "errors"}
    return proc.returncode, out


def test_separated_ranges_are_better_and_overlap_is_noise(tmp_path: Path) -> None:
    for r, (t, i) in enumerate([(120, 8.2), (90, 8.4), (115, 8.3)], 1):
        write_run(tmp_path, "before", 8, r, t, i, 900)
    for r, (t, i) in enumerate([(60, 8.1), (58, 8.5), (62, 8.3)], 1):
        write_run(tmp_path, "tp2x2", 8, r, t, i, 1800)
    code, out = run(
        str(tmp_path), "--baseline", "before", "--gpus", "before=4", "--gpus", "tp2x2=4"
    )
    assert code == 0 and out["next"] == "decide"
    got = {c["metric"]: c["verdict"] for c in out["facts"]["comparisons"]}
    assert got["ttft_avg"] == "better" and got["ttft_p99"] == "better"
    assert got["itl_avg"] == "within_noise"
    assert got["otps_avg"] == "better"
    assert out["facts"]["cases"]["tp2x2"]["8"]["otps_per_gpu"] == 450.0


def test_too_few_repeats_blocks_the_decision(tmp_path: Path) -> None:
    write_run(tmp_path, "before", 8, 1, 100, 8, 900)
    write_run(tmp_path, "cand", 8, 1, 90, 8, 950)
    code, out = run(str(tmp_path), "--baseline", "before")
    assert code == 3 and out["next"] == "measure_more"
    assert {e["code"] for e in out["errors"]} == {"too_few_repeats"}


def test_missing_baseline_is_a_precondition(tmp_path: Path) -> None:
    write_run(tmp_path, "cand", 8, 1, 90, 8, 950)
    code, out = run(str(tmp_path), "--baseline", "before")
    assert code == 3 and out["errors"][0]["code"] == "baseline_missing"


def test_bad_usage_is_contract_json(tmp_path: Path) -> None:
    code, out = run(str(tmp_path))
    assert code == 2 and out["errors"][0]["code"] == "bad_usage"


def write_log(
    root: Path, case: str, conc: int, rep: int, ttft: float, err_rate: float | None
) -> None:
    """An AIPerf console log with only the final summary table (no CSV), as when a copy fails.
    err_rate=None leaves the error-rate row out."""
    d = root / case / f"c{conc}-r{rep}"
    d.mkdir(parents=True)
    bar = "│"
    rows: list[tuple[str, float, float | None]] = [
        ("Time to First Token (ms)", ttft, ttft * 2),
        ("Inter Token Latency (ms)", 5.0, 5.5),
        ("Output Token Throughput (tokens/sec)", 1000.0, None),
    ]
    if err_rate is not None:
        rows.append(("Request Error Rate (%)", err_rate, None))
    lines = ["NVIDIA AIPerf | LLM Metrics"]
    for name, avg, p99 in rows:
        p = "N/A" if p99 is None else f"{p99:,.2f}"
        cells = [name, f"{avg:,.2f}", "1", "2", p, "1", "1", "0"]
        lines.append(bar + bar.join(cells) + bar)
    lines.append("└" + "-" * 10)
    (d / "aiperf.log").write_text("\n".join(lines) + "\n")


def test_console_log_fallback_sla_boundary_and_error_rejection(tmp_path: Path) -> None:
    for conc, t in ((8, 100.0), (32, 300.0), (128, 900.0)):
        for r in (1, 2, 3):
            write_log(tmp_path, "before", conc, r, t, 0.0)
            write_log(tmp_path, "tp1x4", conc, r, t / 3, 0.0)
    write_log(tmp_path, "tp1x4", 8, 4, 10.0, 2.5)  # failed requests: never counted
    report = tmp_path / "final" / "before_after.md"
    code, out = run(
        str(tmp_path),
        "--baseline",
        "before",
        "--sla",
        "ttft_p99<=700",
        "--report",
        str(report),
    )
    # the rejected run is reported, so the agent repeats it before deciding
    assert code == 3 and out["next"] == "measure_more"
    assert [e["code"] for e in out["errors"]] == ["run_had_request_errors"]
    assert out["facts"]["rejected_runs"] == ["tp1x4/c8-r4"]
    assert out["facts"]["cases"]["tp1x4"]["8"]["repeats"] == 3
    b = out["facts"]["sla_boundary"]
    assert b["before"]["max_concurrency"] == 32 and b["before"]["first_failing"] == 128
    assert b["tp1x4"]["max_concurrency"] == 128
    assert out["facts"]["picks"][2]["sla"]["pick"] == "tp1x4"
    assert "Highest load within the target" in report.read_text()


def test_different_measurement_setups_are_refused(tmp_path: Path) -> None:
    for case, osl in (("before", 256), ("tp2x2", 512)):
        for r in (1, 2, 3):
            write_run(tmp_path, case, 8, r, 100, 8, 900)
        (tmp_path / case / "manifest.json").write_text(json.dumps({"workload": {"osl": osl}}))
    code, out = run(str(tmp_path), "--baseline", "before")
    assert code == 5 and out["state"] == "refused"
    assert out["errors"][0]["code"] == "manifest_mismatch"


def test_plateau_tie_goes_to_lower_itl_then_simpler(tmp_path: Path) -> None:
    # "a" and "b" are within 3% on throughput; "b" has the lower ITL p99 so it wins the pick
    for case, otps, itl in (("before", 900, 8.0), ("a", 2000, 5.0), ("b", 1950, 4.0)):
        for r in (1, 2, 3):
            write_run(tmp_path, case, 8, r, 100, itl, otps)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 0
    assert out["facts"]["picks"][0]["sla"]["pick"] == "b"
    assert out["facts"]["recommendation"]["case"] == "b"
    # same TTFT everywhere, so only the highest-throughput case is on the Pareto front
    assert out["facts"]["picks"][0]["pareto"] == ["a"]


def test_excluded_cases_are_never_ranked(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100 + r, 8.0, 900)
        write_run(tmp_path, "crashed", 8, r, 10 + r, 4.0, 5000)  # fast, but failed later
    code, out = run(str(tmp_path), "--baseline", "before", "--exclude", "crashed")
    assert code == 0 and set(out["facts"]["cases"]) == {"before"}
    assert out["facts"]["picks"][0]["best"]["otps_avg"]["case"] == "before"


def test_a_tiny_change_is_noise_even_with_tight_ranges(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100.0 + r * 0.01, 8.0, 1000 + r * 0.1)
        write_run(tmp_path, "x", 8, r, 99.0 + r * 0.01, 8.0, 1010 + r * 0.1)  # -1%, +1%
    _, out = run(str(tmp_path), "--baseline", "before")
    got = {c["metric"]: c["verdict"] for c in out["facts"]["comparisons"]}
    assert got["ttft_avg"] == "within_noise" and got["otps_avg"] == "within_noise"


def test_a_metric_measured_in_too_few_repeats_decides_nothing_and_never_blocks(
    tmp_path: Path,
) -> None:
    # run_campaign counts a run on throughput (and the target's metric) alone, so a descriptive
    # metric missing from one repeat must not block: it only loses its verdict
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "x", 8, r, 50, 8.0, 900)
    csv = tmp_path / "x" / "c8-r2" / "profile_export_aiperf.csv"
    csv.write_text(
        csv.read_text().replace("Time to First Token (ms),50", "Time to First Token (ms),nan")
    )
    code, out = run(str(tmp_path), "--baseline", "before")
    assert code == 0 and out["state"] == "compared", out["errors"]
    assert "ttft_avg" not in out["facts"]["cases"]["x"]["8"]
    assert not any(c["metric"] == "ttft_avg" for c in out["facts"]["comparisons"])
    best = out["facts"]["picks"][0]["best"]["ttft_avg"]  # x is not ranked on two values
    assert best["case"] == "before" and best["tie_with"] == []
    assert out["facts"]["partly_measured"] == [
        {"case": "x", "concurrency": 8, "metric": "ttft_avg", "n": 2, "repeats": 3}
    ]


def test_an_excluded_cases_failed_requests_do_not_block(tmp_path: Path) -> None:
    for conc, t in ((8, 100.0), (32, 300.0)):
        for r in (1, 2, 3):
            write_log(tmp_path, "before", conc, r, t, 0.0)
    write_log(tmp_path, "crashed", 8, 1, 10.0, 5.0)  # failed requests, and the case is excluded
    code, out = run(str(tmp_path), "--baseline", "before", "--exclude", "crashed")
    assert code == 0 and out["facts"]["rejected_runs"] == []


def edit_csv(root: Path, run: str, old: str, new: str) -> None:
    csv = root / run / "profile_export_aiperf.csv"
    text = csv.read_text()
    assert old in text
    csv.write_text(text.replace(old, new))


def test_the_sla_metric_needs_the_full_repeat_count_and_is_never_passed_on_one_value(
    tmp_path: Path,
) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    # only one of three runs reports a request latency p99: that is not the worst of three repeats
    edit_csv(
        tmp_path,
        "before/c8-r1",
        "HTTP Blocked",
        "Request Latency (ms),500,,,,,,,,,,,,800,\nHTTP Blocked",
    )
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "e2e_p99<=1000")
    assert code == 3
    assert any(
        "e2e_p99" in e["message"] and "has 1 repeats" in e["message"]
        for e in out["errors"]
        if e["code"] == "too_few_repeats"
    )
    assert out["facts"]["picks"][0]["sla"]["meets"] == []
    assert out["facts"]["recommendation"] is None


def test_a_target_metric_never_measured_counts_as_zero_repeats(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "e2e_p99<=1000")
    assert code == 3
    assert any("has 0 repeats measuring e2e_p99" in e["message"] for e in out["errors"])


def test_an_infinite_value_is_not_a_measurement(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    edit_csv(
        tmp_path, "before/c8-r2", "Time to First Token (ms),100", "Time to First Token (ms),inf"
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--baseline", "before"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    def strict(name: str) -> None:
        raise ValueError(f"not strict JSON: {name}")

    out = json.loads(proc.stdout, parse_constant=strict)
    assert proc.returncode == 0
    assert "ttft_avg" not in out["facts"]["cases"]["before"]["8"]  # two values are not three
    assert out["facts"]["partly_measured"][0]["n"] == 2


def test_the_best_per_load_ranks_tokens_per_gpu_next_to_raw_tokens(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 2000 + r)  # 4 GPUs: ~500 per GPU
        write_run(tmp_path, "one", 8, r, 100, 8.0, 1000 + r)  # 1 GPU: ~1000 per GPU
    code, out = run(str(tmp_path), "--baseline", "before", "--gpus", "before=4")
    assert code == 0
    best = out["facts"]["picks"][0]["best"]
    assert best["otps_avg"]["case"] == "before"
    assert best["otps_per_gpu"]["case"] == "one" and best["otps_per_gpu"]["tie_with"] == []


def test_a_present_but_unreadable_error_rate_is_not_clean(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    edit_csv(tmp_path, "before/c8-r3", "Request Error Rate (%),0.00", "Request Error Rate (%),N/A")
    edit_csv(tmp_path, "before/c8-r2", "Request Error Rate (%),0.00", "Request Error Rate (%),inf")
    write_log(tmp_path, "before", 8, 4, 100.0, float("nan"))
    code, out = run(str(tmp_path), "--baseline", "before")
    assert code == 3 and out["facts"]["rejected_runs"] == []  # unreadable is not overload
    got = sorted(out["facts"]["runs_without_error_rate"])
    assert got == ["before/c8-r2", "before/c8-r3", "before/c8-r4"]
    assert out["facts"]["cases"]["before"]["8"]["repeats"] == 1


def test_an_export_with_only_an_unreadable_error_row_is_missing_evidence(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    d = tmp_path / "before" / "c8-r4"
    d.mkdir()
    (d / "profile_export_aiperf.csv").write_text("Metric,Value\nRequest Error Rate (%),N/A\n")
    code, out = run(str(tmp_path), "--baseline", "before")
    assert out["facts"]["runs_without_error_rate"] == ["before/c8-r4"], out


def test_a_load_with_only_unreadable_error_rates_is_a_gap_not_overload(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "before", 32, r, 400, 9.0, 2000)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
        write_run(tmp_path, "cand", 32, r, 300, 9.0, 2400)
        edit_csv(tmp_path, f"cand/c32-r{r}", "Request Error Rate (%),0.00",
                 "Request Error Rate (%),N/A")  # fmt: skip
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8,32",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    rec = out["facts"]["recommendation"]
    assert rec["provisional"] is True
    assert any("cand has no counted runs at 32" in m for m in rec["provisional_reasons"])


def test_an_incomplete_comparison_marks_the_recommendation_provisional(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    for r in (1, 2):
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    report = tmp_path / "final" / "before_after.md"
    code, out = run(
        str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000", "--report", str(report)
    )
    assert code == 3 and out["state"] == "incomplete"
    rec = out["facts"]["recommendation"]
    assert rec["case"] == "cand" and rec["provisional"] is True
    assert any("cand at 8 concurrent has 2 repeats" in m for m in rec["provisional_reasons"])
    text = report.read_text()
    assert "Provisional, not a recommendation" in text and "**Recommendation:" not in text


def test_a_complete_comparison_is_not_provisional(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 0 and out["facts"]["recommendation"]["provisional"] is False


def test_a_truncated_metric_value_row_is_unmeasured_not_a_crash(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    edit_csv(
        tmp_path,
        "before/c8-r1",
        "Output Token Throughput (tokens/sec),900",
        "Output Token Throughput (tokens/sec)",
    )
    code, out = run(str(tmp_path), "--baseline", "before")
    assert code == 3
    assert out["facts"]["cases"]["before"]["8"]["otps_avg"]["n"] == 2
    assert any("otps_avg" in e["message"] for e in out["errors"] if e["code"] == "too_few_repeats")


def test_a_load_without_the_target_metric_is_unmeasured_not_failing(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "before", 32, r, 100, 8.0, 900)
        write_run(tmp_path, "before", 64, r, 900, 8.0, 900)  # p99 1800: misses the target
    for r in (1, 2, 3):  # throughput measured at 32, but never the target metric
        edit_csv(tmp_path, f"before/c32-r{r}", "Time to First Token (ms),100,", "X,")
    report = tmp_path / "final" / "before_after.md"
    _, out = run(
        str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000", "--report", str(report)
    )
    b = out["facts"]["sla_boundary"]["before"]
    assert b["max_concurrency"] == 8 and b["first_failing"] == 64 and b["unmeasured"] == [32]
    assert "64 (unmeasured: 32)" in report.read_text()


def test_a_nonpositive_default_gpu_count_is_bad_input(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    code, out = run(str(tmp_path), "--baseline", "before", "--default-gpus", "0")
    assert code == 2 and out["errors"][0]["code"] == "bad_gpus"


def test_a_run_without_an_error_rate_row_is_not_counted(tmp_path: Path) -> None:
    # four repeats, so the three clean ones still meet --min-repeats: the gap alone must block
    for r in (1, 2, 3, 4):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900, err=None if r == 4 else "0.00")
        write_log(tmp_path, "cand", 8, r, 50.0, None if r == 4 else 0.0)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3 and out["state"] == "incomplete"
    assert out["facts"]["runs_without_error_rate"] == ["before/c8-r4", "cand/c8-r4"]
    assert out["facts"]["rejected_runs"] == []
    assert out["facts"]["cases"]["before"]["8"]["repeats"] == 3
    assert out["facts"]["cases"]["cand"]["8"]["repeats"] == 3
    assert [e["code"] for e in out["errors"]] == ["run_error_rate_missing"] * 2
    rec = out["facts"]["recommendation"]
    assert rec["case"] == "cand" and rec["provisional"] is True


def test_a_counted_repeat_without_the_target_metric_leaves_the_pick_provisional(
    tmp_path: Path,
) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    for r in (1, 2, 3, 4):
        write_run(tmp_path, "before", 32, r, 200, 8.0, 900)
    # the fourth repeat at 32 measured TTFT avg but not the p99 the target uses
    edit_csv(
        tmp_path,
        "before/c32-r4",
        "Time to First Token (ms),200,,,,,,,,,,,,400,",
        "Time to First Token (ms),200,,,,,,,,,,,,,",
    )
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3
    assert out["facts"]["cases"]["before"]["32"]["ttft_p99"]["n"] == 3
    assert out["facts"]["sla_boundary"]["before"]["unmeasured"] == [32]
    rec = out["facts"]["recommendation"]
    assert rec["max_concurrency"] == 8 and rec["provisional"] is True
    assert any(
        "before at 32 concurrent has 3 repeats measuring ttft_p99" in m
        for m in rec["provisional_reasons"]
    )


def test_a_candidate_missing_a_requested_load_leaves_the_pick_provisional(
    tmp_path: Path,
) -> None:
    for r in (1, 2, 3):
        for conc in (8, 32):
            write_run(tmp_path, "before", conc, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)  # never measured at 32
    code, out = run(
        str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000", "--loads", "8,32,64"
    )
    assert code == 3
    gaps = [e["message"] for e in out["errors"] if e["code"] == "load_missing"]
    # nobody measured 64: one gap per case, the baseline's not repeated per candidate
    assert gaps == [
        "before has no counted runs at 64 concurrent",
        "cand has no counted runs at 32 concurrent",
        "cand has no counted runs at 64 concurrent",
    ]
    rec = out["facts"]["recommendation"]
    assert rec["case"] == "before" and rec["provisional"] is True


def test_a_load_lost_to_failed_requests_is_a_result_not_a_gap(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        for conc in (8, 32):
            write_run(tmp_path, "before", conc, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
        write_run(tmp_path, "cand", 32, r, 50, 8.0, 1800, err="4.5")  # overloaded at 32
    code, out = run(
        str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000", "--loads", "8,32"
    )
    # every cand run at 32 failed requests: a measured overload, so the comparison is complete
    assert code == 0 and out["ok"] is True and out["state"] == "compared"
    assert {e["code"] for e in out["errors"]} == {"run_had_request_errors"}
    assert out["facts"]["recommendation"]["provisional"] is False


def test_a_case_whose_throughput_was_never_measured_is_incomplete(tmp_path: Path) -> None:
    # every repeat reports infinite throughput, so summarize drops the metric: the case must
    # not be ranked as if it had 0 tokens/s, and the pick stays provisional
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, float("inf"))
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3 and out["state"] == "incomplete"
    assert any("cand at 8 concurrent" in e["message"] and "otps_avg" in e["message"]
               for e in out["errors"])  # fmt: skip
    # an incomplete comparison never carries a final recommendation
    rec = out["facts"]["recommendation"]
    assert rec is not None and rec["provisional"] is True


def test_throughput_missing_from_one_of_four_repeats_leaves_the_pick_provisional(
    tmp_path: Path,
) -> None:
    # three throughput values meet --min-repeats, but the fourth counted repeat could change
    # the ranking, so throughput needs every counted repeat, as the target metric does
    for r in (1, 2, 3, 4):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, float("inf") if r == 4 else 1800)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert out["facts"]["cases"]["cand"]["8"]["repeats"] == 4
    assert out["facts"]["cases"]["cand"]["8"]["otps_avg"]["n"] == 3
    assert code == 3 and out["state"] == "incomplete"
    assert any("cand at 8 concurrent" in e["message"] and "otps_avg" in e["message"]
               for e in out["errors"] if e["code"] == "too_few_repeats")  # fmt: skip
    assert out["facts"]["recommendation"]["provisional"] is True


def test_four_complete_repeats_still_give_a_final_pick(tmp_path: Path) -> None:
    for r in (1, 2, 3, 4):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 0 and out["facts"]["recommendation"]["provisional"] is False


def test_a_baseline_overload_is_not_a_missing_load(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "before", 32, r, 400, 9.0, 2000, err="12.50")  # overloaded at 32
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
        write_run(tmp_path, "cand", 32, r, 300, 9.0, 2400)
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8,32",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    assert "load_missing" not in [e["code"] for e in out["errors"]]
    assert out["facts"]["recommendation"]["provisional"] is False


def test_a_case_with_only_empty_run_folders_is_a_missing_load(tmp_path: Path) -> None:
    # collect() counts no run of "cand" (its folders hold no export), so it is not a case; the
    # requested load is still missing for it and the baseline pick stays provisional
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        (tmp_path / "cand" / f"c8-r{r}").mkdir(parents=True)
        (tmp_path / "skipped" / f"c8-r{r}").mkdir(parents=True)  # excluded: never a gap
        (tmp_path / "cand" / "notes").mkdir(exist_ok=True)  # not a run folder
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000", "--exclude", "skipped")  # fmt: skip
    assert code == 3 and out["state"] == "incomplete"
    gaps = [e["message"] for e in out["errors"] if e["code"] == "load_missing"]
    assert gaps == ["cand has no counted runs at 8 concurrent"]
    rec = out["facts"]["recommendation"]
    assert rec["case"] == "before" and rec["provisional"] is True


def test_a_case_with_only_overloaded_runs_is_not_a_missing_load(tmp_path: Path) -> None:
    # every run of "cand" failed requests: a measured overload, not a gap, as before
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800, err="4.5")
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    assert code == 0 and out["ok"] is True
    assert {e["code"] for e in out["errors"]} == {"run_had_request_errors"}
    assert out["facts"]["recommendation"]["provisional"] is False


def test_without_loads_a_case_with_only_empty_run_folders_is_a_gap(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
        (tmp_path / "empty" / f"c8-r{r}").mkdir(parents=True)  # measured nothing readable
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3 and any("empty has no counted runs" in e["message"] for e in out["errors"])
    assert out["facts"]["recommendation"]["provisional"] is True
    # the same case excluded, or a complete set without it, is not a gap
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000",
                    "--exclude", "empty")  # fmt: skip
    assert code == 0, out


def test_a_stray_failed_run_makes_the_report_provisional(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    write_run(tmp_path, "cand", 8, 4, 50, 8.0, 1800, err="1.5")  # one failed run among good ones
    report = tmp_path / "final" / "before_after.md"
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000",
                    "--report", str(report))  # fmt: skip
    rec = out["facts"]["recommendation"]
    assert code == 3 and rec["provisional"] is True
    assert any("cand/c8-r4" in m for m in rec["provisional_reasons"])
    assert "Provisional" in report.read_text()  # the report never shows it as final


def test_without_loads_each_load_needs_counted_runs_or_an_overload(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800, err="2.0")  # overloaded at 8
    (tmp_path / "cand" / "c32-r1").mkdir()  # an empty folder at 32: no evidence at all
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    msgs = [e["message"] for e in out["errors"] if e["code"] == "load_missing"]
    assert msgs == ["cand has no counted runs at 32 concurrent"]
    assert code == 3 and out["facts"]["recommendation"]["provisional"] is True


def test_an_overloaded_load_is_the_first_failing_one(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "before", 32, r, 100, 8.0, 900, err="9.0")  # every request batch failed
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8,32",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    b = out["facts"]["sla_boundary"]["before"]
    assert b["max_concurrency"] == 8 and b["first_failing"] == 32 and b["unmeasured"] == []


def test_one_failed_attempt_next_to_empty_ones_is_not_an_overload(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "before", 32, r, 400, 9.0, 2000)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    write_run(tmp_path, "cand", 32, 1, 300, 9.0, 2400, err="3.0")  # failed requests
    for r in (2, 3):
        (tmp_path / "cand" / f"c32-r{r}").mkdir()  # no usable result: unresolved
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8,32",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    assert code == 3 and out["facts"]["recommendation"]["provisional"] is True
    assert any("cand has no counted runs at 32" in e["message"] for e in out["errors"])


def test_an_unfinished_search_load_outside_loads_is_a_gap(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    (tmp_path / "cand" / "c32-r1").mkdir()  # an SLA-search attempt that never finished
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    msgs = [e["message"] for e in out["errors"] if e["code"] == "load_missing"]
    assert msgs == ["cand has no counted runs at 32 concurrent"]  # reported once
    assert code == 3 and out["facts"]["recommendation"]["provisional"] is True


def test_a_negative_error_rate_is_unreadable_not_an_overload(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800, err="-1")
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    assert out["facts"]["rejected_runs"] == []  # never a measured overload
    assert len(out["facts"]["runs_without_error_rate"]) == 3
    assert code == 3 and out["facts"]["recommendation"]["provisional"] is True


def test_an_empty_repeat_next_to_counted_runs_leaves_the_pick_provisional(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    (tmp_path / "before" / "c8-r4").mkdir()  # an attempt that never wrote an export
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3 and out["state"] == "incomplete"
    assert [e["message"] for e in out["errors"] if e["code"] == "run_unfinished"] == [
        "before/c8-r4 has no readable export and was not counted"
    ]
    assert out["facts"]["recommendation"]["provisional"] is True


def test_an_export_with_only_a_clean_error_row_is_unfinished(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    d = tmp_path / "before" / "c8-r4"
    d.mkdir()
    (d / "profile_export_aiperf.csv").write_text("Metric,Value\nRequest Error Rate (%),0.00\n")
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3 and out["state"] == "incomplete"
    assert [e["message"] for e in out["errors"] if e["code"] == "run_unfinished"] == [
        "before/c8-r4 has no readable export and was not counted"
    ]
    assert out["facts"]["recommendation"]["provisional"] is True


def test_itl_missing_from_one_of_four_repeats_cannot_settle_a_tie(tmp_path: Path) -> None:
    # equal throughput, so ITL p99 would decide; three ITL values meet --min-repeats, but the
    # fourth counted repeat could change the winner: the tie is not settled on the partial mean
    # (it would pick cand, 8.0 against 9.0), and the pick stays provisional
    for r in (1, 2, 3, 4):
        write_run(tmp_path, "before", 8, r, 100, 9.0, 900)
        write_run(tmp_path, "cand", 8, r, 100, float("inf") if r == 4 else 8.0, 900)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert out["facts"]["cases"]["cand"]["8"]["itl_p99"]["n"] == 3
    rec = out["facts"]["recommendation"]
    assert rec["unsettled_tie"] == ["before", "cand"]
    assert rec["case"] == "before"  # the name order, not cand's partial ITL mean
    assert code == 3 and out["state"] == "incomplete"
    assert [e["code"] for e in out["errors"]] == ["tie_unsettled"]  # ITL blocks nothing else
    assert rec["provisional"] is True
    assert any("ITL p99" in m for m in rec["provisional_reasons"])
    # the same tie with ITL p99 in every repeat of both is settled by it, and final
    edit_csv(tmp_path, "cand/c8-r4", "Inter Token Latency (ms),inf", "Inter Token Latency (ms),8.0")
    edit_csv(tmp_path, "cand/c8-r4", ",,,,,,,,,,,,inf,", ",,,,,,,,,,,,8.8,")
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    rec = out["facts"]["recommendation"]
    assert code == 0 and rec["case"] == "cand" and rec["provisional"] is False
    assert "unsettled_tie" not in rec


def without_itl(root: Path, run: str) -> None:
    """Drop the ITL row, as AIPerf does for outputs of one token."""
    csv = root / run / "profile_export_aiperf.csv"
    csv.write_text("".join(x for x in csv.read_text().splitlines(True) if "Inter Token" not in x))


def test_runs_without_any_itl_give_a_final_pick_when_nothing_ties(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
        without_itl(tmp_path, f"before/c8-r{r}")
        without_itl(tmp_path, f"cand/c8-r{r}")
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000",
                    "--loads", "8")  # fmt: skip
    assert code == 0 and out["state"] == "compared", out["errors"]
    rec = out["facts"]["recommendation"]
    assert rec["case"] == "cand" and rec["provisional"] is False
    assert out["facts"]["partly_measured"] == []  # never measured at all: nothing to list


def test_a_search_point_with_a_failed_repeat_is_a_measured_miss(tmp_path: Path) -> None:
    # the runner stops an SLA-search point at its first repeat with failed requests and calls it
    # a miss; compare must agree, or the case stays provisional with nothing left to measure
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
        write_run(tmp_path, "cand", 16, r, 60, 8.0, 2400)
    write_run(tmp_path, "cand", 32, 1, 70, 8.0, 3000)  # counted
    write_run(tmp_path, "cand", 32, 2, 70, 8.0, 3000, err="2.5")  # failed requests: stop
    args = (str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000", "--loads", "8")
    code, out = run(*args)
    assert code == 0 and out["state"] == "compared", out["errors"]
    b = out["facts"]["sla_boundary"]["cand"]
    assert b["max_concurrency"] == 16 and b["first_failing"] == 32 and b["unmeasured"] == []
    rec = out["facts"]["recommendation"]
    assert rec["case"] == "cand" and rec["provisional"] is False
    # at a fixed load the same pattern is still a stray failure to repeat first
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 32, r, 100, 8.0, 900)
    write_run(tmp_path, "cand", 32, 3, 70, 8.0, 3000)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000",
                    "--loads", "8,32")  # fmt: skip
    assert code == 3 and out["facts"]["recommendation"]["provisional"] is True
    assert any("cand/c32-r2" in m and "repeat it" in m
               for m in out["facts"]["recommendation"]["provisional_reasons"])  # fmt: skip


def test_a_fixed_load_with_unstarted_repeats_is_not_an_overload(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        for conc in (8, 32):
            write_run(tmp_path, "before", conc, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    # one failed repeat at 32, the other two never started: no measured overload yet
    write_run(tmp_path, "cand", 32, 1, 50, 8.0, 1800, err="4.5")
    args = (str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000", "--loads", "8,32")
    code, out = run(*args)
    assert code == 3 and out["state"] == "incomplete"
    assert "cand has no counted runs at 32 concurrent" in [
        e["message"] for e in out["errors"] if e["code"] == "load_missing"
    ]
    assert out["facts"]["recommendation"]["provisional"] is True
    # every repeat at 32 failed requests: now a measured overload, and the comparison is final
    for r in (2, 3):
        write_run(tmp_path, "cand", 32, r, 50, 8.0, 1800, err="4.5")
    # an SLA-search point (not in --loads) stops at its first failed repeat: still a result
    write_run(tmp_path, "before", 64, 1, 100, 8.0, 900, err="9.0")
    code, out = run(*args)
    assert code == 0 and out["state"] == "compared", out["errors"]
    assert out["facts"]["recommendation"]["provisional"] is False


def test_a_load_with_no_counted_run_is_unmeasured_in_the_boundary(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    (tmp_path / "before" / "c32-r1").mkdir()  # an attempt that wrote nothing
    write_run(tmp_path, "before", 64, 1, 100, 8.0, 900, err="9.0")  # search stop: overloaded
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000",
                    "--loads", "8")  # fmt: skip
    b = out["facts"]["sla_boundary"]["before"]
    assert b["max_concurrency"] == 8 and b["first_failing"] == 64
    assert b["unmeasured"] == [32]
    assert code == 3 and out["facts"]["recommendation"]["provisional"] is True
    # without --loads nothing marks 64 as a search point: one failed repeat there is unmeasured
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    b = out["facts"]["sla_boundary"]["before"]
    assert b["first_failing"] is None and b["unmeasured"] == [32, 64]


def test_an_unreadable_export_is_an_unfinished_run_not_a_crash(tmp_path: Path) -> None:
    for r in (1, 2, 3, 4):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
    # the fourth export is not UTF-8: the three good repeats still count, the fourth is a gap
    (tmp_path / "before" / "c8-r4" / "profile_export_aiperf.csv").write_bytes(b"\xff\xfe\x00bad")
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3 and out["state"] == "incomplete"
    assert out["facts"]["cases"]["before"]["8"]["repeats"] == 3
    assert [e["message"] for e in out["errors"] if e["code"] == "run_unfinished"] == [
        "before/c8-r4 has no readable export and was not counted"
    ]
    assert out["facts"]["recommendation"]["provisional"] is True


def test_without_loads_one_failed_repeat_is_not_a_search_miss(tmp_path: Path) -> None:
    # without --loads nothing shows that 32 was an SLA-search point (which stops at its first
    # failed repeat): one failed repeat with the others never started is a gap, never a final
    # pick on partial evidence
    for r in (1, 2, 3):
        for conc in (8, 32):
            write_run(tmp_path, "before", conc, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    write_run(tmp_path, "cand", 32, 1, 50, 8.0, 1800, err="4.5")
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 3 and out["state"] == "incomplete"
    assert "cand has no counted runs at 32 concurrent" in [
        e["message"] for e in out["errors"] if e["code"] == "load_missing"
    ]
    assert out["facts"]["recommendation"]["provisional"] is True
    assert out["facts"]["sla_boundary"]["cand"]["unmeasured"] == [32]
    # --loads naming 8 only marks 32 as a search point: the failed repeat is its measured miss
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000",
                    "--loads", "8")  # fmt: skip
    assert code == 0 and out["facts"]["recommendation"]["provisional"] is False
    assert out["facts"]["sla_boundary"]["cand"]["first_failing"] == 32
    # without --loads, --min-repeats failed repeats are a measured overload again
    for r in (2, 3):
        write_run(tmp_path, "cand", 32, r, 50, 8.0, 1800, err="4.5")
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000")
    assert code == 0 and out["facts"]["recommendation"]["provisional"] is False
    assert out["facts"]["sla_boundary"]["cand"]["first_failing"] == 32


def test_a_case_with_no_passing_load_is_in_the_boundary(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800, err="4.5")  # every repeat failed
    report = tmp_path / "final" / "before_after.md"
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000", "--report", str(report))  # fmt: skip
    assert code == 0, out["errors"]
    assert out["facts"]["sla_boundary"]["cand"] == {
        "max_concurrency": None, "otps_per_gpu": None, "first_failing": 8, "unmeasured": []
    }  # fmt: skip
    assert "| cand | none | - | 8 |" in report.read_text()


def test_a_load_passing_again_above_a_miss_never_lifts_the_boundary(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        for conc, ttft in ((8, 400), (16, 600), (32, 400)):  # p99 = 2x: 16 misses, 32 passes
            write_run(tmp_path, "before", conc, r, ttft, 8.0, 900)
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8,16,32",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    assert code == 0, out["errors"]
    b = out["facts"]["sla_boundary"]["before"]
    assert (b["max_concurrency"], b["first_failing"], b["unmeasured"]) == (8, 16, [])


def test_a_baseline_overloaded_at_every_load_is_a_result_not_missing(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 400, 8.0, 900, err="4.5")  # every repeat failed
        write_run(tmp_path, "cand", 8, r, 400, 8.0, 1800)
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    codes = [e["code"] for e in out["errors"]]
    assert code == 0 and "baseline_missing" not in codes and "load_missing" not in codes, out
    assert out["facts"]["sla_boundary"]["before"]["first_failing"] == 8
    assert out["facts"]["sla_boundary"]["cand"]["max_concurrency"] == 8
    # one stray failed repeat is no overload: still not final, but never "baseline missing"
    for r in (2, 3):
        shutil.rmtree(tmp_path / "before" / f"c8-r{r}")
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    codes = [e["code"] for e in out["errors"]]
    assert code != 0 and "baseline_missing" not in codes and "load_missing" in codes, out


def test_an_overflowing_change_is_null_never_infinity(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 0.001)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1e308)  # finite, but the ratio overflows
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--baseline", "before"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
    )  # fmt: skip

    def strict(name: str) -> None:
        raise AssertionError(f"{name} in the JSON")

    out = json.loads(proc.stdout, parse_constant=strict)
    otps = [c for c in out["facts"]["comparisons"] if c["metric"] == "otps_avg"]
    assert otps and otps[0]["change_pct"] is None and otps[0]["verdict"] == "better"


def test_a_baseline_without_error_rates_says_which_repeats_to_redo(tmp_path: Path) -> None:
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900, err=None)  # no error-rate row
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    code, out = run(str(tmp_path), "--baseline", "before")
    assert code == 3 and [e["code"] for e in out["errors"]] == ["baseline_missing"]
    unread = ["before/c8-r1", "before/c8-r2", "before/c8-r3"]
    assert out["facts"]["runs_without_error_rate"] == unread
    assert all(r in out["errors"][0]["message"] and r in out["errors"][0]["fix"] for r in unread)


def test_an_overload_only_case_from_another_setup_is_refused(tmp_path: Path) -> None:
    # every run of "cand" failed requests, so it has no counted run; its overload is still
    # evidence, and evidence from another client is never compared
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800, err="4.5")
    for case, client in (("before", "a"), ("cand", "b")):
        (tmp_path / case / "manifest.json").write_text(json.dumps({"client": client}))
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    assert code == 5 and [e["code"] for e in out["errors"]] == ["manifest_mismatch"]
    # the same setup: the overload is a result, as before
    (tmp_path / "cand" / "manifest.json").write_text(json.dumps({"client": "a"}))
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8",
                    "--sla", "ttft_p99<=1000")  # fmt: skip
    assert code == 0 and out["state"] == "compared"


def test_without_sla_a_load_outside_loads_is_never_a_search_miss(tmp_path: Path) -> None:
    # no --sla, no search: one failed repeat at 32 next to counted ones is a stray failure
    for r in (1, 2, 3):
        for conc in (8, 32):
            write_run(tmp_path, "before", conc, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
    write_run(tmp_path, "cand", 32, 1, 50, 8.0, 1800)
    write_run(tmp_path, "cand", 32, 2, 50, 8.0, 1800, err="4.5")
    code, out = run(str(tmp_path), "--baseline", "before", "--loads", "8")
    assert code == 3 and out["state"] == "incomplete"


def test_a_search_miss_is_final_even_with_a_repeat_without_an_error_rate(tmp_path: Path) -> None:
    # the search stopped at 32 on a repeat with failed requests (a measured miss the runner
    # never measures again); another repeat there wrote no error-rate row: listed, never a gap
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 8, r, 100, 8.0, 900)
        write_run(tmp_path, "cand", 8, r, 50, 8.0, 1800)
        write_run(tmp_path, "cand", 16, r, 60, 8.0, 2400)
    write_run(tmp_path, "cand", 32, 1, 70, 8.0, 3000, err=None)  # no error-rate row
    write_run(tmp_path, "cand", 32, 2, 70, 8.0, 3000, err="2.5")  # failed requests: stop
    args = (str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000", "--loads", "8")
    code, out = run(*args)
    assert code == 0 and out["state"] == "compared", out["errors"]
    assert "run_error_rate_missing" in [e["code"] for e in out["errors"]]  # still listed
    assert out["facts"]["sla_boundary"]["cand"]["first_failing"] == 32
    assert out["facts"]["recommendation"]["provisional"] is False
    # at a fixed load, and at a search point not measured to miss, it still blocks
    for r in (1, 2, 3):
        write_run(tmp_path, "before", 32, r, 100, 8.0, 900)
    write_run(tmp_path, "cand", 32, 3, 70, 8.0, 3000)
    code, out = run(str(tmp_path), "--baseline", "before", "--sla", "ttft_p99<=1000",
                    "--loads", "8,32")  # fmt: skip
    assert code == 3 and any(
        "cand/c32-r1" in m for m in out["facts"]["recommendation"]["provisional_reasons"]
    )
    write_run(tmp_path, "cand", 64, 1, 70, 8.0, 3000, err=None)  # an unmissed search point
    code, out = run(*args)
    assert code == 3 and any(
        "cand/c64-r1" in m for m in out["facts"]["recommendation"]["provisional_reasons"]
    )
