# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nb_compare.py: baseline matching, verdicts and the report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nb_accuracy  # noqa: E402
import nb_common  # noqa: E402
import nb_compare  # noqa: E402
import nb_perf  # noqa: E402

PERF_SECTION: dict[str, Any] = {
    "aiperf_version": "0.13.0",
    "client_image": "python:3.12-slim",
    "isl": 1024,
    "osl": 256,
    "loads": [8],
    "repeats": 3,
    "seeds": [801, 802, 803],
    "tokenizer": "tok",
    "requests_per_load": {"8": 80},
    "extra_inputs": {"ignore_eos": True},
}
ACC_SECTION: dict[str, Any] = {
    "gym_image": "img@sha256:" + "cd" * 32,
    "sets": ["gsm8k"],
    "limits": {"gsm8k": None},
    "repeats": 1,
    "temperature": 0.0,
    "max_output_tokens": 4096,
    "judge": None,
}


def metrics(ttft: list[float], itl: list[float], otps: list[float]) -> dict[str, Any]:
    def stat(values: list[float]) -> dict[str, Any]:
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    return {
        "ttft_avg": stat(ttft),
        "ttft_p99": stat(ttft),
        "itl_avg": stat(itl),
        "itl_p99": stat(itl),
        "otps_avg": stat(otps),
    }


def make_perf_run(
    folder: Path, ttft: list[float], itl: list[float], otps: list[float], **section: Any
) -> Path:
    manifest = {"model": "m", "endpoint": "http://x", "perf": {**PERF_SECTION, **section}}
    nb_common.write_json(folder / "manifest.json", manifest)
    summary = {"loads": {"8": {"repeats": 3, "metrics": metrics(ttft, itl, otps)}}}
    nb_common.write_json(folder / "perf" / "summary.json", {**summary, "incomplete": []})
    return folder


def make_acc_run(
    folder: Path, rewards: list[float], items_hash: str = "h1", **section: Any
) -> Path:
    manifest = {"model": "m", "accuracy": {**ACC_SECTION, **section}}
    nb_common.write_json(folder / "manifest.json", manifest)
    scores = {"items_hash": items_hash, "rewards": rewards, "plan_id": "p"}
    nb_common.write_json(folder / "accuracy" / "gsm8k" / "scores.json", scores)
    entry = {"items": len(rewards), "score": 90.0, "low": 88.0, "high": 92.0}
    nb_common.write_json(folder / "accuracy" / "summary.json", {"sets": {"gsm8k": entry}})
    return folder


def test_manifest_diff_names_only_the_fields_that_differ() -> None:
    a, b = {"x": 1, "y": 2, "z": 3}, {"x": 1, "y": 9, "z": 3}
    assert nb_compare.manifest_diff(a, b, ("x", "y", "z")) == ["y"]
    assert nb_compare.manifest_diff(a, a, ("x", "y", "z")) == []


def test_lower_is_better_follows_the_metric_name() -> None:
    assert nb_compare.lower_is_better("ttft_p99") and nb_compare.lower_is_better("e2e_avg")
    assert not nb_compare.lower_is_better("otps_avg") and not nb_compare.lower_is_better("rps_avg")


def test_every_key_metric_is_a_metric_the_runs_write() -> None:
    written = set(nb_perf.summarize_run(json.loads((FIXTURE).read_text())))
    assert set(nb_compare.KEY_METRICS) <= written


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aiperf_export_1_4.json"


def test_a_faster_candidate_is_better(tmp_path: Path) -> None:
    cand = make_perf_run(tmp_path / "c", [40, 41, 42], [2.0, 2.0, 2.1], [600, 610, 620])
    base = make_perf_run(tmp_path / "b", [80, 81, 82], [4.0, 4.1, 4.2], [300, 305, 310])
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert problems == []
    assert verdicts["perf"]["loads"]["8"]["verdict"] == "better"
    assert verdicts["perf"]["loads"]["8"]["metrics"]["ttft_p99"]["change_pct"] == -49.38


def test_a_slower_candidate_is_worse_and_one_worse_metric_decides(tmp_path: Path) -> None:
    cand = make_perf_run(tmp_path / "c", [40, 41, 42], [2.0, 2.0, 2.1], [300, 305, 310])
    base = make_perf_run(tmp_path / "b", [40, 41, 42], [2.0, 2.0, 2.1], [600, 610, 620])
    verdicts, _ = nb_compare.compare_dirs(cand, base, 1.0)
    assert verdicts["perf"]["loads"]["8"]["verdict"] == "worse"  # only throughput fell


def test_equal_runs_are_the_same(tmp_path: Path) -> None:
    cand = make_perf_run(tmp_path / "c", [40, 41, 42], [2.0, 2.0, 2.1], [600, 610, 620])
    base = make_perf_run(tmp_path / "b", [40, 41, 42], [2.0, 2.0, 2.1], [600, 610, 620])
    verdicts, _ = nb_compare.compare_dirs(cand, base, 1.0)
    assert verdicts["perf"]["loads"]["8"]["verdict"] == "same"


def test_overlapping_repeats_are_within_noise(tmp_path: Path) -> None:
    cand = make_perf_run(tmp_path / "c", [40, 50, 60], [2.0, 2.5, 3.0], [600, 650, 700])
    base = make_perf_run(tmp_path / "b", [45, 55, 62], [2.2, 2.6, 3.1], [590, 640, 690])
    verdicts, _ = nb_compare.compare_dirs(cand, base, 1.0)
    assert verdicts["perf"]["loads"]["8"]["verdict"] == "within_noise"


def test_a_baseline_with_another_setup_is_refused_with_the_field_names(tmp_path: Path) -> None:
    cand = make_perf_run(tmp_path / "c", [40, 41, 42], [2, 2, 2], [600, 610, 620])
    base = make_perf_run(
        tmp_path / "b", [40, 41, 42], [2, 2, 2], [600, 610, 620], aiperf_version="0.12.0", osl=128
    )
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert verdicts == {}
    assert problems[0]["code"] == "baseline_mismatch"
    assert "aiperf_version, osl" in problems[0]["message"]


def test_endpoint_url_and_model_may_differ(tmp_path: Path) -> None:
    cand = make_perf_run(tmp_path / "c", [40, 41, 42], [2, 2, 2], [600, 610, 620])
    base = make_perf_run(tmp_path / "b", [40, 41, 42], [2, 2, 2], [600, 610, 620])
    manifest = json.loads((base / "manifest.json").read_text())
    manifest.update({"endpoint": "http://other", "model": "other-model"})
    (base / "manifest.json").write_text(json.dumps(manifest))
    assert nb_compare.compare_dirs(cand, base, 1.0)[1] == []


def rewards(wrong: int, total: int = 1000) -> list[float]:
    return [0.0 if i < wrong else 1.0 for i in range(total)]


def test_equal_accuracy_passes(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(100))
    base = make_acc_run(tmp_path / "b", rewards(100))
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert problems == [] and verdicts["accuracy"]["sets"]["gsm8k"]["verdict"] == "pass"


def test_a_real_drop_fails(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(130))
    base = make_acc_run(tmp_path / "b", rewards(100))
    sets = nb_compare.compare_dirs(cand, base, 1.0)[0]["accuracy"]["sets"]
    assert sets["gsm8k"]["verdict"] == "fail" and sets["gsm8k"]["difference"] == -3.0


def test_a_small_drop_with_a_wide_range_is_inconclusive(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(108))
    base = make_acc_run(tmp_path / "b", rewards(100))
    sets = nb_compare.compare_dirs(cand, base, 1.0)[0]["accuracy"]["sets"]
    assert sets["gsm8k"]["verdict"] == "inconclusive"
    assert sets["gsm8k"]["low"] < -1.0 < sets["gsm8k"]["high"]


def test_another_margin_changes_the_verdict(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(108))
    base = make_acc_run(tmp_path / "b", rewards(100))
    sets = nb_compare.compare_dirs(cand, base, 3.0)[0]["accuracy"]["sets"]
    assert sets["gsm8k"]["verdict"] == "pass"


def test_different_items_are_refused(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(100), items_hash="h1")
    base = make_acc_run(tmp_path / "b", rewards(100), items_hash="h2")
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert verdicts["accuracy"] == {"sets": {}}
    assert "items_hash of gsm8k" in problems[0]["message"]


def test_another_gym_image_or_temperature_is_refused(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(100))
    base = make_acc_run(tmp_path / "b", rewards(100), temperature=0.7, repeats=4)
    _verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert "repeats, temperature" in problems[0]["message"]
    assert set(nb_accuracy.MATCH_FIELDS) >= {"gym_image", "temperature", "max_output_tokens"}


def test_unreadable_or_unrelated_folders_are_bad_input(tmp_path: Path) -> None:
    _v, problems = nb_compare.compare_dirs(tmp_path / "none", tmp_path / "other", 1.0)
    assert problems[0]["code"] == "bad_input"
    cand = make_perf_run(tmp_path / "c", [1, 2, 3], [1, 2, 3], [1, 2, 3])
    base = make_acc_run(tmp_path / "b", rewards(10))
    _v, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert "share no mode" in problems[0]["message"]


def test_the_report_has_numbers_and_never_the_raw_text(tmp_path: Path) -> None:
    run = make_perf_run(tmp_path / "r", [40, 41, 42], [2, 2, 2], [600, 610, 620])
    make_acc_run(run, rewards(100))
    manifest = json.loads((run / "manifest.json").read_text())
    text = nb_compare.report_md(manifest, run, {})
    assert "## Speed" in text and "## Accuracy" in text and "no baseline" in text
    assert "| 8 | 41.0 |" in text and "| gsm8k | 1000 | 90.0 | 88.0 to 92.0 |" in text
    assert "prompt" not in text.lower() and "answer" not in text.lower()


def test_the_report_says_inconclusive_is_not_proven(tmp_path: Path) -> None:
    run = make_acc_run(tmp_path / "r", rewards(100))
    verdicts = {"accuracy": {"sets": {"gsm8k": {"verdict": "inconclusive"}}}}
    text = nb_compare.report_md({"model": "m"}, run, verdicts)
    assert "inconclusive (not proven)" in text


def test_an_incomplete_load_is_listed_without_a_verdict(tmp_path: Path) -> None:
    run = make_perf_run(tmp_path / "r", [40, 41, 42], [2, 2, 2], [600, 610, 620])
    summary = json.loads((run / "perf" / "summary.json").read_text())
    summary["incomplete"] = [{"load": 32, "counted": 2, "needed": 3}]
    (run / "perf" / "summary.json").write_text(json.dumps(summary))
    text = nb_compare.report_md({"model": "m"}, run, {})
    assert "Load 32 is incomplete: 2 of 3 repeats counted" in text


def test_script_prints_the_key_metrics() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nb_compare.py"), "--metrics"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert json.loads(proc.stdout)["facts"]["key_metrics"] == ["ttft_p99", "itl_p99", "otps_avg"]


def test_a_baseline_graded_by_another_judge_is_refused(tmp_path: Path) -> None:
    judge = {"endpoint": "http://judge:8000/v1", "model": "a"}
    cand = make_acc_run(tmp_path / "c", rewards(100), judge=judge)
    base = make_acc_run(tmp_path / "b", rewards(100), judge={**judge, "model": "b"})
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert verdicts == {} and "judge" in problems[0]["message"]
    same = make_acc_run(tmp_path / "s", rewards(100), judge=judge)
    assert nb_compare.compare_dirs(cand, same, 1.0)[1] == []


def test_runs_with_different_images_are_refused(tmp_path: Path) -> None:
    cand = make_perf_run(
        tmp_path / "c", [40, 41, 42], [2, 2, 2], [600, 610, 620], images_per_request=2
    )
    base = make_perf_run(
        tmp_path / "b", [40, 41, 42], [2, 2, 2], [600, 610, 620], images_per_request=1
    )
    _v, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert "images_per_request" in problems[0]["message"]


def test_a_set_with_no_scores_on_either_side_gets_no_verdict(tmp_path: Path) -> None:
    sets = ["gsm8k", "ifeval"]
    limits = {"gsm8k": None, "ifeval": None}
    cand = make_acc_run(tmp_path / "c", rewards(100), sets=sets, limits=limits)
    base = make_acc_run(tmp_path / "b", rewards(100), sets=sets, limits=limits)
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)  # ifeval failed on both
    assert problems == []
    found = verdicts["accuracy"]["sets"]
    assert found["gsm8k"]["verdict"] == "pass"
    assert found["ifeval"] == {
        "verdict": "no verdict",
        "why": "no scores in the run and baseline",
    }
    (base / "accuracy" / "gsm8k" / "scores.json").unlink()
    found = nb_compare.compare_dirs(cand, base, 1.0)[0]["accuracy"]["sets"]
    assert found["gsm8k"]["why"] == "no scores in the baseline"
    text = nb_compare.report_md({"model": "m"}, cand, {"accuracy": {"sets": found}})
    assert "no verdict: no scores in the baseline" in text
    assert "Test set ifeval has no score in this run. It has no verdict." in text


def test_scores_left_by_an_earlier_plan_are_not_compared(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(100))
    base = make_acc_run(tmp_path / "b", rewards(100))
    manifest = json.loads((cand / "manifest.json").read_text())
    manifest["plan_ids"] = {"accuracy": "newer-plan"}  # the rerun failed and wrote no scores
    (cand / "manifest.json").write_text(json.dumps(manifest))
    found = nb_compare.compare_dirs(cand, base, 1.0)[0]["accuracy"]["sets"]
    assert found["gsm8k"] == {"verdict": "no verdict", "why": "no scores in the run"}


def test_a_set_with_too_few_items_is_inconclusive(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", [1.0])
    base = make_acc_run(tmp_path / "b", [1.0])
    found = nb_compare.compare_dirs(cand, base, 1.0)[0]["accuracy"]["sets"]["gsm8k"]
    assert found["verdict"] == "inconclusive" and found["why"] == "under 30 items"
    text = nb_compare.report_md({"model": "m"}, cand, {"accuracy": {"sets": {"gsm8k": found}}})
    assert "inconclusive (not proven): under 30 items" in text


def test_compare_dirs_can_be_held_to_one_mode(tmp_path: Path) -> None:
    cand = make_acc_run(tmp_path / "c", rewards(100))
    base = make_acc_run(tmp_path / "b", rewards(100), temperature=0.7)
    for folder in (cand, base):
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest["perf"] = PERF_SECTION
        (folder / "manifest.json").write_text(json.dumps(manifest))
        nb_common.write_json(folder / "perf" / "summary.json", {"loads": {}, "incomplete": []})
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0, ("perf",))
    assert problems == [] and set(verdicts) == {"perf"}  # the old accuracy is not compared
    _v, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert problems[0]["code"] == "baseline_mismatch"


@pytest.mark.parametrize("summary", [None, {}, {"sets": {}}, {"sets": None}, {"sets": {"gsm8k": None}}, {"sets": {"gsm8k": {"plan_id": "different"}}}])
def test_a_score_is_not_evidence_without_a_successful_set_summary(tmp_path: Path, summary: Any) -> None:
    """An undeletable cached score cannot count after its set failed privacy cleanup."""
    cand = make_acc_run(tmp_path / "c", [1.0] * 1000)
    base = make_acc_run(tmp_path / "b", [1.0] * 1000)
    path = cand / "accuracy/summary.json"
    if summary is None:
        path.unlink()
    else:
        path.write_text(json.dumps(summary))
    verdicts, _problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert verdicts["accuracy"]["sets"]["gsm8k"]["verdict"] == "no verdict"


INVALID_SCORE_HASHES: list[Any] = [None, 123, [], {}, False, "", " \t\n"]
INVALID_SCORE_REWARDS: list[Any] = [
    None, "1.0", 1, {}, [], [None], ["1.0"], [True], [[1.0]],
    [float("nan")], [float("inf")], [float("-inf")], [-0.1], [1.1],
]


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize(
    "bad_scores",
    [{}, {"items_hash": "h1"}, {"rewards": [1.0]}]
    + [{"items_hash": h, "rewards": [1.0]} for h in INVALID_SCORE_HASHES]
    + [{"items_hash": "h1", "rewards": r} for r in INVALID_SCORE_REWARDS],
)
def test_invalid_score_fields_produce_no_verdict(
    tmp_path: Path, side: str, bad_scores: dict[str, Any]
) -> None:
    """Missing or malformed comparison inputs must not crash or yield an accuracy verdict."""
    cand = make_acc_run(tmp_path / "candidate", [1.0] * 1000)
    base = make_acc_run(tmp_path / "baseline", [1.0] * 1000)
    folder = cand if side == "candidate" else base
    nb_common.write_json(folder / "accuracy/gsm8k/scores.json", {"plan_id": "p", **bad_scores})
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert problems == []
    assert verdicts["accuracy"]["sets"]["gsm8k"]["verdict"] == "no verdict"


@pytest.mark.parametrize("values", [[0, 1] * 500, [0.0, 0.5, 1.0] * 334])
def test_valid_numeric_rewards_remain_comparable(tmp_path: Path, values: list[float]) -> None:
    """Valid binary and partial-credit rewards retain the normal paired comparison."""
    cand = make_acc_run(tmp_path / "candidate", values)
    base = make_acc_run(tmp_path / "baseline", values)
    verdicts, problems = nb_compare.compare_dirs(cand, base, 1.0)
    assert problems == []
    assert verdicts["accuracy"]["sets"]["gsm8k"]["verdict"] == "pass"
