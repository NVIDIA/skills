# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nb_accuracy.py: Gym commands, result checks and whole test sets."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nb_accuracy  # noqa: E402
import nb_common  # noqa: E402

DIGEST = "@sha256:" + "ab" * 32
IMAGE = "nvcr.io/nvidia/nemo-gym" + DIGEST


def make_plan(**changes: Any) -> dict[str, Any]:
    plan = nb_accuracy.build_plan(
        endpoint_v1="http://gpu-01:8000/v1",
        model="qwen7b",
        sets=["gsm8k", "mmlu_pro"],
        limit=None,
        full=False,
        repeats=1,
        temperature=0.0,
        max_tokens=4096,
        key_env=None,
        image=IMAGE,
    )
    plan.update(changes)
    plan["plan_id"] = "abc123"
    return plan


def rollout(task: int, reward: float, repeat: int = 0, cut: bool = False) -> dict[str, Any]:
    """One rollout line in the shape NeMo Gym 0.7.0 writes."""
    row: dict[str, Any] = {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": reward,
        "agent_ref": {"name": "gsm8k_math_with_judge_simple_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": f"question {task}"}]},
        "response": {"status": "completed", "output": [{"type": "message", "text": "private"}]},
    }
    if cut:
        row["response"] = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
    return row


def aggregate(tasks: int, rollouts: int) -> list[dict[str, Any]]:
    return [
        {
            "agent_ref": {"name": "a"},
            "agent_metrics": {"task_count": tasks, "rollout_count": rollouts},
        }
    ]


def write_set(folder: Path, rows: list[dict[str, Any]], agg: Any) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / nb_accuracy.ROLLOUTS).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if agg is not None:
        (folder / nb_accuracy.AGGREGATE).write_text(json.dumps(agg))


class FakeGym:
    """Stands in for run_command: it plays NeMo Gym by writing results into the mounted folder."""

    def __init__(self, make: Callable[[str], tuple[list[dict[str, Any]], Any]]) -> None:
        self.make = make
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: list[str], timeout: float = 0, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            folder = Path(argv[argv.index("-v") + 1].split(":")[0])
            name = argv[argv.index("--benchmark") + 1]
            rows, agg = self.make(name)
            write_set(folder, rows, agg)
        return subprocess.CompletedProcess(argv, 0, "gym output", "")


def items(n: int, wrong: int = 0, cut: int = 0) -> tuple[list[dict[str, Any]], Any]:
    rows = [rollout(i, 0.0 if i < wrong else 1.0, cut=i < cut) for i in range(n)]
    return rows, aggregate(n, n)


def test_only_a_digest_pins_an_image() -> None:
    assert nb_accuracy.is_pinned(IMAGE)
    assert not nb_accuracy.is_pinned("nvcr.io/nvidia/nemo-gym:latest")
    assert not nb_accuracy.is_pinned("nemo-gym@sha256:abc")


def test_default_limits_run_every_item_except_mmlu_pro() -> None:
    assert make_plan()["limits"] == {"gsm8k": None, "mmlu_pro": 2000}
    assert make_plan(limits={"gsm8k": 5})["limits"] == {"gsm8k": 5}
    full = nb_accuracy.build_plan(
        endpoint_v1="u",
        model="m",
        sets=["mmlu_pro"],
        limit=7,
        full=True,
        repeats=1,
        temperature=0.0,
        max_tokens=1,
        key_env=None,
        image=IMAGE,
    )
    assert full["limits"] == {"mmlu_pro": None}
    capped = nb_accuracy.build_plan(
        endpoint_v1="u",
        model="m",
        sets=["gsm8k"],
        limit=7,
        full=False,
        repeats=1,
        temperature=0.0,
        max_tokens=1,
        key_env=None,
        image=IMAGE,
    )
    assert capped["limits"] == {"gsm8k": 7}


def test_gym_command_is_exact_and_holds_no_secret(tmp_path: Path) -> None:
    plan = make_plan(key_env="MY_KEY")
    argv = nb_accuracy.gym_argv(plan, "mmlu_pro", tmp_path, "nm")
    assert argv[:5] == ["docker", "run", "--rm", "--name", "nm"]
    tail = argv[argv.index(IMAGE) :]
    assert tail[:4] == [IMAGE, "eval", "run", "--benchmark"]
    joined = " ".join(tail)
    assert "--model qwen7b" in joined and "--model-url http://gpu-01:8000/v1" in joined
    assert "--limit 2000" in joined and "--temperature 0.0" in joined
    assert "--max-output-tokens 4096" in joined and "--output /out/rollouts.jsonl" in joined
    assert "+policy_api_key=${oc.env:MY_KEY}" in tail
    assert argv[argv.index("-e", argv.index("HF_TOKEN") + 1) + 1] == "MY_KEY"


def test_an_endpoint_without_a_key_gets_the_documented_placeholder(tmp_path: Path) -> None:
    argv = nb_accuracy.gym_argv(make_plan(), "gsm8k", tmp_path, "nm")
    assert "+policy_api_key=EMPTY" in argv and "--limit" not in argv


def test_rewards_are_averaged_over_repeats_per_task() -> None:
    rows = [rollout(0, 1.0, 0), rollout(0, 0.0, 1), rollout(1, 1.0, 0), rollout(1, 1.0, 1)]
    assert nb_accuracy.rewards_by_task(rows) == {0: 0.5, 1: 1.0}
    with pytest.raises(ValueError, match="no _ng_task_index"):
        nb_accuracy.rewards_by_task([{"reward": 1.0}])


def test_truncation_is_read_from_the_incomplete_details() -> None:
    assert nb_accuracy.is_truncated(rollout(0, 1.0, cut=True))
    assert not nb_accuracy.is_truncated(rollout(0, 1.0))
    assert not nb_accuracy.is_truncated({"response": "text"})


def test_items_hash_depends_on_the_items_not_the_answers() -> None:
    a = [rollout(0, 1.0), rollout(1, 0.0)]
    b = [rollout(0, 0.0), rollout(1, 1.0)]
    c = [rollout(0, 1.0), rollout(2, 0.0)]
    assert nb_accuracy.items_hash(a) == nb_accuracy.items_hash(b)
    assert nb_accuracy.items_hash(a) != nb_accuracy.items_hash(c)


@pytest.mark.parametrize(
    ("rows", "agg", "limit", "repeats", "kind", "word"),
    [
        ([], aggregate(0, 0), None, 1, "tool_failed", "no rollout"),
        ([{"reward": 1.0}], aggregate(1, 1), None, 1, "tool_failed", "_ng_task_index"),
        (items(4)[0], None, None, 1, "tool_failed", "no task_count"),
        (items(4)[0], aggregate(5, 4), None, 1, "tool_failed", "task_count 5"),
        (items(4)[0], aggregate(4, 4), 3, 1, "tool_failed", "more than the limit"),
        (items(4)[0], aggregate(4, 4), None, 2, "tool_failed", "4 rollouts for 4 tasks x 2"),
        (items(4)[0], aggregate(4, 9), None, 1, "tool_failed", "rollout_count"),
        (items(20, cut=2)[0], aggregate(20, 20), None, 1, "answers_truncated", "10.0%"),
    ],
)
def test_check_results_refuses_bad_output(
    rows: list[dict[str, Any]], agg: Any, limit: int | None, repeats: int, kind: str, word: str
) -> None:
    got, why = nb_accuracy.check_results(rows, agg, limit, repeats)
    assert got == kind and word in why


def test_check_results_accepts_good_output_and_a_few_cut_answers() -> None:
    rows, agg = items(100, wrong=10, cut=5)
    assert nb_accuracy.check_results(rows, agg, 100, 1) == ("ok", "")


def test_run_set_scores_and_writes_a_compact_scores_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGym(lambda _n: items(100, wrong=10))
    monkeypatch.setattr(nb_common, "run_command", fake)
    summary, problems = nb_accuracy.run_set(make_plan(), tmp_path, "gsm8k")
    assert problems == [] and summary is not None
    assert summary["score"] == 90.0 and summary["items"] == 100.0
    assert summary["low"] < 90.0 < summary["high"]
    saved = json.loads((tmp_path / "accuracy" / "gsm8k" / "scores.json").read_text())
    assert saved["plan_id"] == "abc123" and len(saved["rewards"]) == 100
    assert "private" not in json.dumps(saved) and "question" not in json.dumps(saved)


def test_a_finished_set_with_the_same_plan_id_is_not_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGym(lambda _n: items(20))
    monkeypatch.setattr(nb_common, "run_command", fake)
    nb_accuracy.run_plan(make_plan(sets=["gsm8k"], limits={"gsm8k": None}), tmp_path)
    assert len(fake.calls) == 1
    nb_accuracy.run_plan(make_plan(sets=["gsm8k"], limits={"gsm8k": None}), tmp_path)
    assert len(fake.calls) == 1  # skipped
    other = make_plan(sets=["gsm8k"], limits={"gsm8k": None})
    other["plan_id"] = "different"
    nb_accuracy.run_plan(other, tmp_path)
    assert len(fake.calls) == 2  # a changed plan runs again


def test_one_failed_set_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make(name: str) -> tuple[list[dict[str, Any]], Any]:
        return items(20) if name == "gsm8k" else ([], None)

    monkeypatch.setattr(nb_common, "run_command", FakeGym(make))
    result = nb_accuracy.run_plan(make_plan(), tmp_path)
    assert list(result["sets"]) == ["gsm8k"]
    assert [p["code"] for p in result["problems"]] == ["tool_failed"]
    assert "mmlu_pro" in result["problems"][0]["message"]
    assert "rewards" not in result["sets"]["gsm8k"]


def test_cut_off_answers_make_the_score_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nb_common, "run_command", FakeGym(lambda _n: items(20, cut=10)))
    summary, problems = nb_accuracy.run_set(make_plan(), tmp_path, "gsm8k")
    assert summary is None and problems[0]["code"] == "answers_truncated"
    assert not (tmp_path / "accuracy" / "gsm8k" / "scores.json").exists()


def test_a_timeout_removes_the_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def slow(argv: list[str], timeout: float = 0, env: dict[str, str] | None = None) -> Any:
        calls.append(argv)
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", slow)
    summary, problems = nb_accuracy.run_set(make_plan(), tmp_path, "gsm8k")
    assert summary is None and "timed out" in problems[0]["message"]
    assert calls[-1][:3] == ["docker", "rm", "-f"]


def test_malformed_rollout_lines_are_a_tool_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make(_n: str) -> tuple[list[dict[str, Any]], Any]:
        return items(3)

    class Corrupt(FakeGym):
        def __call__(self, argv: list[str], timeout: float = 0, env: Any = None) -> Any:
            done = super().__call__(argv, timeout, env)
            folder = Path(argv[argv.index("-v") + 1].split(":")[0])
            (folder / nb_accuracy.ROLLOUTS).write_text("[1, 2]\n")
            return done

    monkeypatch.setattr(nb_common, "run_command", Corrupt(make))
    summary, problems = nb_accuracy.run_set(make_plan(), tmp_path, "gsm8k")
    assert summary is None and problems[0]["code"] == "tool_failed"


def test_script_prints_the_defaults() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nb_accuracy.py"), "--defaults"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    facts = json.loads(proc.stdout)["facts"]
    assert facts["sets"] == ["gsm8k", "ifeval", "mmlu_pro"] and facts["pass_margin_points"] == 1.0
    assert facts["vision_sets"] == ["labbench2_vlm", "gdp_pdf"]


JUDGE = {"endpoint": "http://judge:8000/v1", "model": "big-judge"}


def vision_plan(**changes: Any) -> dict[str, Any]:
    plan = make_plan(
        sets=["labbench2_vlm", "gsm8k"],
        limits={"labbench2_vlm": None, "gsm8k": None},
        judge=JUDGE,
        judge_key_env="JUDGE_KEY",
    )
    plan.update(changes)
    return plan


def test_vision_sets_are_known_judged_and_leave_out_the_self_judged_one() -> None:
    assert nb_accuracy.VISION_SETS == ("labbench2_vlm", "gdp_pdf")
    assert {"labbench2_vlm", "gdp_pdf"} == nb_accuracy.JUDGED_SETS
    assert set(nb_accuracy.VISION_SETS) <= set(nb_accuracy.SETS)
    assert "hle_diamond" not in nb_accuracy.SETS
    assert "judge" in nb_accuracy.MATCH_FIELDS


def test_the_judge_is_part_of_the_plan() -> None:
    built = nb_accuracy.build_plan(
        endpoint_v1="u",
        model="m",
        sets=["gdp_pdf"],
        limit=None,
        full=False,
        repeats=1,
        temperature=0.0,
        max_tokens=1,
        key_env=None,
        image=IMAGE,
        judge=JUDGE,
        judge_key_env="JK",
    )
    assert built["judge"] == JUDGE and built["judge_key_env"] == "JK"
    assert built["limits"] == {"gdp_pdf": None}
    assert nb_common.plan_id(built) != nb_common.plan_id({**built, "judge": None})


def test_a_judged_set_gets_the_judge_by_reference_and_a_text_set_does_not(tmp_path: Path) -> None:
    plan = vision_plan()
    vision = nb_accuracy.gym_argv(plan, "labbench2_vlm", tmp_path, "nm")
    assert "+judge_base_url=http://judge:8000/v1" in vision
    assert "+judge_model_name=big-judge" in vision
    assert "+judge_api_key=${oc.env:JUDGE_KEY}" in vision
    assert "JUDGE_KEY" in vision[: vision.index(IMAGE)]  # passed to the container by name
    assert not any("secret" in a.lower() for a in vision)
    text = nb_accuracy.gym_argv(plan, "gsm8k", tmp_path, "nm")
    assert not any(a.startswith("+judge_") for a in text)


def test_a_judge_without_a_key_gets_the_placeholder(tmp_path: Path) -> None:
    argv = nb_accuracy.gym_argv(vision_plan(judge_key_env=None), "labbench2_vlm", tmp_path, "nm")
    assert "+judge_api_key=EMPTY" in argv


def test_vision_rollouts_hold_images_so_they_are_deleted_after_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nb_common, "run_command", FakeGym(lambda _n: items(20, wrong=2)))
    plan = vision_plan()
    result = nb_accuracy.run_plan(plan, tmp_path)
    assert result["problems"] == [] and set(result["sets"]) == {"labbench2_vlm", "gsm8k"}
    vision_dir, text_dir = (tmp_path / "accuracy" / n for n in ("labbench2_vlm", "gsm8k"))
    assert (vision_dir / "scores.json").exists() and (vision_dir / "gym.log").exists()
    assert not (vision_dir / nb_accuracy.ROLLOUTS).exists()
    assert (text_dir / nb_accuracy.ROLLOUTS).exists()


def test_failed_accuracy_rerun_cannot_reuse_old_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a plan must invalidate both scores and the underlying tool output."""
    monkeypatch.setattr(nb_common, "run_command", FakeGym(lambda _n: items(10)))
    assert nb_accuracy.run_set(make_plan(), tmp_path, "gsm8k")[0] is not None
    monkeypatch.setattr(
        nb_common,
        "run_command",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "failed"),
    )
    changed = make_plan()
    changed["plan_id"] = "changed"
    score, problems = nb_accuracy.run_set(changed, tmp_path, "gsm8k")
    assert score is None and problems[0]["code"] == "tool_failed"
    assert not (tmp_path / "accuracy/gsm8k/scores.json").exists()


@pytest.mark.parametrize("failure", ["timeout", "missing_metrics", "malformed", "invalid"])
def test_vision_rollouts_are_deleted_when_scoring_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A failed vision attempt must not retain private prompts and images."""

    def gym(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if argv[:2] == ["docker", "run"]:
            folder = Path(argv[argv.index("-v") + 1].split(":")[0])
            rows, agg = items(10, cut=10 if failure == "invalid" else 0)
            write_set(folder, rows, None if failure == "missing_metrics" else agg)
            if failure == "malformed":
                (folder / nb_accuracy.ROLLOUTS).write_text("not JSON")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", gym)
    score, problems = nb_accuracy.run_set(vision_plan(), tmp_path, "labbench2_vlm")
    assert score is None and problems
    assert not (tmp_path / "accuracy/labbench2_vlm" / nb_accuracy.ROLLOUTS).exists()


@pytest.mark.parametrize("blocked", ["rollouts.jsonl", "rollouts_aggregate_metrics.json", "scores.json", "gym.log"])
def test_initial_cleanup_failure_skips_the_set_but_runs_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: str
) -> None:
    """An unreadable stale artifact must not launch Gym or abort unrelated test sets."""
    fake = FakeGym(lambda _: items(20))
    unlink = Path.unlink
    attempted: list[str] = []

    def remove(path: Path, missing_ok: bool = False) -> None:
        if path.parent.name == "labbench2_vlm":
            attempted.append(path.name)
            if path.name == blocked:
                raise PermissionError("cleanup denied")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(nb_common, "run_command", fake)
    monkeypatch.setattr(Path, "unlink", remove)
    result = nb_accuracy.run_plan(vision_plan(), tmp_path)
    assert set(result["sets"]) == {"gsm8k"}
    assert result["problems"][0]["code"] == "tool_failed"
    assert "cleanup" in result["problems"][0]["message"].lower()
    assert "Restore" in result["problems"][0]["fix"]
    assert str(tmp_path / "accuracy/labbench2_vlm") in result["problems"][0]["fix"]
    assert {"rollouts.jsonl", "rollouts_aggregate_metrics.json", "scores.json", "gym.log"} <= set(attempted)
    assert [a[a.index("--benchmark") + 1] for a in fake.calls if a[:2] == ["docker", "run"]] == ["gsm8k"]


@pytest.mark.parametrize("failure", ["none", "timeout", "truncated"])
def test_final_cleanup_failure_keeps_prior_errors_and_never_publishes_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A failed final privacy cleanup invalidates the set while keeping its original problem."""
    fake = FakeGym(lambda name: items(20, cut=20 if failure == "truncated" and name == "labbench2_vlm" else 0))
    unlink = Path.unlink
    ran_vision = False

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        nonlocal ran_vision
        done = fake(argv, timeout, env)
        if argv[:2] == ["docker", "run"] and argv[argv.index("--benchmark") + 1] == "labbench2_vlm":
            ran_vision = True
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout)
        return done

    def remove(path: Path, missing_ok: bool = False) -> None:
        if ran_vision and path.parent.name == "labbench2_vlm" and path.name == nb_accuracy.ROLLOUTS:
            raise PermissionError("vision data is protected")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(nb_common, "run_command", command)
    monkeypatch.setattr(Path, "unlink", remove)
    result = nb_accuracy.run_plan(vision_plan(), tmp_path)
    assert set(result["sets"]) == {"gsm8k"}
    assert any("cleanup" in p["message"].lower() and "Restore" in p["fix"] for p in result["problems"])
    if failure == "timeout":
        assert any("timed out" in p["message"] for p in result["problems"])
    elif failure == "truncated":
        assert any(p["code"] == "answers_truncated" for p in result["problems"])
    assert not (tmp_path / "accuracy/labbench2_vlm/scores.json").exists()


INVALID_CACHED_SCORE_VALUES: list[Any] = ["missing", None, "", True, {}, [], float("nan"), float("inf")]


@pytest.mark.parametrize(("field", "value"), [
    (field, value)
    for field in ("items", "score", "low", "high", "truncated_share", "items_hash", "rewards")
    for value in INVALID_CACHED_SCORE_VALUES
] + [
    ("rewards", [value]) for value in (None, "1", True, -0.1, 1.1, float("nan"))
] + [("items", 19), ("truncated_share", -0.1), ("truncated_share", 1.1)])
def test_invalid_cached_score_is_cleared_and_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    fake = FakeGym(lambda _: items(20))
    monkeypatch.setattr(nb_common, "run_command", fake)
    plan = make_plan()
    expected, problems = nb_accuracy.run_set(plan, tmp_path, "gsm8k")
    assert expected is not None and not problems
    folder = tmp_path / "accuracy/gsm8k"
    cached = dict(expected)
    if value == "missing":
        del cached[field]
    else:
        cached[field] = value
    nb_common.write_json(folder / "scores.json", cached)
    for filename in (nb_accuracy.ROLLOUTS, nb_accuracy.AGGREGATE, "gym.log"):
        (folder / filename).write_text("stale")

    def rerun(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        assert all(not (folder / f).exists() for f in (
            nb_accuracy.ROLLOUTS, nb_accuracy.AGGREGATE, "gym.log", "scores.json"
        ))
        return fake(argv, timeout, env)

    monkeypatch.setattr(nb_common, "run_command", rerun)
    summary, problems = nb_accuracy.run_set(plan, tmp_path, "gsm8k")
    assert summary == expected and not problems
    assert len(fake.calls) == 2
    assert nb_common.read_json(folder / "scores.json") == expected


def test_valid_cached_vision_score_still_removes_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGym(lambda _: items(20))
    monkeypatch.setattr(nb_common, "run_command", fake)
    plan = vision_plan()
    expected, problems = nb_accuracy.run_set(plan, tmp_path, "labbench2_vlm")
    assert expected is not None and not problems
    raw = tmp_path / "accuracy/labbench2_vlm" / nb_accuracy.ROLLOUTS
    raw.write_text("private leftover")
    summary, problems = nb_accuracy.run_set(plan, tmp_path, "labbench2_vlm")
    assert summary == expected and not problems
    assert len(fake.calls) == 1
    assert not raw.exists()


def test_cached_vision_score_cannot_bypass_failed_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming a cached score must still check for leftover private vision rollouts."""
    fake = FakeGym(lambda _: items(20))
    monkeypatch.setattr(nb_common, "run_command", fake)
    plan = vision_plan()
    assert nb_accuracy.run_set(plan, tmp_path, "labbench2_vlm")[0] is not None
    raw = tmp_path / "accuracy/labbench2_vlm" / nb_accuracy.ROLLOUTS
    raw.write_text("private leftover")
    unlink = Path.unlink

    def remove(path: Path, missing_ok: bool = False) -> None:
        if path == raw:
            raise PermissionError("cleanup denied")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", remove)
    score, problems = nb_accuracy.run_set(plan, tmp_path, "labbench2_vlm")
    assert score is None and problems[0]["code"] == "tool_failed"
    assert "cleanup" in problems[0]["message"].lower()


@pytest.mark.parametrize("stop_failure", ["exit", "oserror", "timeout"])
def test_unconfirmed_gym_stop_preserves_timeout_and_prevents_other_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_failure: str
) -> None:
    """A timed-out Gym container must stop before another test set sends more load."""
    starts: list[list[str]] = []

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if argv[:2] == ["docker", "run"]:
            starts.append(argv)
            raise subprocess.TimeoutExpired(argv, timeout)
        if stop_failure == "oserror":
            raise OSError("Docker is unavailable")
        if stop_failure == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 1, "", "container is still running")

    monkeypatch.setattr(nb_common, "run_command", command)
    result = nb_accuracy.run_plan(make_plan(), tmp_path)
    assert len(starts) == 1 and result["sets"] == {}
    assert any("timed out" in p["message"] for p in result["problems"])
    assert any("may still run" in p["message"] and "Stop" in p["fix"] for p in result["problems"])
    assert all(p["code"] == "tool_failed" for p in result["problems"])


INVALID_GYM_COUNTS: list[Any] = [
    None, "bad", "4", {}, [], True, False,
    float("nan"), float("inf"), float("-inf"), 4.5, -1,
]


@pytest.mark.parametrize("field", ["task_count", "rollout_count"])
@pytest.mark.parametrize("value", INVALID_GYM_COUNTS)
def test_invalid_gym_counts_fail_only_the_affected_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    """Bad aggregate counts must not crash conversion, count the set, or stop the next set."""
    def make(name: str) -> tuple[list[dict[str, Any]], Any]:
        rows, agg = items(1 if isinstance(value, bool) and name == "gsm8k" else 4)
        if name == "gsm8k":
            agg[0]["agent_metrics"][field] = value
        return rows, agg

    fake = FakeGym(make)
    monkeypatch.setattr(nb_common, "run_command", fake)
    result = nb_accuracy.run_plan(make_plan(), tmp_path)
    assert set(result["sets"]) == {"mmlu_pro"}
    assert len(result["problems"]) == 1
    assert result["problems"][0]["code"] == "tool_failed"
    assert field in result["problems"][0]["message"]
    assert not (tmp_path / "accuracy/gsm8k/scores.json").exists()
    assert not result["client_not_stopped"]
    assert [a[a.index("--benchmark") + 1] for a in fake.calls if a[:2] == ["docker", "run"]] == ["gsm8k", "mmlu_pro"]


@pytest.mark.parametrize("as_float", [False, True])
@pytest.mark.parametrize("include_rollout_count", [False, True])
def test_valid_gym_counts_and_optional_rollout_count_still_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    as_float: bool, include_rollout_count: bool
) -> None:
    """Integral numeric counts and the existing absent-rollout-count fallback remain valid."""
    def make(_name: str) -> tuple[list[dict[str, Any]], Any]:
        rows, agg = items(4)
        metrics = agg[0]["agent_metrics"]
        metrics["task_count"] = 4.0 if as_float else 4
        if include_rollout_count:
            metrics["rollout_count"] = 4.0 if as_float else 4
        else:
            del metrics["rollout_count"]
        return rows, agg

    monkeypatch.setattr(nb_common, "run_command", FakeGym(make))
    result = nb_accuracy.run_plan(make_plan(), tmp_path)
    assert set(result["sets"]) == {"gsm8k", "mmlu_pro"}
    assert result["problems"] == []
