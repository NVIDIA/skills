#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nb_accuracy: accuracy mode. Plans, runs and checks NeMo Gym test sets against one endpoint.

NeMo Gym needs Python 3.13.14 or newer, so it runs in its own pinned container (Docker only).
For each test set the script runs `gym eval run`, then reads the rollout lines and the aggregate
metrics file and checks them: the counts must agree, and answers cut off by the token limit
must stay rare. It turns the rewards into one score per set with its 95% range and an items
hash, so two runs can be compared item by item. A rerun with the same plan id skips test sets
that already finished.
Run it to print the defaults: python3 scripts/nb_accuracy.py --defaults.
Exit codes: 0 ok, 2 bad input. Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_common
import nb_stats
from nb_common import ContractParser, emit, error

# test set name -> default item limit (None runs every item)
SETS: dict[str, int | None] = {
    "gsm8k": None,
    "ifeval": None,
    "mmlu_pro": 2000,
    "labbench2_vlm": None,
    "gdp_pdf": None,
}
TEXT_SETS = ("gsm8k", "ifeval", "mmlu_pro")
# Test sets that read images. Gym grades both with a separate LLM judge: a second model that
# sees the prompts, the images and the answers, so the user must name it. hle_diamond is left
# out on purpose: Gym lets the model under test judge itself, which is unfair between endpoints.
VISION_SETS = ("labbench2_vlm", "gdp_pdf")
JUDGED_SETS = frozenset(VISION_SETS)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 32
TRUNCATION_LIMIT = 0.05  # more cut-off answers than this share make a score invalid
PINNED = re.compile(r"@sha256:[0-9a-f]{64}$")
ROLLOUTS = "rollouts.jsonl"
AGGREGATE = "rollouts_aggregate_metrics.json"
# fields that must match for two runs to be compared
MATCH_FIELDS = (
    "gym_image",
    "sets",
    "limits",
    "repeats",
    "temperature",
    "max_output_tokens",
    "judge",
)


def is_pinned(image: str) -> bool:
    """An image counts as pinned only with a sha256 digest, never a moving tag."""
    return bool(PINNED.search(image))


def build_plan(
    *,
    endpoint_v1: str,
    model: str,
    sets: list[str],
    limit: int | None,
    full: bool,
    repeats: int,
    temperature: float,
    max_tokens: int,
    key_env: str | None,
    image: str,
    judge: dict[str, str] | None = None,
    judge_key_env: str | None = None,
) -> dict[str, Any]:
    """Everything an accuracy run will do. Its hash is the plan id the user approves.

    `judge` is {"endpoint": the /v1 URL, "model": name} for the sets that need an LLM judge.
    """
    limits: dict[str, int | None] = {}
    for name in sets:
        limits[name] = None if full else (limit if limit is not None else SETS[name])
    return {
        "mode": "accuracy",
        "endpoint": endpoint_v1,
        "model": model,
        "sets": sets,
        "limits": limits,
        "repeats": repeats,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "concurrency": DEFAULT_CONCURRENCY,
        "gym_image": image,
        "key_env": key_env,
        "judge": judge,
        "judge_key_env": judge_key_env,
    }


def gym_argv(plan: dict[str, Any], name: str, set_dir: Path, container: str) -> list[str]:
    """The docker command that runs one test set. The key is a reference, never a value."""
    argv = ["docker", "run", "--rm", "--name", container, "--network", "host"]
    argv += ["-v", f"{set_dir.resolve()}:/out", "-e", "HF_TOKEN"]
    for variable in (plan["key_env"], plan.get("judge_key_env")):
        if variable:
            argv += ["-e", variable]
    key = (
        f"+policy_api_key=${{oc.env:{plan['key_env']}}}"
        if plan["key_env"]
        else "+policy_api_key=EMPTY"
    )
    argv += [plan["gym_image"], "eval", "run", "--benchmark", name]
    argv += ["--model", plan["model"], "--model-url", plan["endpoint"], key]
    judge = plan.get("judge") if name in JUDGED_SETS else None
    if judge:
        judge_key = (
            f"+judge_api_key=${{oc.env:{plan['judge_key_env']}}}"
            if plan.get("judge_key_env")
            else "+judge_api_key=EMPTY"
        )
        argv += [
            f"+judge_base_url={judge['endpoint']}",
            f"+judge_model_name={judge['model']}",
            judge_key,
        ]
    limit = plan["limits"][name]
    if limit is not None:
        argv += ["--limit", str(limit)]
    argv += ["--num-repeats", str(plan["repeats"])]
    argv += ["--temperature", str(plan["temperature"])]
    argv += ["--max-output-tokens", str(plan["max_output_tokens"])]
    argv += ["--concurrency", str(plan["concurrency"]), "--output", f"/out/{ROLLOUTS}"]
    return argv


def read_rollouts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("a rollout line is not a JSON object")
            rows.append(row)
    return rows


def is_truncated(row: dict[str, Any]) -> bool:
    """True when the model hit the output token limit before it finished."""
    response = row.get("response")
    if not isinstance(response, dict):
        return False
    details = response.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    return reason == "max_output_tokens"


def rewards_by_task(rows: list[dict[str, Any]]) -> dict[int, float]:
    """One reward per task: the mean over its repeats. Raises when a field is missing."""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if "_ng_task_index" not in row or "reward" not in row:
            raise ValueError("a rollout has no _ng_task_index or reward")
        grouped.setdefault(int(row["_ng_task_index"]), []).append(float(row["reward"]))
    return {task: sum(vals) / len(vals) for task, vals in sorted(grouped.items())}


def items_hash(rows: list[dict[str, Any]]) -> str:
    """A hash of which test items ran, so two runs can be paired item by item."""
    seen: dict[int, Any] = {}
    for row in rows:
        task = int(row["_ng_task_index"])
        params = row.get("responses_create_params")
        seen.setdefault(task, params.get("input") if isinstance(params, dict) else None)
    return nb_common.sha256_text(nb_common.canonical_json(sorted(seen.items())))


def check_results(
    rows: list[dict[str, Any]], aggregate: Any, limit: int | None, repeats: int
) -> tuple[str, str]:
    """Judge one test set's output: ('ok' | 'tool_failed' | 'answers_truncated', why)."""
    if not rows:
        return "tool_failed", "no rollout lines"
    try:
        rewards = rewards_by_task(rows)
    except ValueError as exc:
        return "tool_failed", str(exc)
    entry = aggregate[0] if isinstance(aggregate, list) and aggregate else None
    metrics = entry.get("agent_metrics") if isinstance(entry, dict) else None
    if not isinstance(metrics, dict) or "task_count" not in metrics:
        return "tool_failed", "the aggregate metrics file has no task_count"
    task_count = metrics["task_count"]
    if type(task_count) not in (int, float):
        return "tool_failed", "task_count must be a numeric count"
    if task_count != len(rewards):
        return "tool_failed", f"task_count {task_count} but {len(rewards)} tasks ran"
    if limit is not None and len(rewards) > limit:
        return "tool_failed", f"{len(rewards)} tasks ran, more than the limit {limit}"
    if len(rows) != len(rewards) * repeats:
        return "tool_failed", f"{len(rows)} rollouts for {len(rewards)} tasks x {repeats} repeats"
    rollout_count = metrics.get("rollout_count", len(rows))
    if type(rollout_count) not in (int, float) or rollout_count != len(rows):
        return "tool_failed", "rollout_count does not match the rollout lines"
    cut = sum(is_truncated(r) for r in rows) / len(rows)
    if cut > TRUNCATION_LIMIT:
        return "answers_truncated", f"{100 * cut:.1f}% of answers hit the output token limit"
    return "ok", ""


def _execute_set(
    plan: dict[str, Any], set_dir: Path, name: str
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Run Gym and calculate a score; the caller owns cleanup and publishing."""
    container = f"nim-benchmark-{plan.get('plan_id', 'run')}-{name}"
    try:
        done_proc = nb_common.run_command(
            gym_argv(plan, name, set_dir, container), timeout=6 * 3600
        )
    except subprocess.TimeoutExpired:
        problems = [error(
            "tool_failed", f"{name}: NeMo Gym timed out",
            "Check the Gym client and endpoint health before retrying.",
        )]
        try:
            stopped = nb_common.run_command(["docker", "rm", "-f", container], timeout=60).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            stopped = False
        if not stopped:
            # Internal control code: run_plan emits tool_failed and halts further load.
            problems.append(error(
                "client_not_stopped", f"{name}: Gym may still run in container {container}",
                f"Stop benchmark container {container} before any rerun.",
            ))
        return None, problems
    nb_common.write_text(set_dir / "gym.log", done_proc.stdout + done_proc.stderr)
    rollouts, aggregate = set_dir / ROLLOUTS, set_dir / AGGREGATE
    log = f"Read {set_dir}/gym.log"
    if not rollouts.exists() or not aggregate.exists():
        return None, [
            error("tool_failed", f"{name}: NeMo Gym wrote no rollouts or metrics", log)
        ]
    try:
        rows = read_rollouts(rollouts)
        verdict, why = check_results(
            rows, nb_common.read_json(aggregate), plan["limits"][name], plan["repeats"]
        )
    except ValueError as exc:
        return None, [error("tool_failed", f"{name}: {exc}", log)]
    if verdict != "ok":
        return None, [error(verdict, f"{name}: {why}", log)]
    rewards = rewards_by_task(rows)
    summary = {
        "plan_id": plan.get("plan_id"),
        **nb_stats.score_summary(list(rewards.values())),
        "truncated_share": round(sum(is_truncated(r) for r in rows) / len(rows), 4),
        "items_hash": items_hash(rows),
        "rewards": [rewards[t] for t in sorted(rewards)],
    }
    return summary, []


def run_set(
    plan: dict[str, Any], out: Path, name: str
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Resume or run a set, publishing scores only after privacy cleanup succeeds."""
    set_dir = out / "accuracy" / name
    scores = set_dir / "scores.json"
    fix = f"Restore write/delete access to {set_dir}, remove leftover raw files, and retry {name}."
    summary: dict[str, Any] | None = None
    problems: list[dict[str, str]] = []
    try:
        set_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None, [error("tool_failed", f"{name}: cannot create the output folder", fix)]
    try:
        done = nb_common.read_json(scores) if scores.exists() else None
        if (
            isinstance(done, dict)
            and done.get("plan_id") == plan.get("plan_id")
            and all(
                type(done.get(key)) in (int, float) and abs(done[key]) < float("inf")
                for key in ("items", "score", "low", "high", "truncated_share")
            )
            and isinstance(done.get("items_hash"), str) and done["items_hash"]
            and isinstance(done.get("rewards"), list) and done["rewards"]
            and all(type(v) in (int, float) and 0 <= v <= 1 for v in done["rewards"])
            and done["items"] == len(done["rewards"])
            and 0 <= done["truncated_share"] <= 1
        ):
            summary = done
    except (OSError, ValueError):
        pass  # unreadable or malformed cached output must be cleared before a new attempt
    cached = summary is not None
    if not cached:
        for filename in (ROLLOUTS, AGGREGATE, "scores.json", "gym.log"):
            try:
                (set_dir / filename).unlink(missing_ok=True)
            except OSError:
                problems.append(error("tool_failed", f"{name}: initial cleanup failed for {filename}", fix))
        if problems:
            return None, problems
    try:
        if not cached:
            summary, problems = _execute_set(plan, set_dir, name)
    except (OSError, subprocess.TimeoutExpired) as exc:
        problems.append(error(
            "tool_failed", f"{name}: cannot run or save Gym results ({type(exc).__name__})",
            f"Check the Docker client and output-folder permissions for {set_dir}, then retry {name}.",
        ))
    finally:
        if name in VISION_SETS:
            try:
                (set_dir / ROLLOUTS).unlink(missing_ok=True)
            except OSError:
                problems.append(error("tool_failed", f"{name}: final rollout cleanup failed", fix))
    if problems:
        return None, problems
    if summary is not None and not cached:
        try:
            nb_common.write_json(scores, summary)
        except OSError:
            return None, [error("tool_failed", f"{name}: cannot write scores.json", fix)]
    return summary, []


def run_plan(plan: dict[str, Any], out: Path) -> dict[str, Any]:
    """Run sets independently, but halt if a client may still be sending load."""
    results: dict[str, Any] = {}
    problems: list[dict[str, str]] = []
    client_not_stopped = False
    for name in plan["sets"]:
        summary, errors = run_set(plan, out, name)
        client_not_stopped = any(e["code"] == "client_not_stopped" for e in errors)
        problems += [
            {**e, "code": "tool_failed"} if e["code"] == "client_not_stopped" else e
            for e in errors
        ]
        if summary is not None:
            results[name] = {k: v for k, v in summary.items() if k != "rewards"}
        if client_not_stopped:
            break
    return {"sets": results, "problems": problems, "client_not_stopped": client_not_stopped}


def main(argv: list[str] | None = None) -> None:
    parser = ContractParser(
        description="Accuracy mode helpers for nim-benchmark. Run it to print the defaults.",
        epilog="Example: python3 scripts/nb_accuracy.py --defaults",
    )
    parser.add_argument("--defaults", action="store_true", help="print the default test sets")
    parser.parse_args(argv)
    emit(
        True,
        "defaults",
        "done",
        {
            "sets": list(TEXT_SETS),
            "vision_sets": list(VISION_SETS),
            "limits": SETS,
            "temperature": DEFAULT_TEMPERATURE,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
            "pass_margin_points": nb_stats.DEFAULT_MARGIN_POINTS,
        },
    )


if __name__ == "__main__":
    main()
