# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render DisplayPort campaign reports from recorded state and local logs.

This support module has no CLI and does not launch processes. The supervisor
calls write_report() with a session directory, campaign state, and source links.
Output: atomically updated report.md. File/format errors propagate to the
supervisor so a failed report cannot be mistaken for successful completion.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from storage import read_session
from pathlib import Path

ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")


def final_iteration(attempt: dict) -> int:
    return attempt.get("start_iteration", 0) + attempt["spec"]["max_iterations"] - 1


def trainer_elapsed_at(console: Path, target: int) -> int | None:
    """Read the trainer-reported elapsed time for one numbered PPO update."""
    step = None
    for line in console.read_text(errors="replace").splitlines() if console.is_file() else []:
        match = ITERATION.search(line)
        if match:
            step = int(match.group(1))
        if step == target and "Time elapsed:" in line:
            match = re.search(r"Time elapsed:\s*(\d+):(\d+):(\d+)", line)
            if match:
                hours, minutes, seconds = map(int, match.groups())
                return hours * 3600 + minutes * 60 + seconds
    return None


def wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    if episodes <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    margin = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return center - margin, center + margin


def human_time(value: str) -> str:
    try:
        stamp = datetime.fromisoformat(value).astimezone(timezone.utc)
        return f"{stamp:%b} {stamp.day}, {stamp.year} · {stamp:%H:%M} UTC"
    except (ValueError, TypeError):
        return "unavailable"


def brief(value: object, limit: int = 180) -> str:
    text = " ".join(str(value).replace("|", "/").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def run_label(attempt: dict) -> str:
    number = attempt.get("run_number")
    return (f"R{number:02d} · " if number else "") + brief(attempt["run_id"].replace("_", "-"), 48)


def evaluation_rows(state: dict) -> list[tuple[dict, str, dict]]:
    """All evaluation roles use the same run record and stay visible in reports."""
    result = []
    for item in state.get("attempts", []):
        for step, evaluation in item.get("evaluations", {}).items():
            result.append((item, str(evaluation.get("step", step)), evaluation))
    return result


def write_report(session: Path, state: dict, *, guide: str, eula_scope: str) -> None:
    plan, _ = read_session(session)
    attempts = state.get("attempts", [])
    rows = evaluation_rows(state)
    candidates = [row for row in rows if row[2].get("group", "tuning") != "independent"]
    selected = state.get("selected_checkpoint")
    chosen = next((row for row in candidates if row[2]["checkpoint"] == selected), None)
    chosen = chosen or (max(candidates, key=lambda row: row[2]["rate"]) if candidates else None)
    lines = ["# DisplayPort training report", "",
             f"**{state['status']}** · {human_time(state['updated_utc'])} · "
             f"{state['completed_runs']}/{plan['max_completed_runs']} training runs completed.", "",
             f"**Objective:** {brief(plan.get('objective', 'Compare policy success.'))}", ""]
    if chosen:
        item, step, evaluation = chosen
        lines += [f"**Best observed:** {run_label(item)}, checkpoint {step}: "
                  f"**{evaluation['rate']:.2%}** ({evaluation['successes']}/{evaluation['episodes']}) "
                  f"on {evaluation.get('group', 'tuning')} seeds."]
        holdout = [row for row in rows if row[2].get("group") == "independent"
                   and row[2]["checkpoint"] == evaluation["checkpoint"]]
        if holdout:
            proof = holdout[-1][2]
            low, high = wilson_interval(proof["successes"], proof["episodes"])
            lines += [f"Independent validation: **{proof['rate']:.2%}** "
                      f"({proof['successes']}/{proof['episodes']}); episode-level 95% interval "
                      f"{low:.2%}–{high:.2%}."]
        else:
            lines += ["Independent validation: not recorded for this checkpoint."]
    else:
        lines += ["**Result:** no completed fixed-distribution evaluation yet."]
    latest = attempts[-1] if attempts else None
    if latest and latest.get("progress_evidence"):
        evidence = latest["progress_evidence"]
        stamp = datetime.fromtimestamp(evidence["epoch"], timezone.utc).isoformat()
        lines += [f"**Last confirmed progress:** {run_label(latest)}, update {evidence['step']} "
                  f"via {evidence['source']} · {human_time(stamp)}."]
    if latest and latest.get("stop_origin") == "supervisor":
        lines += [f"**Stop:** {brief(latest['stop_reason'])}."]
    if latest and latest.get("progress_probe_error"):
        lines += [f"**Monitoring issue:** {brief(latest['progress_probe_error'])}"]
    if state.get("failure_phase") == "preflight":
        lines += ["**Preflight failed:** training prerequisites were not verified; no new trainer was launched."]
    lines += ["", "## Comparison", "",
              "| Run | Checkpoint / evaluation | num_steps_per_env | max_iterations¹ | Success |",
              "| --- | --- | ---: | ---: | ---: |"]
    # Keep a baseline, chosen result, independent check, failures, and recent alternatives.
    selected_rows = []
    if rows:
        priorities = [rows[0]] + ([chosen] if chosen else [])
        priorities += [row for row in rows if row[2].get("group") == "independent"][-1:]
        priorities += rows[-3:]
        for row in priorities:
            if row not in selected_rows:
                selected_rows.append(row)
    for item, step, evaluation in selected_rows[:6]:
        spec = item["spec"]
        lines.append(f"| {run_label(item)} | {step} / {evaluation.get('group', 'tuning')} | "
                     f"{spec['num_steps_per_env']} | {spec['max_iterations']} | {evaluation['rate']:.2%} |")
    unseen = [item for item in attempts if not any(row[0] is item for row in selected_rows[:6])]
    remaining = max(0, 6 - len(selected_rows[:6]))
    for item in unseen[-remaining:] if remaining else []:
        spec = item["spec"]
        lines.append(f"| {run_label(item)} | {item['status']} | {spec['num_steps_per_env']} | "
                     f"{spec['max_iterations']} | unavailable |")
    lines += ["", "¹ Additional optimizer updates requested in this run. Saved checkpoint labels can repeat on resume; "
              "they are not total training counts. Full comparisons and parameters are in [session.json](session.json).", ""]
    if chosen:
        item, _, evaluation = chosen
        spec = item["spec"]
        lineage_updates = item.get("total_lineage_updates")
        parent = Path(spec["parent_checkpoint"]).name if spec.get("parent_checkpoint") else "none (fresh)"
        lines += [f"**Selected settings:** num_envs={spec['num_envs']}, seed={spec['seed']}; "
                  f"parent={parent}; completed additional updates="
                  f"{item.get('additional_updates_completed', 'unavailable')}; "
                  f"total lineage updates={lineage_updates if lineage_updates is not None else 'unavailable'}. "]
        overrides = {key: value for key, value in spec.get("overrides", {}).items()
                     if key != "env.events.reset_plug_curriculum.params.num_steps_per_env"}
        lines += [f"Overrides: {brief(', '.join(f'{key}={value}' for key, value in overrides.items()), 300) or 'task defaults'}.", ""]
    findings = state.get("findings", [])[:3]
    if findings:
        lines += ["**Findings:**", ""] + [f"- {brief(finding)}" for finding in findings] + [""]
    lines += [f"**Cost this session:** trainer process time {state['training_seconds'] / 3600:.2f} h; "
              f"evaluation {state['evaluation_seconds'] / 3600:.2f} h (including startup and failed attempts)."]
    if chosen:
        inherited = chosen[0]["spec"].get("parent_training_seconds")
        lines += [f"Selected policy's inherited training: {inherited / 3600:.2f} h."
                  if inherited is not None else
                  "Selected policy's inherited training: " + ("0 h (fresh)." if not chosen[0]["spec"].get("parent_checkpoint") else "unavailable; excluded from session cost.")]
    errors = [item for item in attempts if item.get("error")]
    unavailable = sum(item.get("exit_code") is None for item in attempts)
    issue = brief(state.get("evaluation_error") or state.get("error") or (errors[-1]["error"] if errors else "none recorded"))
    lines += ["", f"**Issues:** {len(errors)} attempts with errors; {unavailable} exit codes unavailable. {issue}.", "",
              "**Measurement:** fixed first episodes, 0% near-goal starts, 20–60 mm approach, "
              "observation noise disabled, 3 mm geometric success. TensorBoard's training metric uses "
              "a changing distribution. Independent seeds assess evaluation uncertainty; they do not measure "
              "variation between training seeds.", ""]
    links = ["[Parameters and results](session.json)", "[Events and repairs](events.jsonl)"]
    config_attempt = chosen[0] if chosen else attempts[-1] if attempts else None
    if config_attempt:
        if config_attempt.get("console"):
            links.append(f"[Training log](<{config_attempt['console']}>)")
        if not chosen and config_attempt.get("run_dir"):
            links.append(f"[Native artifacts (may be partial)](<{config_attempt['run_dir']}>)")
        for config in config_attempt.get("resolved_configs", []):
            if "archive_member" not in config:
                links.append(f"[{Path(config['path']).name}](<{config['path']}>)")
    if chosen:
        item, _, evaluation = chosen
        links.insert(0, f"[Checkpoint](<{evaluation['checkpoint']}>)")
        proof = item.get("verification", {})
        videos = proof.get("videos", [])
        if videos:
            links.append(f"[Beginning training video](<{videos[0]}>)")
            if len(videos) > 1:
                links.append(f"[End training video](<{videos[-1]}>)")
        if item.get("run_dir"):
            links.append(f"[Native artifacts and exports](<{item['run_dir']}>)")
    for milestone in state.get("milestones", [])[-5:]:
        if isinstance(milestone, dict) and milestone.get("video"):
            links.append(f"[{brief(milestone.get('label', 'Milestone'), 30)}](<{milestone['video']}>)")
    lines += ["**Files:** " + " · ".join(links), "",
              "**Next:** " + brief(("Recover the pending checkpoint evaluation with its original config."
                                    if state.get("pending_evaluation") else state.get("next_action")) or
                  ("Review the comparison and choose the next hypothesis." if state['status'] == 'decision_pending'
                   else "Resolve the recorded issue before resuming." if state['status'] == 'blocked'
                   else "Deliver this report and the selected artifacts." if state['status'] in {'complete', 'closed'}
                   else "Monitor training and evaluate completed checkpoints.")), ""]
    temp = session / ".report.md.tmp"
    temp.write_text("\n".join(lines))
    temp.replace(session / "report.md")
