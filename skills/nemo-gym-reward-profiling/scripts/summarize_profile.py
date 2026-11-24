#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Summarize a NeMo Gym reward profile for choosing RL training tasks.

Reads the per-task file that `gym eval profile` writes (`<stem>_reward_profiling.jsonl`) and, when they sit next to
it, the run's materialized inputs (`<stem>_materialized_inputs.jsonl`) and agent metrics (`<stem>_agent_metrics.json`).
Prints coverage first, then sorts the tasks by whether their reward changed across repeats. Needs only the Python standard library.

    python3 summarize_profile.py results/rollouts_reward_profiling.jsonl
    python3 summarize_profile.py results/rollouts_reward_profiling.jsonl --json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROFILE_SUFFIX = "_reward_profiling.jsonl"
# Rewards closer than this count as equal, so float noise does not read as spread.
REWARD_TOLERANCE = 1e-9
# The text report lists at most this many tasks per group, so it stays short on runs with thousands of tasks;
# --json lists every task.
MAX_LISTED_TASKS = 20


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: not valid JSON ({e.msg}); is the file complete?") from None
            if not isinstance(row, dict) or "_ng_task_index" not in row:
                raise ValueError(
                    f"{path}:{line_number}: no _ng_task_index; not a file written by gym eval run or profile"
                )
            yield row


def agent_name(row):
    return (row.get("agent_ref") or {}).get("name")


def run_key(row):
    server = row.get("_ng_environment_server")
    return server if server is not None else agent_name(row)


def run_labels(rows):
    """Label each run the way `gym eval profile` labels the rows of a task run by several agents.

    Mirrors nemo_gym's `rollout_run_labels` and `label_runs`: a run is keyed by its Environment Server, else its
    agent, and labelled by its agent's name when that name identifies exactly one run, else by its key.
    """
    agent_by_key = {}
    for row in rows:
        key = run_key(row)
        if key is not None and key not in agent_by_key:
            agent_by_key[key] = agent_name(row)
    keys_by_agent = defaultdict(set)
    for key, agent in agent_by_key.items():
        if agent is not None:
            keys_by_agent[agent].add(key)
    labels = {
        key: agent if agent is not None and len(keys_by_agent[agent]) == 1 else key
        for key, agent in agent_by_key.items()
    }
    while True:
        counts = Counter(labels.values())
        clashing = [key for key, label in labels.items() if counts[label] > 1 and label != key]
        if not clashing:
            return labels
        for key in clashing:
            labels[key] = key


def planned_rollouts(read_rows):
    """Planned rollouts per (task, run label), the label of each run, whether every row names an agent, and the names
    that several runs have as their agent or key.

    `read_rows` returns a fresh iterator over the materialized inputs; the file is read twice instead of held in memory.
    """
    labels = run_labels(read_rows())
    planned, every_row_names_agent, runs_by_name = Counter(), True, defaultdict(set)
    for row in read_rows():
        key = run_key(row)
        planned[(row["_ng_task_index"], labels.get(key))] += 1
        every_row_names_agent = every_row_names_agent and agent_name(row) is not None
        runs_by_name[key].add(key)
        runs_by_name[agent_name(row)].add(key)
    ambiguous_labels = {name for name, keys in runs_by_name.items() if name is not None and len(keys) > 1}
    return planned, labels, every_row_names_agent, ambiguous_labels


def record_key(record):
    task = record["task"]
    return (record["agent"] or "", (0, task, "") if isinstance(task, int) else (1, 0, str(task)))


def new_record(task, agent, planned, row=None):
    row = row or {}
    return {
        "task": task,
        "agent": agent,
        "measured": row.get("num_rollouts", 0),
        # Planned minus masked: `gym eval profile` drops masked rollouts before it counts.
        "expected": row.get("expected_num_rollouts", row.get("num_rollouts", 0)),
        "planned": planned,
        "mean": row.get("mean/reward"),
        "std": row.get("std/reward"),
        "min": row.get("min/reward"),
        "max": row.get("max/reward"),
        "in_profile_file": bool(row),
    }


def summarize(
    profile_rows,
    planned_by_task_label=None,
    agent_metrics=None,
    every_input_names_agent=True,
    input_run_labels=None,
    input_ambiguous_labels=frozenset(),
):
    """Coverage and task groups for one profile.

    `planned_by_task_label` counts the materialized input rows per (task, run label) and `input_run_labels` labels
    their runs; both are None when the inputs were not found. `input_ambiguous_labels` are the names that several input
    runs have as their agent or key, so gym may label any of those runs with them. `gym eval profile` leaves masked
    rollouts out of a row's counts and writes no row for a task whose every planned rollout was saved and masked, so
    masked rollouts of such tasks show only in the agent metrics' total. That total covers only rows that name an
    agent.
    """
    if any("num_rollouts" not in row for row in profile_rows):
        raise ValueError(
            "No num_rollouts field: pass the <stem>_reward_profiling.jsonl that gym eval profile writes, or profile "
            "the run again with a current nemo-gym."
        )

    # gym names a row's run only when a task ran under several runs, and names the runs that masking left, so masking
    # can drop or rename them. A row with a measured rollout keeps its first input row in `sample`, so read its run
    # from there and name it as the inputs do.
    labels = input_run_labels
    if labels is None:
        labels = run_labels(row["sample"] for row in profile_rows if isinstance(row.get("sample"), dict))
    several_runs = len(set(labels.values())) > 1
    # gym gives a run one label across the profile, so a measured row tells the run of a count-only row with its label.
    input_label_by_profile_label = {
        agent_name(row): labels.get(run_key(row["sample"]))
        for row in profile_rows
        if "agent_ref" in row and isinstance(row.get("sample"), dict)
    }

    def named_run(row):
        """The label of the run the row names, or None when it names none or there is only one run to name."""
        if planned_by_task_label is None and "agent_ref" in row:
            # Without the inputs, a row gym split by run keeps gym's label, even when only one run was measured.
            return agent_name(row)
        if not several_runs:
            return None
        if isinstance(row.get("sample"), dict):
            return labels.get(run_key(row["sample"]))
        label = agent_name(row)
        if label in input_label_by_profile_label:
            return input_label_by_profile_label[label]
        # gym labels a run by its agent or its key, depending on which runs masking left, so a name that several input
        # runs have may be any of them.
        return None if label in input_ambiguous_labels else label

    def mismatch(task):
        return ValueError(
            f"The materialized inputs do not match this profile (task {task}). Use the "
            "<stem>_materialized_inputs.jsonl written next to the same rollouts file."
        )

    records = []
    planned = None
    masked_in_rows = 0
    # Tasks with a row whose run is unknown: the rows then cannot tell masked rollouts from rollouts never saved.
    unresolved_tasks = set()
    if planned_by_task_label is None:
        records = [new_record(row["_ng_task_index"], named_run(row), None, row) for row in profile_rows]
    else:
        planned = Counter()
        runs_by_task = defaultdict(list)
        for (task, label), count in planned_by_task_label.items():
            key = (task, label if several_runs else None)
            if key not in planned:
                runs_by_task[task].append(key[1])
            planned[key] += count
        # Every planned run of a task gets its own record, which takes the row that names it. A row with no measured
        # rollout (written by main only) has no sample: it names its run through a measured row with its label, else
        # by gym's label alone or not at all, so it can miss the inputs' labels. It then takes its task's one run left
        # without a row, and with several left its run stays unknown.
        row_by_run = {}
        unnamed_by_task = defaultdict(list)
        for row in profile_rows:
            key = (row["_ng_task_index"], named_run(row))
            if key in planned and key not in row_by_run:
                row_by_run[key] = row
            else:
                unnamed_by_task[key[0]].append(row)
        for task, rows in unnamed_by_task.items():
            left = [label for label in runs_by_task.get(task, ()) if (task, label) not in row_by_run]
            # A row with measured rollouts names its run in its sample, so it never takes a run by elimination.
            if len(rows) > len(left) or any(row["num_rollouts"] for row in rows):
                raise mismatch(task)
            if len(left) == 1:
                row_by_run[(task, left[0])] = rows[0]
            else:
                unresolved_tasks.add(task)
        for (task, label), count in planned.items():
            # nemo-gym 0.7.0 writes no row for a run with no measured rollout. Main writes one with num_rollouts 0,
            # except for a run whose every planned rollout was saved and masked.
            row = row_by_run.get((task, label))
            record = new_record(task, label, count, row)
            if row is not None:
                if record["expected"] > count:
                    raise mismatch(task)
                masked_in_rows += count - record["expected"]
            elif task in unresolved_tasks:
                # A row of unknown run on this task may be this run's, so whether it is in the file is unknown.
                record["in_profile_file"] = None
            records.append(record)

    masked_reported = None
    if agent_metrics is not None:
        masked_reported = sum(e.get("coverage/masked_rollouts", 0) for e in agent_metrics if isinstance(e, dict))

    groups = {"varies": [], "one_rollout": [], "no_measured_rollout": [], "no_reward": []}
    same_every_repeat = {}
    for record in sorted(records, key=record_key):
        target = record["planned"] if record["planned"] is not None else record["expected"]
        record["partial"] = 0 < record["measured"] < target
        if record["measured"] == 0:
            groups["no_measured_rollout"].append(record)
        elif record["min"] is None or record["max"] is None:
            groups["no_reward"].append(record)
        elif record["measured"] == 1:
            groups["one_rollout"].append(record)
        elif record["max"] - record["min"] > REWARD_TOLERANCE:
            groups["varies"].append(record)
        else:
            same_every_repeat.setdefault(record["min"], []).append(record)

    measured_total = sum(r["measured"] for r in records)
    planned_total = sum(planned.values()) if planned is not None else None
    absent = [r for r in records if r["in_profile_file"] is False]
    # A row's planned minus expected rollouts were masked, and masked rollouts of tasks with no row show only in the
    # agent metrics' total. That total counts only rows that name an agent, so the split between masked and never
    # saved is known only when the inputs are found and every input row names an agent. A row whose run is unknown
    # still counts in that total; only which of its task's runs it belongs to stays unknown.
    masked_total = None
    if planned is not None and masked_reported is not None and every_input_names_agent:
        # Masked rollouts were saved but not measured: at least those the rows leave out, at most all unmeasured ones.
        if not masked_in_rows <= masked_reported <= planned_total - measured_total:
            raise ValueError(
                "The materialized inputs or the agent metrics do not match this profile: the agent metrics count "
                f"{masked_reported} masked rollouts, while the profile rows imply at least {masked_in_rows} and the "
                f"inputs leave only {planned_total - measured_total} unmeasured. Use the "
                "<stem>_materialized_inputs.jsonl and <stem>_agent_metrics.json written with the same rollouts file."
            )
        masked_total = masked_reported
    known = masked_total is not None

    means = {}
    for record in records:
        if record["measured"] and record["mean"] is not None:
            label = record["agent"] or ""
            total, count = means.get(label, (0.0, 0))
            means[label] = (total + record["mean"] * record["measured"], count + record["measured"])

    def ids(selected):
        return [{"task": r["task"], "agent": r["agent"]} for r in sorted(selected, key=record_key)]

    return {
        "coverage": {
            "planned_rollouts": planned_total,
            "measured_rollouts": measured_total,
            "masked_rollouts": masked_total,
            "never_saved_rollouts": planned_total - measured_total - masked_total if known else None,
            "measured_pct": round(100.0 * measured_total / planned_total, 2) if planned_total else None,
            "tasks": len(records),
            "complete_tasks": sum(1 for r in records if r["measured"] and not r["partial"]),
            "partial_tasks": ids(r for r in records if r["partial"]),
            "tasks_without_measured_rollout": ids(groups["no_measured_rollout"]),
            "tasks_not_in_profile_file": ids(absent),
            # Masked rollouts that belong to tasks with no row: 0 means those tasks were never saved, a value equal to
            # their planned rollouts means every one of them was saved and masked, and null means unknown.
            "masked_rollouts_of_tasks_not_in_profile_file": (masked_total - masked_in_rows)
            if known and absent and not unresolved_tasks
            else None,
            "inputs_checked": planned is not None,
            "masked_rollouts_in_agent_metrics": masked_reported,
        },
        "mean_reward_over_measured_rollouts": {
            label: total / count for label, (total, count) in sorted(means.items())
        },
        "varies": groups["varies"],
        "same_reward_every_repeat": {str(value): rs for value, rs in sorted(same_every_repeat.items())},
        "one_rollout": groups["one_rollout"],
        "no_measured_rollout": groups["no_measured_rollout"],
        "no_reward": groups["no_reward"],
    }


def task_list(records):
    names = [
        (str(r["task"]) if r["agent"] is None else f"{r['task']} ({r['agent']})") + ("*" if r["partial"] else "")
        for r in records
    ]
    if len(names) > MAX_LISTED_TASKS:
        return ", ".join(names[:MAX_LISTED_TASKS]) + f", and {len(names) - MAX_LISTED_TASKS} more (see --json)"
    return ", ".join(names) if names else "none"


def print_report(summary):
    cov = summary["coverage"]
    pct = "n/a" if cov["measured_pct"] is None else f"{cov['measured_pct']}%"
    # Without the inputs, a task with no row in the profile file cannot be counted at all.
    unknown = "" if cov["inputs_checked"] else "; tasks with no row in the profile file are unknown"
    if not cov["inputs_checked"]:
        print(
            f"Coverage: {cov['measured_rollouts']} rollouts measured. Materialized inputs not found, so the planned "
            "total, masked rollouts and tasks with no saved rollout are unknown."
        )
        if cov["masked_rollouts_in_agent_metrics"]:
            print(f"  The agent metrics count {cov['masked_rollouts_in_agent_metrics']} masked rollouts.")
    elif cov["never_saved_rollouts"] is None:
        # When every planned rollout was measured, none was masked or never saved.
        rest = (
            "."
            if cov["measured_rollouts"] == cov["planned_rollouts"]
            else "; the rest were never saved or were masked, and these files cannot tell which."
        )
        print(
            f"Coverage: {cov['measured_rollouts']} of {cov['planned_rollouts']} planned rollouts measured ({pct}){rest}"
        )
    else:
        print(
            f"Coverage: {cov['measured_rollouts']} of {cov['planned_rollouts']} planned rollouts measured ({pct}): "
            f"{cov['never_saved_rollouts']} never saved, {cov['masked_rollouts']} masked."
        )
    print(
        f"Tasks: {cov['tasks']}; {cov['complete_tasks']} complete, {len(cov['partial_tasks'])} partial, "
        f"{len(cov['tasks_without_measured_rollout'])} with no measured rollout{unknown}."
    )
    absent = [r for r in summary["no_measured_rollout"] if r["in_profile_file"] is False]
    if absent:
        masked = cov["masked_rollouts_of_tasks_not_in_profile_file"]
        if masked == 0:
            reason = "no saved rollout"
        elif masked == sum(r["planned"] for r in absent):
            reason = "every saved rollout was masked"
        else:
            reason = "never saved, or every saved rollout masked"
        print(f"  Not in the profile file ({reason}): {task_list(absent)}")
    for label, mean in summary["mean_reward_over_measured_rollouts"].items():
        who = f" for {label}" if label else ""
        print(f"Mean reward over measured rollouts{who}: {mean:.4f} (says nothing about tasks with none).")
    print("* = partial: fewer measured rollouts than planned.")
    print()
    print(f"Reward varies across repeats (learning signal): {task_list(summary['varies'])}")
    for value, records in summary["same_reward_every_repeat"].items():
        print(f"Same reward ({value}) on every repeat (no signal): {task_list(records)}")
    if summary["one_rollout"]:
        print(f"Only one measured rollout (spread unknown): {task_list(summary['one_rollout'])}")
    if summary["no_reward"]:
        print(f"Measured rollouts without a reward: {task_list(summary['no_reward'])}")
    print(f"No measured rollout (not measured): {task_list(summary['no_measured_rollout'])}{unknown}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", type=Path, help=f"<stem>{PROFILE_SUFFIX} written by `gym eval profile`")
    parser.add_argument(
        "--inputs", type=Path, help="materialized inputs JSONL (default: <stem>_materialized_inputs.jsonl beside it)"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    if not args.profile.is_file():
        sys.exit(f"Profile file not found: {args.profile}. Run `gym eval profile` first.")
    stem = args.profile.name[: -len(PROFILE_SUFFIX)] if args.profile.name.endswith(PROFILE_SUFFIX) else None
    inputs_path = args.inputs or (args.profile.with_name(stem + "_materialized_inputs.jsonl") if stem else None)
    if args.inputs and not inputs_path.is_file():
        sys.exit(f"Materialized inputs file not found: {inputs_path}")
    metrics_path = args.profile.with_name(stem + "_agent_metrics.json") if stem else None

    try:
        planned, input_run_labels, every_input_names_agent, input_ambiguous_labels = None, None, True, frozenset()
        if inputs_path is not None and inputs_path.is_file():
            planned, input_run_labels, every_input_names_agent, input_ambiguous_labels = planned_rollouts(
                lambda: read_jsonl(inputs_path)
            )
        agent_metrics = None
        if metrics_path is not None and metrics_path.is_file():
            try:
                agent_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"{metrics_path}: not valid JSON ({e.msg}); is the file complete?") from None
            if not isinstance(agent_metrics, list):
                raise ValueError(f"{metrics_path}: expected a list of agent entries from gym eval profile")
        summary = summarize(
            list(read_jsonl(args.profile)),
            planned,
            agent_metrics,
            every_input_names_agent,
            input_run_labels,
            input_ambiguous_labels,
        )
    except ValueError as e:
        sys.exit(str(e))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
