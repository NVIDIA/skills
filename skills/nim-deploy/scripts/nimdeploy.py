#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run state machine for a NIM deployment: new, set, status, list, advance, approval-summary,
approve, record.

intake > plan > render > validate > approve > apply > wait_ready > verify > record > done, with
diagnose and rollback off the side. The run folder state.json is the checkpoint: `status` prints
the state and the exact next command, so a run resumes after the chat context is lost.
Prints one JSON object. Exit codes: 0 ok, 2 bad input, 3 precondition failed, 5 refused.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handoff as ho
import nd_common as nd
import nextstep as ne
import placement
import runners

KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*|env\.[A-Za-z_][A-Za-z0-9_]*")  # env.NAME=value


def refuse(
    state: str,
    code: str,
    message: str,
    fix: str,
    exit_code: int = nd.EXIT_INPUT,
    facts: dict[str, Any] | None = None,
) -> NoReturn:
    nd.emit(False, state, "ask_user", facts, [nd.error(code, message, fix)], exit_code)


# --- helpers --------------------------------------------------------------------------------


def load_or_fail(run_id: str) -> dict[str, Any]:
    try:
        return nd.load_state(run_id)
    except (FileNotFoundError, ValueError) as exc:
        refuse(
            "unknown",
            "run_not_found",
            str(exc),
            f"List runs with: {ne.script('nimdeploy.py list')}",
        )


def parse_pairs(state: str, pairs: list[str]) -> dict[str, str]:
    values = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not KEY_PATTERN.fullmatch(key):
            message = f"expected KEY=VALUE with a lowercase key, got {pair!r}"
            fix = "Pass key=value, e.g. namespace=team-a; extra environment as env.NAME=value"
            refuse(state, "bad_input", message, fix)
        values[key] = value
    return values


RENDERED_STATES = ("validate", "approve")  # files exist and `set` is still allowed


def accept_inputs(state: dict[str, Any], values: dict[str, Any], source: str = "user") -> None:
    """Store inputs; refuse secret-looking values (exit 5) before anything is saved.

    Values from result files (source select or tune) were checked by handoff.py and may be lists
    (extra_args) or a mapping (env)."""
    secret = sorted(key for key, value in values.items() if nd.looks_secret(key, value))
    if secret:
        message = f"input(s) {', '.join(secret)} look like secret values; they are not stored"
        refuse(state["state"], "secret_input", message, nd.SECRET_FIX, nd.EXIT_REFUSED)
    if "model_profile" in values and not str(values["model_profile"]).strip():
        message = "model_profile is blank: that would unpin the profile and let NIM pick one"
        fix = (
            "Never drop model_profile. Give the profile id: --set model_profile=ID when "
            "starting a run with new, or set RUN model_profile=ID on an existing run "
            "(nim-select lists the ids); ask the user if unsure"
        )
        refuse(state["state"], "profile_not_pinned", message, fix)
    if values.get("runner") and values["runner"] not in runners.registry():
        fix = f"Use one of: {', '.join(sorted(runners.registry()))}"
        refuse(state["state"], "bad_runner", f"unknown runner {values['runner']!r}", fix)
    old_image = nd.input_value(state, "image")
    context_changed = (
        source == "user" and "context" in values
        and nd.input_value(state, "context") not in (None, values["context"])
    )
    if context_changed and any(key in state.get("inputs", {}) for key in nd.PLACEMENT_ANSWER):
        state["placement_pending"] = True
    if source == "user" and (context_changed or any(key in values for key in nd.PLACEMENT_ANSWER)):
        # the user's placement replaces nim-select's, with the node list it was worked out from
        for stale in (
            "placement",
            "node_selector",
            "node_hostnames",
            "placement_nodes",
            "placement_source",
        ):
            if stale not in values:
                state.get("inputs", {}).pop(stale, None)
    for key, value in values.items():
        if key.startswith("env."):  # one variable at a time, merged into the env mapping
            merged = dict(nd.input_value(state, "env") or {})
            merged[key[4:]] = value
            state.setdefault("inputs", {})["env"] = {"value": merged, "source": source}
            continue
        if key == "env" and isinstance(value, dict):  # a whole mapping from a result file
            merged = {**(nd.input_value(state, "env") or {}), **value}
            state.setdefault("inputs", {})["env"] = {"value": merged, "source": source}
            continue
        state.setdefault("inputs", {})[key] = {"value": value, "source": source}
    if "image" in values and state.get("docs_check") and values["image"] != old_image:
        del state["docs_check"]  # the docs were read for another image
    model = state.get("inputs", {}).get("model") or {}
    if (
        "image" in values
        and "model" not in values
        and old_image
        and model.get("value") == old_image
    ):
        # `nim run <model>` starts what model names: a model that followed the image follows it
        state["inputs"]["model"] = {"value": values["image"], "source": source}
    state["backend"] = nd.input_value(state, "backend")
    if state["target"] == "docker" and ("runner" in values or not state.get("runner")):
        # saved once, so a later NIM_DEPLOY_RUNNER cannot swap the runner of an existing run
        state["runner"] = runners.name_of({"inputs": state.get("inputs", {})})


def wrong_state(state: dict[str, Any], wanted: str) -> None:
    if state["state"] != wanted and not (wanted == "approve" and state["state"] == "rollback"):
        next_, command = ne.next_step(state)
        message = f"run is in {state['state']!r}, not {wanted!r}"
        error = nd.error("wrong_state", message, f"Run: {command}")
        nd.emit(False, state["state"], next_, ne.status_facts(state), [error], nd.EXIT_PRECONDITION)


def summary_or_fail(state: dict[str, Any]) -> dict[str, Any]:
    try:
        return ne.approval_summary(state)
    except FileNotFoundError as exc:
        fix = "Re-render the run; an approved artifact is gone"
        refuse(state["state"], "artifact_missing", str(exc), fix, nd.EXIT_PRECONDITION)


def apply_defaults(state: dict[str, Any]) -> None:
    """Record what the runner can fill in (name, gpus, container port, ...) with its source.

    Entries the skill chose (source in AUTO_SOURCES) are recomputed; the user's and docs-supplied
    values always win."""
    if state["target"] != "docker":
        return
    make = getattr(runners.runner_of(state), "default_inputs", None)
    if make is None:
        return
    inputs = state.setdefault("inputs", {})
    auto = nd.AUTO_SOURCES
    given = {k: e["value"] for k, e in inputs.items() if e.get("source") not in auto}
    produced = make(given, previous=dict(inputs))
    for key in [k for k, e in inputs.items() if e.get("source") in auto and k not in produced]:
        del inputs[key]
    for key, entry in produced.items():
        if key not in inputs or inputs[key].get("source") in auto:
            inputs[key] = dict(entry)


def enter_render_when_ready(state: dict[str, Any]) -> None:
    """plan -> render once every required input is set, so the renderer's state check holds."""
    if state["state"] == "plan" and not nd.missing_inputs(state):
        nd.transition(state, "render")


def report(state: dict[str, Any]) -> NoReturn:
    nd.emit(True, state["state"], ne.next_step(state)[0], ne.status_facts(state))


# --- subcommands ----------------------------------------------------------------------------


def used_handoff_files() -> dict[str, str]:
    """sha256 of every result file an earlier run was created from -> that run id."""
    used: dict[str, str] = {}
    for path in sorted((nd.runs_root() / "runs").glob("*/state.json")):
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for entry in (state.get("handoff") or {}).get("files", []):
            used.setdefault(str(entry.get("sha256")), str(state.get("run_id")))
    return used


def read_handoff(args: argparse.Namespace, values: dict[str, str]) -> ho.Plan | None:
    """The run inputs from `--from` files (scripts/handoff.py), or None without files."""
    if not args.from_files:
        if not args.target:
            refuse("intake", "bad_input", "--target is required", "Add --target TARGET")
        return None
    runner = runners.name_for(values.get("runner"))
    try:
        return ho.build(args.from_files, args.target, runner, values, used_handoff_files())
    except ho.HandoffError as exc:
        refuse("intake", exc.code, exc.message, exc.fix, exc.exit_code)


def cmd_new(args: argparse.Namespace) -> None:
    values = parse_pairs("intake", args.set or [])
    plan = read_handoff(args, values)
    target = plan.target if plan else args.target
    given = {key: entry["value"] for key, entry in (plan.values if plan else {}).items()}
    image = str(values.get("image") or given.get("image") or "")
    image_leaf = image.split("@", 1)[0].rsplit("/", 1)[-1].split(":", 1)[0]
    base = nd.new_run_id(target, args.model or image_leaf or "run", nd.utc_now())
    run_id, suffix = base, 1
    while (nd.runs_root() / "runs" / run_id / "state.json").exists():
        suffix += 1
        run_id = f"{base}-{suffix}"
    state: dict[str, Any] = {
        "run_id": run_id,
        "state": "intake",
        "target": target,
        "backend": None,
        "inputs": {},
        "artifacts": {},
        "approvals": [],
        "attempts": {},
        "history": [],
    }
    if plan:
        state["handoff"] = plan.record
        for source in ("select", "tune"):  # tune after select: its digest pin wins
            chosen = {k: e["value"] for k, e in plan.values.items() if e["source"] == source}
            accept_inputs(state, chosen, source)
    accept_inputs(state, {**({"model": args.model} if args.model else {}), **values})
    apply_defaults(state)
    nd.transition(state, "plan")
    enter_render_when_ready(state)
    report(state)


def cmd_set(args: argparse.Namespace) -> None:
    state = load_or_fail(args.run)
    if state["state"] in nd.APPROVED_STATES:
        message = (
            f"run {args.run} is in {state['state']!r}: its inputs were approved and are frozen"
        )
        fix = "Start a new run, or from diagnose advance to render, set, and approve again."
        refuse(state["state"], "run_approved", message, fix, nd.EXIT_REFUSED)
    before = nd.run_inputs(state)
    accept_inputs(state, parse_pairs(state["state"], args.pairs), args.source)
    apply_defaults(state)
    nd.save_state(state)
    enter_render_when_ready(state)
    if state["state"] in RENDERED_STATES and nd.run_inputs(state) != before:
        return_to_render(state)
    report(state)


def return_to_render(state: dict[str, Any]) -> None:
    """An input changed after render: the rendered files and the validation no longer match it.

    Forget the artifacts and go back to render, so the files are rebuilt, validated and approved
    again; the approval token binds the new inputs."""
    state["artifacts"] = {}
    try:
        nd.transition(state, "render")
    except ValueError as exc:
        fix = "Stop and ask the user: this run changed its plan too many times; start a new run."
        refuse(state["state"], "too_many_renders", str(exc), fix, nd.EXIT_PRECONDITION)


def cmd_docs_check(args: argparse.Namespace) -> None:
    """Record that the official docs were read (or, with the user's say-so, could not be)."""
    state = load_or_fail(args.run)
    if state["state"] in nd.APPROVED_STATES:
        message = f"run {args.run} is already approved; its docs check is part of the approval"
        refuse(state["state"], "run_approved", message, "Start a new run.", nd.EXIT_REFUSED)
    urls = args.url or []
    if args.offline:
        if urls or not args.reason:
            fix = "Use --offline --reason TEXT and no --url, only after the user agreed."
            refuse(state["state"], "bad_docs_check", "--offline needs --reason and no --url", fix)
        check: dict[str, Any] = {
            "mode": "offline",
            "urls": [],
            "reason": args.reason,
            "note": "",
            "doc_version": "",
        }
    else:
        bad = [u for u in urls if not re.fullmatch(r"https?://\S+", u)]
        if not urls or bad or not args.doc_version or not args.note:
            fix = "Pass --url URL (repeat), --doc-version V and --note 'port=...; env=...'."
            refuse(
                state["state"],
                "bad_docs_check",
                "an online docs check needs urls, a version and a note",
                fix,
            )
        check = {
            "mode": "online",
            "urls": urls,
            "doc_version": args.doc_version,
            "note": args.note,
            "reason": "",
        }
    texts = [str(check.get("note", "")), str(check.get("reason", ""))]
    if any(nd.looks_secret("note", text) for text in texts):
        refuse(
            state["state"],
            "secret_input",
            "the note looks like a secret",
            nd.SECRET_FIX,
            nd.EXIT_REFUSED,
        )
    state["docs_check"] = {**check, "fetched_at": nd.utc_now().isoformat()}
    nd.save_state(state)
    report(state)


def cmd_status(args: argparse.Namespace) -> None:
    report(load_or_fail(args.run))


def cmd_list(_args: argparse.Namespace) -> None:
    runs = []
    for path in sorted((nd.runs_root() / "runs").glob("*/state.json")):
        state = json.loads(path.read_text())
        keys = ("run_id", "state", "target")
        runs.append({**{key: state[key] for key in keys}, "model": nd.input_value(state, "model")})
    nd.emit(True, "list", "status" if runs else "ask_user", {"runs": runs})


def cmd_advance(args: argparse.Namespace) -> None:
    state = load_or_fail(args.run)
    current, to = state["state"], args.to
    if to == current:  # new/set already moved plan -> render: nothing to do
        report(state)
    if to not in nd.EDGES.get(current, ()):
        allowed = ", ".join(nd.EDGES.get(current, ())) or "none"
        message = f"cannot go {current} -> {to}"
        refuse(
            current, "illegal_transition", message, f"Legal next states from {current}: {allowed}"
        )
    visits = state.get("attempts", {}).get(to, 0)
    if visits >= nd.MAX_VISITS:
        fix = "Stop retrying. Show the user the errors so far and ask how to proceed."
        message = f"state {to!r} already visited {visits} times"
        facts = {"attempts": state["attempts"]}
        refuse(current, "too_many_attempts", message, fix, nd.EXIT_PRECONDITION, facts)
    try:
        nd.transition(state, to, [str(Path(path).resolve()) for path in args.artifact or []])
    except FileNotFoundError as exc:
        refuse(
            current,
            "artifact_missing",
            str(exc),
            "Pass an existing file to --artifact",
            nd.EXIT_PRECONDITION,
        )
    report(state)


def docs_gate(state: dict[str, Any]) -> None:
    if state["state"] == "rollback":  # renewing the approval of a rollback needs no docs
        return
    problem = nd.docs_problem(state)
    if problem:
        code, message, fix = problem
        refuse(state["state"], code, message, fix, nd.EXIT_PRECONDITION)


def cmd_approval_summary(args: argparse.Namespace) -> None:
    state = load_or_fail(args.run)
    wrong_state(state, "approve")
    docs_gate(state)
    nd.emit(True, state["state"], "ask_user", summary_or_fail(state))


def cmd_approve(args: argparse.Namespace) -> None:
    state = load_or_fail(args.run)
    wrong_state(state, "approve")
    docs_gate(state)
    facts = summary_or_fail(state)
    if args.token != facts["token"]:
        show = ne.script(f"nimdeploy.py approval-summary {args.run}")
        fix = f"Run {show}, show it to the user and use the token they send"
        message = "token does not match the current artifacts, context and namespace"
        refuse(state["state"], "token_mismatch", message, fix, nd.EXIT_REFUSED)
    placement_confirmation = None
    if state["state"] != "rollback" and state["target"] in ("helm", "nim-operator"):
        # the second look, right before apply: the cluster may have changed since validate
        shown = (state.get("placement_check") or {}).get("status")
        checked, found = placement.check(state)
        if not found and placement.weaker(checked["status"], shown):
            message = (
                f"placement was {shown} in the summary the user approved, but is {checked['status']}"
                f" now{': ' + checked['reason'] if checked.get('reason') else ''}"
            )
            fix = (
                f"Run {ne.script(f'nimdeploy.py advance {args.run} --to render')}, render and "
                "validate again, and show the user the new summary"
            )
            refuse(state["state"], "placement_weaker", message, fix, nd.EXIT_PRECONDITION)
        if found:
            refuse(
                state["state"],
                found[0]["code"],
                found[0]["message"],
                found[0]["fix"],
                nd.EXIT_PRECONDITION,
            )
        if checked["status"] == "skipped" or (
            checked["status"] == "unverified"
            and (shown != "unverified" or not args.confirm_placement)
        ):
            code = (
                "placement_unchecked" if checked["status"] == "skipped"
                else "placement_confirmation_required"
            )
            refuse(
                state["state"], code,
                "Placement must be checked and its current status shown before approval",
                f"Run {ne.script(f'nimdeploy.py advance {args.run} --to render')}, render and "
                "validate on the target cluster, then show the new approval summary. "
                "If unverified, ask the user to confirm the labels, then approve with their "
                "token and --confirm-placement",
                nd.EXIT_PRECONDITION,
            )
        if checked["status"] == "unverified":
            placement_confirmation = {
                "selector": checked.get("selector"), "token": args.token,
                "context": nd.input_value(state, "context"),
            }
        state["placement_check"] = checked
        nd.save_state(state)
    now = nd.utc_now()
    artifacts = {
        str(ne.artifact_path(state, k).resolve()): sha
        for k, sha in ne.current_artifacts(state).items()
    }
    record = {
        **nd.approval_bindings(state),
        "handoff_files": (state.get("handoff") or {}).get("files", []),
        "token": args.token,
        "at": now.isoformat(),
        "artifacts": artifacts,
        "expires_at": (now + nd.APPROVAL_TTL).isoformat(),
    }
    if placement_confirmation is not None:
        record["placement_confirmation"] = {**placement_confirmation, "at": now.isoformat()}
    state["approvals"].append(record)
    if state["state"] == "rollback":  # the user renewed the approval; the run stays in rollback
        nd.save_state(state)
        report(state)
    nd.transition(state, "apply")
    nd.emit(
        True,
        state["state"],
        ne.next_step(state)[0],
        {**ne.status_facts(state), "expires_at": record["expires_at"]},
    )


def cmd_record(args: argparse.Namespace) -> None:
    state = load_or_fail(args.run)
    current = state["state"]
    if current != "record" and "record" not in nd.EDGES.get(current, ()):
        fix = "Record is reachable from verify, diagnose or rollback; advance there first"
        refuse(current, "illegal_transition", f"cannot record from {current!r}", fix)
    if current != "record":
        nd.transition(state, "record")
    nd.transition(state, "done")
    entry = {
        "run_id": state["run_id"],
        "target": state["target"],
        "model": nd.input_value(state, "model"),
        "result": args.result,
        "reason": args.reason,
        "attempts": state["attempts"],
        "ts": nd.utc_now().isoformat(),
    }
    fd = os.open(nd.runs_root() / "runs.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    nd.emit(True, "done", "done", {"run_id": state["run_id"], "result": args.result})


def build_parser() -> argparse.ArgumentParser:
    parser = nd.JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/nimdeploy.py new --target helm --set namespace=nim "
        "--set context=kind-a; from nim-select and nim-tune results: nimdeploy.py new --target "
        "docker --from ~/.cache/nim-select/select.json --from final/recommended_config.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="create a run ('render' when every required input is set)")
    new.add_argument(
        "--target",
        choices=nd.TARGETS,
        help="required unless --from files name it (a host select.json means docker)",
    )
    new.add_argument(
        "--from",
        dest="from_files",
        action="append",
        metavar="FILE",
        help="a nim-select select.json or env.json, or a nim-tune recommended_config.json "
        "(repeatable); --set values win over the files",
    )
    new.add_argument("--model", help="needed for model-free images and the nim CLI runner")
    new.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="user-given input (repeatable)"
    )
    new.set_defaults(func=cmd_new)
    set_ = sub.add_parser("set", help="record user-given inputs (refused once approved or secret)")
    set_.add_argument("run")
    set_.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    set_.add_argument(
        "--source",
        choices=("user", "docs"),
        default="user",
        help="where the values come from: the user, or facts read from the docs",
    )
    set_.set_defaults(func=cmd_set)
    status = sub.add_parser("status", help="state and the exact next command (the resume point)")
    status.add_argument("run")
    status.set_defaults(func=cmd_status)
    sub.add_parser("list", help="list runs").set_defaults(func=cmd_list)
    docs = sub.add_parser(
        "docs-check", help="record that the official docs were read (before validate)"
    )
    docs.add_argument("run")
    docs.add_argument("--url", action="append", metavar="URL", help="a docs page read (repeat)")
    docs.add_argument("--doc-version", help="the docs version, e.g. 2.1.4 or 2.1.x")
    docs.add_argument(
        "--note", help="the facts taken from the docs, e.g. 'port=8000; env=NGC_API_KEY'"
    )
    docs.add_argument(
        "--offline", action="store_true", help="the docs could not be reached (the user agreed)"
    )
    docs.add_argument("--reason", help="why the docs were not checked (with --offline)")
    docs.set_defaults(func=cmd_docs_check)
    advance = sub.add_parser("advance", help="move the run along a legal edge")
    advance.add_argument("run")
    advance.add_argument("--to", required=True, choices=nd.STATES)
    advance.add_argument(
        "--artifact", action="append", metavar="PATH", help="file to hash into the run"
    )
    advance.set_defaults(func=cmd_advance)
    summary = sub.add_parser(
        "approval-summary", help="what will be applied, plus the approval token"
    )
    summary.add_argument("run")
    summary.set_defaults(func=cmd_approval_summary)
    approve = sub.add_parser("approve", help="record the user's approval token and move to 'apply'")
    approve.add_argument("run")
    approve.add_argument("--token", required=True)
    approve.add_argument("--confirm-placement", action="store_true",
                         help="record the user's explicit confirmation of shown unverified labels")
    approve.set_defaults(func=cmd_approve)
    record = sub.add_parser("record", help="append the outcome to runs.jsonl and finish the run")
    record.add_argument("run")
    record.add_argument("--result", required=True, choices=("ok", "fail"))
    record.add_argument("--reason", default="")
    record.set_defaults(func=cmd_record)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
