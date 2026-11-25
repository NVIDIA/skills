#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a rendered run before approval.

The rendered files must equal a fresh render of the run's inputs (a hand edit is drift), then
per target:
  nim-operator       structure checks, then `kubectl apply --dry-run=server`
  helm               namespace, context and values-file checks (no --set), then `helm template`
  docker             the run's runner checks the host (see runners/docker.py: NVIDIA runtime,
                     free port, free container name)
--offline skips the live checks. On pass the run moves to `approve` and the approval summary is
printed; on failure it goes back to `render`. Prints one JSON object.
Exit codes: 0 ok, 3 precondition failed.
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_common as nd
import nextstep
import nimdeploy
import placement
import render_k8s
import runners

# Errors the user must resolve (environment or choices), not the renderer.
USER_ERRORS = {
    "docker_missing",
    "no_nvidia_runtime",
    "port_in_use",
    "name_in_use",
    "context_missing",
    "cluster_auth",
    "crd_missing",
    "nim_missing",
    "placement_no_nodes",
    "placement_other_nodes",
}
# Errors where going on would destroy something the run does not own: refuse (exit 5).
REFUSE_ERRORS = {"name_collision"}
NO_CLUSTER = placement.NO_CLUSTER
Errors = list[dict[str, str]]


def tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def run_tool(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    nd.log("validate: running " + " ".join(argv))
    return subprocess.run(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def drift(path: Path, expected: str) -> Errors:
    """The file on disk must be byte-identical to a fresh render."""
    if not path.is_file():
        return [nd.error("artifact_missing", f"{path.name} not found", "Re-render the run")]
    if path.read_text() != expected:
        fix = "Never edit rendered files. Change an input with nimdeploy.py set, then re-render"
        return [
            nd.error(
                "file_drift", f"{path.name} differs from a fresh render of the run inputs", fix
            )
        ]
    return []


def check_operator(documents: list[dict[str, Any]], state: dict[str, Any]) -> Errors:
    kinds = [document.get("kind") for document in documents]
    errors: Errors = []
    if kinds.count("NIMService") != 1:
        errors.append(
            nd.error("wrong_kind", f"expected one NIMService, found {kinds}", "Re-render")
        )
    for document in documents:
        if not str(document.get("apiVersion", "")).startswith("apps.nvidia.com/"):
            message = f"{document.get('kind')} is not apps.nvidia.com"
            errors.append(nd.error("wrong_api_version", message, "Re-render"))
        if document["metadata"].get("namespace") != nd.input_value(state, "namespace"):
            message = f"{document.get('kind')} namespace differs from the run"
            errors.append(nd.error("namespace_mismatch", message, "Re-render"))
    return errors


def check_helm(command: str, state: dict[str, Any]) -> Errors:
    argv = shlex.split(command)
    errors: Errors = []
    scopes = (
        (("-n", "--namespace"), "namespace", "namespace_mismatch"),
        (("--kube-context",), "context", "context_mismatch"),
    )
    for flags, wanted, code in scopes:
        got = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token in flags]
        if got != [nd.input_value(state, wanted)]:
            want = nd.input_value(state, wanted)
            message = f"helm command has {flags[-1]} {got or 'missing'}, run wants {want!r}"
            errors.append(nd.error(code, message, "Re-render"))
    if any(token.startswith("--set") for token in argv):
        message = "helm command uses --set; values live in values.yaml"
        errors.append(nd.error("helm_set_flag", message, "Re-render"))
    return errors


# --- live checks ----------------------------------------------------------------------------


def server_dry_run(path: Path, state: dict[str, Any], offline: bool) -> tuple[str, Errors]:
    if offline or not shutil.which("kubectl"):
        return "skipped", []
    context, namespace = nd.input_value(state, "context"), nd.input_value(state, "namespace")
    argv = [
        "kubectl",
        "apply",
        "--dry-run=server",
        "-f",
        str(path),
        "--context",
        context,
        "-n",
        namespace,
    ]
    try:
        proc = run_tool(argv)
    except subprocess.TimeoutExpired:
        return "skipped", []
    if proc.returncode == 0:
        return "ok", []
    output = tail(proc.stderr or proc.stdout)
    if re.search(r"context .* does not exist|no context exists", output):
        fix = (
            "Ask the user for the right context (kubectl config get-contexts), "
            "then nimdeploy.py set RUN context=..."
        )
        return "failed", [
            nd.error("context_missing", f"kube context {context!r} does not exist", fix)
        ]
    if NO_CLUSTER.search(output):
        return "skipped", []
    if re.search(r"Unauthorized|forbidden|provide credentials", output, re.I):
        return "failed", [
            nd.error("cluster_auth", output, "Ask the user to log in to the cluster or fix RBAC")
        ]
    if re.search(r"no matches for kind|ensure CRDs are installed", output):
        fix = "Ask the user to install the NIM Operator CRDs on this cluster"
        return "failed", [nd.error("crd_missing", output, fix)]
    return "failed", [nd.error("dry_run_rejected", output, "Fix the inputs and re-render")]


def helm_template(
    folder: Path, command: str, state: dict[str, Any], offline: bool
) -> tuple[str, Errors]:
    if offline or not shutil.which("helm"):
        return "skipped", []
    argv = shlex.split(command)
    positional = [token for token in argv[2:] if not token.startswith("-")]
    release, chart = [*positional, "", ""][:2]
    version = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "--version"]
    tool = [
        "helm",
        "template",
        release,
        chart,
        "-f",
        "values.yaml",
        "-n",
        nd.input_value(state, "namespace"),
    ]
    try:
        proc = run_tool([*tool, *(["--version", version[0]] if version else [])], cwd=folder)
    except subprocess.TimeoutExpired:
        return "skipped", []
    if proc.returncode == 0:
        return "ok", []
    fix = "Check the chart reference (helm repo add/update) and values.yaml, then re-render"
    return "failed", [nd.error("helm_template_failed", tail(proc.stderr), fix)]


# --- main -----------------------------------------------------------------------------------


def validate_local(
    state: dict[str, Any], inputs: dict[str, Any], offline: bool
) -> tuple[Errors, str, list[Path]]:
    """The docker target: every artifact must equal a fresh render by the run's runner."""
    runner, folder = runners.runner_of(state), nd.run_dir(state["run_id"])
    errors: Errors = []
    with tempfile.TemporaryDirectory() as scratch:
        fresh = runner.render(inputs, Path(scratch))
        for path in fresh:
            errors += drift(folder / path.name, path.read_text())
    files = [folder / path.name for path in fresh]
    preflight = getattr(runner, "preflight", None)
    if errors or preflight is None:
        return errors, "skipped", files
    live, found = preflight(state, offline)
    return found, live, files


def validate_run(state: dict[str, Any], offline: bool) -> tuple[Errors, dict[str, str], list[Path]]:
    folder, target = nd.run_dir(state["run_id"]), state["target"]
    inputs, checks = nd.run_inputs(state), {"live": "skipped"}
    if state.get("handoff") and not inputs.get("model_profile"):
        message = "this run came from nim-select or nim-tune results but has no pinned model_profile"
        fix = "Never drop model_profile. Set it with set RUN model_profile=ID, then render again"
        return [nd.error("profile_not_pinned", message, fix)], checks, []
    live: Errors = []
    try:
        if target == "docker":
            errors, checks["live"], files = validate_local(state, inputs, offline)
            return errors, checks, files
        if target == "nim-operator":
            documents = render_k8s.render_operator(inputs)
            files = [folder / "nimservice.yaml"]
            errors = drift(files[0], render_k8s.operator_text(inputs)) or check_operator(
                documents, state
            )
            if not errors:
                checks["live"], live = server_dry_run(files[0], state, offline)
        else:
            values, command = render_k8s.render_helm(inputs)
            files = [folder / "values.yaml", folder / "helm-cmd.txt"]
            errors = drift(files[0], nd.to_yaml(values)) + drift(files[1], command)
            errors = errors or check_helm(command, state)
            if not errors:
                checks["live"], live = helm_template(folder, command, state, offline)
        if not errors:  # the selector must match the GPU's nodes on this very cluster
            state["placement_check"], found = placement.check(state, offline)
            checks["placement"] = state["placement_check"]["status"]
            live = live + found
        return errors + live, checks, files
    except nd.RenderError as exc:
        fix = "Fix the input with nimdeploy.py set, then re-render"
        return [nd.error("render_input", str(exc), fix)], checks, []


def main(argv: list[str] | None = None) -> None:
    parser = nd.JsonArgumentParser(
        description=__doc__,
        epilog="Example: python3 scripts/validate.py --run RUN --offline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--offline", action="store_true", help="skip cluster, helm and docker checks"
    )
    args = parser.parse_args(argv)
    state = nimdeploy.load_or_fail(args.run)
    nimdeploy.wrong_state(state, "validate")
    problem = nd.docs_problem(state)
    if problem:  # the docs are read before the checks, so the approval shows what they said
        code, message, fix = problem
        error = nd.error(code, message, fix)
        nd.emit(
            False,
            state["state"],
            "docs_check",
            nextstep.status_facts(state),
            [error],
            nd.EXIT_PRECONDITION,
        )
    errors, checks, files = validate_run(state, args.offline)
    facts: dict[str, Any] = {
        "target": state["target"],
        "checks": checks,
        "files": [str(p) for p in files],
    }
    if errors:
        next_ = "ask_user" if any(error["code"] in USER_ERRORS for error in errors) else "render"
        try:
            nd.transition(state, "render")
        except ValueError as exc:
            errors.append(
                nd.error("too_many_attempts", str(exc), "Stop and show these errors to the user")
            )
            next_ = "ask_user"
        refused = any(error["code"] in REFUSE_ERRORS for error in errors)
        exit_code = nd.EXIT_REFUSED if refused else nd.EXIT_PRECONDITION
        facts = {**facts, **nextstep.status_facts(state)}
        nd.emit(False, state["state"], "ask_user" if refused else next_, facts, errors, exit_code)
    try:
        nd.transition(state, "approve")
    except ValueError as exc:
        fix = "Stop and ask the user how to proceed"
        error = nd.error("too_many_attempts", str(exc), fix)
        nd.emit(False, state["state"], "ask_user", facts, [error], nd.EXIT_PRECONDITION)
    nd.emit(True, state["state"], "ask_user", {**facts, **nextstep.approval_summary(state)})


if __name__ == "__main__":
    main()
