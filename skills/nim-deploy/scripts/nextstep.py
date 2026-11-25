#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What to run next for a run: the command for each state and the approval summary.

Used by nimdeploy.py, validate.py and verify.py. The docker target's start, stop, endpoint and
container come from its runner (runners/__init__.py); Kubernetes targets use kubectl and helm
forms built from the approved inputs. `python3 scripts/nextstep.py RUN` prints the same JSON as
`nimdeploy.py status RUN`. Prints one JSON object. Exit codes: 0 ok, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handoff
import nd_common as nd
import render_k8s
import runners

RENDERERS = {
    "docker": "render_docker.py --run {run}",
    "helm": "render_k8s.py --run {run} --kind helm",
    "nim-operator": "render_k8s.py --run {run} --kind nim-operator",
}
INSTRUCTION = (
    "Show this summary to the user. Run approve only after the user replies with the token."
)
APPLY_NOTE = (
    "Run apply_command alone in its own Bash call, exactly as printed; "
    "the guard denies anything else."
)


def script(command: str) -> str:
    """`python3 <absolute, shell-quoted path of the script> ARGS`: runs from any folder, from any
    install location, even when the path has a space."""
    name, _, args = command.partition(" ")
    return f"{nd.script_command(name)} {args}".rstrip()


def artifact_path(state: dict[str, Any], key: str) -> Path:
    path = Path(key)
    if path.is_absolute():
        return path
    in_run = nd.run_dir(state["run_id"]) / path
    return in_run if in_run.exists() else path.resolve()


def current_artifacts(state: dict[str, Any]) -> dict[str, str]:
    """Re-hash every recorded artifact from disk; FileNotFoundError when one is gone."""
    keys = sorted(state.get("artifacts", {}))
    return {key: nd.sha256_file(artifact_path(state, key)) for key in keys}


def docs_signature(state: dict[str, Any]) -> str:
    """The docs check and, for a run made with --from, the result files: the token binds both."""
    check = state.get("docs_check") or {}
    text = f"docs:{check.get('mode')}:{check.get('doc_version')}:{','.join(check.get('urls', []))}"
    files = (state.get("handoff") or {}).get("files") or []
    return text + "".join(f":{entry['kind']}={entry['sha256']}" for entry in files)


def approval_token(artifacts: dict[str, str], context: str, namespace: str, docs: str = "") -> str:
    lines = sorted(f"{sha}:{path}" for path, sha in artifacts.items())
    return hashlib.sha256("\n".join([*lines, context, namespace, docs]).encode()).hexdigest()[:8]


def version_check(image: str, doc_version: str) -> str:
    """Compare the version in the image tag with the docs version; say if they differ."""
    tag = image.rsplit("/", 1)[-1].partition(":")[2]
    found = re.match(r"v?(\d+)\.(\d+)(?:\.(\d+))?", tag)
    docs = re.match(r"v?(\d+)\.(\d+)(?:\.(\d+|x))?", doc_version.strip().lower())
    if not image:
        return "not compared: the run names no image (the tool picks it)"
    if not found:
        return f"the image tag has no version, so it cannot be compared with docs {doc_version}"
    if not docs:
        return (
            f"docs version {doc_version!r} is not numeric; it cannot be compared with image {tag}"
        )
    image_v = (int(found[1]), int(found[2]), int(found[3] or 0))
    docs_v = (int(docs[1]), int(docs[2]), int(docs[3]) if (docs[3] or "x").isdigit() else None)
    relation = None
    if image_v[:2] != docs_v[:2]:
        relation = "newer" if image_v[:2] > docs_v[:2] else "older"
    elif docs_v[2] is not None and image_v[2] != docs_v[2]:
        relation = "newer" if image_v[2] > docs_v[2] else "older"
    if relation:
        return f"image {relation} than docs: image {tag} vs docs {doc_version}"
    if re.search(r"-(rc|alpha|beta|dev)", tag):
        return f"image {tag} is a pre-release of docs {doc_version}; docs describe the release"
    return f"image {tag} matches docs {doc_version}"


def docs_summary(state: dict[str, Any]) -> dict[str, Any]:
    check = state.get("docs_check") or {}
    image = str(nd.input_value(state, "image") or "")
    if check.get("mode") == "offline":
        warning = f"DOCS NOT CHECKED (offline): {check.get('reason')}"
        basis = "documented default (offline): references/docker.md; DOCS NOT CHECKED"
        return {
            **check,
            "version_check": "not compared (offline)",
            "warning": warning,
            "run_basis": basis,
        }
    compare = version_check(image, str(check.get("doc_version", "")))
    warning = (
        compare
        if re.match(
            r"image (newer|older) than docs|image .* is a pre-release|the image tag|docs version",
            compare,
        )
        else ""
    )
    urls = ", ".join(check.get("urls", []))
    read = f"{check.get('doc_version')} read {check.get('fetched_at')}"
    basis = f"NIM docs {read} from {urls}: {check.get('note')}"
    return {**check, "version_check": compare, "warning": warning, "run_basis": basis}


def container_name(state: dict[str, Any]) -> str:
    return str(runners.runner_of(state).container_name(nd.run_inputs(state)))


def scope(state: dict[str, Any]) -> tuple[str, str]:
    """(context, namespace) of a Kubernetes run."""
    return str(nd.input_value(state, "context")), str(nd.input_value(state, "namespace"))


def kube_flags(state: dict[str, Any]) -> list[str]:
    context, namespace = scope(state)
    return ["--context", context, "-n", namespace]


def helm_argv(folder: Path) -> list[str]:
    """helm-cmd.txt with its relative -f path made absolute, so it runs from any folder."""
    argv = shlex.split((folder / "helm-cmd.txt").read_text())
    for index in range(1, len(argv)):
        if argv[index - 1] in ("-f", "--values") and not Path(argv[index]).is_absolute():
            argv[index] = str(folder / argv[index])
    return argv


def apply_command(state: dict[str, Any]) -> str:
    """One plain command; the guard allows only this exact form, alone in its Bash call."""
    target, folder = state["target"], nd.run_dir(state["run_id"])
    if target in nd.MANIFESTS:
        manifest = str(folder / nd.MANIFESTS[target])
        return shlex.join(["kubectl", "apply", "-f", manifest, *kube_flags(state)])
    if target == "helm":
        if not (folder / "helm-cmd.txt").exists():
            return "<render first: helm-cmd.txt missing>"
        return shlex.join(helm_argv(folder))
    return shlex.join(runners.runner_of(state).apply_command(state))


def rollback_command(state: dict[str, Any]) -> str:
    target, folder = state["target"], nd.run_dir(state["run_id"])
    if target in nd.MANIFESTS:
        manifest = str(folder / nd.MANIFESTS[target])
        return shlex.join(["kubectl", "delete", "-f", manifest, *kube_flags(state)])
    if target == "helm":
        context, namespace = scope(state)
        name = str(nd.input_value(state, "name"))
        return shlex.join(["helm", "uninstall", name, "-n", namespace, "--kube-context", context])
    return shlex.join(runners.runner_of(state).rollback_command(state))


def port_forward(state: dict[str, Any]) -> str | None:
    """Kubernetes only: the background command that makes the service reachable on localhost."""
    name, target = str(nd.input_value(state, "name")), state["target"]
    service = {
        "helm": f"svc/{name}-nim-llm",
        "nim-operator": f"svc/{name}",
    }
    if target not in service:
        return None
    port = nd.input_value(state, "service_port") or nd.input_value(state, "http_port") or 8000
    return shlex.join(
        ["kubectl", "port-forward", service[target], f"8000:{port}", *kube_flags(state)]
    )


def endpoint_url(state: dict[str, Any]) -> str:
    if nd.input_value(state, "url"):
        return str(nd.input_value(state, "url"))
    if state["target"] == "docker":
        return str(runners.runner_of(state).ready_url(nd.run_inputs(state)))
    return "http://localhost:8000"  # Kubernetes: reached through port_forward


def triage_command(state: dict[str, Any]) -> str:
    run = state["run_id"]
    if state["target"] == "docker":
        return script(f"verify.py triage --docker {shlex.quote(container_name(state))} --run {run}")
    context, namespace = (shlex.quote(value) for value in scope(state))
    return script(
        f"verify.py triage --k8s {namespace}/POD --context {context} --run {run}"
        f"  # POD from: kubectl get pods --context {context} -n {namespace}"
    )


def next_step(state: dict[str, Any]) -> tuple[str, str]:
    """(next, next_command) for the run's current state."""
    run, current = state["run_id"], state["state"]
    missing = nd.missing_inputs(state)
    if current in ("validate", "approve") and nd.docs_problem(state):
        hint = f"nimdeploy.py docs-check {run} --url URL --doc-version V --note TEXT"
        return "docs_check", script(f"{hint}  # see references/docs-sources.md")
    if current in ("intake", "plan") and missing:
        keys = " ".join(f"{key}=<ask user>" for key in missing)
        return "ask_user", script(f"nimdeploy.py set {run} {keys}")
    if current in ("intake", "plan"):
        return "render", script(f"nimdeploy.py advance {run} --to render")
    if current == "render":
        return "render", script(RENDERERS[state["target"]].format(run=run))
    url = shlex.quote(endpoint_url(state))
    model = nd.input_value(state, "model")
    model_flag = f" --model {shlex.quote(str(model))}" if model else ""  # else: the one served id
    table = {
        "validate": ("validate", script(f"validate.py --run {run}")),
        "approve": ("ask_user", script(f"nimdeploy.py approval-summary {run}")),
        "apply": ("apply", apply_command(state)),
        "wait_ready": ("wait_ready", script(f"verify.py ready --url {url} --run {run}")),
        "verify": ("verify", script(f"verify.py smoke --url {url}{model_flag} --run {run}")),
        "diagnose": ("diagnose", triage_command(state)),
        "rollback": ("rollback", rollback_command(state)),
        "record": ("record", script(f"nimdeploy.py record {run} --result ok")),
    }
    return table.get(current, ("done", ""))


def defaulted(state: dict[str, Any]) -> dict[str, Any]:
    """Inputs the skill chose (source "default"), for the user to see and change."""
    return {
        key: entry["value"]
        for key, entry in state.get("inputs", {}).items()
        if entry.get("source") == "default"
    }


def status_facts(state: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "run_id": state["run_id"],
        "state": state["state"],
        "target": state["target"],
        "next_command": next_step(state)[1],
        "inputs": {key: entry["value"] for key, entry in state.get("inputs", {}).items()},
        "attempts": state.get("attempts", {}),
        "artifacts": state.get("artifacts", {}),
        "missing": nd.missing_inputs(state),
        "defaulted": sorted(defaulted(state)),
        **({"handoff": handoff.summary(state)} if state.get("handoff") else {}),
        **({"questions": nd.handoff_pending(state)} if nd.handoff_pending(state) else {}),
        "docs_check": {
            k: (state.get("docs_check") or {}).get(k) for k in ("mode", "doc_version", "fetched_at")
        },
    }
    then = {"apply": "wait_ready", "rollback": "record"}.get(state["state"])
    if then:  # a separate command, never chained with &&
        facts["then_command"] = script(f"nimdeploy.py advance {state['run_id']} --to {then}")
    forward = port_forward(state)
    if forward and state["state"] in ("wait_ready", "verify"):
        facts["port_forward"] = forward
    return facts


def describe_objects(state: dict[str, Any], artifacts: dict[str, str]) -> list[dict[str, Any]]:
    if state["target"] == "docker":
        return [{"kind": "container", "name": container_name(state)}]
    objects: list[dict[str, Any]] = []
    for key in artifacts:
        path = artifact_path(state, key)
        if path.name == "helm-cmd.txt":
            objects.append({"kind": "helm release", "name": nd.input_value(state, "name")})
        elif path.suffix == ".yaml" and path.name != "values.yaml":
            for doc in re.split(r"^---\s*$", path.read_text(), flags=re.M):
                kind = re.search(r"^kind:\s*(\S+)", doc, re.M)
                name = re.search(
                    r"^metadata:\n(?:[ \t]+.*\n)*?[ \t]+name:\s*\"?([^\s\"]+)", doc, re.M
                )
                if kind:
                    objects.append({"kind": kind.group(1), "name": name.group(1) if name else None})
    return objects


def instruction(state: dict[str, Any], rolling_back: bool = False) -> str:
    if rolling_back:
        return (
            "This approves the ROLLBACK: apply_command removes the deployment. Show it. "
            "Run approve only after the user replies with the token."
        )
    text = f"{INSTRUCTION} Show `run_basis`, and `docs_warning` first when it is not empty."
    if state.get("handoff"):
        text += (
            " Show `handoff`: the profile id and name, chosen in nim-select, and how it is "
            "pinned; `tuning`; every value from select or tune and each user override in "
            "`sources`; and each of `handoff.warnings`."
        )
    if not defaulted(state):
        return text
    return (
        f"{text} Say which values are defaults ({', '.join(sorted(defaulted(state)))}, listed in "
        "`defaults`) and that the user can change them before approving."
    )


def placement_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Where the pod may run and what the cluster said about it (validate and approve look)."""
    inputs = nd.run_inputs(state)
    check = state.get("placement_check") or {}
    status = check.get("status") or "not checked"
    summary: dict[str, Any] = {
        "node_selector": render_k8s.node_selector(inputs),
        "node_hostnames": render_k8s.node_hostnames(inputs),
        "source": (state.get("inputs", {}).get("node_selector") or {}).get("source")
        or (state.get("inputs", {}).get("node_hostnames") or {}).get("source"),
        "found_by": inputs.get("placement_source"),
        "verification": status,
        "nodes_matched": check.get("nodes_matched"),
        "gpu_name": inputs.get("placement_gpu_name"),
        "gpu_memory_mib": inputs.get("placement_gpu_memory_mib"),
    }
    notes = {
        "none": "The user chose to deploy without placement: any node may be used.",
        "unverified": "The nodes could not be listed on this cluster, so the labels are "
        "UNVERIFIED: ask the user to confirm them before approving.",
        "skipped": "The selector was not checked (offline or kubectl unavailable); approval "
        "requires a live check.",
        "nodes_only": "The selector matches nodes, but the recorded identity or current GPU "
        "labels do not establish the selected GPU configuration.",
    }
    summary["note"] = notes.get(status)
    return summary


def approval_summary(state: dict[str, Any]) -> dict[str, Any]:
    """What the agent shows the user before asking for the token."""
    artifacts = current_artifacts(state)
    context = nd.input_value(state, "context") or ""
    namespace = nd.input_value(state, "namespace") or ""
    rolling_back = state["state"] == "rollback"
    token = approval_token(artifacts, context, namespace, docs_signature(state))
    docs = docs_summary(state)
    port_entry = state.get("inputs", {}).get("container_port") or {}
    why = port_entry.get("note") or "given by the user or the docs"
    port_source = f"{port_entry.get('source')}: {why}" if port_entry else None
    return {
        "run_id": state["run_id"],
        "target": state["target"],
        "model": nd.input_value(state, "model"),
        "image": nd.input_value(state, "image"),
        "namespace": namespace,
        "context": context,
        "gpus": nd.input_value(state, "gpus"),
        "placement": placement_summary(state)
        if state["target"] in ("helm", "nim-operator") else None,
        "name": nd.input_value(state, "name"),
        "port": nd.input_value(state, "port"),
        "defaults": defaulted(state),
        "container_port": nd.input_value(state, "container_port"),
        "port_source": port_source,
        "docs": {
            k: docs.get(k)
            for k in (
                "mode",
                "urls",
                "doc_version",
                "fetched_at",
                "note",
                "reason",
                "version_check",
            )
        },
        "docs_warning": docs["warning"],
        "run_basis": docs["run_basis"],
        "model_profile": nd.input_value(state, "model_profile"),
        "extra_args": nd.input_value(state, "extra_args"),
        "env": nd.input_value(state, "env"),
        "handoff": handoff.summary(state),
        "objects": describe_objects(state, artifacts),
        "artifacts": [
            {"path": str(artifact_path(state, k)), "sha256": s} for k, s in artifacts.items()
        ],
        "apply_command": rollback_command(state) if rolling_back else apply_command(state),
        "token": token,
        "approve_command": script(f"nimdeploy.py approve {state['run_id']} --token {token}"),
        "instruction": instruction(state, rolling_back),
        "apply_note": APPLY_NOTE,
    }


def main(argv: list[str] | None = None) -> None:
    parser = nd.JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/nextstep.py 20260101T000000-helm-qwen",
    )
    parser.add_argument("run", help="run id")
    args = parser.parse_args(argv)
    try:
        state = nd.load_state(args.run)
    except (FileNotFoundError, ValueError) as exc:
        fix = f"List runs with: {script('nimdeploy.py list')}"
        nd.emit(False, "unknown", "ask_user", errors=[nd.error("run_not_found", str(exc), fix)])
    nd.emit(True, state["state"], next_step(state)[0], status_facts(state))


if __name__ == "__main__":
    main()
