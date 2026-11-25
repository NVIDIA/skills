# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a Kubernetes run's placement on the cluster it will be applied to.

The selector (node labels, or node names for the hostname fallback) is looked up with
`kubectl get nodes -l SELECTOR` on the run's own context, so a selector carried to another
cluster, or a pin that went stale, is caught wherever the run is applied. It must match at least
one node and, when nim-select recorded the nodes of the selected GPU (`placement_nodes`), only
those nodes. Called by validate.py (before approval) and by `nimdeploy.py approve` (right
before apply). Library only.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

import nd_common as nd
import render_k8s

NO_CLUSTER = re.compile(
    r"Unable to connect to the server|connection refused|no such host|i/o timeout|dial tcp"
    r"|TLS handshake timeout",
    re.I,
)
DENIED = re.compile(r"forbidden|Unauthorized|provide credentials", re.I)


# How much a status proves, weakest first; approve refuses to go below what the user approved.
STRENGTH = {"skipped": 0, "unverified": 1, "nodes_only": 2, "verified": 3}


def weaker(now: str | None, before: str | None) -> bool:
    return STRENGTH.get(str(now), 0) < STRENGTH.get(str(before), 0)


def label_selector(inputs: dict[str, Any]) -> str | None:
    """The kubectl -l form of the run's placement, or None when it has none."""
    selector = render_k8s.node_selector(inputs)
    if selector:
        return ",".join(f"{key}={value}" for key, value in sorted(selector.items()))
    names = render_k8s.node_hostnames(inputs)
    if names:
        return f"{render_k8s.HOSTNAME_LABEL} in ({','.join(names)})"
    return None


def list_nodes(argv: list[str]) -> subprocess.CompletedProcess[str]:
    nd.log("placement: running " + " ".join(argv))
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _failed(result: dict[str, Any], message: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    return {**result, "status": "failed", "reason": message}, [
        nd.error("placement_check_failed", message, "Check the context and the cluster, then validate again")
    ]


def check(
    state: dict[str, Any], offline: bool = False, lister: Any = list_nodes
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return (result, errors). result["status"] is one of:
    none        the run has no placement (the user chose to deploy without)
    verified    the current GPU product and memory labels match the selected configuration
    nodes_only  nodes match, but the recorded identity or current labels cannot verify the GPU
    unverified  the nodes could not be listed (no permission); the user must confirm the labels
    skipped     not checked (offline or no kubectl); cannot authorize approval
    failed      the selector matches no node, or nodes outside the selected GPU's, or kubectl
                failed for another reason
    """
    inputs = nd.run_inputs(state)
    try:
        selector = label_selector(inputs)
    except nd.RenderError as exc:
        return {"status": "failed", "reason": str(exc)}, [
            nd.error("render_input", str(exc), "Fix the input with nimdeploy.py set, then re-render")
        ]
    if selector is None:
        return {"status": "none"}, []
    result: dict[str, Any] = {"status": "skipped", "selector": selector}
    if offline or not shutil.which("kubectl"):
        return result, []
    context, namespace = inputs.get("context"), inputs.get("namespace")
    argv = ["kubectl", "get", "nodes", "-l", selector, "-o", "json"]
    argv += ["--context", str(context), "-n", str(namespace)]
    try:
        proc = lister(argv)
    except subprocess.TimeoutExpired:
        return _failed(result, "kubectl node query timed out")
    except OSError as exc:
        return _failed(result, f"kubectl node query failed: {exc}")
    output = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-5:])
    if proc.returncode != 0:
        if DENIED.search(output):
            # Without permission to list nodes the labels cannot be checked: say so, do not guess.
            return {**result, "status": "unverified", "reason": output}, []
        message = f"kubectl could not list the nodes of context {context}: {output}"
        return _failed(result, message)
    try:
        items = json.loads(proc.stdout)["items"]
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        names = [item["metadata"]["name"] for item in items]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("node name must be a non-empty string")
        if any(not isinstance(item["metadata"].get("labels", {}), dict) for item in items):
            raise ValueError("node labels must be a mapping")
        names.sort()
    except (ValueError, KeyError, TypeError, AttributeError):
        return _failed(result, "kubectl returned unreadable node JSON")
    result["nodes_matched"] = names
    fix = (
        "Inspect this cluster again with nim-select, or ask the user for labels that select "
        "the GPU's nodes, then nimdeploy.py set RUN node_selector=..."
    )
    if not names:
        message = f"no node on context {context} matches {selector}"
        return {**result, "status": "failed", "reason": message}, [
            nd.error("placement_no_nodes", message, fix)
        ]
    expected = inputs.get("placement_nodes")
    if isinstance(expected, list):
        extra = [name for name in names if name not in expected]
        if extra:
            message = f"{selector} also matches nodes outside the selected GPU's: {extra}"
            return {**result, "status": "failed", "reason": message}, [
                nd.error("placement_other_nodes", message, fix)
            ]
    gpu = inputs.get("placement_gpu_name")
    memory = inputs.get("placement_gpu_memory_mib")
    verified = (
        isinstance(gpu, str) and bool(gpu) and isinstance(memory, int)
        and not isinstance(memory, bool) and memory > 0
    )
    different = []
    for item in items:
        labels = item["metadata"].get("labels", {})
        product = labels.get("nvidia.com/gpu.product")
        raw_memory = str(labels.get("nvidia.com/gpu.memory", ""))
        current_memory = int(raw_memory) if raw_memory.isascii() and raw_memory.isdigit() else None
        if (gpu and product and product != gpu) or (
            memory is not None and current_memory is not None and current_memory != memory
        ):
            different.append(item["metadata"]["name"])
        verified = verified and product == gpu and current_memory == memory
    if different:
        message = f"{selector} matches nodes with a different GPU configuration: {different}"
        return {**result, "status": "failed", "reason": message}, [
            nd.error("placement_other_nodes", message, fix)
        ]
    result["status"] = "verified" if verified else "nodes_only"
    return result, []
