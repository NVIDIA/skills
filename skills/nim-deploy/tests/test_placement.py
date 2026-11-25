# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live placement check: `kubectl get nodes -l SELECTOR` on the run's own context."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

import _testkit  # noqa: F401  (puts scripts/ on sys.path)
import placement


def state_with(**inputs: Any) -> dict[str, Any]:
    base = {"context": "prod-a", "namespace": "nim-prod", **inputs}
    inputs_ = {key: {"value": value, "source": "user"} for key, value in base.items()}
    return {"run_id": "r1", "inputs": inputs_}


def lister(names: list[str] | None = None, code: int = 0, err: str = "") -> Any:
    seen: list[list[str]] = []

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(argv)
        items = [{"metadata": {"name": name}} for name in names or []]
        return subprocess.CompletedProcess(argv, code, json.dumps({"items": items}), err)

    run.seen = seen  # type: ignore[attr-defined]
    return run


@pytest.fixture(autouse=True)
def kubectl_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(placement.shutil, "which", lambda _name: "/usr/bin/kubectl")


def test_no_placement_is_reported_as_none() -> None:
    assert placement.check(state_with(), lister=lister()) == ({"status": "none"}, [])
    assert placement.check(state_with(placement="none", node_selector={"a": "b"}))[0] == {
        "status": "none"
    }


def test_any_label_is_checked_on_the_runs_own_context() -> None:
    run = lister(["n1", "n2"])
    state = state_with(node_selector={"pool": "gpu", "acc": "a100"}, placement_nodes=["n1", "n2"])
    result, errors = placement.check(state, lister=run)
    assert (result["status"], errors) == ("nodes_only", [])
    argv = run.seen[0]
    assert argv[argv.index("-l") + 1] == "acc=a100,pool=gpu"
    assert argv[argv.index("--context") + 1] == "prod-a"


def test_hostname_fallback_uses_an_in_selector() -> None:
    run = lister(["n1", "n2"])
    result, _ = placement.check(state_with(node_hostnames=["n1", "n2"]), lister=run)
    assert run.seen[0][run.seen[0].index("-l") + 1] == "kubernetes.io/hostname in (n1,n2)"
    assert result["status"] == "nodes_only"


def test_a_selector_that_matches_nothing_is_refused() -> None:
    result, errors = placement.check(state_with(node_selector={"a": "b"}), lister=lister([]))
    assert result["status"] == "failed"
    assert errors[0]["code"] == "placement_no_nodes"


def test_a_selector_that_also_matches_other_gpu_nodes_is_refused() -> None:
    state = state_with(node_selector={"a": "b"}, placement_nodes=["n1"])
    result, errors = placement.check(state, lister=lister(["n1", "other"]))
    assert result["status"] == "failed"
    assert errors[0]["code"] == "placement_other_nodes" and "other" in errors[0]["message"]


def test_a_selector_without_a_node_list_is_nodes_only() -> None:
    result, errors = placement.check(state_with(node_selector={"a": "b"}), lister=lister(["n1"]))
    assert (result["status"], errors) == ("nodes_only", [])


def test_no_permission_to_list_nodes_is_unverified_not_an_error() -> None:
    run = lister(code=1, err='nodes is forbidden: User "u" cannot list resource "nodes"')
    result, errors = placement.check(state_with(node_selector={"a": "b"}), lister=run)
    assert (result["status"], errors) == ("unverified", [])
    assert "forbidden" in result["reason"]


def test_offline_validation_skips_the_check() -> None:
    offline = placement.check(state_with(node_selector={"a": "b"}), offline=True)
    assert offline[0]["status"] == "skipped"


def test_another_kubectl_failure_is_an_error_not_unverified() -> None:
    run = lister(code=1, err='error: context "prod-a" does not exist')
    result, errors = placement.check(state_with(node_selector={"a": "b"}), lister=run)
    assert result["status"] == "failed"
    assert errors[0]["code"] == "placement_check_failed" and "prod-a" in errors[0]["message"]


@pytest.mark.parametrize(
    ("now", "before", "is_weaker"),
    [
        ("unverified", "verified", True),
        ("skipped", "verified", True),
        ("nodes_only", "verified", True),
        ("skipped", "unverified", True),
        ("verified", "verified", False),
        ("verified", "unverified", False),
        ("verified", None, False),
        ("none", "none", False),
    ],
)
def test_a_weaker_status_than_the_approved_one_is_detected(
    now: str, before: str | None, is_weaker: bool
) -> None:
    assert placement.weaker(now, before) is is_weaker


@pytest.mark.parametrize(
    "output",
    ["not json", "[]", '{"items":[{}]}', '{"items":[{"metadata":{"name":null}}]}'],
)
def test_review_malformed_node_output_fails_closed(output: str) -> None:
    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, output, "")

    result, errors = placement.check(
        state_with(node_selector={"pool": "a"}), lister=run
    )
    assert result["status"] == "failed"
    assert errors[0]["code"] == "placement_check_failed"


def test_review_unreachable_cluster_fails_closed() -> None:
    result, errors = placement.check(
        state_with(node_selector={"pool": "a"}),
        lister=lister(code=1, err="connection refused"),
    )
    assert result["status"] == "failed"
    assert errors[0]["code"] == "placement_check_failed"


def test_review_node_query_timeout_fails_closed() -> None:
    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 60)

    result, errors = placement.check(
        state_with(node_selector={"pool": "a"}), lister=run
    )
    assert result["status"] == "failed"
    assert errors[0]["code"] == "placement_check_failed"


@pytest.mark.parametrize(
    ("product", "memory", "status"),
    [
        ("L40S", "46068", "verified"),
        ("OTHER-GPU", "46068", "failed"),
        ("L40S", "24000", "failed"),
        (None, None, "nodes_only"),
    ],
)
def test_review_live_gpu_identity_is_checked_even_when_node_names_match(
    product: str | None, memory: str | None, status: str
) -> None:
    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        labels = {"nvidia.com/gpu.product": product, "nvidia.com/gpu.memory": memory}
        body = {"items": [{"metadata": {"name": "n1", "labels": labels}}]}
        return subprocess.CompletedProcess(argv, 0, json.dumps(body), "")

    state = state_with(
        node_selector={"pool": "a"},
        placement_nodes=["n1"],
        placement_gpu_name="L40S",
        placement_gpu_memory_mib=46068,
    )
    result, errors = placement.check(state, lister=run)
    assert result["status"] == status
    if status == "failed":
        assert errors[0]["code"] == "placement_other_nodes"
