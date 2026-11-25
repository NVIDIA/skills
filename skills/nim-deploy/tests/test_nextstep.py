# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nextstep.py: the command for each state and the approval summary."""

from __future__ import annotations

import os

import nd_common as nd
import nextstep
import pytest
from _testkit import (  # noqa: F401
    DOCKER_VALUES,
    K8S_VALUES,
    make_run,
    run_cli,
    runs_home,
    with_docs,
)

pytestmark = pytest.mark.usefixtures("runs_home")


def test_kubectl_forms_carry_context_and_namespace_for_kubernetes_targets() -> None:
    state = make_run(K8S_VALUES, target="nim-operator", state="apply")
    assert nextstep.kube_flags(state) == [
        "--context",
        K8S_VALUES["context"],
        "-n",
        K8S_VALUES["namespace"],
    ]
    assert nextstep.apply_command(state).endswith(
        f"--context {K8S_VALUES['context']} -n {K8S_VALUES['namespace']}"
    )
    assert nextstep.rollback_command(state).startswith("kubectl delete -f ")


def test_docker_commands_come_from_the_runner() -> None:
    state = make_run(DOCKER_VALUES, target="docker", state="apply")
    folder = nd.run_dir("run1")
    assert nextstep.apply_command(state) == f"bash {folder}/docker-run.sh"
    assert nextstep.rollback_command(state) == "docker rm -f nim-llama"
    assert nextstep.container_name(state) == "nim-llama"
    assert nextstep.endpoint_url(state) == "http://localhost:8000"
    assert nextstep.port_forward(state) is None


def test_the_url_input_overrides_the_endpoint() -> None:
    state = make_run({**DOCKER_VALUES, "url": "http://10.0.0.5:9000"}, target="docker")
    assert nextstep.endpoint_url(state) == "http://10.0.0.5:9000"


def test_next_step_walks_the_states() -> None:
    expected = {
        "intake": ("ask_user", "set run1 image="),
        "render": ("render", "render_k8s.py --run run1 --kind nim-operator"),
        "validate": ("validate", "validate.py --run run1"),
        "approve": ("ask_user", "approval-summary run1"),
        "diagnose": ("diagnose", "verify.py triage --k8s nim-prod/POD --context prod-a"),
        "done": ("done", ""),
    }
    for state_name, (next_, text) in expected.items():
        values = {} if state_name == "intake" else K8S_VALUES
        state = with_docs(make_run(values, state=state_name))
        got_next, command = nextstep.next_step(state)
        assert got_next == next_ and text in command, state_name


def test_approval_summary_binds_the_scope_into_the_token() -> None:
    state = with_docs(make_run(K8S_VALUES, state="approve"))
    path = nd.run_dir("run1") / "nimservice.yaml"
    path.write_text("kind: NIMService\nmetadata:\n  name: llama\n")
    state["artifacts"][str(path)] = nd.sha256_file(path)
    summary = nextstep.approval_summary(state)
    assert summary["objects"] == [{"kind": "NIMService", "name": "llama"}]
    sign = nextstep.docs_signature(state)
    artifacts = nextstep.current_artifacts(state)
    assert summary["token"] == nextstep.approval_token(artifacts, "prod-a", "nim-prod", sign)
    assert summary["token"] != nextstep.approval_token(artifacts, "other", "nim-prod", sign)
    assert summary["token"] != nextstep.approval_token(
        artifacts, "prod-a", "nim-prod", "docs:offline::"
    )
    assert summary["approve_command"].endswith(f"approve run1 --token {summary['token']}")


def test_cli_prints_the_same_facts_as_status() -> None:
    with_docs(make_run(K8S_VALUES, state="validate"))
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, payload = run_cli("nextstep.py", "run1", env=env)
    assert (code, payload["next"]) == (0, "validate")
    assert payload["facts"] == nextstep.status_facts(nd.load_state("run1"))
    code, payload = run_cli("nextstep.py", "nope", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "run_not_found")
