# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/render_k8s.py: Helm (nim-llm) and NIM Operator (NIMCache, NIMService)."""

from __future__ import annotations

import os
import shlex
from typing import Any

import nd_common as nd
import pytest
import render_k8s
from _testkit import GOLDEN, make_run, run_cli
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

BASE: dict[str, Any] = {
    "run_id": "golden",
    "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8",
    "namespace": "nim-service",
    "context": "prod-east",
    "name": "llama-31-8b",
    "ngc_secret": "ngc-api-secret",
    "gpus": 1,
    "pull_secret": "ngc-secret",
    "model": "meta/llama-3.1-8b-instruct",
}
MODEL_FREE: dict[str, Any] = {
    "image": "nvcr.io/nim/nvidia/llm-nim:2.0.0",
    "model_path": "hf://Qwen/Qwen3-0.6B",
}


def check_golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(text)
    assert text == path.read_text()


@pytest.mark.parametrize(
    "renderer", [render_k8s.render_helm, render_k8s.render_operator]
)
def test_gpu_product_placement_is_rendered(renderer: Any) -> None:
    selector = {"nvidia.com/gpu.product": "NVIDIA-GB300", "kubernetes.io/arch": "arm64"}
    result = renderer({**BASE, "node_selector": selector})
    spec = result[0] if isinstance(result, tuple) else result[-1]["spec"]
    assert spec["nodeSelector"] == selector


def test_cache_and_service_use_the_same_gpu_placement() -> None:
    selector = {"nvidia.com/gpu.product": "NVIDIA-GB300"}
    cache, service = render_k8s.render_operator({**BASE, "node_selector": selector})
    assert cache["spec"]["nodeSelector"] == service["spec"]["nodeSelector"] == selector


@pytest.mark.parametrize("renderer", [render_k8s.render_helm, render_k8s.render_operator])
def test_selector_json_is_rendered(renderer: Any) -> None:
    result = renderer({**BASE, "node_selector": '{"nvidia.com/gpu.product":"NVIDIA-GB300"}'})
    spec = result[0] if isinstance(result, tuple) else result[-1]["spec"]
    assert spec["nodeSelector"] == {"nvidia.com/gpu.product": "NVIDIA-GB300"}


@pytest.mark.parametrize(
    "value",
    [
        {},
        "[]",
        "null",
        None,
        "{bad",
        {"gpu": 1},
    ],
)
@pytest.mark.parametrize("renderer", [render_k8s.render_helm, render_k8s.render_operator])
def test_invalid_selector_is_refused(value: Any, renderer: Any) -> None:
    with pytest.raises(nd.RenderError, match="node_selector"):
        renderer({**BASE, "node_selector": value})


def test_any_label_is_a_valid_selector() -> None:
    selector = {"cloud.example/accelerator": "a100", "pool": "gpu"}
    values, _ = render_k8s.render_helm({**BASE, "node_selector": selector})
    assert values["nodeSelector"] == selector


@pytest.mark.parametrize(
    "selector",
    [
        {"pool": "a "},
        {"bad/key/extra": "a"},
        {"Bad.Example/pool": "a"},
        {"pool": "a,b"},
        {"a" * 64: "x"},
        {"pool": "a" * 64},
        {"pool": "-bad"},
    ],
)
def test_review_invalid_kubernetes_labels_are_rejected(
    selector: dict[str, str],
) -> None:
    with pytest.raises(nd.RenderError, match="node_selector"):
        render_k8s.render_helm({**BASE, "node_selector": selector})


def test_hostname_fallback_is_node_affinity_in_helm() -> None:
    values, _ = render_k8s.render_helm({**BASE, "node_hostnames": ["n1", "n2"]})
    term = values["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert term["nodeSelectorTerms"][0]["matchExpressions"] == [
        {"key": "kubernetes.io/hostname", "operator": "In", "values": ["n1", "n2"]}
    ]
    assert "nodeSelector" not in values


def test_operator_takes_one_hostname_as_a_selector_and_refuses_several() -> None:
    cache, service = render_k8s.render_operator({**BASE, "node_hostnames": ["n1"]})
    expected = {"kubernetes.io/hostname": "n1"}
    assert cache["spec"]["nodeSelector"] == service["spec"]["nodeSelector"] == expected
    with pytest.raises(nd.RenderError, match="label selector"):
        render_k8s.render_operator({**BASE, "node_hostnames": ["n1", "n2"]})


@pytest.mark.parametrize("renderer", [render_k8s.render_helm, render_k8s.render_operator])
def test_selector_and_hostnames_together_are_refused(renderer: Any) -> None:
    with pytest.raises(nd.RenderError, match="not both"):
        renderer({**BASE, "node_selector": {"a": "b"}, "node_hostnames": ["n1"]})


@pytest.mark.parametrize("value", [[], "[]", "{bad", [""], [1], "null"])
def test_invalid_hostnames_are_refused(value: Any) -> None:
    with pytest.raises(nd.RenderError, match="node_hostnames"):
        render_k8s.render_helm({**BASE, "node_hostnames": value})


@pytest.mark.parametrize("renderer", [render_k8s.render_helm, render_k8s.render_operator])
def test_placement_none_renders_no_placement(renderer: Any) -> None:
    result = renderer({**BASE, "placement": "none", "node_selector": {"a": "b"}})
    spec = result[0] if isinstance(result, tuple) else result[-1]["spec"]
    assert "nodeSelector" not in spec and "affinity" not in spec


@pytest.mark.parametrize(
    ("case", "extra"),
    [
        ("basic", {}),
        (
            "model-free",
            {
                **MODEL_FREE,
                "gpus": 2,
                "served_model_name": "qwen3",
                "hf_token_secret": "hf-token",
                "model_profile": "vllm-bf16-tp2",
                "cache_size": "100Gi",
                "storage_class": "nfs",
                "env": {"NIM_MAX_MODEL_LEN": "32768"},
            },
        ),
    ],
)
def test_helm_matches_golden(case: str, extra: dict[str, Any]) -> None:
    values, command = render_k8s.render_helm({**BASE, **extra})
    check_golden(f"helm-{case}-values.yaml", nd.to_yaml(values))
    check_golden(f"helm-{case}-cmd.txt", command)


@pytest.mark.parametrize(
    ("case", "extra"),
    [
        ("basic", {}),
        (
            "profile-pvc",
            {
                "model_profile": "abc123",
                "tp": 2,
                "gpus": 2,
                "engine": "tensorrt_llm",
                "startup_failure_threshold": 240,
                "service_type": "NodePort",
            },
        ),
        ("model-free", MODEL_FREE),
    ],
)
def test_operator_matches_golden(case: str, extra: dict[str, Any]) -> None:
    check_golden(f"nim-operator-{case}.yaml", render_k8s.operator_text({**BASE, **extra}))


def test_helm_command_has_scope_and_values_file_and_no_set() -> None:
    _, command = render_k8s.render_helm({**BASE, "helm_chart_version": "2.1.4"})
    argv = shlex.split(command)
    assert argv[:4] == ["helm", "upgrade", "--install", "llama-31-8b"]
    assert argv[argv.index("--kube-context") + 1] == "prod-east"
    assert argv[argv.index("-n") + 1] == "nim-service"
    assert argv[argv.index("-f") + 1] == "values.yaml"
    assert argv[argv.index("--version") + 1] == "2.1.4"
    assert not any(word.startswith("--set") for word in argv)


def test_helm_values_reference_secrets_by_name_and_use_custom_args() -> None:
    values, _ = render_k8s.render_helm(
        {**BASE, "extra_args": "--enable-prefix-caching --max-num-batched-tokens 8192"}
    )
    assert values["model"]["ngcAPISecret"] == "ngc-api-secret"
    assert values["imagePullSecrets"] == [{"name": "ngc-secret"}]
    assert values["customArgs"] == ["--enable-prefix-caching", "--max-num-batched-tokens", "8192"]
    assert "NIM_PASSTHROUGH_ARGS" not in str(values)


def test_operator_cache_and_service_are_linked_with_default_startup_window() -> None:
    cache, service = render_k8s.render_operator({**BASE, "extra_args": "--max-model-len 4096"})
    assert (cache["kind"], service["kind"]) == ("NIMCache", "NIMService")
    assert cache["apiVersion"] == service["apiVersion"] == "apps.nvidia.com/v1alpha1"
    assert service["spec"]["storage"]["nimCache"]["name"] == cache["metadata"]["name"]
    startup = service["spec"]["startupProbe"]["probe"]
    assert (startup["failureThreshold"], startup["periodSeconds"]) == (120, 10)  # 20 minutes
    assert startup["httpGet"]["path"] == "/v1/health/ready"
    assert service["spec"]["livenessProbe"]["probe"]["httpGet"]["path"] == "/v1/health/live"
    assert service["spec"]["args"] == ["--max-model-len", "4096"]
    assert cache["spec"]["source"]["ngc"]["authSecret"] == "ngc-api-secret"


def test_operator_without_cache_uses_a_pvc_and_env_profile() -> None:
    documents = render_k8s.render_operator({**BASE, "use_nimcache": "false", "model_profile": "p"})
    assert [doc["kind"] for doc in documents] == ["NIMService"]
    assert "pvc" in documents[0]["spec"]["storage"]
    assert {"name": "NIM_MODEL_PROFILE", "value": "p"} in documents[0]["spec"]["env"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image": "nvcr.io/nim/x"}, "explicit tag"),
        ({"image": "nvcr.io/nim/x@sha256:" + "a" * 64}, "digests are not supported"),
        ({"gpus": "all"}, "GPU count"),
        ({"name": "Bad"}, "name must be a lowercase DNS label"),
        ({"ngc_secret": "A_B"}, "ngc_secret must be a lowercase DNS label"),
        ({"context": "has space"}, "context must be a kube context"),
        ({"cache_size": "50GB"}, "cache_size must look like"),
        ({"env": {"HF_TOKEN": "x"}}, "looks like a credential"),
        ({"extra_args": "--token abc"}, "looks like a credential"),
    ],
)
def test_bad_inputs_are_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(nd.RenderError, match=message):
        render_k8s.render_helm({**BASE, **change})


def test_operator_rejects_a_bad_service_type_and_long_helm_names() -> None:
    with pytest.raises(nd.RenderError, match="service_type"):
        render_k8s.render_operator({**BASE, "service_type": "Magic"})
    with pytest.raises(nd.RenderError, match="at most 53"):
        render_k8s.render_helm({**BASE, "name": "a" * 60})


@pytest.mark.parametrize(
    ("target", "files"),
    [("helm", ["values.yaml", "helm-cmd.txt"]), ("nim-operator", ["nimservice.yaml"])],
)
def test_cli_writes_files_and_advances(target: str, files: list[str]) -> None:
    make_run({k: v for k, v in BASE.items() if k != "run_id"}, target=target)
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, payload = run_cli("render_k8s.py", "--run", "run1", "--kind", target, env=env)
    assert (code, payload["state"]) == (0, "validate")
    assert all((nd.runs_root() / "runs" / "run1" / name).is_file() for name in files)
    assert set(nd.load_state("run1")["artifacts"]) == {
        str(nd.runs_root() / "runs" / "run1" / name) for name in files
    }


def test_cli_kind_must_match_the_run_target_and_bad_input_exits_2() -> None:
    make_run({k: v for k, v in BASE.items() if k != "run_id"}, target="helm")
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, payload = run_cli("render_k8s.py", "--run", "run1", "--kind", "nim-operator", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "wrong_target")
    make_run({**{k: v for k, v in BASE.items() if k != "run_id"}, "gpus": "all"}, target="helm")
    code, payload = run_cli("render_k8s.py", "--run", "run1", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "k8s_input")
