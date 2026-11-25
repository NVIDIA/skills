#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a Kubernetes NIM deployment for a run and advance it from render to validate.

--kind helm          <run>/values.yaml for the nim-llm chart and <run>/helm-cmd.txt, one helm
                     command (its `-f values.yaml` is relative to the run folder)
--kind nim-operator  <run>/nimservice.yaml with a NIMCache and a NIMService
                     (apps.nvidia.com/v1alpha1)

Secrets are referenced by name only. Prints one JSON object. Exit codes: 0 ok, 2 bad input,
3 precondition failed.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_common as nd
from nd_common import RenderError, whole_number

KINDS = ("helm", "nim-operator")
parse_extra_args = nd.parse_extra_args  # shared with the docker runner


INPUT_HELP = """\
run inputs read (required: image namespace context name ngc_secret gpus):
  image              NIM image with an explicit tag, e.g. nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8
  namespace, name    lowercase DNS labels; context is written into the helm command (--kube-context)
  ngc_secret         Secret holding NGC_API_KEY (helm model.ngcAPISecret, operator authSecret)
  gpus               GPUs per pod (nvidia.com/gpu limit)
  node_selector      node labels as a JSON object that select the GPU's nodes; from nim-select's
                     inspection of the cluster, or the user's verified labels
  node_hostnames     JSON list of actual kubernetes.io/hostname label values (not node names),
                     the fallback when no placement key qualifies; the operator accepts one
  pull_secret        image pull secret (default: none)
  replicas           pod count (default 1)
  model_profile      helm env NIM_MODEL_PROFILE; operator NIMCache profiles + nimCache.profile
  model_path         model-free source, NIM_MODEL_PATH (default: unset)
  served_model_name  helm model.name (default: `model`); operator NIM_SERVED_MODEL_NAME
  hf_token_secret    Secret with key HF_TOKEN (default: none)
  extra_args         backend args (shell string): helm customArgs, operator spec.args
  env                extra non-secret environment, {NAME: value}
  cache_size         model cache size (default 50Gi); storage_class (default: cluster default)
helm only:     helm_chart (default nim/nim-llm), helm_chart_version (default: latest)
operator only: use_nimcache (default true; false when model_path is set),
               engine (tensorrt_llm|vllm|sglang), tp (default gpus),
               service_type (ClusterIP|NodePort|LoadBalancer), service_port (8000),
               startup_failure_threshold (default 120 x 10 s = 20 min)
"""
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"nvapi-|^hf_[A-Za-z0-9]{8,}")
_QUANTITY = re.compile(r"[1-9][0-9]*(Mi|Gi|Ti)")
_SERVICE_TYPES = ("ClusterIP", "NodePort", "LoadBalancer")
_LABEL_PART = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?")
HOSTNAME_LABEL = "kubernetes.io/hostname"


def label_key(value: str) -> bool:
    """A Kubernetes label name, optionally qualified by a DNS subdomain."""
    prefix, slash, name = value.rpartition("/")
    return bool(_LABEL_PART.fullmatch(name)) and (
        not slash or (len(prefix) <= 253 and all(nd.is_dns_label(p) for p in prefix.split(".")))
    )


def node_selector(inputs: dict[str, Any]) -> dict[str, str] | None:
    """Read the node labels a pod must match. Any label is accepted: which labels select the
    GPU type is checked against the target cluster (placement.py), not decided here."""
    if inputs.get("placement") == "none":  # the user chose to deploy without placement
        return None
    if "placement" in inputs:
        raise RenderError("placement must be none; otherwise give node_selector or node_hostnames")
    selector = inputs.get("node_selector")
    if isinstance(selector, str):
        try:
            selector = json.loads(selector)
        except ValueError:
            raise RenderError("node_selector must be a JSON object of node labels") from None
    if "node_selector" in inputs and (
        not isinstance(selector, dict)
        or not selector
        or any(
            not isinstance(k, str) or not label_key(k) or not isinstance(v, str)
            or (v != "" and not _LABEL_PART.fullmatch(v))
            for k, v in selector.items()
        )
    ):
        raise RenderError("node_selector must be a non-empty mapping of label names to strings")
    return selector


def node_hostnames(inputs: dict[str, Any]) -> list[str] | None:
    """Read actual kubernetes.io/hostname label values, which can differ from node names."""
    if inputs.get("placement") == "none":
        return None
    names = inputs.get("node_hostnames")
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except ValueError:
            raise RenderError("node_hostnames must be a JSON list of hostname label values") from None
    if "node_hostnames" in inputs and (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or not _LABEL_PART.fullmatch(n) for n in names)
    ):
        raise RenderError("node_hostnames must be a non-empty list of hostname label values")
    if names and node_selector(inputs):
        raise RenderError("give node_selector or node_hostnames, not both")
    return names


def _env(extra: Any) -> list[dict[str, str]]:
    if extra in (None, {}):
        return []
    if not isinstance(extra, dict):
        raise RenderError("env must be a mapping of NAME to value")
    result = []
    for variable in sorted(extra):
        value = str(extra[variable])
        if not _ENV_NAME.fullmatch(variable):
            raise RenderError(f"env name {variable!r} is not a valid variable name")
        if _SECRET_NAME.search(variable) or _SECRET_VALUE.search(value):
            raise RenderError(
                f"env {variable} looks like a credential; use a Secret reference input"
            )
        result.append({"name": variable, "value": value})
    return result


def _common(inputs: dict[str, Any]) -> dict[str, Any]:
    for label in ("name", "namespace", "ngc_secret", "pull_secret", "hf_token_secret"):
        value = inputs.get(label)
        if (
            value is not None or label in ("name", "namespace", "ngc_secret")
        ) and not nd.is_dns_label(value):
            raise RenderError(f"{label} must be a lowercase DNS label of at most 63 characters")
    image = inputs.get("image")
    if not isinstance(image, str) or not nd.IMAGE_REFERENCE.fullmatch(image) or "@" in image:
        raise RenderError(
            "image must be a container image reference with a tag (digests are not supported)"
        )
    repository, _, tag = image.rpartition(":")
    if not repository or "/" in tag or not _TAG.fullmatch(tag):
        raise RenderError(
            "image needs an explicit tag, e.g. nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8"
        )
    if inputs.get("gpus") == "all":
        raise RenderError("gpus must be a GPU count for Kubernetes targets, not 'all'")
    cache_size = str(inputs.get("cache_size", "50Gi"))
    if not _QUANTITY.fullmatch(cache_size):
        raise RenderError("cache_size must look like 50Gi")
    context = inputs.get("context")
    if not isinstance(context, str) or not context or any(ord(char) < 33 for char in context):
        raise RenderError("context must be a kube context name without spaces")
    return {
        "repository": repository,
        "tag": tag,
        "gpus": whole_number(inputs, "gpus"),
        "replicas": whole_number(inputs, "replicas", 1),
        "pull_secret": inputs.get("pull_secret"),
        "cache_size": cache_size,
        "env": _env(inputs.get("env")),
        "args": parse_extra_args(inputs.get("extra_args")),
        "node_selector": node_selector(inputs),
        "node_hostnames": node_hostnames(inputs),
    }


def render_helm(inputs: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return (values, command). The command reads `-f values.yaml` relative to the run folder."""
    common = _common(inputs)
    if len(inputs["name"]) > 53:
        raise RenderError("name must be at most 53 characters for a Helm release")
    model: dict[str, Any] = {"ngcAPISecret": inputs["ngc_secret"]}
    served = inputs.get("served_model_name") or inputs.get("model")
    if served:  # the chart uses model.name for `helm test`
        model["name"] = str(served)
    for key, field in (("model_path", "modelPath"), ("hf_token_secret", "hfTokenSecret")):
        if inputs.get(key):
            model[field] = str(inputs[key])
    persistence: dict[str, Any] = {"enabled": True, "size": common["cache_size"]}
    if inputs.get("storage_class"):
        persistence["storageClass"] = str(inputs["storage_class"])
    pull = [{"name": common["pull_secret"]}] if common["pull_secret"] else []
    values: dict[str, Any] = {
        "image": {
            "repository": common["repository"],
            "tag": common["tag"],
            "pullPolicy": "IfNotPresent",
        },
        "imagePullSecrets": pull,
        "replicaCount": common["replicas"],
        "model": model,
        "resources": {"limits": {"nvidia.com/gpu": common["gpus"]}},
        "persistence": persistence,
        "podAnnotations": {"nim-deploy/run-id": str(inputs.get("run_id", ""))},
    }
    if common["node_selector"]:
        values["nodeSelector"] = common["node_selector"]
    if common["node_hostnames"]:
        values["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": HOSTNAME_LABEL,
                                    "operator": "In",
                                    "values": common["node_hostnames"],
                                }
                            ]
                        }
                    ]
                }
            }
        }
    env = list(common["env"])
    if inputs.get(
        "model_profile"
    ):  # model.profile only feeds multi-node; single-node reads the env var
        env.append({"name": "NIM_MODEL_PROFILE", "value": str(inputs["model_profile"])})
    if env:
        values["env"] = env
    if common[
        "args"
    ]:  # never NIM_PASSTHROUGH_ARGS in env: Kubernetes would replace the image value
        values["customArgs"] = common["args"]
    words = [
        "helm",
        "upgrade",
        "--install",
        inputs["name"],
        str(inputs.get("helm_chart", "nim/nim-llm")),
    ]
    words += ["--kube-context", inputs["context"], "-n", inputs["namespace"], "-f", "values.yaml"]
    if inputs.get("helm_chart_version"):
        words += ["--version", str(inputs["helm_chart_version"])]
    return values, shlex.join(words) + "\n"


def _probe(path: str, initial_delay: int, failure_threshold: int) -> dict[str, Any]:
    http_get = {"path": path, "port": "api"}
    return {
        "enabled": True,
        "probe": {
            "httpGet": http_get,
            "initialDelaySeconds": initial_delay,
            "timeoutSeconds": 1,
            "periodSeconds": 10,
            "successThreshold": 1,
            "failureThreshold": failure_threshold,
        },
    }


def render_operator(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    common = _common(inputs)
    name, namespace = inputs["name"], inputs["namespace"]
    pinned = common["node_selector"]
    if common["node_hostnames"]:
        if len(common["node_hostnames"]) > 1:
            raise RenderError(
                "the NIM Operator takes a label selector, not several hostname values: set "
                "node_selector to a label that only the selected GPU's nodes carry"
            )
        pinned = {HOSTNAME_LABEL: common["node_hostnames"][0]}
    metadata = {
        "name": name,
        "namespace": namespace,
        "labels": {"app.kubernetes.io/managed-by": "nim-deploy"},
        "annotations": {"nim-deploy/run-id": str(inputs.get("run_id", ""))},
    }
    use_cache = nd.flag_input(inputs, "use_nimcache", not inputs.get("model_path"))
    pvc = {
        "create": True,
        "storageClass": str(inputs.get("storage_class", "")),
        "size": common["cache_size"],
        "volumeAccessMode": "ReadWriteOnce",
    }
    documents: list[dict[str, Any]] = []
    if use_cache:
        model_filter: dict[str, Any] = {}
        if inputs.get("model_profile"):
            model_filter["profiles"] = [str(inputs["model_profile"])]
        else:
            if inputs.get("engine"):
                model_filter["engine"] = str(inputs["engine"])
            model_filter["tensorParallelism"] = str(whole_number(inputs, "tp", common["gpus"]))
        ngc: dict[str, Any] = {"modelPuller": inputs["image"]}
        if common["pull_secret"]:
            ngc["pullSecret"] = common["pull_secret"]
        ngc["authSecret"] = inputs["ngc_secret"]
        ngc["model"] = model_filter
        cache_spec: dict[str, Any] = {"source": {"ngc": ngc}, "storage": {"pvc": pvc}}
        if pinned:
            cache_spec["nodeSelector"] = pinned
        documents.append(
            {
                "apiVersion": "apps.nvidia.com/v1alpha1",
                "kind": "NIMCache",
                "metadata": metadata,
                "spec": cache_spec,
            }
        )
        storage: dict[str, Any] = {
            "nimCache": {"name": name, "profile": str(inputs.get("model_profile", ""))}
        }
    else:
        storage = {"pvc": {"name": f"{name}-model-store", **pvc}}
    image: dict[str, Any] = {
        "repository": common["repository"],
        "tag": common["tag"],
        "pullPolicy": "IfNotPresent",
    }
    if common["pull_secret"]:
        image["pullSecrets"] = [common["pull_secret"]]
    env = list(common["env"])
    if inputs.get("model_path"):
        env.insert(0, {"name": "NIM_MODEL_PATH", "value": str(inputs["model_path"])})
    if inputs.get("served_model_name"):
        env.append({"name": "NIM_SERVED_MODEL_NAME", "value": str(inputs["served_model_name"])})
    if not use_cache and inputs.get("model_profile"):
        env.append({"name": "NIM_MODEL_PROFILE", "value": str(inputs["model_profile"])})
    if inputs.get("hf_token_secret"):
        ref = {"secretKeyRef": {"name": inputs["hf_token_secret"], "key": "HF_TOKEN"}}
        env.append({"name": "HF_TOKEN", "valueFrom": ref})
    service_type = inputs.get("service_type", "ClusterIP")
    if service_type not in _SERVICE_TYPES:
        raise RenderError(f"service_type must be one of {', '.join(_SERVICE_TYPES)}")
    spec: dict[str, Any] = {
        "image": image,
        "authSecret": inputs["ngc_secret"],
        "storage": storage,
        "replicas": common["replicas"],
        "resources": {"limits": {"nvidia.com/gpu": common["gpus"]}},
        "expose": {
            "service": {"type": service_type, "port": whole_number(inputs, "service_port", 8000)}
        },
    }
    if pinned:
        spec["nodeSelector"] = pinned
    if env:
        spec["env"] = env
    if common["args"]:  # each option and value is its own list item
        spec["args"] = common["args"]
    # Operator defaults, written out so a reviewer sees them and large models can raise them.
    spec["startupProbe"] = _probe(
        "/v1/health/ready", 30, whole_number(inputs, "startup_failure_threshold", 120)
    )
    spec["readinessProbe"] = _probe("/v1/health/ready", 15, 3)
    spec["livenessProbe"] = _probe("/v1/health/live", 15, 3)
    documents.append(
        {
            "apiVersion": "apps.nvidia.com/v1alpha1",
            "kind": "NIMService",
            "metadata": metadata,
            "spec": spec,
        }
    )
    return documents


def operator_text(inputs: dict[str, Any]) -> str:
    return "---\n".join(nd.to_yaml(document) for document in render_operator(inputs))


def main(argv: list[str] | None = None) -> None:
    parser = nd.render_parser(
        "Render a Helm or NIM Operator deployment and advance the run to validate.", INPUT_HELP
    )
    parser.add_argument("--kind", choices=KINDS, help="target kind (default: the run's target)")
    args = parser.parse_args(argv)
    state, inputs = nd.open_run(args.run, (args.kind,) if args.kind else KINDS, args.preview)
    folder = nd.run_dir(state["run_id"])
    try:
        if state["target"] == "helm":
            values, command = render_helm(inputs)
        else:
            text = operator_text(inputs)
    except RenderError as exc:
        nd.reject_input(state, "k8s_input", exc)
    if state["target"] == "helm":
        values_path, command_path = folder / "values.yaml", folder / "helm-cmd.txt"
        nd.write_atomic(values_path, nd.to_yaml(values))
        nd.write_atomic(command_path, command)
        facts = {
            "path": str(values_path),
            "sha256": nd.sha256_file(values_path),
            "command": command.strip(),
        }
        nd.close_run(state, [values_path, command_path], facts, args.preview)
    path = folder / "nimservice.yaml"
    nd.write_atomic(path, text)
    nd.close_run(state, [path], {"path": str(path), "sha256": nd.sha256_file(path)}, args.preview)


if __name__ == "__main__":
    main()
