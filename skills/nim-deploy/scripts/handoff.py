#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the nim-select and nim-tune result files for `nimdeploy.py new --from FILE`.

Kinds: a nim-select `select.json` (the `select --out` facts), a nim-select `env.json`
(`schema: nim-select/environment/v1`, only together with its select.json) and a nim-tune result
(`apiVersion: nim-tune/v1`, `kind: RecommendedConfig`, references/tune-result.schema.json).
`build` checks them and maps them to run inputs with source `select` or `tune`; it never guesses:
a malformed file is HandoffError exit 2, a refusal (does not fit, files disagree) exit 3.
Things the user must answer (stale file, other environment, GPU ids) become `pending` questions.
`python3 scripts/handoff.py FILE...` prints what the files give, without creating a run.
Stdlib only. Prints one JSON object. Exit codes: 0 ok, 2 bad input, 3 precondition failed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_common as nd

TUNE_API, TUNE_KIND = "nim-tune/v1", "RecommendedConfig"
ENV_SCHEMA = "nim-select/environment/v1"
SCHEMA_FILE = Path(__file__).resolve().parent.parent / "references" / "tune-result.schema.json"
MAX_AGE = timedelta(days=7)
CONFIRM_KEY = nd.HANDOFF_CONFIRM
K8S_TARGETS = ("helm", "nim-operator")
SELECT_VERDICTS = ("fits", "unknown", "does_not_fit")
SCHEMA_FIX = (
    "Use the file nim-tune wrote (final/recommended_config.json) unchanged, or fix the field so it "
    "matches references/tune-result.schema.json; never fill a field in by guessing"
)
SELECT_FIX = (
    "Pass the file written by nimselect.py select ... --out ~/.cache/nim-select/select.json, unchanged"
)
# Environment the run sets through its own inputs (so it is never rendered twice).
ENV_INPUTS = {"NIM_MODEL_PATH": "model_path", "NIM_SERVED_MODEL_NAME": "served_model_name"}
SECRET_ENV = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


class HandoffError(Exception):
    """A file that cannot be used: (code, message, fix, exit code)."""

    def __init__(self, code: str, message: str, fix: str, exit_code: int = nd.EXIT_INPUT) -> None:
        super().__init__(message)
        self.code, self.message, self.fix, self.exit_code = code, message, fix, exit_code


# --- a JSON Schema subset (stdlib only) -----------------------------------------------------

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _is_type(value: Any, name: str) -> bool:
    if name in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[name])


def schema_errors(value: Any, schema: dict[str, Any], where: str = "$") -> list[str]:
    """Problems of VALUE against SCHEMA, one line each with its JSON path. Supports type, const,
    enum, required, properties, additionalProperties, propertyNames, items, minLength, maxLength,
    minimum and pattern: the subset references/tune-result.schema.json uses."""
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, name) for name in names):
            return [f"{where}: expected {' or '.join(names)}, got {json.dumps(value)[:60]}"]
    if "const" in schema and value != schema["const"]:
        return [f"{where}: must be {json.dumps(schema['const'])}, got {json.dumps(value)[:60]}"]
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(json.dumps(item) for item in schema["enum"])
        return [f"{where}: must be one of {allowed}, got {json.dumps(value)[:60]}"]
    found: list[str] = []
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            found.append(f"{where}: must not be empty")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            found.append(f"{where}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            found.append(f"{where}: {value!r} does not match {schema['pattern']}")
    if _is_type(value, "number") and "minimum" in schema and value < schema["minimum"]:
        found.append(f"{where}: must be at least {schema['minimum']}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            found += schema_errors(item, schema["items"], f"{where}[{index}]")
    if isinstance(value, dict):
        found += [
            f"{where}: missing {key!r}" for key in schema.get("required", []) if key not in value
        ]
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if "propertyNames" in schema:
                found += schema_errors(key, schema["propertyNames"], f"{where}.{key} (name)")
            if key in properties:
                found += schema_errors(item, properties[key], f"{where}.{key}")
            elif extra is False:
                found.append(f"{where}: unknown field {key!r}")
            elif isinstance(extra, dict):
                found += schema_errors(item, extra, f"{where}.{key}")
    return found


# --- reading the files ----------------------------------------------------------------------


@dataclass
class HandoffFile:
    kind: str  # select | tune | env
    path: str
    sha256: str
    data: dict[str, Any]
    created_at: str | None
    run_id: str | None  # the tune campaign; select and env have none

    def entry(self, now: datetime) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "age_days": age_days(self.created_at, now),
        }


def parse_time(text: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)  # noqa: UP017


def age_days(created_at: str | None, now: datetime) -> float | None:
    moment = parse_time(created_at) if created_at else None
    return round((now - moment).total_seconds() / 86400, 1) if moment else None


def read_file(path: str) -> tuple[dict[str, Any], str]:
    file = Path(path).expanduser()
    if not file.is_file():
        raise HandoffError(
            "handoff_missing",
            f"{path} does not exist",
            "Pass the path the nim-select or nim-tune step printed; ask the user if unsure",
        )
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise HandoffError(
            "handoff_unreadable", f"{path} is not JSON: {exc}"[:300], SELECT_FIX
        ) from None
    if not isinstance(data, dict):
        raise HandoffError("handoff_unknown", f"{path} is not a JSON object", SELECT_FIX)
    return data, nd.sha256_file(file)


def kind_of(path: str, data: dict[str, Any]) -> str:
    """select, tune or env; HandoffError (exit 2) for anything else or an unknown version."""
    api = data.get("apiVersion")
    if api is not None or data.get("kind") == TUNE_KIND:
        if api != TUNE_API or data.get("kind") != TUNE_KIND:
            message = (
                f"{path} has apiVersion {api!r}, kind {data.get('kind')!r}; this nim-deploy reads "
                f"apiVersion {TUNE_API!r}, kind {TUNE_KIND!r}"
            )
            raise HandoffError("handoff_version", message, SCHEMA_FIX)
        return "tune"
    schema = data.get("schema")
    if isinstance(schema, str) and schema.startswith("nim-select/"):
        if schema != ENV_SCHEMA:
            message = f"{path} has schema {schema!r}; this nim-deploy reads {ENV_SCHEMA!r}"
            raise HandoffError("handoff_version", message, SELECT_FIX)
        return "env"
    if "handoff" in data or "images_considered" in data or "environment_summary" in data:
        return "select"
    message = f"{path} is neither a nim-select select.json or env.json nor a nim-tune result"
    raise HandoffError("handoff_unknown", message, SELECT_FIX)


def require(path: str, problems: list[str], fix: str) -> None:
    if problems:
        message = f"{path}: " + "; ".join(problems[:5])
        raise HandoffError("handoff_invalid", message[:600], fix)


def check_select(path: str, data: dict[str, Any]) -> None:
    """Fields deploy reads; a select that found nothing to run is a refusal (exit 3)."""
    verdict = data.get("verdict")
    handoff = data.get("handoff")
    if verdict == "does_not_fit" or (verdict is None and "images_considered" in data):
        reason = "does not fit" if verdict else "has no profile (no metadata for any image)"
        raise HandoffError(
            "select_does_not_fit",
            f"{path}: nim-select says the model {reason}; there is no profile to deploy",
            "Stop and ask the user: run nim-select again with more GPUs, another GPU type or tag. "
            "Never deploy without a selected profile",
            nd.EXIT_PRECONDITION,
        )
    problems: list[str] = []
    if verdict not in SELECT_VERDICTS:
        problems.append(f"verdict must be one of {', '.join(SELECT_VERDICTS)}, got {verdict!r}")
    environment = data.get("environment_summary")
    if not isinstance(environment, dict) or environment.get("target") not in ("host", "k8s"):
        problems.append("environment_summary.target must be host or k8s")
    require(path, problems, SELECT_FIX)
    if not isinstance(handoff, dict) or not handoff.get("model_profile"):
        raise HandoffError(
            "select_no_profile",
            f"{path} has no recommended profile (no handoff.model_profile)",
            "Stop and ask the user: finish nim-select (the user confirms a profile) first",
            nd.EXIT_PRECONDITION,
        )
    for key, kinds in (("image", str), ("model_profile", str), ("gpu_count", int)):
        if not isinstance(handoff.get(key), kinds) or handoff.get(key) in ("", None):
            problems.append(f"handoff.{key} is missing or not a {kinds.__name__}")
    ids = handoff.get("gpu_ids")
    if ids is not None and not (isinstance(ids, list) and all(isinstance(i, int) for i in ids)):
        problems.append("handoff.gpu_ids must be a list of GPU indices or null")
    require(path, problems, SELECT_FIX)


def check_tune(path: str, data: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    problems = schema_errors(data, schema)
    if not problems and parse_time(data["created_at"]) is None:
        problems.append(f"$.created_at: {data['created_at']!r} is not an ISO 8601 time")
    if not problems and data["baseline_won"] and data["args"]:
        problems.append("$.args: must be [] when baseline_won is true (the untuned config won)")
    require(path, problems, SCHEMA_FIX)


def check_env(path: str, data: dict[str, Any]) -> None:
    problems = []
    if data.get("target") not in ("host", "k8s"):
        problems.append("target must be host or k8s")
    if not isinstance(data.get("captured_at"), str):
        problems.append("captured_at is missing")
    require(path, problems, "Pass the file written by nimselect.py inspect --out env.json")


def load(paths: list[str]) -> dict[str, HandoffFile]:
    files: dict[str, HandoffFile] = {}
    for path in paths:
        data, sha = read_file(path)
        kind = kind_of(path, data)
        if kind in files:
            message = f"two {kind} files: {files[kind].path} and {path}"
            raise HandoffError("handoff_duplicate", message, "Pass each kind of file once")
        {"select": check_select, "tune": check_tune, "env": check_env}[kind](path, data)
        created = {
            "select": (data.get("environment_summary") or {}).get("captured_at"),
            "tune": data.get("created_at"),
            "env": data.get("captured_at"),
        }[kind]
        run_id = str(data["campaign"]) if kind == "tune" else None
        resolved = str(Path(path).expanduser().resolve())
        files[kind] = HandoffFile(kind, resolved, sha, data, created, run_id)
    if "env" in files and "select" not in files:
        raise HandoffError(
            "handoff_env_alone",
            "env.json gives only the environment; deploy needs the select.json made with it",
            "Pass --from ~/.cache/nim-select/select.json too (or drop the env.json)",
        )
    return files


# --- mapping to run inputs ------------------------------------------------------------------


def split_digest(image: str) -> tuple[str, str | None]:
    name, _, digest = image.partition("@")
    return name, digest or None


def same_model(left: str, right: str) -> bool:
    def norm(text: str) -> str:
        return text.strip().lower().removeprefix("nim/")

    return norm(left) == norm(right)


def profile_name(select: dict[str, Any] | None, profile: str) -> str | None:
    """backend-precision-tpN-ppN of the profile in select's records (the metadata has no name)."""
    if not select:
        return None
    for key in ("recommended", "best_unconfirmed"):
        record = select.get(key)
        if isinstance(record, dict) and record.get("profile_id") == profile:
            parts = [record.get("backend"), record.get("precision")]
            parts += [f"tp{record['tp']}" if record.get("tp") else None]
            parts += [f"pp{record['pp']}" if record.get("pp") else None]
            return "-".join(str(part) for part in parts if part) or None
    return None


def pinned_as(target: str, runner: str) -> str:
    if target == "helm":
        return "env NIM_MODEL_PROFILE"
    if target == "nim-operator":
        return "NIMCache spec.source.ngc.model.profiles and NIMService storage.nimCache.profile"
    return "--profile" if runner == "nimcli" else "-e NIM_MODEL_PROFILE"


def derive_target(files: dict[str, HandoffFile]) -> str:
    select, tune = files.get("select"), files.get("tune")
    host = select and select.data["environment_summary"]["target"] == "host"
    if host or (not select and tune and tune.data["target"] == "docker"):
        return "docker"
    if not select and tune:
        return str(tune.data["target"])
    raise HandoffError(
        "target_required",
        "select.json is for Kubernetes; Helm or the NIM Operator is the user's choice",
        "Ask the user: Kubernetes with Helm or with the NIM Operator? Then add --target",
    )


@dataclass
class Plan:
    target: str
    values: dict[str, dict[str, Any]] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)

    def put(self, key: str, value: Any, source: str) -> None:
        self.values[key] = {"value": value, "source": source}


def gpu_key(name: str) -> str:
    """A GPU model name compared without case, spacing or the NVIDIA prefix."""
    return " ".join(name.lower().replace("nvidia", " ").split())


def gpu_notes(plan: Plan, select: HandoffFile | None, tune: HandoffFile) -> None:
    """What the files say about the GPU model: the result must not be used on other hardware
    unnoticed. A mismatch between two named GPUs is a refusal (disagreements); an unknown name
    is a warning and a question."""
    measured = tune.data["gpu_name"]
    if measured is None:
        plan.record["warnings"].append(
            "the tune result does not say which GPU model it was measured on (gpu_name is null): "
            "its settings were not checked against this hardware"
        )
        plan.record["confirm"].append("the tune result has no gpu_name")
        return
    checked = select.data["handoff"].get("gpu_name") if select else None
    if not checked:
        where = "nim-select did not name its GPU" if select else "no nim-select file was given"
        plan.record["warnings"].append(
            f"measured on {measured}: {where}, so check that the GPUs you chose are {measured}"
        )


def disagreements(select: HandoffFile, tune: HandoffFile) -> list[str]:
    handoff, data = select.data["handoff"], tune.data
    found = []
    if handoff.get("model") and not same_model(str(handoff["model"]), str(data["model"])):
        found.append(f"model: select {handoff['model']!r}, tune {data['model']!r}")
    select_image, select_digest = split_digest(str(handoff["image"]))
    if select_image != data["image"]:
        found.append(f"image: select {select_image!r}, tune {data['image']!r}")
    tune_digest = data.get("image_digest")
    if select_digest and tune_digest and select_digest != tune_digest:
        found.append(f"image digest: select {select_digest!r}, tune {tune_digest!r}")
    if handoff["model_profile"] != data["model_profile"]:
        found.append(
            f"model_profile: select {handoff['model_profile']!r}, tune {data['model_profile']!r}"
        )
    select_gpu, tune_gpu = handoff.get("gpu_name"), data["gpu_name"]
    if select_gpu and tune_gpu and gpu_key(str(select_gpu)) != gpu_key(str(tune_gpu)):
        found.append(f"gpu_name: select {select_gpu!r}, tune {tune_gpu!r}")
    if int(handoff["gpu_count"]) != int(data["gpus"]):
        found.append(f"gpus: select {handoff['gpu_count']}, tune {data['gpus']}")
    return found


def map_placement(
    plan: Plan, handoff: dict[str, Any], environment: dict[str, Any], user: dict[str, Any]
) -> None:
    """Take the placement nim-select worked out for the selected GPU from this cluster's
    inspection ({selector | hostnames, nodes_matched, source}); without it, ask the user."""
    product = handoff.get("gpu_name")
    same_cluster = environment.get("target") == "k8s" and (
        not user.get("context") or user["context"] == environment.get("context")
    )
    groups = [
            group
            for group in environment.get("groups") or []
            if same_cluster
            and isinstance(group, dict)
            and isinstance(product, str)
            and product
            and group.get("name") == product
            and ("gpu_memory_mib" not in handoff
                 or group.get("memory_mib") == handoff["gpu_memory_mib"])
    ]
    # Older select files have no memory identity. Never guess between same-name groups.
    if groups and all(group.get("unified_memory") is True for group in groups):
        # The saved capacity is shared node memory, not a dedicated-GPU memory label.
        plan.values.pop("placement_gpu_memory_mib", None)
    found = groups[0].get("placement") if len(groups) == 1 else None
    if not isinstance(found, dict):
        found = None
    if found and found.get("selector"):
        plan.put("node_selector", found["selector"], "select")
    elif found and found.get("hostnames"):
        plan.put("node_hostnames", found["hostnames"], "select")
        plan.record["warnings"].append(
            "No node label selects only the selected GPU, so the pod is pinned to its nodes by "
            "kubernetes.io/hostname; this breaks when those nodes are replaced."
        )
    else:
        plan.record["pending"].append(
            {
                "key": "placement",
                "question": "nim-select found no placement for the selected GPU on this cluster. "
                "Which node labels select its nodes (set node_selector as a JSON object, checked "
                "with kubectl get nodes -l), or which kubernetes.io/hostname label values "
                "(node_hostnames as a JSON list, not node names), "
                "or should it deploy without placement (placement=none)?",
            }
        )
        return
    if not isinstance(found.get("nodes_matched"), list):
        raise HandoffError(
            "handoff_invalid",
            "Selected GPU group placement.nodes_matched must be a list",
            SELECT_FIX,
        )
    plan.put("placement_nodes", found["nodes_matched"], "select")
    plan.put("placement_source", str(found.get("source")), "select")
    plan.record["warnings"].append(
        "GPU placement does not confirm free GPU capacity; the pod may remain Pending."
    )


def map_select(plan: Plan, select: HandoffFile, runner: str, user: dict[str, Any]) -> None:
    handoff, environment = select.data["handoff"], select.data["environment_summary"]
    plan.put("image", str(handoff["image"]), "select")
    plan.put("model_profile", str(handoff["model_profile"]), "select")
    ids = handoff.get("gpu_ids")
    if plan.target in K8S_TARGETS:
        plan.put("gpus", str(handoff["gpu_count"]), "select")
        if handoff.get("gpu_name"):
            plan.put("placement_gpu_name", handoff["gpu_name"], "select")
        if handoff.get("gpu_memory_mib") is not None:
            plan.put("placement_gpu_memory_mib", handoff["gpu_memory_mib"], "select")
        for key in ("context", "namespace"):
            if environment.get("target") == "k8s" and environment.get(key):
                plan.put(key, str(environment[key]), "select")
        map_placement(plan, handoff, environment, user)
    elif ids:
        joined = ",".join(str(i) for i in ids)
        plan.put("gpus", joined if runner == "nimcli" else f"device={joined}", "select")
    else:
        # nim-select could not tell which GPUs are idle; the runner default (all GPUs) is unsafe
        plan.record["pending"].append(
            {
                "key": "gpus",
                "question": f"nim-select did not say which GPUs are idle. Which GPUs should it use "
                f"({handoff['gpu_count']} GPU(s), device ids such as 0,1)?",
            }
        )
    if handoff.get("verdict") == "unknown":
        plan.record["warnings"].append(
            "nim-select verdict is unknown: the profile is not confirmed to fit these GPUs"
        )
    if runner == "nimcli":
        plan.put("model", str(handoff["image"]), "select")
    deploy_k8s = plan.target in K8S_TARGETS
    if (environment.get("target") == "k8s") != deploy_k8s:
        where = "Kubernetes" if environment.get("target") == "k8s" else "this GPU host"
        plan.record["warnings"].append(
            f"nim-select inspected {where}, but this run deploys with target {plan.target}"
        )
        plan.record["confirm"].append(f"nim-select checked {where}, not the {plan.target} target")
    for key in ("context", "namespace"):
        given, checked = user.get(key), environment.get(key)
        if deploy_k8s and given and checked and given != checked:
            plan.record["warnings"].append(
                f"nim-select inspected {key} {checked!r}; you set {given!r}"
            )
            plan.record["confirm"].append(f"select checked {key} {checked!r}, not {given!r}")


def map_tune(
    plan: Plan, tune: HandoffFile, runner: str, has_select: bool, select_digest: str | None = None
) -> None:
    data = tune.data
    image = str(data["image"])
    digest = data.get("image_digest")
    if digest and plan.target == "docker" and runner == "docker":
        plan.put("image", f"{image}@{digest}", "tune")
        if select_digest != digest:
            # nim-select writes the tag only, so this is the usual case: the digest is trusted
            where = "nim-select gave only the tag" if has_select else "only the tag was given"
            plan.record["warnings"].append(
                f"the image is pinned to the digest {digest} from the tune result; {where} "
                f"{image.rpartition(':')[2]} and nim-deploy did not check that the tag still "
                "resolves to that digest"
            )
    else:
        if select_digest and not digest:
            pass  # nim-select pinned a digest and the tune result has none: keep the pinned image
        else:
            plan.put("image", image, "select" if has_select else "tune")
        if digest:
            plan.record["warnings"].append(
                f"the tuned image digest {digest} is shown, not pinned: this target renders tags "
                f"only; check that {image} still resolves to it"
            )
    if not has_select:
        plan.put("model_profile", str(data["model_profile"]), "tune")
        if plan.target in K8S_TARGETS:
            plan.put("gpus", str(data["gpus"]), "tune")
        else:
            plan.record["pending"].append(
                {
                    "key": "gpus",
                    "question": f"Which GPUs should it use ({data['gpus']} GPU(s), device ids "
                    "such as 0,1)?",
                }
            )
        if runner == "nimcli":
            plan.put("model", image, "tune")
    if plan.target in K8S_TARGETS:
        plan.put("replicas", str(data["replicas"]), "tune")
    elif int(data["replicas"]) > 1:
        raise HandoffError(
            "replicas_unsupported",
            f"{tune.path} was measured with {data['replicas']} replicas; a {plan.target} run "
            "starts one container",
            "Ask the user: deploy one replica (the tuned throughput will not be reached) or "
            "deploy to Kubernetes",
            nd.EXIT_PRECONDITION,
        )
    if data["target"] != plan.target:
        plan.record["warnings"].append(
            f"measured on {data['target']}: the tuned settings are mapped to {plan.target}; "
            "the measured numbers may differ there"
        )
    map_tuned_settings(plan, tune, runner)


def map_tuned_settings(plan: Plan, tune: HandoffFile, runner: str) -> None:
    data = tune.data
    args = list(data["args"])
    if args and runner == "nimcli":
        raise HandoffError(
            "runner_cannot_pass_args",
            "the nim CLI runner cannot pass engine arguments, and the tuned config has "
            + " ".join(args),
            "Ask the user: deploy with the docker runner (drop --set runner=nimcli), which "
            "passes the arguments after the image",
            nd.EXIT_PRECONDITION,
        )
    if args:
        plan.put("extra_args", args, "tune")
    env: dict[str, str] = {}
    for name, value in sorted(data["env"].items()):
        if name == "NIM_PASSTHROUGH_ARGS":
            raise HandoffError(
                "handoff_invalid",
                f"{tune.path}: env NIM_PASSTHROUGH_ARGS is not accepted; engine arguments go "
                "in args",
                SCHEMA_FIX,
            )
        if SECRET_ENV.search(name) or nd.looks_secret(name, value):
            raise HandoffError(
                "handoff_invalid",
                f"{tune.path}: env {name} looks like a credential; secrets go in env_from_host "
                "by name",
                SCHEMA_FIX,
            )
        if name == "NIM_MODEL_PROFILE":
            if value != data["model_profile"]:
                raise HandoffError(
                    "handoff_disagree",
                    f"{tune.path}: env NIM_MODEL_PROFILE {value!r} differs from model_profile "
                    f"{data['model_profile']!r}",
                    "Stop and ask the user which profile was measured",
                    nd.EXIT_PRECONDITION,
                )
            continue  # pinned through model_profile
        if name in ENV_INPUTS:
            plan.put(ENV_INPUTS[name], value, "tune")
            continue
        env[name] = value
    if env:
        plan.put("env", env, "tune")
    for name in data["env_from_host"]:
        if name == "NGC_API_KEY":
            continue  # every target passes it by name already
        if name == "HF_TOKEN":
            if plan.target in K8S_TARGETS:
                plan.record["pending"].append(
                    {
                        "key": "hf_token_secret",
                        "question": "The tuned config needs HF_TOKEN: which Kubernetes Secret "
                        "holds it (key HF_TOKEN)?",
                    }
                )
            elif runner == "docker":
                plan.put("pass_hf_token", "true", "tune")
            continue  # the nim CLI reads HF_TOKEN itself
        raise HandoffError(
            "secret_unsupported",
            f"{tune.path} needs the secret {name}, which nim-deploy cannot pass by name",
            f"Ask the user how the NIM gets {name}; nim-deploy passes only NGC_API_KEY and "
            "HF_TOKEN by name. Never put the value in a file or command",
        )


def tuning_text(tune: HandoffFile | None) -> str:
    if tune is None:
        return "not tuned: the selected config, without tuned arguments"
    data = tune.data
    if data["baseline_won"]:
        return (
            f"baseline won in campaign {data['campaign']}: no tuned change, the selected config is "
            f"deployed with no extra arguments ({data['verdict']})"
        )
    shown = " ".join(data["args"]) or "no arguments"
    return f"tuned in campaign {data['campaign']}: {shown} ({data['verdict']})"


def build(
    paths: list[str],
    target: str | None,
    runner: str,
    user: dict[str, Any],
    used_by: dict[str, str] | None = None,
    now: datetime | None = None,
) -> Plan:
    """Read, check and map the files; `used_by` maps a file sha256 to the run that used it."""
    now = now or nd.utc_now()
    files = load(paths)
    plan = Plan(target or derive_target(files))
    plan.record = {"files": [], "warnings": [], "pending": [], "confirm": []}
    select, tune = files.get("select"), files.get("tune")
    if select and tune:
        found = disagreements(select, tune)
        if found:
            raise HandoffError(
                "handoff_disagree",
                f"{select.path} and {tune.path} disagree: " + "; ".join(found),
                "Stop and show the user both values. Profile ids belong to one image version: "
                "tune the image and profile chosen in nim-select, or select again",
                nd.EXIT_PRECONDITION,
            )
    env = files.get("env")
    if select and env:
        summary = select.data["environment_summary"]
        keys = ("captured_at", "target", "context", "namespace")
        if any(summary.get(key) != env.data.get(key) for key in keys):
            plan.record["warnings"].append(
                f"env.json ({env.data.get('captured_at')}) is not the environment select.json "
                f"used ({summary.get('captured_at')})"
            )
            plan.record["confirm"].append("env.json comes from another nim-select run")
    for file in files.values():
        age = age_days(file.created_at, now)
        if age is not None and age > MAX_AGE.days:
            plan.record["warnings"].append(
                f"{file.kind} file from {file.created_at} ({age:g} days old)"
            )
            plan.record["confirm"].append(f"the {file.kind} file is {age:g} days old")
        earlier = (used_by or {}).get(file.sha256)
        if earlier:
            plan.record["warnings"].append(f"{file.kind} file already used by run {earlier}")
            plan.record["confirm"].append(f"the {file.kind} file was used by run {earlier}")
    if select:
        map_select(plan, select, runner, user)
    if tune:
        select_digest = split_digest(str(select.data["handoff"]["image"]))[1] if select else None
        map_tune(plan, tune, runner, select is not None, select_digest)
        gpu_notes(plan, select, tune)
    profile = str(plan.values["model_profile"]["value"])
    plan.record["profile"] = {
        "id": profile,
        "name": profile_name(select.data if select else None, profile),
        "chosen_in": "nim-select" if select else "nim-tune (not screened by nim-select)",
        "pinned_as": pinned_as(plan.target, runner),
        "gpus_needed": int(select.data["handoff"]["gpu_count"])
        if select
        else int(tune.data["gpus"])
        if tune
        else 0,
    }
    plan.record["tuning"] = tuning_text(tune)
    plan.record["measured_on"] = tune.data["target"] if tune else None
    plan.record["files"] = [file.entry(now) for file in files.values()]
    plan.record["values"] = plan.values
    if plan.record["confirm"]:
        reasons = "; ".join(plan.record["confirm"])
        plan.record["pending"].append(
            {
                "key": CONFIRM_KEY,
                "question": f"Use these result files anyway ({reasons})? If yes: set {CONFIRM_KEY}"
                "=yes",
            }
        )
    del plan.record["confirm"]
    return plan


# --- reading the record back ----------------------------------------------------------------


def pending(state: dict[str, Any]) -> list[dict[str, str]]:
    """Questions from the files the user has not answered yet."""
    return nd.handoff_pending(state)


def source_label(state: dict[str, Any], key: str, given: dict[str, Any]) -> str:
    entry = state.get("inputs", {}).get(key) or {}
    if entry.get("source") == given["source"] or entry.get("value") == given["value"]:
        return str(given["source"])
    return f"{entry.get('source', 'user')} (overrides {given['source']})"


def summary(state: dict[str, Any]) -> dict[str, Any] | None:
    """The approval-summary block: files, profile, tuning, warnings and each value's source."""
    record = state.get("handoff")
    if not record:
        return None
    sources: dict[str, str] = {}
    for key, given in record.get("values", {}).items():
        if key == "env":
            current = nd.input_value(state, "env") or {}
            for name, value in given["value"].items():
                same = current.get(name) == value
                label = given["source"] if same else f"user (overrides {given['source']})"
                sources[f"env.{name}"] = label
        else:
            sources[key] = source_label(state, key, given)
    profile = dict(record["profile"])
    current = nd.input_value(state, "model_profile")
    if current != profile["id"]:
        profile["overridden_by_user"] = current
    warnings = list(record.get("warnings", []))
    image = nd.input_value(state, "image")
    given_image = (record.get("values", {}).get("image") or {}).get("value")
    if given_image and image != given_image:
        warnings.append(
            f"image changed to {image}: the profile was chosen for {given_image}; profile ids "
            "belong to one image version"
        )
    return {
        "files": record["files"],
        "profile": profile,
        "tuning": record["tuning"],
        "measured_on": record.get("measured_on"),
        "sources": sources,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> None:
    parser = nd.JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/handoff.py ~/.cache/nim-select/select.json --target docker",
    )
    parser.add_argument("files", nargs="+", metavar="FILE")
    parser.add_argument("--target", choices=nd.TARGETS)
    parser.add_argument("--runner", default="docker", choices=("docker", "nimcli"))
    args = parser.parse_args(argv)
    try:
        plan = build(args.files, args.target, args.runner, {})
    except HandoffError as exc:
        error = nd.error(exc.code, exc.message, exc.fix)
        nd.emit(False, "handoff", "ask_user", {}, [error], exc.exit_code)
    nd.emit(True, "handoff", "done", {"target": plan.target, **plan.record})


if __name__ == "__main__":
    main()
