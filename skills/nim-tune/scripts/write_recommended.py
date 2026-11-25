#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""write_recommended: write OUT/final/recommended_config.json from a finished campaign.

The file is the machine-readable result nim-deploy reads (`nimdeploy.py new --from FILE`). It
names the config that won, and the model, image, digest, profile and GPU model that were
MEASURED. It is checked against references/recommended-config.schema.json before it is written.
Nothing is guessed: a missing model, profile or image digest is an error with the fix. The digest
comes only from the measured replicas; campaign image_digest can only confirm it, and every
measured case must have run one recorded image (`image_changed` otherwise). The
profile must be pinned in the measured NIM's env (NIM_MODEL_PROFILE); a campaign model_profile
label alone is refused (`profile_not_pinned`). A
docker.env_file is refused (`env_file_unrecorded`) because its contents are never read; the flag
--env-file-secrets-only accepts a file that holds only secrets and adds a warning.

The winner is the case `compare_aiperf.py` recommends (highest load within the latency target).
It is promoted only when it is `better` than the untuned baseline on the target metric or on
tokens/s, is never `worse` on the target metric, was fully measured, and its settings were
engaged. Otherwise the baseline won: `baseline_won` true, `args` [] and `env` the shared base
env it was measured with (a candidate win adds its own env to the same base env), so a deploy
matches what was measured. A baseline win is written only when the untuned NIM itself meets the
latency target at some load (else `no_recommendation`). A run that is not --dry-run first removes
an earlier OUT/final/recommended_config.json, so a refusal never leaves a stale result behind.

Prints one JSON object: {"ok", "state", "next", "facts", "errors": [{"code", "message", "fix"}]}.
Exit codes: 0 written, 2 bad input, 3 precondition failed (no results, no winner, a gate), 4 not
used, 5 not used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_campaign as rc
import targets

SCHEMA = Path(__file__).resolve().parent.parent / "references" / "recommended-config.schema.json"
DIGEST_IN_ID = re.compile(r"@(sha256:[a-f0-9]{64})$")
_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": int | float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class ContractParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        rc.emit(
            False,
            "error",
            "fix_errors",
            errors=[rc.err("bad_usage", message, "Run with --help and fix the arguments.")],
            code=rc.EXIT_INPUT,
        )


def fail(code: str, message: str, fix: str, exit_code: int = rc.EXIT_PRECONDITION) -> NoReturn:
    rc.emit(False, "refused", "fix_errors", {}, [rc.err(code, message, fix)], exit_code)


def schema_errors(value: Any, schema: dict[str, Any], where: str = "$") -> list[str]:
    """Problems of VALUE against SCHEMA (stdlib subset: type, const, enum, required, properties,
    additionalProperties, propertyNames, items, minLength, maxLength, minimum, pattern)."""
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        ok = any(
            isinstance(value, _TYPES[n])
            and not (n in ("integer", "number") and isinstance(value, bool))
            for n in names
        )
        if not ok:
            return [f"{where}: expected {' or '.join(names)}, got {json.dumps(value)[:60]}"]
    if "const" in schema and value != schema["const"]:
        return [f"{where}: must be {json.dumps(schema['const'])}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{where}: must be one of {schema['enum']}, got {json.dumps(value)[:60]}"]
    found: list[str] = []
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            found.append(f"{where}: must not be empty")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            found.append(f"{where}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            found.append(f"{where}: {value!r} does not match {schema['pattern']}")
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and value < schema.get("minimum", value)
    ):
        found.append(f"{where}: must be at least {schema['minimum']}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            found += schema_errors(item, schema["items"], f"{where}[{i}]")
    if isinstance(value, dict):
        found += [f"{where}: missing {k!r}" for k in schema.get("required", []) if k not in value]
        props, extra = schema.get("properties", {}), schema.get("additionalProperties", True)
        for key, item in value.items():
            if "propertyNames" in schema:
                found += schema_errors(key, schema["propertyNames"], f"{where}.{key} (name)")
            if key in props:
                found += schema_errors(item, props[key], f"{where}.{key}")
            elif extra is False:
                found.append(f"{where}: unknown field {key!r}")
            elif isinstance(extra, dict):
                found += schema_errors(item, extra, f"{where}.{key}")
    return found


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(
            "results_unreadable",
            f"{path}: {exc}",
            "Run the campaign to the end (run_campaign.py ... --approved PLAN_ID) in this folder.",
        )


def split_image(image: str) -> tuple[str, str | None]:
    name, _, digest = image.partition("@")
    return name, digest or None


def repository(ref: str) -> str:
    """The repository of an image reference, without tag or digest, in Docker's normalized form
    (`nginx` and `docker.io/library/nginx` are one repository)."""
    name = ref.split("://", 1)[-1].partition("@")[0]  # docker-pullable://repo@sha256:...
    head, sep, tail = name.rpartition("/")
    name = head + sep + tail.partition(":")[0]  # a ':' after the last '/' starts the tag
    first = name.split("/", 1)[0]
    if "/" not in name or not ("." in first or ":" in first or first == "localhost"):
        name = "docker.io/" + name
    if name.startswith("index.docker.io/"):  # Docker's legacy name for the same registry
        name = "docker.io/" + name.split("/", 1)[1]
    if name.startswith("docker.io/") and name.count("/") == 1:
        name = "docker.io/library/" + name.split("/", 1)[1]
    return name


def operator_config(out: Path, cid: str) -> tuple[str, dict[str, str], list[str]]:
    """Image, plain env and secret names of what was deployed: the saved NIMService."""
    record = read_json(out / cid / "deploy.json")
    spec = record.get("spec", {}) if isinstance(record, dict) else {}
    image = spec.get("image") or {}
    if not isinstance(image, dict) or not image.get("repository") or not image.get("tag"):
        fail(
            "image_unknown",
            f"{out / cid / 'deploy.json'} has no spec.image repository and tag",
            "Put the image with its tag in the campaign's base_manifest (spec.image), then "
            "measure again; the result is written from what was deployed.",
        )
    env: dict[str, str] = {}
    secret_names: list[str] = []
    for item in spec.get("env", []):
        if "value" in item:
            env[item["name"]] = str(item["value"])
        elif "valueFrom" in item:
            secret_names.append(item["name"])
    return f"{image['repository']}:{image['tag']}", env, secret_names


def docker_config(c: dict[str, Any], cand: dict[str, Any]) -> tuple[str, dict[str, str], list[str]]:
    d = c["docker"]
    env = {k: str(v) for k, v in {**d["env"], **cand["env"]}.items()}
    return str(d["image"]), env, list(d["env_from_host"])


def pod_shape_ok(pod: Any) -> bool:
    """A pod record whose nested parts have a pod's types (a null or missing part is allowed)."""
    if not isinstance(pod, dict):
        return False
    parts = {k: pod.get(k) for k in ("spec", "metadata", "status")}
    if not all(v is None or isinstance(v, dict) for v in parts.values()):
        return False
    try:  # every kind of container (the NIM can be an annotated init or ephemeral one), and
        # the metadata and annotations that name it
        targets.pod_container(pod)
    except ValueError:
        return False
    lists = [(parts["status"] or {}).get(status) for _, status in targets.CONTAINER_KINDS]
    return all(
        x is None or (isinstance(x, list) and all(isinstance(i, dict) for i in x)) for x in lists
    )


def state_shape_ok(st: Any) -> bool:
    """A candidate state whose run lists and engagement the promotion check can read (a missing
    run list is allowed, a null one is not: it is read as a list)."""
    if not isinstance(st, dict):
        return False
    runs = [st[k] for k in ("failed_runs", "overloaded_runs") if k in st]
    if not all(isinstance(r, list) and all(isinstance(i, str) for i in r) for r in runs):
        return False
    engaged = st.get("engagement")
    return engaged is None or (
        isinstance(engaged, dict) and all(isinstance(e, dict) for e in engaged.values())
    )


def replica_digest(pod: dict[str, Any], want: str | None) -> str | None:
    """The digest of the measured NIM container in one pod record (its imageID), or None.

    With WANT, only containers of that repository count, and the imageID that supplies the
    digest must name it too: a sidecar's digest is never the NIM's, nor one of another
    repository. A record that does not name the measured container counts only when exactly one
    status matches, since two of the same repository cannot be told apart.
    """
    try:
        found = targets.pod_container(pod)
    except ValueError:  # a malformed container list: no evidence of which container ran
        return None
    field = found[0] if found else "containerStatuses"  # an init or ephemeral NIM: its own list
    nim = found[1].get("name") if found else None
    statuses = (pod.get("status") or {}).get(field) or []
    # another container's digest, even of the same repository, is not the NIM's
    mine = [s for s in statuses if not nim or s.get("name") == nim]
    named = mine if nim else []
    if want:
        mine = [
            s
            for s in mine
            if any(r and repository(str(r)) == want for r in (s.get("image"), s.get("imageID")))
        ]
        # the measured container (named, else the one status of WANT) ran another repository's
        # image (a rewritten spec.image): contradicting evidence, never a missing digest. Only an
        # imageID of another name beside a status.image of WANT is the runtime naming the same
        # image by another of its names (a mirror pre-pull): no digest of WANT, so missing
        for s in named if len(named) == 1 else mine if len(mine) == 1 else []:
            other = str(s.get("imageID", ""))
            ran = s.get("image")
            if (
                DIGEST_IN_ID.search(other)
                and repository(other) != want
                and not (ran and repository(str(ran)) == want)
            ):
                fail(
                    "image_mismatch",
                    f"the measured NIM container ran {repository(other)}, not {want}",
                    "Put the repository the cluster really runs (the mirror) in the campaign's "
                    "image (base manifest spec.image) and measure again in a new --out folder; a "
                    "digest of another repository is never pinned.",
                )
    if len(mine) != 1:
        return None
    image_id = str(mine[0].get("imageID", ""))
    match = DIGEST_IN_ID.search(image_id)
    if not match or (want and repository(image_id) != want):
        return None
    return match.group(1)


def complete(per_replica: list[str | None], image: str | None) -> set[str]:
    """The digests of all measured replicas; a replica without one while another has one is
    incomplete evidence (refused): one replica's digest is never assumed for the others."""
    found = {d for d in per_replica if d}
    if found and None in per_replica:
        fail(
            "image_digest_incomplete",
            f"{per_replica.count(None)} of {len(per_replica)} measured replicas report no "
            f"pullable digest of {image or 'the image'}, the others report {sorted(found)}",
            "Measure again in a new --out folder so every replica runs the image pulled by "
            "that repository; a digest is pinned only when every replica shows it.",
        )
    return found


def records(out: Path, cid: str, replicas: list[str] | None) -> list[Path]:
    """The pod records to read: those of the measured REPLICAS (a record left by an earlier
    deploy is not their evidence), or every record of the candidate when none are named."""
    if replicas is None:
        return sorted((out / cid).glob("*.pod.json"))
    return [out / cid / f"{p}.pod.json" for p in sorted(replicas)]


def pod_digests(
    out: Path, cid: str, image: str | None = None, replicas: list[str] | None = None
) -> set[str]:
    """Repository digests the cluster reported for the pods (imageID) of one candidate, one per
    measured replica (`replica_digest`); an unreadable pod record is a replica without one.
    Docker's own `Image` field is the image id, not a pullable digest, so it is not used.
    """
    want = repository(image) if image else None
    per_replica: list[str | None] = []
    for path in records(out, cid, replicas):
        try:
            pod = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            per_replica.append(None)
            continue
        if not pod_shape_ok(pod):
            fail(
                "pod_record_malformed",
                f"{path} is not a pod object (spec, metadata, status and their container lists "
                "have the wrong types)",
                "Restore that record from the campaign's capture, or measure again in a new "
                "--out folder; it is evidence of the measured replica, never deleted.",
            )
        per_replica.append(replica_digest(pod, want))
    return complete(per_replica, image)


def docker_digests(
    c: dict[str, Any], out: Path, cid: str, replicas: list[str] | None = None
) -> set[str]:
    """Resolve pullable digests only from the measured containers' immutable image IDs."""
    target = targets.DockerTarget(c, out, lambda _: None)
    # one image ID can carry digests of several repositories; only the handoff's own counts
    want = repository(str(c["docker"]["image"]))
    per_replica: list[str | None] = []
    paths = records(out, cid, replicas)
    try:
        if not paths:
            raise ValueError("no measured container records")
        for path in paths:
            boxes = read_json(path)
            if not isinstance(boxes, list) or not boxes:
                raise ValueError(f"{path} has no measured container")
            for box in boxes:
                image_id = box.get("Image", "")
                if not isinstance(image_id, str) or not rc.IMAGE_DIGEST.fullmatch(image_id):
                    raise ValueError(f"{path} has no immutable image ID")
                proc = target.docker("image", "inspect", image_id)
                if proc.returncode != 0:
                    raise ValueError("Docker cannot inspect the measured image")
                images = json.loads(proc.stdout)
                if not isinstance(images, list) or not images:
                    raise ValueError("Docker returned no measured image")
                mine = {
                    match.group(1)
                    for image in images
                    for ref in image.get("RepoDigests") or []
                    if (match := DIGEST_IN_ID.search(str(ref))) and repository(str(ref)) == want
                }
                # one entry per container; two digests of one container go on to the conflict
                per_replica += sorted(mine) or [None]
    except (OSError, ValueError, TypeError, AttributeError, targets.KubeError) as exc:
        fail(
            "image_digest_unavailable",
            f"cannot obtain the measured Docker image digest: {exc}",
            "Restore access to the measured image on the campaign's Docker host, or measure "
            "again; campaign image_digest and reference digests cannot replace Docker evidence.",
        )
    return complete(per_replica, str(c["docker"]["image"]))


def check_image(
    out: Path, c: dict[str, Any], states: dict[str, Any], chosen: str, engaged: dict[str, Any]
) -> None:
    """Every measured case's runs were on one image (the runner records it as image_id and
    measures again when it changes), and the chosen case's saved records, whose digest is
    pinned, still show that image."""
    ids = {
        x["id"]: states[x["id"]].get("image_id")
        for x in c["candidates"]
        if states.get(x["id"], {}).get("status") == "measured"
    }
    unrecorded = sorted(k for k, v in ids.items() if not v)
    if unrecorded:
        fail(
            "measured_config_unrecorded",
            f"{', '.join(unrecorded)} has no record of the image its runs were measured on in "
            f"{out}",
            "Measure the campaign again with this version of run_campaign.py in a new --out "
            "folder; a result is written only from an image the runner recorded.",
        )
    records: list[Any] = []
    for p in sorted(engaged):
        try:
            records.append(json.loads((out / chosen / f"{p}.pod.json").read_text("utf-8")))
        except (OSError, json.JSONDecodeError):
            records.append(None)
    shown = rc.image_identity(records)
    if len(set(ids.values())) > 1 or shown != ids[chosen]:
        fail(
            "image_changed",
            f"the measured cases ran different images ({ids}), or the saved records of "
            f"{chosen} show {shown}",
            "The tag resolved to another image during the campaign: pin the image by digest "
            "(docker.image, or the base manifest's spec.image) and measure again in a new --out "
            "folder.",
        )


def changed_since_measured(
    recorded: dict[str, Any], c: dict[str, Any], cand: dict[str, Any]
) -> list[str]:
    """Names of the settings in campaign.json that differ from what the runner deployed."""
    current: dict[str, Any] = {k: cand.get(k) for k in ("gpus", "replicas", "args", "env")}
    current["workload"] = c.get("workload")  # the exports were measured with this workload
    if c["target"] == "docker":
        keys = ("image", "env", "env_from_host", "env_file")
        current["docker"] = {k: c["docker"].get(k) for k in keys}
    changed = [k for k in current if k != "docker" and current[k] != recorded.get(k)]
    if "docker" in current:
        before = recorded.get("docker") or {}
        changed += [f"docker.{k}" for k, v in current["docker"].items() if v != before.get(k)]
    return changed


def promotion_problem(
    cid: str,
    state: dict[str, Any],
    accuracy_checked: bool,
) -> tuple[str, str, str] | None:
    """(code, message, fix) when the winner may not be promoted as measured, else None."""
    # a load lost to failed requests is a measured overload, as compare_aiperf keeps it
    overloaded = set(state.get("overloaded_runs", []))
    missing = [r for r in state.get("failed_runs", []) if r not in overloaded]
    if state.get("status") != "measured" or missing:
        return ("winner_not_measured", f"{cid} was not fully measured", "Rerun the campaign.")
    engaged = [e.get("state") for e in (state.get("engagement") or {}).values()]
    if not engaged or any(s != "engaged" for s in engaged):
        return (
            "winner_not_engaged",
            f"{cid}: the engine did not prove the setting reached it (states: {engaged})",
            "Add an engagement_pattern to the candidate and measure it again.",
        )
    if state.get("needs_accuracy_check") and not accuracy_checked:
        return (
            "accuracy_check_needed",
            f"{cid} changes precision or speculative decoding and has no accuracy check",
            "Run the accuracy check, then pass --accuracy-checked; or keep the baseline.",
        )
    return None


def separated(
    comparisons: list[dict[str, Any]],
    cid: str,
    sla_metric: str,
    boundary: dict[str, Any] | None = None,
    baseline: str | None = None,
    overloaded: list[list[Any]] | None = None,
    picks: list[dict[str, Any]] | None = None,
) -> str | None:
    """Why the winner is not clearly better than the baseline, or None when it is.

    An untuned NIM that never meets the target and failed requests on every repeat at its first
    failing load (a measured overload: no counted run there, so nothing to compare) is beaten
    by a winner measured to meet the target at that same load. Otherwise the winner must be better outside
    the repeat spread somewhere: a higher boundary alone is no separation."""
    mine = [x for x in comparisons if x["case"] == cid]
    if any(x["metric"] == sla_metric and x["verdict"] == "worse" for x in mine):
        return f"{cid} is worse than the untuned NIM on {sla_metric}"
    b, w = (boundary or {}).get(baseline or "", {}), (boundary or {}).get(cid, {})
    missed, held = b.get("first_failing"), w.get("max_concurrency")
    never = b.get("max_concurrency") is None
    # measured to meet the target at that very load, not only a boundary at or above it
    met = any(
        p.get("concurrency") == missed and cid in ((p.get("sla") or {}).get("meets") or [])
        for p in picks or []
    )
    if never and [baseline, missed] in (overloaded or []) and held is not None and met:
        return None
    if not any(x["verdict"] == "better" and x["metric"] in (sla_metric, "otps_avg") for x in mine):
        return f"{cid} is not better than the untuned NIM outside the repeat spread"
    return None


def verdict_text(
    cid: str, baseline: str, rec: dict[str, Any], boundary: dict[str, Any], sla: dict[str, Any]
) -> str:
    base = boundary.get(baseline, {})
    target = f"{sla['metric']}<={sla['max_ms']} ms"
    if cid == baseline:
        return f"untuned NIM kept: within {target} up to {rec['max_concurrency']} concurrent"
    return (
        f"{cid} holds {rec['max_concurrency']} concurrent within {target} "
        f"(untuned: {base.get('max_concurrency')}), {rec['otps_per_gpu']} tokens/s per GPU"
    )[:300]


def main(argv: list[str] | None = None) -> None:
    parser = ContractParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/write_recommended.py campaign.json --out .",
    )
    parser.add_argument("campaign", type=Path, help="the campaign file of the final campaign")
    parser.add_argument("--out", type=Path, required=True, help="the run folder holding results")
    parser.add_argument(
        "--accuracy-checked",
        action="store_true",
        help="the user ran the accuracy check a precision or speculative winner needs",
    )
    parser.add_argument(
        "--allow-unpinned-image",
        action="store_true",
        help="write image_digest null when no digest is known (the deploy cannot pin the image)",
    )
    parser.add_argument(
        "--env-file-secrets-only",
        action="store_true",
        help="docker.env_file holds only secrets (no tuned settings): write the result with a "
        "warning that the file's contents are not recorded",
    )
    parser.add_argument("--created-at", help="ISO 8601 time to record (default: now, UTC)")
    parser.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    args = parser.parse_args(argv)

    out: Path = args.out
    target = out / "final" / "recommended_config.json"
    if not args.dry_run and target.exists():
        # a refusal below, an input error included, must not leave an earlier result that
        # nim-deploy --from reads as current
        try:
            target.unlink()
        except OSError as exc:
            fail(
                "write_failed",
                f"cannot remove the earlier {target}: {exc}",
                "Make OUT/final writable, then rerun.",
            )
        print(
            f"removed the earlier {target}; it is written again only if this run passes",
            file=sys.stderr,
        )
    c, errors = rc.load_campaign(args.campaign)
    if errors:
        rc.emit(False, "refused", "fix_errors", {}, errors, rc.EXIT_INPUT)
    state = read_json(out / "campaign_state.json")
    states = state.get("candidates") if isinstance(state, dict) else None
    if not isinstance(states, dict) or not all(state_shape_ok(v) for v in states.values()):
        fail(
            "results_unreadable",
            f"{out / 'campaign_state.json'} is not a campaign state (an object whose candidates "
            "maps each id to its state, with lists of run names and an engagement object)",
            "Run the campaign to the end (run_campaign.py ... --approved PLAN_ID) in this folder.",
        )
    baseline = c["baseline"]
    if states.get(baseline, {}).get("status") != "measured":
        fail(
            "baseline_unmeasured",
            f"the untuned NIM {baseline!r} has no complete measurement in {out}",
            "Run the campaign to the end; nothing can be recommended without the baseline.",
        )
    unfinished = sorted(
        f"{k} ({v.get('status')})"
        for k, v in states.items()
        if v.get("status") not in ("measured", "pruned", "rejected")
    )
    if unfinished:
        fail(
            "candidate_not_measured",
            f"not fully measured: {', '.join(unfinished)}; a recommendation would hide that",
            "Rerun the campaign to finish them, or write the result from a final campaign that "
            "holds only the untuned NIM and the winner.",
        )
    if not c.get("sla"):
        fail(
            "sla_missing",
            "the campaign has no sla, so no case can be said to meet a latency target",
            'Add "sla": {"metric": "ttft_p99", "max_ms": N} to campaign.json and rerun it.',
        )
    # compare ranks by tokens/s per GPU from each case's gpus x replicas: an edit to any measured
    # case's GPU count after the run would change the winner without changing a measurement, so
    # every ranked case needs its recorded layout. Every measured case is checked, a superset of
    # the ranked ones (compare excludes one with a missing run): fail-closed, since a measured
    # case without a record comes from a runner that recorded no settings at all
    unrecorded = [
        x["id"]
        for x in c["candidates"]
        if states.get(x["id"], {}).get("status") == "measured"
        and not isinstance(states[x["id"]].get("measured"), dict)
    ]
    if unrecorded:
        fail(
            "measured_config_unrecorded",
            f"{', '.join(unrecorded)} has no record of the settings it was measured with in {out}",
            "Measure the campaign again with this version of run_campaign.py in a new --out "
            "folder; a result is written only from settings the runner recorded.",
        )
    moved = [
        f"{x['id']}.{k}"
        for x in c["candidates"]
        if isinstance(states.get(x["id"], {}).get("measured"), dict)
        for k in ("gpus", "replicas")
        if x.get(k) != states[x["id"]]["measured"].get(k)
    ]
    if moved:
        fail(
            "campaign_changed",
            f"campaign.json differs from what was measured: {', '.join(moved)}",
            "Restore those settings in campaign.json to the ones that were run, or measure the "
            "changed campaign again in a new --out folder.",
        )
    result = rc.compare(c, out, states)
    if not result.get("ok"):
        rc.emit(
            False, "incomplete", "fix_errors", {}, result.get("errors", []), rc.EXIT_PRECONDITION
        )
    facts = result["facts"]
    rec = facts.get("recommendation")
    if not rec:
        fail(
            "no_recommendation",
            f"no case meets {c['sla']['metric']}<={c['sla']['max_ms']} ms at any measured load",
            "Tell the user the target is not met; ask for a looser target or other hardware. "
            "Do not write a recommendation.",
        )

    winner: str = rec["case"]
    reason = None
    if winner != baseline:
        problem = promotion_problem(winner, states.get(winner, {}), args.accuracy_checked)
        if problem:
            fail(*problem)
        reason = separated(
            facts["comparisons"],
            winner,
            c["sla"]["metric"],
            facts.get("sla_boundary"),
            baseline,
            facts.get("overloaded"),
            facts.get("picks"),
        )
    baseline_won = winner == baseline or reason is not None
    if baseline_won and not (facts.get("sla_boundary") or {}).get(baseline, {}).get(
        "max_concurrency"
    ):
        # the candidate met the target but is not promotable; the untuned NIM never met it
        fail(
            "no_recommendation",
            f"{winner} is not promoted ({reason}) and the untuned NIM {baseline!r} misses "
            f"{c['sla']['metric']}<={c['sla']['max_ms']} ms at every measured load",
            "Tell the user neither configuration can be recommended; measure the candidate "
            "again (more repeats), or ask for a looser target. Do not write a recommendation.",
        )
    chosen = baseline if baseline_won else winner
    cand = next(x for x in c["candidates"] if x["id"] == chosen)
    recorded = states.get(chosen, {}).get("measured")
    if not isinstance(recorded, dict) or "workload" not in recorded:
        # a record from an earlier runner has no workload: unrecorded, not a changed campaign
        fail(
            "measured_config_unrecorded",
            f"{chosen} has no record of the settings and workload it was measured with in {out}",
            "Measure the campaign again with this version of run_campaign.py in a new --out "
            "folder; a result is written only from settings the runner recorded.",
        )
    changed = changed_since_measured(recorded, c, cand)
    if changed:
        fail(
            "campaign_changed",
            f"campaign.json differs from what {chosen} was measured with: {', '.join(changed)}",
            "Restore those settings in campaign.json to the ones that were run, or measure the "
            "changed campaign again in a new --out folder.",
        )

    if c["target"] == "docker":
        # what was measured: the shared base env, plus the winner's own env (none for the baseline)
        image_ref, env, secret_names = docker_config(
            c, {**cand, "env": {}} if baseline_won else cand
        )
    else:
        image_ref, env, secret_names = operator_config(out, chosen)
    env_file = c["docker"].get("env_file") if c["target"] == "docker" else None
    if env_file and not args.env_file_secrets_only:
        fail(
            "env_file_unrecorded",
            f"docker.env_file {env_file} was passed to the measured NIM; this writer never reads "
            "secret values, so it cannot tell which settings or secrets the file supplied and the "
            "result could differ from what was measured",
            "Move the non-secret settings into docker.env and the secret names into "
            "docker.env_from_host, then measure again. With docker.sudo the names cannot be "
            "passed that way: if the file holds only secrets (such as HF_TOKEN), run this "
            "again with --env-file-secrets-only to write the result with a warning.",
        )
    image, ref_digest = split_image(image_ref)
    if not re.fullmatch(r"[^@\s]+:[^@:/\s]+", image):
        fail(
            "image_untagged",
            f"image {image!r} has no version tag",
            "Use the image with its tag (nvcr.io/nim/...:2.1.4) in the campaign and measure again.",
        )

    model = c.get("model")
    if not model:
        fail(
            "model_unknown",
            "the campaign does not name the NIM model (workload.model is the served name)",
            'Add "model": "<name as nim-select gives it>" to campaign.json (nim-select '
            "facts.handoff.model), then run this again.",
        )
    # The profile counts only when the measured NIM was pinned to it: model_profile is a label the
    # runner does not use, and an unpinned NIM may have picked another profile on its own.
    pinned = env.get("NIM_MODEL_PROFILE")
    label = c.get("model_profile")
    if not pinned:
        fail(
            "profile_not_pinned" if label else "profile_unknown",
            "the measured NIM environment has no NIM_MODEL_PROFILE"
            + (f" (campaign model_profile {label!r} is only a label)" if label else "")
            + "; an unpinned NIM may have chosen another profile, so the result would not "
            "reproduce what was measured",
            "Set NIM_MODEL_PROFILE=<profile id> in docker.env (or in the NIMService spec.env of "
            'the base manifest), add "model_profile" with the same id to campaign.json '
            "(nim-select facts.handoff.model_profile), and measure again.",
        )
    if label and label != pinned:
        fail(
            "profile_conflict",
            f"campaign model_profile and NIM_MODEL_PROFILE name different profiles: "
            f"{sorted({label, pinned})}",
            "Fix campaign.json so one profile id is named; use the one the NIM was measured on.",
        )
    profile = pinned

    # the digest is pinned only from every measured replica's record: a missing one is a replica
    # whose image is unknown, never left out of the check
    engaged = states[chosen].get("engagement") or {}
    lost = sorted(p for p in engaged if not (out / chosen / f"{p}.pod.json").is_file())
    if not engaged or lost:
        fail(
            "image_digest_incomplete",
            f"{chosen}: the records of measured replicas {lost or '(none recorded)'} are missing "
            f"from {out / chosen}",
            "Restore the saved OUT/ID/REPLICA.pod.json records, or measure again in a new --out "
            "folder; a digest is pinned only when every measured replica shows it.",
        )
    measured_digests = (
        docker_digests(c, out, chosen, list(engaged))
        if c["target"] == "docker"
        else pod_digests(out, chosen, image, list(engaged))
    )
    # campaign image_digest and a digest in the image reference only confirm what the measured
    # replicas show: never a digest in their place
    claimed = {d for d in (c.get("image_digest"), ref_digest) if d}
    if measured_digests and len(claimed | measured_digests) > 1:
        fail(
            "image_digest_conflict",
            f"different digests are recorded for {image}: {sorted(claimed | measured_digests)}",
            "Keep the digest of the image that was measured in campaign.json image_digest.",
        )
    if not measured_digests and not args.allow_unpinned_image:
        fail(
            "image_digest_incomplete",
            f"no measured replica of {chosen} shows a pullable digest of {image}"
            + (f"; {sorted(claimed)} in the campaign is not measured evidence" if claimed else ""),
            "A mirror or preloaded image never shows the campaign repository's digest: set the "
            "campaign image to the repository the nodes really pull (the mirror) and measure "
            "again in a new --out folder, or, with the user's yes, pass --allow-unpinned-image "
            "to write null (the deploy then runs the tag as it is).",
        )
    digest = next(iter(measured_digests), None)
    check_image(out, c, states, chosen, engaged)

    secrets = sorted(set(secret_names) | set(c.get("docker", {}).get("env_from_host", [])))
    record = {
        "apiVersion": "nim-tune/v1",
        "kind": "RecommendedConfig",
        "created_at": args.created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "campaign": c["name"],
        "model": model,
        "image": image,
        "image_digest": digest,
        "model_profile": profile,
        "target": c["target"],
        "gpus": cand["gpus"],
        "gpu_name": c.get("gpu_name"),
        "replicas": cand["replicas"],
        "args": [] if baseline_won else list(cand["args"]),
        "env": env,
        "env_from_host": secrets,
        "baseline_won": baseline_won,
        "verdict": reason
        if reason
        else verdict_text(chosen, baseline, rec, facts["sla_boundary"], c["sla"]),
    }
    record = json.loads(rc.SCRUB.dumps(record))
    problems = schema_errors(record, json.loads(SCHEMA.read_text(encoding="utf-8")))
    if problems:
        fail(
            "schema_violation",
            "; ".join(problems),
            "Fix the named campaign field (model, image, digest, env names) and run again.",
            rc.EXIT_INPUT,
        )
    if not args.dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    facts_out: dict[str, Any] = {
        "path": str(target),
        "written": not args.dry_run,
        "winner": chosen,
        "baseline_won": baseline_won,
        "recommended_config": record,
    }
    warnings = []
    if digest is None:
        warnings.append("image_digest is null: the deploy cannot pin the measured image")
    if record["gpu_name"] is None:
        warnings.append(
            "gpu_name is null: campaign.json has no gpu_name, so the deploy cannot check that "
            "the GPU matches the one measured and will ask the user"
        )
    if env_file:
        warnings.append(
            f"docker.env_file {env_file} was passed to the measured NIM and is not recorded: the "
            "secrets it supplies (for example HF_TOKEN) are not in env_from_host, so the deploy "
            "must be given them another way"
        )
    left_out = sorted(k for k, v in states.items() if v.get("unrepaired") and k != baseline)
    if left_out:
        warnings.append(
            f"{', '.join(left_out)} left out of the comparison: runs still did not count after "
            "the repair passes (candidate_unrepaired)"
        )
    if warnings:
        facts_out["warnings"] = warnings
    rc.emit(True, "written", "done", facts_out)


if __name__ == "__main__":
    main()
