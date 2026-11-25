#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nimbench: measure a running NIM endpoint for speed (AIPerf) and accuracy (NeMo Gym).

Commands:
  check     read /v1/models of the endpoint (and the baseline endpoint); send no load.
  perf      plan, then run, a speed benchmark. --dry-run prints the plan and its plan id.
  accuracy  plan, then run, accuracy test sets. --dry-run prints the plan and its plan id.
  compare   compare a run folder with a baseline run folder and write the verdict.

A live perf or accuracy run needs --approved PLAN_ID, the value --dry-run printed. The plan id
covers the endpoint and the baseline endpoint, so load on either one is approved. A baseline
is optional: with one the run gives a verdict, without one it reports numbers only.

Prints one JSON object: {"ok", "state", "next", "facts", "errors": [{"code", "message", "fix"}]}.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool failed, 5 refused.
Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_accuracy
import nb_common
import nb_compare
import nb_perf
import nb_stats
from nb_common import (
    EXIT_EXTERNAL,
    EXIT_INPUT,
    EXIT_PRECONDITION,
    EXIT_REFUSED,
    ContractParser,
    emit,
    error,
)

Errors = list[dict[str, str]]


def is_url(text: str) -> bool:
    return text.startswith(("http://", "https://"))


def split_ints(text: str, name: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        values = []
    if not values or any(v < 1 for v in values):
        raise ValueError(f"{name} must be positive whole numbers, comma separated")
    return values


def input_errors(args: argparse.Namespace, argv: list[str]) -> Errors:
    """Validate shared client flags and refuse credentials given as flag values."""
    errors: Errors = []
    for flag in ("api_key_env", "baseline_api_key_env", "judge_api_key_env"):
        name = getattr(args, flag, None)
        if name and not nb_common.VARIABLE_NAME.match(name):
            errors.append(
                error(
                    "literal_secret",
                    f"--{flag.replace('_', '-')} must be a variable name",
                    "Put the key in an environment variable and give its name",
                )
            )
    if any(nb_common.TOKEN_SHAPES.search(text) for text in argv) and not errors:
        errors.append(
            error(
                "literal_secret",
                "a flag value looks like a credential",
                "Pass the name of an environment variable, never the value",
            )
        )
    if getattr(args, "client", None) == "kubernetes" and not (args.namespace and args.pod):
        errors.append(
            error(
                "bad_input",
                "--client kubernetes needs --namespace and --pod",
                "Name the client pod",
            )
        )
    return errors


def update_manifest(
    out: Path,
    args: argparse.Namespace,
    names: list[str],
    section: str,
    data: dict[str, Any],
    pid: str,
) -> dict[str, Any]:
    """Add one mode's section to the run's manifest.json and return the manifest."""
    path = out / "manifest.json"
    manifest: dict[str, Any] = nb_common.read_json(path) if path.exists() else {}
    manifest.update(
        {
            "schema": "nim-benchmark/manifest-v1",
            "skill_version": nb_common.VERSION,
            "endpoint": nb_common.split_endpoint(args.endpoint)[0],
            "model": args.model,
            "endpoint_models": names,
        }
    )
    manifest.setdefault("plan_ids", {})[section] = pid
    manifest[section] = {
        k: v for k, v in data.items() if k not in ("key_env", "judge_key_env", "endpoint", "model")
    }
    nb_common.write_json(path, manifest)
    return manifest


def baseline_spec(args: argparse.Namespace) -> dict[str, Any] | None:
    """What the user gave as a baseline, in a form the plan id can cover."""
    if not args.baseline:
        return None
    if is_url(args.baseline):
        return {
            "kind": "endpoint",
            "endpoint": nb_common.split_endpoint(args.baseline)[0],
            "model": args.baseline_model or args.model,
            "key_env": args.baseline_api_key_env,
        }
    manifest = Path(args.baseline) / "manifest.json"
    digest = nb_common.sha256_text(manifest.read_text()) if manifest.exists() else "missing"
    return {"kind": "run", "path": str(Path(args.baseline).resolve()), "manifest": digest}


def check_endpoint(
    args: argparse.Namespace,
    base: str,
    model: str,
    key_env: str | None = None,
) -> tuple[list[str], Errors]:
    """Probe from the Kubernetes client pod, or from this host for other clients."""
    if getattr(args, "client", None) == "kubernetes":
        return nb_common.check_endpoint(base, model, key_env, pod=(args.namespace, args.pod))
    return nb_common.check_endpoint(base, model, key_env)


def client_pod_status(args: argparse.Namespace) -> tuple[bool, Errors]:
    """The client pod's own state (it exists, aiperf runs in it), looked at once per command."""
    status = getattr(args, "client_pod_status", None)
    if status is None:
        status = nb_common.check_client_pod(args.namespace, args.pod)
        args.client_pod_status = status
    return status  # type: ignore[no-any-return]


def precondition_errors(
    args: argparse.Namespace,
    spec: dict[str, Any] | None,
    need_docker: bool,
    plan: dict[str, Any] | None = None,
) -> Errors:
    """Check the endpoints and the client before any load. Sends no load."""
    client = getattr(args, "client", "none")
    if client == "kubernetes" and not shutil.which("kubectl"):
        return [error("client_missing", "kubectl is not on the PATH", "Install kubectl")]
    errors: Errors = []
    if client == "kubernetes":
        _ok, errors = client_pod_status(args)
        if errors:  # no probe can work through a pod that is missing or has no aiperf
            return list(errors)
    base, _ = nb_common.split_endpoint(args.endpoint)
    _names, found = check_endpoint(args, base, args.model, args.api_key_env)
    errors += found
    if spec and spec["kind"] == "endpoint":
        _n, found = check_endpoint(args, spec["endpoint"], spec["model"], spec["key_env"])
        errors += [{**e, "message": "baseline: " + e["message"]} for e in found]
    info = plan or {}
    judge = info.get("judge")
    if judge:
        judge_base, _ = nb_common.split_endpoint(judge["endpoint"])
        _j, found = check_endpoint(args, judge_base, judge["model"], info.get("judge_key_env"))
        errors += [{**e, "message": "judge: " + e["message"]} for e in found]
    if spec and spec["kind"] == "run" and spec["manifest"] == "missing":
        errors.append(
            error("bad_input", f"{spec['path']} has no manifest.json", "Give a finished run folder")
        )
    if (need_docker or client == "docker") and not shutil.which("docker"):
        errors.append(
            error(
                "docker_missing",
                "docker is not on the PATH",
                "Install Docker, or run performance mode with --client local",
            )
        )
    if client == "local" and not shutil.which("aiperf"):
        errors.append(
            error(
                "client_missing",
                "aiperf is not on the PATH",
                "pip install aiperf==" + nb_common.AIPERF_VERSION,
            )
        )
    if client == "local" and args.command == "perf" and shutil.which("aiperf"):
        try:
            done = nb_common.run_command(["aiperf", "--version"], timeout=60)
            valid = done.returncode == 0 and nb_common.aiperf_version_ok(done.stdout + done.stderr)
        except (OSError, subprocess.TimeoutExpired):
            valid = False
        if not valid:
            errors.append(
                error(
                    "client_version_mismatch",
                    f"the local client must run AIPerf {nb_common.AIPERF_VERSION}",
                    "Install aiperf==" + nb_common.AIPERF_VERSION + " on this host",
                )
            )
    return errors


def finish_plan(args: argparse.Namespace, plan: dict[str, Any], need_docker: bool) -> str:
    """Handle --dry-run and the approval gate. Returns the plan id when the run may start."""
    spec = plan["baseline"]
    pid = nb_common.plan_id(plan)
    errors = precondition_errors(args, spec, need_docker, plan)
    if args.dry_run:
        emit(
            not errors,
            "planned",
            "ask_user_to_approve" if not errors else "fix_errors",
            {"plan_id": pid, "plan": plan},
            errors,
            EXIT_PRECONDITION if errors else 0,
        )
    if args.approved != pid:
        emit(
            False,
            "not_approved",
            "dry_run_again",
            {"plan_id": pid, "approved": args.approved or ""},
            [
                error(
                    "not_approved",
                    "the plan id was not approved, or the plan changed",
                    "Run with --dry-run, show the plan, and use the plan id the user approves",
                )
            ],
            EXIT_REFUSED,
        )
    if errors:
        emit(
            False, "precondition_failed", "fix_errors", {"plan_id": pid}, errors, EXIT_PRECONDITION
        )
    return pid


def with_endpoint(plan: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The same plan pointed at the baseline endpoint."""
    return {
        **plan,
        "endpoint": spec["endpoint"],
        "model": spec["model"],
        "key_env": spec["key_env"],
    }


def verdicts_path(out: Path) -> Path:
    return out / "compare.json"


def result_fingerprint(folder: Path, mode: str) -> str | None:
    """Bind a verdict to its mode's manifest, summary, and item scores."""
    try:
        manifest = nb_common.read_json(folder / "manifest.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get(mode), dict):
            return None
        plans = manifest.get("plan_ids", {})
        if not isinstance(plans, dict):
            return None
        paths = sorted((folder / mode).rglob("scores.json")) if mode == "accuracy" else []
        paths.append(folder / mode / "summary.json")
        evidence = {str(p.relative_to(folder)): p.read_text() for p in paths}
        evidence["manifest"] = nb_common.canonical_json(
            {
                "mode": manifest[mode],
                "plan_id": plans.get(mode),
                "endpoint": manifest.get("endpoint"),
                "model": manifest.get("model"),
            }
        )
        return nb_common.sha256_text(nb_common.canonical_json(evidence))
    except (OSError, ValueError, KeyError):
        return None



def finish_run(
    out: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    verdicts: dict[str, Any],
    problems: Errors,
    facts: dict[str, Any],
    modes: tuple[str, ...],
) -> None:
    """Save the verdicts, write the report and print the result.

    The verdicts of `modes` replace the saved ones, so a rerun with no baseline, or a refused
    one, never shows an earlier run's verdict. Other modes keep theirs only while both
    candidate and baseline evidence still match.
    """
    saved = nb_common.read_json(verdicts_path(out)) if verdicts_path(out).exists() else {}
    if not isinstance(saved, dict):
        saved = {}
    previous = saved.pop("_provenance", {})
    if not isinstance(previous, dict):
        previous = {}
    preserved: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for mode, value in saved.items():
        proof = previous.get(mode, {})
        if mode not in nb_compare.MODES or mode in modes or not isinstance(proof, dict):
            continue
        baseline = proof.get("baseline")
        if not isinstance(baseline, str) or not baseline:
            continue
        candidate_hash = result_fingerprint(out, mode)
        baseline_hash = result_fingerprint(Path(baseline), mode)
        if (
            candidate_hash
            and baseline_hash
            and proof.get("candidate_hash") == candidate_hash
            and proof.get("baseline_hash") == baseline_hash
        ):
            preserved[mode] = value
            provenance[mode] = proof
    preserved.update(verdicts)
    for mode in verdicts:
        baseline = out / "baseline" if is_url(args.baseline) else Path(args.baseline)
        provenance[mode] = {
            "candidate_hash": result_fingerprint(out, mode),
            "baseline": str(baseline.resolve()),
            "baseline_hash": result_fingerprint(baseline, mode),
        }
    saved = {**preserved, **({"_provenance": provenance} if provenance else {})}
    nb_common.write_json(verdicts_path(out), saved)
    nb_common.write_text(out / "report.md", nb_compare.report_md(manifest, out, saved))
    facts = {**facts, "report": str(out / "report.md"), "verdicts": verdicts}
    refused = [p for p in problems if p["code"] == "baseline_mismatch"]
    if refused:
        emit(False, "baseline_mismatch", "report_without_verdict", facts, problems, EXIT_REFUSED)
    if problems:
        emit(False, "tool_failed", "fix_errors", facts, problems, EXIT_EXTERNAL)
    emit(True, "done", "report_to_user", facts)


def run_baseline(
    out: Path, spec: dict[str, Any] | None, margin: float, mode: str
) -> tuple[dict[str, Any], Errors]:
    """Compare one mode with the baseline. An endpoint baseline was already run into out/baseline."""
    if spec is None:
        return {}, []
    base_dir = out / "baseline" if spec["kind"] == "endpoint" else Path(spec["path"])
    return nb_compare.compare_dirs(out, base_dir, margin, (mode,))


def cmd_perf(args: argparse.Namespace) -> None:
    try:
        loads = split_ints(args.loads, "--loads")
    except ValueError as exc:
        emit(
            False,
            "bad_arguments",
            "fix_errors",
            errors=[error("bad_input", str(exc), "Fix --loads")],
            code=EXIT_INPUT,
        )
    errors: Errors = []
    if args.repeats < nb_common.MIN_REPEATS:
        errors.append(
            error(
                "bad_input",
                f"--repeats {args.repeats} is below {nb_common.MIN_REPEATS}",
                "Use at least 3 repeats",
            )
        )
    if args.images_per_request < 0:
        errors.append(
            error("bad_input", "--images-per-request must be 0 or more", "Use 0 for a text model")
        )
    sized = args.image_width is not None or args.image_height is not None
    if sized and args.images_per_request == 0:
        errors.append(
            error(
                "bad_input",
                "--image-width and --image-height need --images-per-request",
                "Give the number of images per request, or drop the image flags for a text model",
            )
        )
    if any(v is not None and v < 1 for v in (args.image_width, args.image_height)):
        errors.append(
            error(
                "bad_input", "image sizes must be 1 or more", "Fix --image-width and --image-height"
            )
        )
    if errors:
        emit(False, "bad_arguments", "fix_errors", errors=errors, code=EXIT_INPUT)
    base, _ = nb_common.split_endpoint(args.endpoint)
    plan = nb_perf.build_plan(
        endpoint=base,
        model=args.model,
        tokenizer=args.tokenizer,
        loads=loads,
        repeats=args.repeats,
        isl=args.isl,
        osl=args.osl,
        key_env=args.api_key_env,
        target=args.client,
        image=args.client_image,
        namespace=args.namespace,
        pod=args.pod,
        images=args.images_per_request,
        image_width=args.image_width,
        image_height=args.image_height,
        image_format=args.image_format,
    )
    plan["baseline"] = baseline_spec(args)
    pid = finish_plan(args, plan, need_docker=False)
    out = Path(args.out)
    run_plan = {**plan, "plan_id": pid}
    names, _ = check_endpoint(args, base, args.model, args.api_key_env)
    result = nb_perf.run_plan(run_plan, out)
    nb_common.write_json(
        out / "perf" / "summary.json", {k: result[k] for k in ("loads", "incomplete", "cache_salt")}
    )
    manifest = update_manifest(out, args, names, "perf", plan, pid)
    spec = plan["baseline"]
    problems: Errors = list(result["problems"])
    client_not_stopped = result["client_not_stopped"]
    if spec and spec["kind"] == "endpoint" and not client_not_stopped:
        side = out / "baseline"
        base_plan = {**with_endpoint(run_plan, spec)}
        base_result = nb_perf.run_plan(base_plan, side)
        nb_common.write_json(
            side / "perf" / "summary.json",
            {k: base_result[k] for k in ("loads", "incomplete", "cache_salt")},
        )
        base_names, _ = check_endpoint(args, spec["endpoint"], spec["model"], spec["key_env"])
        side_args = argparse.Namespace(endpoint=spec["endpoint"], model=spec["model"])
        update_manifest(side, side_args, base_names, "perf", {**with_endpoint(plan, spec)}, pid)
        problems += base_result["problems"]
        client_not_stopped = base_result["client_not_stopped"]
    verdicts, refused = (
        ({}, []) if client_not_stopped
        else run_baseline(out, spec, nb_stats.DEFAULT_MARGIN_POINTS, "perf")
    )
    finish_run(
        out,
        args,
        manifest,
        verdicts,
        problems + refused,
        {
            "plan_id": pid,
            "loads": result["loads"],
            "incomplete": result["incomplete"],
            "run_had_request_errors": result["run_had_request_errors"],
            "cache_salt": result["cache_salt"],
        },
        ("perf",),
    )


def judge_errors(args: argparse.Namespace, sets: list[str]) -> Errors:
    """The judge rules. A judged test set needs a named judge that is not a model under test."""
    judged = [s for s in sets if s in nb_accuracy.JUDGED_SETS]
    given = bool(args.judge_endpoint or args.judge_model)
    if judged and not (args.judge_endpoint and args.judge_model):
        return [
            error(
                "judge_required",
                f"{', '.join(judged)} is graded by an LLM judge",
                "Give --judge-endpoint and --judge-model: a model that is not the one measured",
            )
        ]
    if given and not judged:
        return [
            error(
                "bad_input",
                "no selected test set uses a judge, so the judge flags would do nothing",
                "Drop the judge flags, or select a vision test set",
            )
        ]
    if not judged:
        return []
    judge = (nb_common.split_endpoint(args.judge_endpoint)[0], args.judge_model)
    measured = [(nb_common.split_endpoint(args.endpoint)[0], args.model)]
    if args.baseline and is_url(args.baseline):
        measured.append(
            (nb_common.split_endpoint(args.baseline)[0], args.baseline_model or args.model)
        )
    if judge in measured:
        return [
            error(
                "self_judging",
                "the judge is the same endpoint and model as one being measured",
                "Use a different judge model: a model must not grade its own answers",
            )
        ]
    return []


def cmd_accuracy(args: argparse.Namespace) -> None:
    errors: Errors = []
    default_sets = nb_accuracy.VISION_SETS if args.vlm else nb_accuracy.TEXT_SETS
    sets = [s.strip() for s in (args.sets or ",".join(default_sets)).split(",") if s.strip()]
    unknown = [s for s in sets if s not in nb_accuracy.SETS]
    if not sets or unknown:
        errors.append(
            error(
                "bad_input",
                f"unknown test set: {', '.join(unknown) or '(none)'}",
                "Use: " + ", ".join(nb_accuracy.SETS),
            )
        )
    if not nb_accuracy.is_pinned(args.gym_image):
        errors.append(
            error(
                "image_not_pinned",
                "--gym-image must end in @sha256:<64 hex>",
                "Pin the NeMo Gym image by digest",
            )
        )
    if args.limit is not None and args.limit < 1:
        errors.append(error("bad_input", "--limit must be 1 or more", "Fix --limit"))
    errors += judge_errors(args, sets)
    if errors:
        emit(False, "bad_arguments", "fix_errors", errors=errors, code=EXIT_INPUT)
    _, v1 = nb_common.split_endpoint(args.endpoint)
    judge = None
    if args.judge_endpoint:
        judge = {
            "endpoint": nb_common.split_endpoint(args.judge_endpoint)[1],
            "model": args.judge_model,
        }
    plan = nb_accuracy.build_plan(
        endpoint_v1=v1,
        model=args.model,
        sets=sets,
        limit=args.limit,
        full=args.full,
        repeats=args.repeats,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        key_env=args.api_key_env,
        image=args.gym_image,
        judge=judge,
        judge_key_env=args.judge_api_key_env,
    )
    plan["baseline"] = baseline_spec(args)
    plan["margin"] = args.margin  # it changes the verdict, so the approval covers it
    pid = finish_plan(args, plan, need_docker=True)
    out = Path(args.out)
    run_plan = {**plan, "plan_id": pid}
    names, _ = check_endpoint(
        args, nb_common.split_endpoint(args.endpoint)[0], args.model, args.api_key_env
    )
    result = nb_accuracy.run_plan(run_plan, out)
    nb_common.write_json(out / "accuracy" / "summary.json", {"sets": result["sets"]})
    manifest = update_manifest(out, args, names, "accuracy", plan, pid)
    spec = plan["baseline"]
    problems: Errors = list(result["problems"])
    client_not_stopped = result["client_not_stopped"]
    if spec and spec["kind"] == "endpoint" and not client_not_stopped:
        side = out / "baseline"
        base_plan = with_endpoint(run_plan, spec)
        base_plan["endpoint"] = nb_common.split_endpoint(spec["endpoint"])[1]
        base_result = nb_accuracy.run_plan(base_plan, side)
        nb_common.write_json(side / "accuracy" / "summary.json", {"sets": base_result["sets"]})
        side_args = argparse.Namespace(endpoint=spec["endpoint"], model=spec["model"])
        update_manifest(side, side_args, [], "accuracy", base_plan, pid)
        problems += base_result["problems"]
        client_not_stopped = base_result["client_not_stopped"]
    verdicts, refused = (
        ({}, []) if client_not_stopped
        else run_baseline(out, spec, args.margin, "accuracy")
    )
    finish_run(
        out,
        args,
        manifest,
        verdicts,
        problems + refused,
        {"plan_id": pid, "sets": result["sets"]},
        ("accuracy",),
    )


def cmd_compare(args: argparse.Namespace) -> None:
    if is_url(args.baseline):
        emit(
            False,
            "bad_arguments",
            "fix_errors",
            errors=[
                error(
                    "bad_input",
                    "compare reads run folders",
                    "To compare with an endpoint, use perf or accuracy with --baseline URL",
                )
            ],
            code=EXIT_INPUT,
        )
    out = Path(args.run)
    verdicts, problems = nb_compare.compare_dirs(out, Path(args.baseline), args.margin)
    manifest = (
        nb_common.read_json(out / "manifest.json") if (out / "manifest.json").exists() else {}
    )
    finish_run(out, args, manifest, verdicts, problems, {}, nb_compare.MODES)


def cmd_check(args: argparse.Namespace) -> None:
    spec = baseline_spec(args)
    errors = precondition_errors(args, spec, need_docker=False)
    base, _ = nb_common.split_endpoint(args.endpoint)
    in_pod = getattr(args, "client", None) == "kubernetes"
    pod_ready = False
    if in_pod and shutil.which("kubectl"):
        pod_ready = client_pod_status(args)[0]
    names: list[str] = []
    if not in_pod or pod_ready:
        names, _ = check_endpoint(args, base, args.model, args.api_key_env)
    emit(
        not errors,
        "checked",
        "ask_user" if not errors else "fix_errors",
        {
            "served_models": names,
            "docker": bool(shutil.which("docker")),
            # With a Kubernetes client the pod runs AIPerf, so that is the fact that matters.
            "aiperf": pod_ready if in_pod else bool(shutil.which("aiperf")),
            "aiperf_in": f"pod {args.namespace}/{args.pod}" if in_pod else "this machine",
        },
        errors,
        EXIT_PRECONDITION if errors else 0,
    )


def add_endpoint_flags(p: argparse.ArgumentParser, with_run: bool) -> None:
    p.add_argument(
        "--endpoint", required=True, help="OpenAI-compatible URL, for example http://host:8000/v1"
    )
    p.add_argument("--model", required=True, help="served model name, as /v1/models lists it")
    p.add_argument("--api-key-env", help="NAME of the variable that holds the endpoint key")
    p.add_argument("--baseline", help="a baseline endpoint URL, or a finished run folder")
    p.add_argument("--baseline-model", help="served model name at the baseline endpoint")
    p.add_argument(
        "--baseline-api-key-env", help="NAME of the variable that holds the baseline key"
    )
    if with_run:
        p.add_argument("--out", required=True, help="run folder to write")
        p.add_argument(
            "--dry-run", action="store_true", help="print the plan and its plan id; send no load"
        )
        p.add_argument("--approved", help="the plan id the user approved")


def build_parser() -> argparse.ArgumentParser:
    parser = ContractParser(
        description="Measure a running NIM endpoint for speed and accuracy.",
        epilog="Example: python3 scripts/nimbench.py perf --endpoint http://gpu-01:8000/v1 "
        "--model qwen7b --tokenizer Qwen/Qwen2.5-7B-Instruct --out runs/a --dry-run",
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ContractParser)
    check = sub.add_parser("check", help="read /v1/models; send no load")
    add_endpoint_flags(check, with_run=False)
    check.add_argument(
        "--client",
        choices=("local", "kubernetes"),
        default="local",
        help="where endpoint checks run",
    )
    check.add_argument("--namespace", help="namespace of the client pod")
    check.add_argument("--pod", help="client pod with python3 installed")
    perf = sub.add_parser("perf", help="speed benchmark with AIPerf")
    add_endpoint_flags(perf, with_run=True)
    perf.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer id of the model")
    perf.add_argument("--loads", default="8,32,128", help="concurrency levels, comma separated")
    perf.add_argument(
        "--repeats", type=int, default=nb_common.MIN_REPEATS, help="repeats per load, at least 3"
    )
    perf.add_argument("--isl", type=int, default=nb_perf.DEFAULT_ISL, help="input tokens")
    perf.add_argument("--osl", type=int, default=nb_perf.DEFAULT_OSL, help="output tokens")
    perf.add_argument(
        "--client",
        choices=("docker", "local", "kubernetes"),
        default="docker",
        help="where AIPerf runs",
    )
    perf.add_argument(
        "--client-image", default=nb_common.DEFAULT_CLIENT_IMAGE, help="image for --client docker"
    )
    perf.add_argument(
        "--images-per-request",
        type=int,
        default=0,
        help="images in every request, for a vision-language model; 0 means a text model",
    )
    perf.add_argument("--image-width", type=int, help="image width in pixels, default 512")
    perf.add_argument("--image-height", type=int, help="image height in pixels, default 512")
    perf.add_argument(
        "--image-format",
        choices=nb_perf.IMAGE_FORMATS,
        default=nb_perf.DEFAULT_IMAGE_FORMAT,
        help="image file format",
    )
    perf.add_argument("--namespace", help="namespace of the client pod, for --client kubernetes")
    perf.add_argument("--pod", help="client pod that has aiperf installed, for --client kubernetes")
    acc = sub.add_parser("accuracy", help="accuracy test sets with NeMo Gym, in Docker")
    add_endpoint_flags(acc, with_run=True)
    acc.add_argument(
        "--gym-image", required=True, help="NeMo Gym image pinned by digest, name@sha256:..."
    )
    acc.add_argument(
        "--sets",
        help="test sets, comma separated; default is the text sets, or the vision sets with --vlm",
    )
    acc.add_argument(
        "--vlm",
        action="store_true",
        help="the model reads images: default to the vision sets labbench2_vlm and gdp_pdf",
    )
    acc.add_argument("--judge-endpoint", help="OpenAI-compatible URL of the judge, for vision sets")
    acc.add_argument("--judge-model", help="served model name of the judge")
    acc.add_argument("--judge-api-key-env", help="NAME of the variable that holds the judge key")
    acc.add_argument(
        "--limit", type=int, help="items per test set; default is every item, MMLU-Pro 2000"
    )
    acc.add_argument("--full", action="store_true", help="run every item of every test set")
    acc.add_argument(
        "--repeats", type=int, default=1, help="answers per item; use more with temperature above 0"
    )
    acc.add_argument(
        "--temperature",
        type=float,
        default=nb_accuracy.DEFAULT_TEMPERATURE,
        help="sampling temperature",
    )
    acc.add_argument(
        "--max-tokens", type=int, default=nb_accuracy.DEFAULT_MAX_TOKENS, help="output token limit"
    )
    acc.add_argument(
        "--margin", type=float, default=nb_stats.DEFAULT_MARGIN_POINTS, help="pass margin in points"
    )
    cmp_ = sub.add_parser("compare", help="compare a run folder with a baseline run folder")
    cmp_.add_argument("run", help="the run folder")
    cmp_.add_argument("--baseline", required=True, help="the baseline run folder")
    cmp_.add_argument(
        "--margin", type=float, default=nb_stats.DEFAULT_MARGIN_POINTS, help="pass margin in points"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(raw)
    errors = input_errors(args, raw)
    if errors:
        emit(False, "bad_arguments", "fix_errors", errors=errors, code=EXIT_INPUT)
    {"check": cmd_check, "perf": cmd_perf, "accuracy": cmd_accuracy, "compare": cmd_compare}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
