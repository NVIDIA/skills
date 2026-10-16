#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomically commit one verified stage to the DEFT OD AOI state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deft_od_aoi_round_robin_contract import ADMISSION_INDEX_WIDTH

NEXT = {"synthesis_bootstrap": "candidate_cache", "candidate_cache": "baseline_measurement",
        "baseline_measurement": "baseline_gaps",
        "baseline_gaps": "iteration_retrieval", "iteration_retrieval": "iteration_admission",
        "iteration_admission": "iteration_training", "iteration_synthesis": "iteration_training",
        "iteration_training": "iteration_measurement",
        "iteration_measurement": "iteration_gaps"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(artifacts: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    if name not in artifacts:
        raise ValueError(f"stage commit requires artifact {name}")
    value = json.loads(Path(artifacts[name]["path"]).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact {name} must be a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonzero_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("query counts must be a JSON object")
    result = {}
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("query counts must be nonnegative integers")
        if count:
            result[str(key)] = count
    return result


def _validate_admission_index(artifact: dict[str, Any]) -> None:
    index = np.load(artifact["path"], mmap_mode="r")
    if index.ndim != 2 or index.shape[1] != ADMISSION_INDEX_WIDTH:
        raise ValueError(f"invalid round-robin admission index shape {index.shape}")


def _validate_retrieval(iteration: int, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifest = _read_json(artifacts, "query_manifest")
    if manifest.get("status") != "COMPLETE" or int(manifest.get("iteration", -1)) != iteration:
        raise ValueError("query manifest is incomplete or for another iteration")
    counts = _nonzero_counts(manifest.get("query_counts"))
    enabled = set(manifest.get("enabled_roles") or [])
    if not enabled.issubset(counts):
        raise ValueError("enabled retrieval roles disagree with nonzero query counts")
    role_status = manifest.get("role_status")
    if not isinstance(role_status, dict) or set(role_status) != {"real", "clean"}:
        raise ValueError("retrieval manifest lacks per-role exhaustion evidence")
    selected_counts = None
    if manifest.get("selection_strategy") == "round_robin_similarity":
        report = _read_json(artifacts, "selection_report")
        if (report.get("status") != "COMPLETE"
                or int(report.get("iteration", -1)) != iteration
                or report.get("selection_strategy") != "round_robin_similarity"):
            raise ValueError("round-robin selection report is incomplete or mismatched")
        selected_counts = report.get("selected_counts")
        if not isinstance(selected_counts, dict):
            raise ValueError("round-robin selection report has no selected counts")
        if "admission_index" not in artifacts:
            raise ValueError("round-robin retrieval requires artifact admission_index")
        _validate_admission_index(artifacts["admission_index"])
        if Path(str(report.get("admission_index"))).resolve() != Path(
                artifacts["admission_index"]["path"]).resolve():
            raise ValueError("round-robin selection report has a mismatched admission index")
        selection_role_status = report.get("role_status")
        if not isinstance(selection_role_status, dict):
            raise ValueError("round-robin selection report has no role outcomes")
    for role, count in counts.items():
        evidence = role_status.get(role) or {}
        if int(evidence.get("query_count", -1)) != count:
            raise ValueError(f"{role} role evidence disagrees with query count")
        if evidence.get("status") not in {"READY", "EXHAUSTED"}:
            raise ValueError(f"{role} role has invalid retrieval status")
        if int(evidence.get("candidate_count", -1)) != (
                int(evidence.get("excluded_count", -1))
                + int(evidence.get("remaining_candidate_count", -1))):
            raise ValueError(f"{role} exclusion audit does not reconcile")
        for suffix in (("queries", "exclusions", "query_embeddings", "mined")
                       if role in enabled else ("queries", "exclusions")):
            name = f"{role}_{suffix}"
            if name not in artifacts:
                raise ValueError(f"iteration retrieval requires artifact {name}")
        if len(pd.read_parquet(artifacts[f"{role}_queries"]["path"])) != count:
            raise ValueError(f"{role} query count disagrees with its manifest")
        exclusions = pd.read_parquet(artifacts[f"{role}_exclusions"]["path"])
        if "filepath" not in exclusions or exclusions.filepath.nunique() != int(
                evidence["excluded_count"]):
            raise ValueError(f"{role} exclusion manifest disagrees with its audit")
        if role in enabled and len(pd.read_parquet(
                artifacts[f"{role}_query_embeddings"]["path"])) != count:
            raise ValueError(f"{role} embedding count disagrees with its manifest")
        if role not in enabled:
            continue
        mined = pd.read_parquet(artifacts[f"{role}_mined"]["path"])
        if selected_counts is not None and selected_counts.get(role) != len(mined):
            raise ValueError(f"{role} selection count disagrees with its report")
        if selected_counts is not None:
            evidence = selection_role_status.get(role) or {}
            expected = "SELECTED" if len(mined) else "NO_MATCHES"
            if (evidence.get("status") != expected
                    or evidence.get("selected_count") != len(mined)):
                raise ValueError(f"{role} selection outcome disagrees with its artifact")
        elif "filepath" not in mined:
            raise ValueError(f"{role} mining output lacks filepath")
    if bool(manifest.get("converged")) != (not enabled and not manifest.get("synthesis_pending")):
        raise ValueError("retrieval convergence evidence is inconsistent")
    return manifest


def _validate_admission(iteration: int,
                        artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = _read_json(artifacts, "admission_report")
    if report.get("status") != "COMPLETE" or int(report.get("iteration", -1)) != iteration:
        raise ValueError("admission report is incomplete or for another iteration")
    if report.get("selection_strategy") == "round_robin_similarity":
        if "admission_index" not in artifacts:
            raise ValueError("round-robin admission requires artifact admission_index")
        _validate_admission_index(artifacts["admission_index"])
        if Path(str(report.get("admission_index"))).resolve() != Path(
                artifacts["admission_index"]["path"]).resolve():
            raise ValueError("round-robin admission report has a mismatched admission index")
    admitted = report.get("admitted")
    if not isinstance(admitted, dict):
        raise ValueError("admission report has no admitted counts")
    for role in ("real", "clean", "synthetic"):
        count = admitted.get(role, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("admission report has invalid admitted counts")
    outcomes = report.get("role_status")
    valid_outcomes = isinstance(outcomes, dict)
    for value in outcomes.values() if valid_outcomes else []:
        count = value.get("selected_count") if isinstance(value, dict) else None
        valid_outcomes = (
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            and value.get("status") == ("SELECTED" if count else "NO_MATCHES")
        )
        if not valid_outcomes:
            break
    if not valid_outcomes:
        raise ValueError("admission report has invalid role outcomes")
    admitted = report.get("admitted") or {}
    expected_new = sum(int(admitted.get(role, -1)) for role in ("real", "clean", "synthetic"))
    if expected_new < 0 or report.get("new_training_images") != expected_new:
        raise ValueError("admission report new-image count does not reconcile")
    return report


def _synthetic_room(admission: dict[str, Any]) -> int:
    evidence = admission.get("synthetic_admission")
    room = evidence.get("available_room_before_admission") if isinstance(evidence, dict) else None
    if isinstance(room, bool) or not isinstance(room, int) or room < 0:
        raise ValueError("admission report lacks valid synthetic capacity")
    return room


def _committed_admission(state: dict[str, Any], iteration: int) -> dict[str, Any]:
    previous = next((event for event in reversed(state.get("events", []))
                     if event.get("stage") == "iteration_admission"
                     and event.get("iteration") == iteration), {})
    artifacts = previous.get("artifacts") or {}
    if "admission_report" not in artifacts:
        raise ValueError("synthesis requires committed initial admission evidence")
    for artifact in artifacts.values():
        path = Path(artifact["path"])
        if not path.is_file() or _sha(path) != artifact["sha256"]:
            raise ValueError("committed initial admission evidence changed")
    return _validate_admission(iteration, artifacts)


def _validate_budget_skip(state: dict[str, Any], iteration: int,
                          artifacts: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    request = _read_json(artifacts, "synthesis_request")
    admission = _validate_admission(iteration, artifacts)
    if set(artifacts) - {"synthesis_request", "admission_report", "admission_index"}:
        raise ValueError("a budget skip must not include generation artifacts")
    previous = next((event for event in reversed(state.get("events", []))
                     if event.get("stage") == "iteration_admission"
                     and event.get("iteration") == iteration), {})
    committed = (previous.get("artifacts") or {}).get("admission_report") or {}
    supplied = artifacts["admission_report"]
    if (committed.get("path") != supplied["path"]
            or committed.get("sha256") != supplied["sha256"]):
        raise ValueError("budget skip requires the committed admission report")
    if (request.get("status") != "SKIPPED"
            or request.get("reason") != "no_synthetic_budget"):
        raise ValueError("budget skip must declare SKIPPED/no_synthetic_budget")
    if request.get("selection_mode") != "generated_per_type_plan":
        raise ValueError(
            "no_synthetic_budget requires selection_mode=generated_per_type_plan; "
            "all_eligible does not use pre-generation budgeting"
        )
    planning = request.get("planning")
    names = ("new_image_budget", "images_per_fn", "eligible_fn_count",
             "selected_fn_count", "planned_images", "unplanned_budget")
    if (not isinstance(planning, dict)
            or any(isinstance(planning.get(name), bool)
                   or not isinstance(planning.get(name), int)
                   or planning[name] < 0 for name in names)):
        raise ValueError("budget skip has invalid planning counts")
    if (request.get("fn_count") != 0
            or request.get("eligible_fn_count") != planning["eligible_fn_count"]
            or planning["eligible_fn_count"] < 1
            or planning["new_image_budget"] != _synthetic_room(admission)
            or not planning["new_image_budget"] < planning["images_per_fn"]
            or planning["selected_fn_count"] != 0
            or planning["planned_images"] != 0
            or planning["unplanned_budget"] != planning["new_image_budget"]):
        raise ValueError("budget skip has inconsistent whole-FN planning evidence")
    return admission, "no_synthetic_budget"


def _validate_synthesis(iteration: int, artifacts: dict[str, dict[str, Any]],
                        state: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    if "synthesis_request" in artifacts:
        request = _read_json(artifacts, "synthesis_request")
        if request.get("reason") == "no_synthetic_budget":
            return _validate_budget_skip(state, iteration, artifacts)
    skip_contracts = {
        "synthesis_request": "no_routed_false_negatives",
        "synthesis_preparation": "no_eligible_false_negatives",
    }
    for name, reason in skip_contracts.items():
        if name not in artifacts:
            continue
        report = _read_json(artifacts, name)
        if set(artifacts) != {name}:
            raise ValueError("a skipped synthesis stage accepts only its skip contract")
        if report.get("status") != "SKIPPED" or report.get("reason") != reason:
            raise ValueError(f"{name} is not a recognized synthesis skip contract")
        if name == "synthesis_request":
            by_dataset = report.get("skipped_unrouted_by_dataset")
            skipped = int(report.get("skipped_unrouted_fn_count", -1))
            observed = report.get("observed_dataset_ids")
            routes = report.get("configured_route_keys")
            if (int(report.get("fn_count", -1)) != 0 or not isinstance(by_dataset, dict)
                    or sum(map(int, by_dataset.values())) != skipped
                    or not isinstance(observed, list) or not isinstance(routes, list)
                    or set(by_dataset) != set(observed) or set(observed) & set(routes)
                    or not str(report.get("message") or "").strip()):
                raise ValueError("skipped synthesis request has invalid FN counts")
        else:
            skipped = int(report.get("skipped_fn_count", -1))
            skipped_fns, skip_counts = report.get("skipped_fns"), report.get("skip_counts")
            if (int(report.get("eligible_fn_count", -1)) != 0 or skipped < 1
                    or int(report.get("selection_candidate_fn_count", -1)) != skipped
                    or not isinstance(skipped_fns, list) or len(skipped_fns) != skipped
                    or not isinstance(skip_counts, dict)
                    or sum(map(int, skip_counts.values())) != skipped):
                raise ValueError("skipped synthesis preparation has invalid FN counts")
            clean_reference_skips = int(skip_counts.get("no_clean_reference_images", 0))
            if clean_reference_skips:
                warnings = report.get("warnings")
                clean_reference_rows = [
                    row for row in skipped_fns
                    if isinstance(row, dict)
                    and row.get("reason") == "no_clean_reference_images"
                ]
                if (len(clean_reference_rows) != clean_reference_skips
                        or not isinstance(warnings, list)
                        or not any(warning.get("code") == "no_clean_reference_images"
                                   and int(warning.get("fn_count", -1))
                                   == clean_reference_skips
                                   and str(warning.get("message") or "").strip()
                                   for warning in warnings if isinstance(warning, dict))):
                    raise ValueError("no-clean-reference skip lacks typed warning evidence")
        return report, None
    generation = _read_json(artifacts, "generation_report")
    admission = _read_json(artifacts, "admission_report")
    if generation.get("status") != "COMPLETE":
        raise ValueError("generation report is incomplete")
    generated = blocked = requested = 0
    groups = generation.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("generation report has no dataset groups")
    for group in groups:
        group_requested = int(group.get("requested", -1))
        group_generated = int(group.get("generated", -1))
        group_blocked = int(group.get("guardrail_blocked", -1))
        if min(group_requested, group_generated, group_blocked) < 0:
            raise ValueError("generation report contains negative counts")
        if group_generated + group_blocked != group_requested:
            raise ValueError("generated and blocked counts do not reconcile")
        requested += group_requested
        generated += group_generated
        blocked += group_blocked
    if int(generation.get("generated", -1)) != generated or generated + blocked != requested:
        raise ValueError("generation totals disagree with dataset groups")
    report = _validate_admission(iteration, artifacts)
    admitted = int((admission.get("admitted") or {}).get("synthetic", -1))
    if admitted < 0 or admitted > generated:
        raise ValueError("synthetic admission count exceeds generated images")
    return report, None


def _retrieval_had_output(state: dict[str, Any], iteration: int) -> bool | None:
    for event in reversed(state.get("events", [])):
        if event.get("stage") != "iteration_retrieval" or event.get("iteration") != iteration:
            continue
        artifact = (event.get("artifacts") or {}).get("query_manifest")
        if not artifact:
            raise ValueError("synthesis cannot verify retrieval availability")
        path = Path(str(artifact.get("path") or ""))
        if not path.is_file() or _sha(path) != artifact.get("sha256"):
            raise ValueError("committed retrieval manifest failed integrity validation")
        manifest = json.loads(path.read_text())
        enabled = manifest.get("enabled_roles")
        if not isinstance(enabled, list):
            raise ValueError("committed retrieval manifest lacks enabled roles")
        return bool(enabled)
    return None


def _validate_training(artifacts: dict[str, dict[str, Any]]) -> None:
    report = _read_json(artifacts, "checkpoint_selection")
    if report.get("status") != "COMPLETE" or report.get("action") != "select":
        raise ValueError("checkpoint selection is not final")
    try:
        epoch = int(report["best_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint selection has no valid epoch") from error
    if epoch < 0 or _finite(report.get("best_kpi_mAP50"), "best_kpi_mAP50") < 0:
        raise ValueError("checkpoint selection has invalid KPI evidence")
    checkpoint = Path(str(report.get("selected_checkpoint") or "")).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.name != f"model_epoch_{epoch:03d}.pth":
        raise ValueError("checkpoint selection does not bind the KPI-best checkpoint")
    statuses = report.get("status_files")
    if not isinstance(statuses, list) or not statuses:
        raise ValueError("checkpoint selection has no status evidence")
    if any(not Path(str(path)).is_file() for path in statuses):
        raise ValueError("checkpoint selection references missing status evidence")


def _committed_checkpoint(state: dict[str, Any], iteration: int) -> tuple[Path, str]:
    for event in reversed(state.get("events", [])):
        if event.get("stage") != "iteration_training" or event.get("iteration") != iteration:
            continue
        artifact = (event.get("artifacts") or {}).get("checkpoint_selection")
        if artifact and _sha(Path(artifact["path"])) == artifact["sha256"]:
            report = json.loads(Path(artifact["path"]).read_text())
            checkpoint = Path(str(report.get("selected_checkpoint") or "")).resolve()
            if checkpoint.is_file():
                return checkpoint, _sha(checkpoint)
        break
    raise ValueError("measurement has no valid committed checkpoint")


def _validate_measurement(state: dict[str, Any], iteration: int,
                          artifacts: dict[str, dict[str, Any]]) -> None:
    manifest = _read_json(artifacts, "measurement_manifest")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("measurement manifest is incomplete")
    checkpoint, checkpoint_sha = _committed_checkpoint(state, iteration)
    if (Path(str(manifest.get("checkpoint") or "")).resolve() != checkpoint
            or manifest.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("measurement does not bind the committed checkpoint")
    roles = manifest.get("inference_roles")
    if not isinstance(roles, dict) or set(roles) != {"kpi", "test"}:
        raise ValueError("measurement lacks KPI/test inference evidence")
    for role, evidence in roles.items():
        expected = int(evidence.get("expected_images", -1))
        predictions = Path(str(evidence.get("predictions") or ""))
        actual = len(list(predictions.glob("*.txt"))) if predictions.is_dir() else -1
        if expected < 1 or actual != expected:
            raise ValueError(f"{role} inference cardinality mismatch: expected {expected}, got {actual}")


def _gap_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) - {"FP", "FN"}:
        raise ValueError(f"{label} must contain only FP/FN counts")
    result = {"FP": 0, "FN": 0}
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label} must contain nonnegative integer counts")
        result[key] = count
    return result


def _validate_gaps(artifacts: dict[str, dict[str, Any]]) -> None:
    confidences = {}
    for kind in ("loose", "strict"):
        report = _read_json(artifacts, f"{kind}_gap_report")
        totals = _gap_counts(report.get("counts_by_type"), f"{kind}.counts_by_type")
        classes = report.get("counts_by_class")
        if report.get("kpi") != f"kpi_{kind}" or not isinstance(classes, dict):
            raise ValueError(f"{kind} gap report has invalid identity or class counts")
        aggregate = {"FP": 0, "FN": 0}
        for name, counts in classes.items():
            normalized = _gap_counts(counts, f"{kind}.counts_by_class.{name}")
            for gap_type in aggregate:
                aggregate[gap_type] += normalized[gap_type]
        if aggregate != totals:
            raise ValueError(f"{kind} gap report type and class counts disagree")
        boxes = f"{kind}_box_gaps"
        if boxes not in artifacts or len(pd.read_parquet(artifacts[boxes]["path"])) != sum(totals.values()):
            raise ValueError(f"{kind} gap parquet and report counts disagree")
        settings = report.get("settings") or {}
        confidences[kind] = _finite(settings.get("conf_threshold"), f"{kind}.conf_threshold")
        iou = _finite(settings.get("iou_threshold"), f"{kind}.iou_threshold")
        if not 0 <= confidences[kind] <= 1 or not 0 <= iou <= 1:
            raise ValueError(f"{kind} gap thresholds are out of range")
    if confidences["loose"] >= confidences["strict"]:
        raise ValueError("loose gap confidence must be lower than strict gap confidence")


def commit(state_path: Path, stage: str, iteration: int, values: list[str]) -> dict[str, Any]:
    state = json.loads(state_path.read_text())
    if state.get("status") not in {"READY", "RUNNING"} or state.get("next_stage") != stage:
        raise ValueError(f"expected stage {state.get('next_stage')}, not {stage}")
    expected = int(state["current_iteration"])
    if stage == "iteration_retrieval" and state.get("last_stage") in {"baseline_gaps", "iteration_gaps"}:
        expected += 1
    if iteration != expected:
        raise ValueError(f"expected iteration {expected}, not {iteration}")
    artifacts = {}
    for value in values:
        name, separator, raw = value.partition("=")
        path = Path(raw).expanduser().resolve()
        if not separator or not name or not path.is_file():
            raise ValueError(f"artifact must be name=existing-file: {value}")
        artifacts[name] = {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}
    if not artifacts:
        raise ValueError("at least one completion artifact is required")
    retrieval = None
    synthesis = None
    admission = None
    synthesis_skip_reason = None
    if stage == "iteration_retrieval":
        retrieval = _validate_retrieval(iteration, artifacts)
    elif stage == "iteration_admission":
        admission = _validate_admission(iteration, artifacts)
    elif stage == "iteration_synthesis":
        synthesis, synthesis_skip_reason = _validate_synthesis(
            iteration, artifacts, state
        )
        if synthesis.get("status") == "COMPLETE":
            admission = synthesis
    elif stage == "iteration_training":
        _validate_training(artifacts)
    elif stage == "iteration_measurement":
        _validate_measurement(state, iteration, artifacts)
    elif stage == "iteration_gaps":
        _validate_gaps(artifacts)
    next_stage, status = NEXT.get(stage), "RUNNING"
    synthesis_decision = None
    if stage == "iteration_retrieval" and retrieval and retrieval.get("converged"):
        next_stage, status = None, "COMPLETE"
        state["completion_reason"] = "mining_exhausted"
    if (stage == "iteration_synthesis" and synthesis
            and synthesis.get("status") == "SKIPPED"
            and _retrieval_had_output(state, iteration) is False):
        next_stage, status = None, "COMPLETE"
        state["completion_reason"] = "all_producers_exhausted"
    if stage == "iteration_admission" and state.get("synthesis_enabled"):
        room = _synthetic_room(admission)
        if room:
            next_stage = "iteration_synthesis"
        else:
            synthesis_decision = {"status": "SKIPPED", "reason": "no_synthetic_budget",
                                  "available_room_before_admission": room}
    if synthesis_skip_reason:
        synthesis_decision = {"status": "SKIPPED", "reason": synthesis_skip_reason,
                              "available_room_before_admission": _synthetic_room(admission)}
    if stage == "iteration_synthesis" and admission and not admission["new_training_images"]:
        admission = _committed_admission(state, iteration)
    if (next_stage != "iteration_synthesis"
            and stage in {"iteration_admission", "iteration_synthesis"} and admission and not int(
                admission["new_training_images"])):
        next_stage, status = None, "COMPLETE"
        outcomes = list(admission["role_status"].values())
        state["completion_reason"] = (
            "retrieval_no_matches"
            if outcomes and all(row["status"] == "NO_MATCHES" for row in outcomes)
            else "retrieval_no_new_data"
        )
    if stage == "iteration_gaps":
        if iteration >= int(state["max_iterations"]):
            next_stage, status = None, "COMPLETE"
        else:
            next_stage = "iteration_retrieval"
    state.update(status=status, current_iteration=iteration, last_stage=stage,
                 next_stage=next_stage)
    event = {"stage": stage, "iteration": iteration, "artifacts": artifacts,
             "committed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if synthesis_decision:
        event["synthesis_decision"] = synthesis_decision
    state.setdefault("events", []).append(event)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, state_path)
    with (state_path.parent / "loop_log.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(NEXT) + ("iteration_gaps",), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()
    result = commit(args.state.resolve(), args.stage, args.iteration, args.artifact)
    summary = {"status": result["status"], "next_stage": result["next_stage"]}
    if result.get("completion_reason"):
        summary["completion_reason"] = result["completion_reason"]
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
