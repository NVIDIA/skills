#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Atomically commit one Cosmos3 DEFT AOI stage to ``deft_state.json``.

The state file contains both the resume snapshot and ordered stage events, so
the run has one durable source of truth.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

RCCA_ARTIFACT_MANIFEST = (
    pathlib.Path(__file__).resolve().parents[1]
    / "references"
    / "rcca-artifact-manifest.json"
)

from record_metric_result import commit as commit_metric_result
from render_report import render as render_html_report


STAGES = (
    "train",
    "evaluate_proxy",
    "proxy_rcca",
    "evaluate_benchmark",
    "benchmark_metrics",
    "routing",
    "anomalygen",
    "data_mining",
    "assemble_data",
    "validate_data",
    "loop_stop",
)
SKIPPABLE_STAGES = ("anomalygen",)


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: pathlib.Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _migrate_execution_policy(state: dict[str, Any]) -> None:
    if isinstance(state.get("execution_policy"), dict):
        return
    offline = os.environ.get("AIR_GAPPED") == "1"
    state["execution_policy"] = {
        "network_mode": "airgap" if offline else "network-enabled",
        "activation_source": "legacy-state:AIR_GAPPED" if offline else "legacy-state:default",
        "allow_package_install": not offline,
        "allow_remote_fetch": not offline,
        "allow_container_pull": not offline,
        "allow_registry_login": not offline,
        "python_launcher": "scripts/deft_python.sh",
        "python_executable": str(pathlib.Path(sys.executable).resolve()),
        "hf_offline": offline,
    }


def _required_file(value: pathlib.Path | None, flag: str) -> str:
    if value is None:
        raise ValueError(f"{flag} is required")
    path = value.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{flag} must be absolute: {value}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{flag} must be an existing non-empty file: {value}")
    return str(path.resolve())


def _normalized_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _rcca_manifest_entries(
    manifest: dict[str, Any], class_name: str
) -> list[dict[str, Any]]:
    classes = manifest.get("artifact_classes")
    entries = classes.get(class_name) if isinstance(classes, dict) else None
    if not isinstance(entries, list) or not entries or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(
            f"RCCA artifact manifest {class_name} must be a non-empty array"
        )
    return entries


def _rcca_manifest_path(
    output_dir: pathlib.Path, entry: dict[str, Any]
) -> pathlib.Path:
    value = entry.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("RCCA artifact manifest entry has no non-empty path")
    relative = pathlib.Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"RCCA artifact manifest path must be relative: {value}"
        )
    return output_dir / relative


def _rcca_argument(
    args: argparse.Namespace, entry: dict[str, Any]
) -> tuple[pathlib.Path | None, str, str]:
    state_field = entry.get("state_field")
    if not isinstance(state_field, str) or not state_field:
        raise ValueError(
            f"RCCA artifact {entry.get('path')} must declare state_field"
        )
    argument = state_field.removesuffix("_json")
    flag = f"--{argument.replace('_', '-')}"
    return getattr(args, argument, None), flag, state_field


def _validate_rcca_report(
    report: pathlib.Path, entry: dict[str, Any], flag: str
) -> None:
    validation = entry.get("validation")
    headings = (
        validation.get("required_headings")
        if isinstance(validation, dict)
        and validation.get("type") == "markdown_sections"
        else None
    )
    if not isinstance(headings, list) or not all(
        isinstance(heading, str) and heading for heading in headings
    ):
        raise ValueError(
            "RCCA report manifest must declare markdown required_headings"
        )
    try:
        text = report.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{flag} must be UTF-8 text: {report}") from exc
    actual = {
        _normalized_heading(match.group(1))
        for match in re.finditer(
            r"^##[ \t]+(?:[0-9]+\.[ \t]*)?(.+?)[ \t]*$",
            text,
            flags=re.MULTILINE,
        )
    }
    missing = [
        heading
        for heading in headings
        if _normalized_heading(heading) not in actual
    ]
    if missing:
        raise ValueError(
            f"{flag} is missing required section heading(s): "
            f"{', '.join(missing)}; follow references/RCCA_REPORT_TEMPLATE.md"
        )


def _required_rcca_artifacts(
    args: argparse.Namespace, phase_root: pathlib.Path
) -> dict[str, str]:
    try:
        manifest = json.loads(
            RCCA_ARTIFACT_MANIFEST.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load RCCA artifact manifest: {RCCA_ARTIFACT_MANIFEST} ({exc})"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("RCCA artifact manifest root must be an object")

    output_dir = phase_root / "proxy_rcca"
    state_artifacts: dict[str, str] = {}
    for class_name in (
        "container_or_script_produced_required",
        "agent_produced_required",
    ):
        for entry in _rcca_manifest_entries(manifest, class_name):
            if entry.get("kind") != "file":
                raise ValueError(
                    f"RCCA artifact {entry.get('path')} must have kind=file"
                )
            value, flag, state_field = _rcca_argument(args, entry)
            artifact = pathlib.Path(_required_file(value, flag))
            if class_name == "agent_produced_required":
                expected = _rcca_manifest_path(output_dir, entry).resolve()
                if artifact != expected:
                    raise ValueError(
                        f"{flag} must point to the manifest artifact: {expected}"
                    )
            else:
                artifact = pathlib.Path(
                    _within(str(artifact), phase_root, flag)
                )
            validation = entry.get("validation")
            if isinstance(validation, dict):
                _validate_rcca_report(artifact, entry, flag)
            elif validation != "non_empty":
                raise ValueError(
                    f"RCCA artifact {entry.get('path')} has unsupported validation"
                )
            state_artifacts[state_field] = str(artifact)
    return state_artifacts


def _parquet_row_count(path: str, flag: str) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            f"{flag} validation requires pyarrow in the selected DEFT Python"
        ) from exc
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:  # noqa: BLE001 - normalize parquet parser failures
        raise ValueError(f"{flag} must be a readable parquet file: {path} ({exc})") from exc


def _required_json_file(value: pathlib.Path | None, flag: str) -> str:
    """Like _required_file, but say so plainly when the file is not JSON.

    Routing and mining both deal in parquet, so a parquet lands on a `*_json`
    flag easily. Without this the failure surfaces much later as a UTF-8 decode
    error on a parquet magic byte, which names neither the expected format nor
    the offending flag.
    """
    path = _required_file(value, flag)
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head == b"PAR1":
        raise ValueError(
            f"{flag} expects a JSON array but got a parquet file: {path}. "
            "Routing produces both — the JSON for state, the parquet for the "
            "embedding container. Pass the JSON here."
        )
    try:
        payload = json.loads(pathlib.Path(path).read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{flag} must be a JSON array: {path} ({exc})") from exc
    if not isinstance(payload, list):
        raise ValueError(
            f"{flag} must be a JSON array, got {type(payload).__name__}: {path}"
        )
    return path


def _required_allocation(
    value: pathlib.Path | None, flag: str
) -> tuple[str, int]:
    path = _required_file(value, flag)
    try:
        payload = json.loads(pathlib.Path(path).read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{flag} must be a JSON object: {path} ({exc})") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{flag} must be a non-empty defect-to-count JSON object")
    invalid = {
        str(defect): count
        for defect, count in payload.items()
        if not isinstance(count, int) or isinstance(count, bool) or count < 0
    }
    if invalid:
        raise ValueError(f"{flag} contains invalid allocation counts: {invalid}")
    allocated = sum(payload.values())
    if allocated <= 0:
        raise ValueError(f"{flag} must allocate at least one sample")
    return path, allocated


def _required_checkpoint(value: pathlib.Path | None, flag: str) -> str:
    if value is None:
        raise ValueError(f"{flag} is required")
    path = value.expanduser()
    if not path.is_absolute() or not path.exists():
        raise ValueError(f"{flag} must be an existing absolute path: {value}")
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"{flag} must not be empty: {value}")
    if path.is_dir() and not any(path.iterdir()):
        raise ValueError(f"{flag} directory must not be empty: {value}")
    return str(path.resolve())


def _within(path: str, root: pathlib.Path, flag: str) -> str:
    resolved = pathlib.Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{flag} must be under {root}: {resolved}") from exc
    return str(resolved)


def _append_event(
    state: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    events = state.setdefault("events", [])
    if not isinstance(events, list):
        raise ValueError("state.events must be an array")
    sequence = max(
        (
            event.get("seq", 0)
            for event in events
            if isinstance(event, dict)
            and isinstance(event.get("seq"), int)
            and not isinstance(event.get("seq"), bool)
        ),
        default=0,
    ) + 1
    event = {
        "seq": sequence,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "iter": args.iter_label,
        "stage": args.stage,
        "status": "skipped" if args.skip else args.status,
        "summary": args.summary,
        "duration_sec": args.duration_sec,
        "context_tokens": 0,
    }
    if args.skip:
        event["skip_reason"] = args.summary.strip()
    events.append(event)
    return event


def _apply_success(
    phase: dict[str, Any],
    args: argparse.Namespace,
    results_dir: pathlib.Path,
    iterations: dict[str, Any],
) -> None:
    stage = args.stage
    phase_root = results_dir / args.iter_label
    if stage == "train":
        phase["best_ckpt_path"] = _within(
            _required_checkpoint(args.best_ckpt, "--best-ckpt"),
            phase_root / "train",
            "--best-ckpt",
        )
        phase["training_spec"] = _required_file(
            args.training_spec, "--training-spec"
        )
    elif stage == "evaluate_proxy":
        phase["proxy_results_json"] = _within(
            _required_file(args.proxy_results, "--proxy-results"),
            phase_root,
            "--proxy-results",
        )
    elif stage == "proxy_rcca":
        phase.update(_required_rcca_artifacts(args, phase_root))
    elif stage == "evaluate_benchmark":
        phase["benchmark_results_json"] = _within(
            _required_file(args.benchmark_results, "--benchmark-results"),
            phase_root,
            "--benchmark-results",
        )
    elif stage == "benchmark_metrics":
        phase["benchmark_metrics_summary"] = _within(
            _required_file(
                args.benchmark_metrics_summary, "--benchmark-metrics-summary"
            ),
            phase_root,
            "--benchmark-metrics-summary",
        )
    elif stage == "routing":
        phase["mining_targets_json"] = _within(
            _required_json_file(args.mining_targets, "--mining-targets"),
            phase_root,
            "--mining-targets",
        )
    elif stage == "anomalygen":
        if args.skip:
            # Documented branch skip: the driving Proxy RCCA found no false
            # accepts, so there is no under-detection gap for synthetic
            # defects to close.
            match = re.fullmatch(r"iter([1-9][0-9]*)", args.iter_label)
            driving_label = (
                "baseline"
                if match and int(match.group(1)) == 1
                else f"iter{int(match.group(1)) - 1}" if match else args.iter_label
            )
            driving_phase = iterations.get(driving_label, {})
            false_accepts = (
                driving_phase.get("false_accepts_json")
                if isinstance(driving_phase, dict)
                else None
            )
            if not isinstance(false_accepts, str):
                raise ValueError(
                    "anomalygen --skip requires proxy_rcca false_accepts_json evidence"
                )
            try:
                false_accept_rows = json.loads(pathlib.Path(false_accepts).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot validate false_accepts_json: {false_accepts} ({exc})"
                ) from exc
            if not isinstance(false_accept_rows, list) or false_accept_rows:
                raise ValueError(
                    "anomalygen --skip requires false_accepts_json to be an empty JSON array"
                )
            phase["anomalygen_skipped"] = True
            phase["anomalygen_skip_reason"] = args.summary.strip()
        else:
            phase["anomalygen_sdg_csv"] = _within(
                _required_file(args.anomalygen_sdg, "--anomalygen-sdg"),
                phase_root,
                "--anomalygen-sdg",
            )
            allocation, allocated = _required_allocation(
                args.anomalygen_allocation, "--anomalygen-allocation"
            )
            phase["anomalygen_allocation_json"] = _within(
                allocation,
                phase_root,
                "--anomalygen-allocation",
            )
            phase["anomalygen_amp_allocated"] = allocated
            phase["anomalygen_sharegpt_json"] = _within(
                _required_json_file(
                    args.anomalygen_sharegpt, "--anomalygen-sharegpt"
                ),
                phase_root,
                "--anomalygen-sharegpt",
            )
    elif stage == "data_mining":
        artifacts = {
            "mining_mined_parquet": (
                args.mining_parquet,
                "--mining-parquet",
            ),
            "mining_candidate_parquet": (
                args.mining_candidates,
                "--mining-candidates",
            ),
            "mining_summary": (args.mining_summary, "--mining-summary"),
            "mining_history_summary": (
                args.mining_history_summary,
                "--mining-history-summary",
            ),
            "mining_target_embeddings": (
                args.mining_target_embeddings,
                "--mining-target-embeddings",
            ),
            "mining_source_embeddings": (
                args.mining_source_embeddings,
                "--mining-source-embeddings",
            ),
        }
        for field, (value, flag) in artifacts.items():
            phase[field] = _within(
                _required_file(value, flag), phase_root, flag
            )
        phase["mining_history"] = _within(
            _required_file(args.mining_history, "--mining-history"),
            results_dir,
            "--mining-history",
        )
        if args.mining_count is None or args.mining_count <= 0:
            raise ValueError("--mining-count must be > 0")
        actual_count = _parquet_row_count(
            phase["mining_mined_parquet"], "--mining-parquet"
        )
        if args.mining_count != actual_count:
            raise ValueError(
                f"--mining-count={args.mining_count} does not match "
                f"mined parquet rows={actual_count}"
            )
        phase["mining_mined_count"] = args.mining_count
    elif stage == "assemble_data":
        phase["mined_sharegpt_json"] = _within(
            _required_json_file(args.mined_sharegpt, "--mined-sharegpt"),
            phase_root,
            "--mined-sharegpt",
        )
        phase["combined_training_json"] = _within(
            _required_json_file(args.combined_training, "--combined-training"),
            phase_root,
            "--combined-training",
        )
        phase["assemble_summary"] = _within(
            _required_file(args.assemble_summary, "--assemble-summary"),
            phase_root,
            "--assemble-summary",
        )
    elif stage == "validate_data":
        phase["validation_report"] = _within(
            _required_file(args.validation_report, "--validation-report"),
            phase_root,
            "--validation-report",
        )
    elif stage != "loop_stop":
        raise ValueError(f"unsupported stage: {stage}")

    if stage != "loop_stop":
        phase["stage_completed"] = stage
        # benchmark_metrics completes a stopping iteration (record_metric_result
        # sets status=complete). A continuing iteration becomes complete again
        # only after Proxy RCCA has seeded the next round.
        if stage == "proxy_rcca":
            phase["status"] = "complete"
        elif stage != "benchmark_metrics":
            phase["status"] = "in_progress"


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("--iter-label must be baseline or iterN")
    if not isinstance(args.duration_sec, int) or isinstance(args.duration_sec, bool):
        raise ValueError(
            "--duration-sec is required and must be a positive measured duration"
        )
    if args.skip and args.duration_sec < 0:
        raise ValueError("--duration-sec must be >= 0 for a skipped stage")
    if not args.skip and args.duration_sec <= 0:
        raise ValueError(
            "--duration-sec is required and must be a positive measured duration"
        )
    if getattr(args, "skip", False) and args.stage not in SKIPPABLE_STAGES:
        raise ValueError(
            f"--skip is valid only for: {', '.join(SKIPPABLE_STAGES)}"
        )
    if args.skip and args.status == "error":
        raise ValueError("--skip and --status error are mutually exclusive")
    if args.skip and not args.summary.strip():
        raise ValueError("--summary must contain the reason for a skipped stage")
    results_dir = args.results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    original_state = state_path.read_text()
    state = json.loads(original_state)
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")

    try:
        _migrate_execution_policy(state)
        state["version"] = 5
        iterations = state.get("iterations")
        if not isinstance(iterations, dict):
            raise ValueError("state.iterations must be an object")
        phase = iterations.setdefault(args.iter_label, {"status": "in_progress"})
        if not isinstance(phase, dict):
            raise ValueError(
                f"state.iterations.{args.iter_label} must be an object"
            )
        if args.status == "error":
            phase["status"] = "failed"
            state["status"] = "failed"
            state["completed_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")
        elif args.stage != "loop_stop":
            if args.stage == "benchmark_metrics":
                commit_metric_result(
                    argparse.Namespace(
                        state_path=state_path,
                        iter_label=args.iter_label,
                        result_json=args.metric_result,
                        best_ckpt=(
                            pathlib.Path(phase["best_ckpt_path"])
                            if phase.get("best_ckpt_path")
                            else None
                        ),
                        benchmark_results=pathlib.Path(
                            phase["benchmark_results_json"]
                        ),
                        training_spec=(
                            pathlib.Path(phase["training_spec"])
                            if phase.get("training_spec")
                            else None
                        ),
                    )
                )
                state = json.loads(state_path.read_text())
                iterations = state["iterations"]
                phase = iterations[args.iter_label]
                _apply_success(phase, args, results_dir, iterations)
            else:
                _apply_success(phase, args, results_dir, iterations)
            state["status"] = "in_progress"
            state.pop("completed_at", None)
        else:
            baseline = iterations.get("baseline")
            if not isinstance(baseline, dict) or baseline.get("status") != "complete":
                raise ValueError(
                    "loop_stop requires iterations.baseline.status=complete"
                )
            if phase.get("status") != "complete":
                raise ValueError(
                    f"loop_stop requires iterations.{args.iter_label}.status=complete"
                )
            result = phase.get("metric_result")
            passed = isinstance(result, dict) and result.get("passed") is True
            if not phase.get("benchmark_metrics_summary") or not isinstance(
                result, dict
            ):
                raise ValueError(
                    "loop_stop requires final benchmark_metrics evidence"
                )
            if args.stop_reason == "metric_met":
                if not passed:
                    raise ValueError(
                        "--stop-reason metric_met requires final metric_result.passed=true"
                    )
                args.summary = "Stopped because the final Benchmark metric contract was met."
            elif args.stop_reason == "max_iterations":
                match = re.fullmatch(r"iter([1-9][0-9]*)", args.iter_label)
                if not match or int(match.group(1)) < int(state["max_iterations"]):
                    raise ValueError(
                        "--stop-reason max_iterations requires iterN at or beyond max_iterations"
                    )
                if passed:
                    raise ValueError(
                        "--stop-reason max_iterations conflicts with metric_result.passed=true"
                    )
                args.summary = "Stopped because the configured iteration limit was reached."
            else:
                raise ValueError("loop_stop requires --stop-reason")
            final_report = _within(
                _required_file(args.final_report, "--final-report"),
                results_dir,
                "--final-report",
            )
            state["final_artifacts"] = {"report": final_report}
            state["status"] = "complete"
            state["completed_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")

        match = re.fullmatch(r"iter([1-9][0-9]*)", args.iter_label)
        if match:
            state["current_iteration"] = max(
                int(match.group(1)), int(state.get("current_iteration", 0))
            )
        event = _append_event(state, args)
        _atomic_json(state_path, state)
    except Exception:
        _atomic_text(state_path, original_state)
        raise
    report = {
        "status": str(state.get("status", "in_progress")).upper(),
        "terminal": state.get("status") in {"complete", "failed"},
        "last_committed": event,
    }
    # Keep reporting outside the state transaction: a presentation bug is
    # surfaced to the caller without invalidating an otherwise valid commit.
    try:
        output = render_html_report(results_dir)
        report["report_path"] = str(output)
    except Exception as exc:  # noqa: BLE001 - hook failures are non-transactional
        report["report_render_error"] = str(exc)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--status", choices=("ok", "error"), default="ok")
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--duration-sec",
        required=True,
        type=int,
        help="Measured wall-clock seconds; must be positive, or >= 0 with --skip",
    )
    parser.add_argument("--best-ckpt", type=pathlib.Path)
    parser.add_argument("--training-spec", type=pathlib.Path)
    parser.add_argument("--proxy-results", type=pathlib.Path)
    parser.add_argument("--proxy-gaps-summary", type=pathlib.Path)
    parser.add_argument("--false-accepts", type=pathlib.Path)
    parser.add_argument("--false-rejects", type=pathlib.Path)
    parser.add_argument(
        "--rcca-report",
        type=pathlib.Path,
        help="Required proxy_rcca/RCCA_Report.md path for a successful commit",
    )
    parser.add_argument("--benchmark-results", type=pathlib.Path)
    parser.add_argument("--benchmark-metrics-summary", type=pathlib.Path)
    parser.add_argument("--metric-result", type=pathlib.Path)
    parser.add_argument("--mining-targets", type=pathlib.Path)
    parser.add_argument("--anomalygen-sdg", type=pathlib.Path)
    parser.add_argument("--anomalygen-allocation", type=pathlib.Path)
    parser.add_argument("--anomalygen-sharegpt", type=pathlib.Path)
    parser.add_argument(
        "--skip",
        action="store_true",
        help=(
            "Record a documented branch skip with status=skipped and a skip "
            f"reason. Valid only for: {', '.join(SKIPPABLE_STAGES)}."
        ),
    )
    parser.add_argument("--mining-parquet", type=pathlib.Path)
    parser.add_argument("--mining-candidates", type=pathlib.Path)
    parser.add_argument("--mining-summary", type=pathlib.Path)
    parser.add_argument("--mining-history", type=pathlib.Path)
    parser.add_argument("--mining-history-summary", type=pathlib.Path)
    parser.add_argument("--mining-target-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-source-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-count", type=int)
    parser.add_argument("--mined-sharegpt", type=pathlib.Path)
    parser.add_argument("--combined-training", type=pathlib.Path)
    parser.add_argument("--assemble-summary", type=pathlib.Path)
    parser.add_argument("--validation-report", type=pathlib.Path)
    parser.add_argument(
        "--stop-reason", choices=("metric_met", "max_iterations")
    )
    parser.add_argument("--final-report", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = commit(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"commit_stage: {exc}", file=sys.stderr)
        return 2
    last = report["last_committed"]
    print(
        f"committed seq={last['seq']} {last['iter']}/{last['stage']} "
        f"status={last['status']} run={report['status']}"
    )
    if report.get("report_render_error"):
        print(
            f"commit_stage: report hook failed: {report['report_render_error']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
