# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomically commit one DEFT stage to ``deft_state.json``.

Use this instead of inline Python, jq, or hand-authored JSON.  The state file
contains both the resume snapshot and the ordered stage events, so a run has a
single durable source of truth.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

RCA_ARTIFACT_MANIFEST = (
    pathlib.Path(__file__).resolve().parents[1]
    / "references"
    / "rca-artifact-manifest.json"
)

from record_metric_result import commit as commit_metric_result
from render_report import render as render_html_report


class _CommitArgumentParser(argparse.ArgumentParser):
    """Add guidance for the repeatable single-value RCA label flag."""

    _argv: list[str]

    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        self._argv = list(sys.argv[1:] if args is None else args)
        return super().parse_args(self._argv, namespace)

    def error(self, message: str) -> None:
        if (
            message.startswith("unrecognized arguments:")
            and "--rca-target-defect" in getattr(self, "_argv", [])
        ):
            message += (
                "\n--rca-target-defect accepts exactly one label per occurrence; "
                "repeat the flag for each label"
            )
        super().error(message)


def _rca_target_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise argparse.ArgumentTypeError(
            "--rca-target-defect label must not be empty"
        )
    return label


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
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
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
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


def _required_file(path: pathlib.Path | None, name: str) -> str:
    if path is None:
        raise ValueError(f"{name} is required")
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{name} must be absolute: {path}")
    if not expanded.is_file() or expanded.stat().st_size == 0:
        raise ValueError(f"{name} must be an existing non-empty file: {path}")
    return str(expanded.resolve())


def _rca_manifest_entries(
    manifest: dict[str, Any], class_name: str
) -> list[dict[str, Any]]:
    classes = manifest.get("artifact_classes")
    entries = classes.get(class_name) if isinstance(classes, dict) else None
    if not isinstance(entries, list) or not entries or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(
            f"RCA artifact manifest {class_name} must be a non-empty array"
        )
    return entries


def _rca_manifest_path(output_dir: pathlib.Path, entry: dict[str, Any]) -> pathlib.Path:
    value = entry.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("RCA artifact manifest entry has no non-empty path")
    relative = pathlib.Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"RCA artifact manifest path must be relative: {value}")
    return output_dir / relative


def _validate_rca_report(
    report: pathlib.Path,
    entry: dict[str, Any],
    *,
    unreachable: bool,
) -> None:
    validation = entry.get("validation")
    headings = (
        validation.get("required_headings")
        if isinstance(validation, dict)
        else None
    )
    if not isinstance(headings, list) or not all(
        isinstance(heading, str) and heading for heading in headings
    ):
        raise ValueError("RCA report manifest must declare required_headings")
    try:
        text = report.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"--rca-report must be UTF-8 text: {report}") from exc
    actual = [
        re.sub(r"[^a-z0-9]+", " ", match.group(1).lower()).strip()
        for match in re.finditer(
            r"^##[ \t]+(?:[0-9]+\.[ \t]*)?(.+?)[ \t]*$",
            text,
            flags=re.MULTILINE,
        )
    ]
    missing = [
        heading
        for heading in headings
        if not any(
            re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip() in found
            for found in actual
        )
    ]
    if not missing:
        return
    abridged = validation.get("abridged_heading")
    normalized_abridged = (
        re.sub(r"[^a-z0-9]+", " ", abridged.lower()).strip()
        if isinstance(abridged, str)
        else ""
    )
    if unreachable and len(actual) == 1 and normalized_abridged in actual[0]:
        body = re.split(r"^##[^\n]*$", text, maxsplit=1, flags=re.MULTILINE)
        if len(body) == 2 and body[1].strip():
            return
        raise ValueError(
            "abridged --rca-report must explain the unreachable KPI and "
            "recommend retraining or relabeling"
        )
    suffix = (
        f"; with unreachable_kpi.txt, use one '## {abridged}' section"
        if unreachable and normalized_abridged
        else ""
    )
    raise ValueError(
        "--rca-report is missing required section heading(s): "
        f"{', '.join(missing)}{suffix}; follow references/output-template.md"
    )


def _required_rca_artifacts(
    gaps: pathlib.Path | None,
    report: pathlib.Path | None,
) -> dict[str, str]:
    report_path = pathlib.Path(_required_file(report, "--rca-report"))
    if report_path.name != "RCA_Report.md":
        raise ValueError(
            "--rca-report must name RCA_Report.md in the timestamped RCA output "
            f"directory: {report_path}"
        )
    try:
        manifest = json.loads(RCA_ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load RCA artifact manifest: {RCA_ARTIFACT_MANIFEST} ({exc})"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("RCA artifact manifest root must be an object")
    output_dir = report_path.parent
    report_entry = _rca_manifest_entries(
        manifest, "agent_produced_required"
    )[0]
    expected_report = _rca_manifest_path(output_dir, report_entry).resolve()
    if report_path != expected_report:
        raise ValueError(f"--rca-report must point to {expected_report}")

    failure = manifest.get("failure_artifact")
    if not isinstance(failure, dict):
        raise ValueError("RCA artifact manifest failure_artifact must be an object")
    unreachable_path = _rca_manifest_path(output_dir, failure)
    unreachable = unreachable_path.exists()
    state_artifacts: dict[str, str] = {}
    if unreachable:
        state_field = failure.get("state_field")
        if not isinstance(state_field, str) or not state_field:
            raise ValueError("RCA failure artifact must declare state_field")
        state_artifacts[state_field] = _required_file(
            unreachable_path, "RCA artifact unreachable_kpi.txt"
        )
    else:
        for entry in _rca_manifest_entries(
            manifest, "container_produced_required"
        ):
            artifact = pathlib.Path(
                _required_file(
                    _rca_manifest_path(output_dir, entry),
                    f"RCA artifact {entry.get('path')}",
                )
            )
            if entry.get("validation") == "float_text":
                try:
                    value = float(artifact.read_text(encoding="utf-8").strip())
                except (OSError, UnicodeDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"RCA artifact must contain one numeric value: {artifact}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"RCA artifact must contain one finite numeric value: {artifact}"
                    )
            state_field = entry.get("state_field")
            if not isinstance(state_field, str) or not state_field:
                raise ValueError(
                    f"RCA artifact {entry.get('path')} must declare state_field"
                )
            state_artifacts[state_field] = str(artifact)
        gaps_path = pathlib.Path(_required_file(gaps, "--rca-gaps"))
        declared_gaps = state_artifacts.get("rca_gaps_parquet")
        if gaps_path != pathlib.Path(str(declared_gaps)):
            raise ValueError(
                f"--rca-gaps must point to the manifest artifact: {declared_gaps}"
            )

    _validate_rca_report(report_path, report_entry, unreachable=unreachable)
    report_field = report_entry.get("state_field")
    if not isinstance(report_field, str) or not report_field:
        raise ValueError("RCA report artifact must declare state_field")
    state_artifacts[report_field] = str(report_path)

    if not unreachable:
        images_entry = _rca_manifest_entries(
            manifest, "agent_produced_conditionally_required"
        )[0]
        images_dir = _rca_manifest_path(output_dir, images_entry)
        if not images_dir.is_dir():
            raise ValueError(
                f"RCA artifact rca_images/ must be a directory: {images_dir}"
            )
        image_files = [
            path
            for path in images_dir.rglob("*")
            if path.is_file()
            and path.stat().st_size > 0
            and path.suffix.lower()
            in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        ]
        if not image_files:
            raise ValueError(
                "RCA artifact rca_images/ must contain at least one non-empty "
                f"image file: {images_dir}"
            )
        images_field = images_entry.get("state_field")
        if not isinstance(images_field, str) or not images_field:
            raise ValueError("RCA images artifact must declare state_field")
        state_artifacts[images_field] = str(images_dir.resolve())
    return state_artifacts


def _parquet_row_count(path: str, name: str) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            f"{name} validation requires pyarrow in the selected DEFT Python"
        ) from exc
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:  # noqa: BLE001 - normalize parquet parser failures
        raise ValueError(f"{name} must be a readable parquet file: {path} ({exc})") from exc


def _required_log(path: pathlib.Path | None, name: str) -> str:
    resolved = _required_file(path, name)
    try:
        text = pathlib.Path(resolved).read_text(errors="replace").strip().lower()
    except OSError as exc:
        raise ValueError(f"{name} cannot be read: {resolved} ({exc})") from exc
    placeholder_tokens = {"placeholder", "recorded", "todo", "n/a", "none", "ok"}
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if not normalized or normalized in placeholder_tokens:
        raise ValueError(f"{name} must contain real command output, not a placeholder")
    return resolved


def _required_allocation(
    path: pathlib.Path | None, name: str
) -> tuple[str, int]:
    resolved = _required_file(path, name)
    try:
        payload = json.loads(pathlib.Path(resolved).read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a JSON object: {resolved} ({exc})") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{name} must be a non-empty defect-to-count JSON object")
    invalid = {
        str(defect): count
        for defect, count in payload.items()
        if not isinstance(count, int) or isinstance(count, bool) or count < 0
    }
    if invalid:
        raise ValueError(f"{name} contains invalid allocation counts: {invalid}")
    allocated = sum(payload.values())
    if allocated <= 0:
        raise ValueError(f"{name} must allocate at least one sample")
    return resolved, allocated


def _require_within(path: str, root: pathlib.Path, name: str) -> str:
    resolved = pathlib.Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}: {resolved}") from exc
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
    stage: str,
    args: argparse.Namespace,
    results_dir: pathlib.Path,
    iter_label: str,
) -> None:
    if stage == "train":
        checkpoint = _required_file(args.best_ckpt, "--best-ckpt")
        expected_train_dir = results_dir / iter_label / "train"
        phase["best_ckpt_path"] = _require_within(
            checkpoint, expected_train_dir, "--best-ckpt"
        )
        phase["training_spec"] = _required_file(
            args.training_spec, "--training-spec"
        )
        if args.best_ckpt_kind is not None:
            phase["best_ckpt_kind"] = args.best_ckpt_kind
        if args.val_loss is not None:
            phase["val_loss"] = args.val_loss
    elif stage == "evaluate":
        required = ("best_ckpt_path", "inference_csv", "metric_result")
        missing = [field for field in required if not phase.get(field)]
        if missing or phase.get("status") != "complete":
            raise ValueError(
                "evaluate metric commit is incomplete; missing "
                f"{missing or ['status=complete']}"
            )
    elif stage == "rca":
        phase.update(_required_rca_artifacts(args.rca_gaps, args.rca_report))
        if args.rca_threshold is not None:
            phase["rca_threshold"] = args.rca_threshold
        if args.rca_target_defect:
            phase["rca_target_defects"] = _ordered_unique(args.rca_target_defect)
    elif stage == "routing":
        phase["routing_mining_parquet"] = _required_file(
            args.routing_mining, "--routing-mining"
        )
        phase["routing_anomalygen_parquet"] = _required_file(
            args.routing_anomalygen, "--routing-anomalygen"
        )
    elif stage == "anomalygen":
        if args.skip:
            routed = phase.get("routing_anomalygen_parquet")
            if not isinstance(routed, str) or _parquet_row_count(
                routed, "routing_anomalygen_parquet"
            ) != 0:
                raise ValueError(
                    "anomalygen --skip requires routing_anomalygen_parquet with zero rows"
                )
            phase["anomalygen_skipped"] = True
            phase["anomalygen_skip_reason"] = args.summary.strip()
        else:
            phase["anomalygen_sdg_csv"] = _required_file(
                args.anomalygen_sdg, "--anomalygen-sdg"
            )
            allocation, allocated = _required_allocation(
                args.anomalygen_allocation, "--anomalygen-allocation"
            )
            phase["anomalygen_allocation_json"] = allocation
            phase["anomalygen_amp_allocated"] = allocated
    elif stage == "data_mining":
        if args.skip:
            routed = phase.get("routing_mining_parquet")
            if not isinstance(routed, str) or _parquet_row_count(
                routed, "routing_mining_parquet"
            ) != 0:
                raise ValueError(
                    "data_mining --skip requires routing_mining_parquet with zero rows"
                )
            phase["data_mining_skipped"] = True
            phase["data_mining_skip_reason"] = args.summary.strip()
        else:
            phase_root = results_dir / iter_label
            mining_artifacts = {
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
            for field, (path, flag) in mining_artifacts.items():
                phase[field] = _require_within(
                    _required_file(path, flag), phase_root, flag
                )
            for field, path, flag in (
                ("mining_target_log", args.mining_target_log, "--mining-target-log"),
                ("mining_source_log", args.mining_source_log, "--mining-source-log"),
                ("mining_knn_log", args.mining_knn_log, "--mining-knn-log"),
            ):
                phase[field] = _require_within(
                    _required_log(path, flag), phase_root, flag
                )
            phase["mining_history"] = _require_within(
                _required_file(args.mining_history, "--mining-history"),
                results_dir,
                "--mining-history",
            )
            if args.mining_count is None or args.mining_count < 0:
                raise ValueError("--mining-count is required and must be >= 0")
            actual_count = _parquet_row_count(
                phase["mining_mined_parquet"], "--mining-parquet"
            )
            if args.mining_count != actual_count:
                raise ValueError(
                    f"--mining-count={args.mining_count} does not match "
                    f"mined parquet rows={actual_count}"
                )
            phase["mining_mined_count"] = args.mining_count
    elif stage == "data_merge":
        phase["combined_training_csv"] = _required_file(
            args.combined_csv, "--combined-csv"
        )
        phase["provenance_csv"] = _required_file(
            args.provenance_csv, "--provenance-csv"
        )
        phase["merge_validation_report"] = _required_file(
            args.merge_validation_report, "--merge-validation-report"
        )
    elif stage not in {"anomalygen_finetune", "loop_stop"}:
        raise ValueError(f"unsupported stage: {stage}")

    if stage != "loop_stop":
        phase["stage_completed"] = stage
    if stage != "evaluate" and phase.get("status") != "complete":
        phase["status"] = "in_progress"


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("--iter-label must be baseline or iterN (N >= 1)")
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
    if args.skip and args.stage not in {"anomalygen", "data_mining"}:
        raise ValueError("--skip is valid only for anomalygen or data_mining")
    if args.skip and args.status == "error":
        raise ValueError("--skip and --status error are mutually exclusive")
    if args.skip and not args.summary.strip():
        raise ValueError("--summary must contain the reason for a skipped stage")

    results_dir = args.results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    original_state_text = state_path.read_text()
    state = json.loads(original_state_text)
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")
    try:
        if args.stage == "evaluate" and args.status == "ok":
            commit_metric_result(
                argparse.Namespace(
                    state_path=state_path,
                    iter_label=args.iter_label,
                    result_json=args.metric_result,
                    best_ckpt=args.best_ckpt,
                    inference_csv=args.inference_csv,
                    training_spec=args.training_spec,
                    threshold=args.threshold,
                )
            )
            state = json.loads(state_path.read_text())
        _migrate_execution_policy(state)
        state["version"] = 4

        iterations = state.get("iterations")
        if not isinstance(iterations, dict):
            raise ValueError("state.iterations must be an object")
        existing = iterations.setdefault(args.iter_label, {"status": "in_progress"})
        if not isinstance(existing, dict):
            raise ValueError(
                f"state.iterations.{args.iter_label} must be an object"
            )
        if args.status == "error":
            existing["status"] = "failed"
            state["status"] = "failed"
            state["completed_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")
        else:
            if args.stage != "loop_stop":
                _apply_success(
                    existing, args.stage, args, results_dir, args.iter_label
                )
                state["status"] = "in_progress"
                state.pop("completed_at", None)
            else:
                baseline = iterations.get("baseline")
                if not isinstance(baseline, dict) or baseline.get("status") != "complete":
                    raise ValueError(
                        "loop_stop requires iterations.baseline.status=complete"
                    )
                if existing.get("status") != "complete":
                    raise ValueError(
                        f"loop_stop requires iterations.{args.iter_label}.status=complete"
                    )
                result = existing.get("metric_result")
                passed = isinstance(result, dict) and result.get("passed") is True
                if args.stop_reason == "metric_met":
                    if not passed:
                        raise ValueError(
                            "--stop-reason metric_met requires final metric_result.passed=true"
                        )
                    args.summary = "Stopped because the final metric contract was met."
                elif args.stop_reason == "max_iterations":
                    match = re.fullmatch(r"iter([1-9][0-9]*)", args.iter_label)
                    if not match or int(match.group(1)) < int(state["max_iterations"]):
                        raise ValueError(
                            "--stop-reason max_iterations requires iterN at or "
                            "beyond max_iterations"
                        )
                    if passed:
                        raise ValueError(
                            "--stop-reason max_iterations conflicts with metric_result.passed=true"
                        )
                    args.summary = "Stopped because the configured iteration limit was reached."
                else:
                    raise ValueError("loop_stop requires --stop-reason")
                best_model = _require_within(
                    _required_file(args.best_model, "--best-model"),
                    results_dir,
                    "--best-model",
                )
                inference_spec = _require_within(
                    _required_file(args.inference_spec, "--inference-spec"),
                    results_dir,
                    "--inference-spec",
                )
                state["final_artifacts"] = {
                    "best_model_json": best_model,
                    "inference_spec": inference_spec,
                }
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
        _atomic_text(state_path, original_state_text)
        raise
    report = {
        "status": str(state.get("status", "in_progress")).upper(),
        "terminal": state.get("status") in {"complete", "failed"},
        "last_committed": event,
    }
    # Report rendering is a deterministic post-commit hook.  A presentation
    # failure must be visible, but it must not roll back a valid GPU-stage
    # commit and leave callers unable to advance the state machine.
    try:
        output = render_html_report(results_dir)
        report["report_path"] = str(output)
    except Exception as exc:  # noqa: BLE001 - hook failures are non-transactional
        report["report_render_error"] = str(exc)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = _CommitArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Repeatable RCA syntax: [--rca-target-defect <label>]...\n"
            "Example: --rca-target-defect missing --rca-target-defect shift "
            "--rca-target-defect lifted_lead"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "train",
            "evaluate",
            "rca",
            "routing",
            "anomalygen_finetune",
            "anomalygen",
            "data_mining",
            "data_merge",
            "loop_stop",
        ),
    )
    parser.add_argument("--status", choices=("ok", "error"), default="ok")
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--duration-sec",
        required=True,
        type=int,
        help="Measured wall-clock seconds; must be positive, or >= 0 with --skip",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help=(
            "Record a documented skip with status=skipped and a skip reason. "
            "Valid only for anomalygen or data_mining."
        ),
    )
    parser.add_argument("--best-ckpt", type=pathlib.Path)
    parser.add_argument("--best-ckpt-kind", choices=("best_val", "latest"))
    parser.add_argument("--training-spec", type=pathlib.Path)
    parser.add_argument("--val-loss", type=float)
    parser.add_argument("--metric-result", type=pathlib.Path)
    parser.add_argument("--inference-csv", type=pathlib.Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--rca-gaps", type=pathlib.Path)
    parser.add_argument(
        "--rca-report",
        type=pathlib.Path,
        help="Required RCA_Report.md path for a successful RCA commit",
    )
    parser.add_argument("--rca-threshold", type=float)
    parser.add_argument(
        "--rca-target-defect",
        action="append",
        default=[],
        type=_rca_target_label,
        metavar="<label>",
        help="RCA target label; repeat this flag for each label",
    )
    parser.add_argument("--routing-mining", type=pathlib.Path)
    parser.add_argument("--routing-anomalygen", type=pathlib.Path)
    parser.add_argument("--anomalygen-sdg", type=pathlib.Path)
    parser.add_argument("--anomalygen-allocation", type=pathlib.Path)
    parser.add_argument("--mining-parquet", type=pathlib.Path)
    parser.add_argument("--mining-candidates", type=pathlib.Path)
    parser.add_argument("--mining-summary", type=pathlib.Path)
    parser.add_argument("--mining-history", type=pathlib.Path)
    parser.add_argument("--mining-history-summary", type=pathlib.Path)
    parser.add_argument("--mining-target-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-source-embeddings", type=pathlib.Path)
    parser.add_argument("--mining-target-log", type=pathlib.Path)
    parser.add_argument("--mining-source-log", type=pathlib.Path)
    parser.add_argument("--mining-knn-log", type=pathlib.Path)
    parser.add_argument("--mining-count", type=int)
    parser.add_argument("--combined-csv", type=pathlib.Path)
    parser.add_argument("--provenance-csv", type=pathlib.Path)
    parser.add_argument("--merge-validation-report", type=pathlib.Path)
    parser.add_argument(
        "--stop-reason", choices=("metric_met", "max_iterations")
    )
    parser.add_argument("--best-model", type=pathlib.Path)
    parser.add_argument("--inference-spec", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = commit(args)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
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
