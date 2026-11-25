#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze Cosmos evaluator OK/NG results for DEFT AOI.

NG is the positive class:
  TP = NG -> NG
  FN = NG -> OK  (false accept: a defective component passed)
  FP = OK -> NG  (false reject: a good component was scrapped)
  TN = OK -> OK
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from typing import Any


LABEL_RE = re.compile(r"\b(OK|NG)\b", re.IGNORECASE)
VALID = {"OK", "NG"}


def normalize_label(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    upper = text.upper()
    if upper in VALID:
        return upper
    matches = LABEL_RE.findall(text)
    if matches:
        return matches[-1].upper()
    return "UNKNOWN"


def _load_samples(path: pathlib.Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict):
        for key in ("results", "samples", "predictions", "individual_results"):
            value = payload.get(key)
            if isinstance(value, list):
                samples = value
                break
        else:
            raise ValueError(
                f"{path} does not contain a sample list under results/samples/predictions/individual_results"
            )
    else:
        raise ValueError(f"{path} must contain a JSON object or list")
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError(f"{path}: every sample must be a JSON object")
    return samples


def _field(sample: dict, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in sample:
            return sample[name]
    return None


def analyze(
    samples: list[dict],
    *,
    kpi_metric: str = "recall_ng",
    kpi_threshold: float = 1.0,
    evaluation_role: str = "proxy",
    report_only: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    if evaluation_role not in {"proxy", "benchmark"}:
        raise ValueError(
            f"evaluation_role must be proxy or benchmark, got {evaluation_role!r}"
        )
    aggregate_only = report_only or evaluation_role == "benchmark"
    gate_eligible = evaluation_role == "benchmark" and not report_only
    rows: list[dict] = []
    false_accepts: list[dict] = []
    false_rejects: list[dict] = []
    unknowns: list[dict] = []
    tp = fp = tn = fn = 0

    for idx, sample in enumerate(samples):
        gt_raw = _field(sample, ("gt", "ground_truth", "answer", "label", "target"))
        pred_raw = _field(sample, ("response", "prediction", "pred", "model_response", "output"))
        gt = normalize_label(gt_raw)
        pred = normalize_label(pred_raw)
        row = {
            "index": idx,
            "id": _field(sample, ("video_id", "sample_id", "id", "image_id")) or idx,
            "gt": gt,
            "pred": pred,
            "gt_raw": gt_raw,
            "response": pred_raw,
            "question": _field(sample, ("question", "prompt")),
            "images": _field(sample, ("images", "video", "media")),
        }

        if gt not in VALID or pred not in VALID:
            unknowns.append(row)
        elif gt == "NG" and pred == "NG":
            tp += 1
        elif gt == "NG" and pred == "OK":
            fn += 1
            false_accepts.append(row)
        elif gt == "OK" and pred == "NG":
            fp += 1
            false_rejects.append(row)
        elif gt == "OK" and pred == "OK":
            tn += 1
        rows.append(row)

    actual_ng = tp + fn
    actual_ok = tn + fp
    predicted_ng = tp + fp
    recall_ng = tp / actual_ng if actual_ng else 0.0
    precision_ng = tp / predicted_ng if predicted_ng else 0.0
    false_accept_rate = fn / actual_ng if actual_ng else 0.0
    false_reject_rate = fp / actual_ok if actual_ok else 0.0
    # Unknown predictions are incorrect predictions, not samples to omit from
    # the reported accuracy. Keep the class-specific metrics based on their
    # parseable confusion-matrix populations, but use the full evaluation set
    # for accuracy so unparseable responses cannot inflate it.
    accuracy = (tp + tn) / len(samples) if samples else 0.0
    f1_ng = (
        2 * precision_ng * recall_ng / (precision_ng + recall_ng)
        if precision_ng + recall_ng
        else 0.0
    )

    metrics = {
        "recall_ng": recall_ng,
        "precision_ng": precision_ng,
        "f1_ng": f1_ng,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "accuracy": accuracy,
    }
    if kpi_metric not in metrics:
        raise ValueError(
            f"kpi_metric must be one of {sorted(metrics)}, got {kpi_metric!r}"
        )

    summary = {
        "evaluation_role": evaluation_role,
        "report_only": report_only,
        "aggregate_only": aggregate_only,
        "samples": len(samples),
        "parseable_samples": tp + fp + tn + fn,
        "unknown_samples": len(unknowns),
        "confusion": {
            "tp_ng_to_ng": tp,
            "fn_ng_to_ok_false_accept": fn,
            "fp_ok_to_ng_false_reject": fp,
            "tn_ok_to_ok": tn,
        },
        "metrics": metrics,
        "kpi": {
            "metric": kpi_metric,
            "threshold": kpi_threshold if gate_eligible else None,
            "value": metrics[kpi_metric],
            "met": (
                metrics[kpi_metric] >= kpi_threshold and len(unknowns) == 0
                if gate_eligible
                else None
            ),
            "primary": kpi_metric,
            "tie_breaker": "precision_ng" if gate_eligible else None,
            "gate_eligible": gate_eligible,
            "unknown_predictions_block_gate": gate_eligible,
        },
    }
    return summary, false_accepts, false_rejects, unknowns


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    columns = ["index", "id", "gt", "pred", "gt_raw", "response", "question", "images"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            if isinstance(serial.get("images"), (list, dict)):
                serial["images"] = json.dumps(serial["images"], sort_keys=True)
            writer.writerow({col: serial.get(col) for col in columns})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--kpi-metric",
        default="recall_ng",
        choices=(
            "recall_ng",
            "precision_ng",
            "f1_ng",
            "false_accept_rate",
            "false_reject_rate",
            "accuracy",
        ),
        help=(
            "Metric used for the configured run gate. Default recall_ng. "
            "Use accuracy when the user requests an accuracy KPI."
        ),
    )
    parser.add_argument(
        "--kpi-threshold",
        default=1.0,
        type=float,
        help="KPI threshold compared as metric >= threshold. Default 1.0.",
    )
    parser.add_argument(
        "--evaluation-role",
        choices=("proxy", "benchmark"),
        default="proxy",
        help="Proxy emits RCCA artifacts; Benchmark emits aggregate metrics and owns the stop gate.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Disable the stop gate and emit aggregate metrics only. Benchmark is aggregate-only even without this flag.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        samples = _load_samples(args.results_json)
        summary, false_accepts, false_rejects, unknowns = analyze(
            samples,
            kpi_metric=args.kpi_metric,
            kpi_threshold=args.kpi_threshold,
            evaluation_role=args.evaluation_role,
            report_only=args.report_only,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if summary["aggregate_only"]:
            _write_json(args.output_dir / "metrics_summary.json", summary)
            metric_result = {
                "name": args.kpi_metric,
                "value": summary["metrics"][args.kpi_metric],
                "unit": "",
                "samples": summary["samples"],
                "parseable_samples": summary["parseable_samples"],
                "unknown_samples": summary["unknown_samples"],
                "constraints": {
                    "unknown_predictions": summary["unknown_samples"],
                },
                "metrics": summary["metrics"],
                "confusion": summary["confusion"],
            }
            _write_json(args.output_dir / "metric_result.json", metric_result)
        else:
            _write_json(args.output_dir / "gaps_summary.json", summary)
            _write_json(args.output_dir / "false_accepts.json", false_accepts)
            _write_json(args.output_dir / "false_rejects.json", false_rejects)
            _write_json(args.output_dir / "unknown_predictions.json", unknowns)
            _write_csv(args.output_dir / "false_accepts.csv", false_accepts)
            _write_csv(args.output_dir / "false_rejects.csv", false_rejects)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"analyze_gaps: {exc}", file=sys.stderr)
        return 2

    metrics = summary["metrics"]
    confusion = summary["confusion"]
    kpi_text = summary["kpi"]["metric"]
    if summary["kpi"]["threshold"] is not None:
        kpi_text += f">={summary['kpi']['threshold']:.6f}"
    print(
        "analyze_gaps: "
        f"role={summary['evaluation_role']} "
        f"report_only={summary['report_only']} "
        f"aggregate_only={summary['aggregate_only']} "
        f"recall_ng={metrics['recall_ng']:.6f} "
        f"precision_ng={metrics['precision_ng']:.6f} "
        f"false_accepts={confusion['fn_ng_to_ok_false_accept']} "
        f"false_rejects={confusion['fp_ok_to_ng_false_reject']} "
        f"samples={summary['samples']} "
        f"parseable_samples={summary['parseable_samples']} "
        f"unknown_samples={summary['unknown_samples']} "
        f"kpi={kpi_text} "
        f"kpi_met={summary['kpi']['met']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
