#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute binary collision-question metrics from Cosmos Reason results.json."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from workflow_common import absolute_path, atomic_write_json, load_json_array


BINARY_TOKEN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
LEADING_BINARY_TOKEN = re.compile(r"^\s*[\W_]*(yes|no)\b", re.IGNORECASE)
ANSWER_BINARY_TOKEN = re.compile(
    r"\b(?:final\s+)?answer\s*(?:is|:)?\s*[\W_]*(yes|no)\b",
    re.IGNORECASE,
)
PREDICTION_FIELDS = ("response", "answer", "prediction")
GROUND_TRUTH_FIELDS = ("gt", "ground_truth")


def extract_yes_no(value: str) -> str | None:
    """Extract an unambiguous binary label from a short or free-form answer."""
    if re.fullmatch(r"\s*[\W_]*(?:yes\s+or\s+no|no\s+or\s+yes)[\W_]*", value, re.IGNORECASE):
        return None
    tokens = [match.group(1).lower() for match in BINARY_TOKEN.finditer(value)]
    if not tokens:
        return None
    if len(set(tokens)) == 1:
        return tokens[0]

    leading = LEADING_BINARY_TOKEN.search(value)
    if leading:
        return leading.group(1).lower()

    answer_matches = list(ANSWER_BINARY_TOKEN.finditer(value))
    if answer_matches:
        answer_labels = {match.group(1).lower() for match in answer_matches}
        if len(answer_labels) == 1:
            return answer_matches[-1].group(1).lower()
    return None


def required_answer(
    row: dict[str, Any],
    fields: tuple[str, ...],
    *,
    source: Path,
    row_number: int,
    label: str,
) -> str:
    """Return the first supported non-empty string field for one answer."""
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    joined = ", ".join(fields)
    raise ValueError(
        f"{source}: item {row_number} is missing a non-empty {label} "
        f"(supported fields: {joined})"
    )


def compute_metrics(records: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    """Compute confusion counts, accuracy, and balanced accuracy."""
    if not records:
        raise ValueError(f"{source}: results array is empty")

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    positive_samples = 0
    negative_samples = 0
    unparseable_predictions = 0
    unparseable_positive_predictions = 0
    unparseable_negative_predictions = 0

    for row_number, row in enumerate(records, start=1):
        prediction_raw = required_answer(
            row,
            PREDICTION_FIELDS,
            source=source,
            row_number=row_number,
            label="prediction",
        )
        ground_truth_raw = required_answer(
            row,
            GROUND_TRUTH_FIELDS,
            source=source,
            row_number=row_number,
            label="ground truth",
        )
        ground_truth = extract_yes_no(ground_truth_raw)
        if ground_truth is None:
            raise ValueError(
                f"{source}: item {row_number} ground truth does not contain one clear "
                f"yes/no label: {ground_truth_raw!r}"
            )

        prediction = extract_yes_no(prediction_raw)
        if ground_truth == "yes":
            positive_samples += 1
        else:
            negative_samples += 1

        if prediction is None:
            unparseable_predictions += 1
            if ground_truth == "yes":
                unparseable_positive_predictions += 1
            else:
                unparseable_negative_predictions += 1
            continue

        if prediction == "yes" and ground_truth == "yes":
            true_positives += 1
        elif prediction == "no" and ground_truth == "no":
            true_negatives += 1
        elif prediction == "yes" and ground_truth == "no":
            false_positives += 1
        else:
            false_negatives += 1

    if positive_samples == 0 or negative_samples == 0:
        raise ValueError(
            f"{source}: balanced accuracy requires both ground-truth classes; "
            f"found yes={positive_samples}, no={negative_samples}"
        )

    total_samples = len(records)
    accuracy = (true_positives + true_negatives) / total_samples
    true_positive_rate = true_positives / positive_samples
    true_negative_rate = true_negatives / negative_samples
    return {
        "schema_version": 1,
        "results_json": str(absolute_path(source)),
        "total_samples": total_samples,
        "parsed_predictions": total_samples - unparseable_predictions,
        "unparseable_predictions": unparseable_predictions,
        "unparseable_positive_predictions": unparseable_positive_predictions,
        "unparseable_negative_predictions": unparseable_negative_predictions,
        "positive_ground_truth": positive_samples,
        "negative_ground_truth": negative_samples,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "accuracy": accuracy,
        "balanced_accuracy": (true_positive_rate + true_negative_rate) / 2,
    }


def compute_metrics_file(results_json: Path, output_json: Path) -> dict[str, Any]:
    """Compute metrics from a results file and persist the output."""
    results_json = absolute_path(results_json)
    output_json = absolute_path(output_json)
    if not results_json.is_file():
        raise FileNotFoundError(f"results JSON does not exist: {results_json}")
    metrics = compute_metrics(load_json_array(results_json), results_json)
    atomic_write_json(output_json, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    metrics = compute_metrics_file(args.results_json, args.output_json)
    print(f"metrics_json: {absolute_path(args.output_json)}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"false_positives: {metrics['false_positives']}")
    print(f"false_negatives: {metrics['false_negatives']}")
    print(f"unparseable_predictions: {metrics['unparseable_predictions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
