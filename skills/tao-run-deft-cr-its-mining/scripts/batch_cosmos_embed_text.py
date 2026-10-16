#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare bounded Cosmos Embed text configs, run the native CLI, and merge outputs."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
from typing import Any

import numpy as np

from native_process import live_output, run_child
from validate_cosmos_embed_output import (
    ACCEPTED_EXIT_CODES,
    check_completion,
    file_sha256,
    spec_context,
    validate_completion,
    validate_outputs,
)
from workflow_common import absolute_path, atomic_write_json, load_yaml, write_yaml


class Cancelled(Exception):
    """A signal to the wrapper is cancellation, not native teardown success."""


class Cancellation:
    def __init__(self):
        self.signal = None

    def request(self, signum, _frame):
        # Do not raise inside Popen: the child may exist before its handle returns.
        if self.signal is None:
            self.signal = signum

    def check(self):
        if self.signal is not None:
            raise Cancelled()


@contextmanager
def cancellation_signals():
    cancellation = Cancellation()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, cancellation.request)
        yield cancellation
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)




def text_spec(spec_path: Path) -> dict[str, Any]:
    """Require a text query spec with a selected checkpoint."""
    if spec_context(spec_path)["mode"] != "text":
        raise ValueError("Batching requires inference.mode=text")
    spec = load_yaml(spec_path)
    checkpoint = spec["inference"].get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("Batching requires a selected inference.checkpoint")
    return spec


def batch_spec(spec: dict[str, Any], directory: Path, start: int, count: int) -> dict[str, Any]:
    """Change only the query slice and batch-local writable output paths."""
    partial = copy.deepcopy(spec)
    partial["results_dir"] = str(directory / "results")
    partial["inference"]["save_dataset_pkl"] = str(directory / "results" / "embeddings.pkl")
    partial["inference"]["query"]["input_texts"] = (
        spec["inference"]["query"]["input_texts"][start:start + count]
    )
    return partial


def prepare(spec_path: Path, chunk_size: int) -> Path:
    """Write a fresh batch plan without changing the full inference spec."""
    spec_path = absolute_path(spec_path)
    spec = text_spec(spec_path)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    queries = spec["inference"]["query"]["input_texts"]
    stage = Path(spec["results_dir"])
    if (stage / "inference").exists():
        raise FileExistsError("Full-spec outputs already exist; use a fresh results_dir")
    (stage / "batches").mkdir(parents=True, exist_ok=False)
    entries = []
    for number, start in enumerate(range(0, len(queries), chunk_size), 1):
        directory = stage / "batches" / f"batch_{number:03}"
        count = min(chunk_size, len(queries) - start)
        path = directory / "spec.yaml"
        write_yaml(path, batch_spec(spec, directory, start, count))
        entries.append({
            "number": number, "start": start, "count": count,
            "spec": str(path), "sha256": file_sha256(path),
        })
    manifest = stage / "batch-plan.json"
    atomic_write_json(manifest, {
        "full_spec": str(spec_path), "full_spec_sha256": file_sha256(spec_path),
        "chunk_size": chunk_size, "total_queries": len(queries), "batches": entries,
    })
    return manifest


def load_plan(spec_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the full spec and every ordered query slice before execution."""
    spec = text_spec(spec_path)
    stage = Path(spec["results_dir"])
    plan = json.loads((stage / "batch-plan.json").read_text(encoding="utf-8"))
    if plan["full_spec"] != str(spec_path) or plan["full_spec_sha256"] != file_sha256(spec_path):
        raise ValueError("Full spec changed since batch preparation")
    size = plan["chunk_size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("Invalid batch-plan chunk_size")
    total = len(spec["inference"]["query"]["input_texts"])
    entries = plan["batches"]
    if plan["total_queries"] != total or len(entries) != (total + size - 1) // size:
        raise ValueError("Batch plan does not cover the full query list")
    for number, (start, entry) in enumerate(zip(range(0, total, size), entries, strict=True), 1):
        directory = stage / "batches" / f"batch_{number:03}"
        path = directory / "spec.yaml"
        count = min(size, total - start)
        if entry != {"number": number, "start": start, "count": count,
                     "spec": str(path), "sha256": file_sha256(path)}:
            raise ValueError("Batch plan or spec changed since preparation")
        if load_yaml(path) != batch_spec(spec, directory, start, count):
            raise ValueError("Batch spec differs from its full-spec query slice")
    return spec, entries


def merge_outputs(spec_path: Path, entries: list[dict[str, Any]]) -> Path:
    """Merge validated native outputs while preserving every ordered occurrence."""
    spec = text_spec(spec_path)
    queries = spec["inference"]["query"]["input_texts"]
    matrices, records, outcomes = [], [], []
    for entry in entries:
        checked = check_completion(Path(entry["spec"]))
        metadata = json.loads(Path(checked["metadata_path"]).read_text(encoding="utf-8"))
        matrix = np.load(checked["npy_path"], allow_pickle=False)
        if metadata.get("checkpoint") != spec["inference"]["checkpoint"]:
            raise ValueError("Batch checkpoint mismatch")
        ordered = sorted(metadata["results"], key=lambda row: row["npy_row"])
        if [row["text"] for row in ordered] != queries[entry["start"]:entry["start"] + entry["count"]]:
            raise ValueError("Batch text order mismatch")
        offset = len(records)
        records.extend({**row, "npy_row": offset + row["npy_row"]} for row in ordered)
        matrices.append(matrix)
        outcomes.append({**entry, "exit_code": checked["exit_code"], "status": checked["status"]})
    matrix = np.concatenate(matrices, axis=0)
    if len(records) != len(queries) or matrix.shape[0] != len(queries) or not np.isfinite(matrix).all():
        raise ValueError("Aggregate coverage/shape/finite validation failed")
    output = Path(spec["results_dir"]) / "inference"
    output.mkdir(exist_ok=False)
    np.save(output / "text_embeddings.npy", matrix)
    atomic_write_json(output / "text_embeddings.json", {
        "mode": "text", "checkpoint": spec["inference"]["checkpoint"],
        "num_items": len(records), "num_embedded": len(matrix), "embedding_dim": matrix.shape[1],
        "npy_file": "text_embeddings.npy", "results": records,
    })
    atomic_write_json(output / "batch-provenance.json", {
        "full_query_count": len(queries), "batches": outcomes,
        "aggregate_validation": validate_outputs(spec_path),
    })
    return validate_completion(spec_path, 0)


def run_plan(spec_path, stage, entries, cancellation):
    outcomes = []
    for entry in entries:
        cancellation.check()
        path = Path(entry["spec"])
        print(f"BATCH_START number={entry['number']} count={entry['count']}", flush=True)
        # The native CLI may signal its process group during torchrun teardown.
        with (path.parent / "container-child.log").open("xb") as log:
            child = run_child(
                ["cosmos-embed1", "inference", "-e", str(path)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                cancellation=cancellation,
            )
        (path.parent / "child-exit-code.txt").write_text(str(child.returncode) + "\n", encoding="utf-8")
        cancellation.check()
        if child.returncode not in ACCEPTED_EXIT_CODES:
            return child.returncode if 0 < child.returncode < 256 else 1
        completion = validate_completion(path, child.returncode)
        outcomes.append({**entry, "exit_code": child.returncode, "validation": str(completion)})
        atomic_write_json(stage / "batch-progress.json", outcomes)
        print(f"BATCH_VALIDATED number={entry['number']} exit={child.returncode}", flush=True)
    cancellation.check()
    merge_outputs(spec_path, entries)
    cancellation.check()
    return 0


def run(spec_path: Path) -> int:
    """Run inside the approved model container; stop at the first failed child."""
    spec_path = absolute_path(spec_path)
    spec, entries = load_plan(spec_path)
    stage = Path(spec["results_dir"])
    if (stage / "inference").exists() or (stage / "batch-cancellation.json").exists() or any(
        (Path(entry["spec"]).parent / "container-child.log").exists() for entry in entries
    ):
        raise FileExistsError("Batch execution already attempted; inspect evidence before a new plan")
    with live_output(), cancellation_signals() as cancellation:
        try:
            return run_plan(spec_path, stage, entries, cancellation)
        except Cancelled:
            atomic_write_json(stage / "batch-cancellation.json", {
                "status": "canceled", "signal": cancellation.signal,
                "full_spec_sha256": file_sha256(spec_path),
            })
            (stage / "inference" / "completion_validation.json").unlink(missing_ok=True)
            print(f"BATCH_CANCELED signal={cancellation.signal}", flush=True)
            return 128 + cancellation.signal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--inference-spec", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, help="Required for prepare; choose for the selected GPU/model")
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.chunk_size is None:
            parser.error("prepare requires --chunk-size")
        print(prepare(args.inference_spec, args.chunk_size))
        return 0
    if args.chunk_size is not None:
        parser.error("run uses the prepared plan; do not override --chunk-size")
    return run(args.inference_spec)


if __name__ == "__main__":
    raise SystemExit(main())
