# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the AOI card pack's similarity, history, and training-row evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data/tao-mine-aoi-images/scripts"))
from filter_mined_history import select_novel_samples  # noqa: E402


def prepare(results_dir: Path, workspace: Path, iteration: str) -> dict:
    results_dir, workspace = results_dir.resolve(), workspace.resolve()
    match = re.fullmatch(r"iter([1-9][0-9]*)", iteration)
    if match is None:
        raise ValueError("iteration must be iterN, N >= 1")
    state = json.loads((results_dir / "deft_state.json").read_text())
    config = state["config"]["mining_filter"]
    if config["metric"] != "cosine" or config["history_aware"]["enabled"] is not True:
        raise ValueError("this card requires cosine mining with history enabled")
    history = Path(config["history_aware"]["history_file"]).resolve()
    history.relative_to(results_dir)
    threshold = float(config["min_similarity"])
    if not 0 <= threshold <= 1:
        raise ValueError("min_similarity must be between 0 and 1")
    root = results_dir / iteration / "mining_filter"
    mined = pd.read_parquet(root / "mined.parquet")
    if "filepath" not in mined:
        raise ValueError("mined.parquet must contain filepath")
    scores = [col for col in mined if "similarity" in col.lower()]
    if len(scores) != 1:
        raise ValueError("expected one cosine similarity column; do not infer similarity from distance")
    candidates = mined[mined[scores[0]] >= threshold].drop_duplicates("filepath").reset_index(drop=True)
    candidate_count = len(mined.drop_duplicates("filepath"))
    # Validate before appending the immutable history ledger. Never silently
    # drop missing images: output parquet, ledger, and CSV counts must agree.
    names = set()
    for value in candidates["filepath"]:
        source = Path(str(value))
        if not source.is_file():
            raise ValueError(f"mined image is missing: {source}")
        if not source.name.endswith("_SolderLight.jpg"):
            raise ValueError(f"expected NV_PCB_Siamese *_SolderLight.jpg: {source}")
        if source.name in names:
            raise ValueError(f"mined image basename collision: {source.name}")
        names.add(source.name)
    candidate_path = root / "mining_candidates.parquet"
    if candidate_path.exists():
        if not pd.read_parquet(candidate_path).equals(candidates):
            raise ValueError("candidate evidence changed; refusing to overwrite a resumed selection")
    else:
        candidates.to_parquet(candidate_path, index=False)
    # kept_count is PRE-history, as required by audit_deft_run.py.
    pd.DataFrame([{
        "candidate_count": candidate_count,
        "kept_count": len(candidates),
        "rejected_count": candidate_count - len(candidates),
        "similarity_threshold": threshold,
    }]).to_csv(root / "knn_summary.csv", index=False)
    summary = select_novel_samples(
        candidate_parquet=candidate_path,
        output_parquet=root / "mined_filtered.parquet",
        history_file=history,
        summary_file=root / "mining_history_summary.json",
        iteration=int(match[1]),
        topn=int(config["top_k_per_target"]),
        resume=True,
    )
    selected = pd.read_parquet(root / "mined_filtered.parquet")
    image_root = results_dir / iteration / "dataset/images"
    inputs, golden = image_root / "mined_input", image_root / "mined_golden"
    input_relative = inputs.relative_to(workspace).as_posix()
    golden_relative = golden.relative_to(workspace).as_posix()
    inputs.mkdir(parents=True, exist_ok=True)
    golden.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in selected["filepath"]:
        source = Path(str(value))
        shutil.copy2(source, inputs / source.name)
        shutil.copy2(source, golden / source.name)
        rows.append({
            "input_path": input_relative,
            "golden_path": golden_relative,
            "label": "PASS",
            "object_name": source.name.removesuffix("_SolderLight.jpg"),
        })
    # Retain headers even when history removes every candidate.
    pd.DataFrame(rows, columns=["input_path", "golden_path", "label", "object_name"]).to_csv(
        root / "mining_pool.csv", index=False
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--iter-label", required=True)
    args = parser.parse_args()
    try:
        summary = prepare(args.results_dir, args.workspace, args.iter_label)
    except (OSError, ValueError, KeyError) as exc:
        print(f"prepare_card_mining: {exc}", file=sys.stderr)
        return 2
    print(f"selected={summary['selected_count']} already_mined={summary['already_mined_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
