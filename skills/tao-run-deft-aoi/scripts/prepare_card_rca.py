# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the AOI card pack's RCA report and spot-check images from gap-analysis output.

Every number comes from the container outputs and the committed inference CSV.
The pack's executor models are text-only, so spot-check verdicts are recorded
as pending operator review rather than guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

SPOT_CHECK_PER_SIDE = 5
TOP_K_ROWS_PER_LABEL = 10
PENDING = "pending operator review (text-only executor)"


def _is_pass(labels: pd.Series) -> pd.Series:
    return labels.astype(str).str.upper() == "PASS"


def _signed_weakness(frame: pd.DataFrame, threshold: float) -> pd.Series:
    score = frame["siamese_score"].astype(float)
    return (score - threshold).where(_is_pass(frame["label"]), threshold - score)


def _table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _sample_key(frame: pd.DataFrame) -> list[str]:
    keys = [column for column in ("input_path", "object_name") if column in frame]
    return keys or ["filepath"]


def _resolve_image(value: str, workspace: Path) -> Path:
    path = Path(value)
    for candidate in (path, workspace / path, workspace / "kpi/images" / path):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise ValueError(f"spot-check image is missing: {value}")


def _abridged(out: Path, unreachable: Path) -> dict:
    text = unreachable.read_text(errors="replace").strip()
    (out / "RCA_Report.md").write_text(
        "# VCN Gap Analysis Report\n\n"
        "## KPI Unreachable\n"
        f"The gap analysis container wrote `unreachable_kpi.txt`:\n\n```\n{text}\n```\n\n"
        "The model cannot meet the KPI at any threshold. Recommendation: retrain or relabel "
        "before running another DEFT iteration.\n"
    )
    return {"unreachable": True, "report": str(out / "RCA_Report.md"), "target_defects": []}


def prepare(results_dir: Path, workspace: Path, iteration: str, out: Path,
            min_recall: float = 1.0) -> dict:
    results_dir, workspace, out = results_dir.resolve(), workspace.resolve(), out.resolve()
    out.relative_to(results_dir / iteration / "rca_results")
    unreachable = out / "unreachable_kpi.txt"
    if unreachable.is_file() and unreachable.stat().st_size > 0:
        return _abridged(out, unreachable)

    state = json.loads((results_dir / "deft_state.json").read_text())
    inference_csv = Path(state["iterations"][iteration]["inference_csv"])
    threshold = float((out / "threshold.txt").read_text().strip())
    breakdown = (out / "weak_samples_breakdown.txt").read_text(errors="replace").strip()
    gaps = pd.read_parquet(out / "kpi_gaps.parquet")
    for column in ("filepath", "label", "siamese_score", "weakness"):
        if column not in gaps:
            raise ValueError(f"kpi_gaps.parquet must contain {column}")
    inference = pd.read_csv(inference_csv)
    for column in ("label", "siamese_score"):
        if column not in inference:
            raise ValueError(f"inference CSV must contain {column}: {inference_csv}")

    inference["weakness"] = _signed_weakness(inference, threshold)
    inference["misclassified"] = inference["weakness"] > 0
    positive = ~_is_pass(inference["label"])
    predicted = inference["siamese_score"].astype(float) >= threshold
    tp, fn = int((positive & predicted).sum()), int((positive & ~predicted).sum())
    fp, tn = int((~positive & predicted).sum()), int((~positive & ~predicted).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_label = inference.groupby("label").agg(
        total=("weakness", "size"), mean=("weakness", "mean"), median=("weakness", "median"),
        max=("weakness", "max"), misclassified=("misclassified", "sum"),
    ).sort_values("misclassified", ascending=False)
    target_defects = [
        str(label) for label, row in per_label.iterrows()
        if row["misclassified"] > 0 and str(label).upper() != "PASS"
    ]

    samples = gaps.sort_values("weakness", ascending=False).drop_duplicates(_sample_key(gaps))
    images = out / "rca_images"
    images.mkdir(exist_ok=True)
    spot_rows = []
    for side in (_is_pass(samples["label"]), ~_is_pass(samples["label"])):
        for _, row in samples[side].head(SPOT_CHECK_PER_SIDE).iterrows():
            source = _resolve_image(str(row["filepath"]), workspace)
            name = f"{len(spot_rows):02d}_{source.name}"
            shutil.copy2(source, images / name)
            spot_rows.append([row["label"], row.get("object_name", source.stem),
                              f"{row['siamese_score']:.4f}", f"{row['weakness']:.4f}",
                              f"![](rca_images/{name})", PENDING])
    if not spot_rows:
        raise ValueError("kpi_gaps.parquet produced no spot-check samples")

    top_rows = []
    for label, group in samples.groupby("label", sort=False):
        for _, row in group.head(TOP_K_ROWS_PER_LABEL).iterrows():
            top_rows.append([label, row.get("object_name", ""), row.get("input_path", row["filepath"]),
                             f"{row['siamese_score']:.4f}", f"{row['weakness']:.4f}",
                             "yes" if row["weakness"] > 0 else "no"])
    lightings = max(1, len(gaps) // max(1, len(samples)))
    top3 = ", ".join(f"{label} ({int(row['misclassified'])})" for label, row in per_label.head(3).iterrows())

    report = f"""# VCN Gap Analysis Report: {results_dir.name} / {iteration}

## 1. Verdict
- Chosen threshold: {threshold:.6g} (precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f} on NO_PASS; target recall >= {min_recall:g})
- KPI reachability: {"yes" if recall >= min_recall else "no"} (achieved NO_PASS recall {recall:.4f})
- Total samples: {len(inference)} | Total weak samples kept: {len(samples)} | Misclassified: {int(inference["misclassified"].sum())}
- Top-3 labels by misclassification: {top3}
- {len(samples)} weak samples written to kpi_gaps.parquet for augmentation

## 2. Threshold Selection
- Target NO_PASS recall: {min_recall:g}
- Chosen threshold (from `threshold.txt`): {threshold:.6g}; the container does not emit its candidate sweep.

{_table(["", "Predicted NO_PASS", "Predicted PASS"], [["Actual NO_PASS", f"TP={tp}", f"FN={fn}"], ["Actual PASS", f"FP={fp}", f"TN={tn}"]])}

## 3. Weakness Distribution
{_table(["Label", "Total Samples", "Mean Weakness", "Median Weakness", "Max Weakness", "# Misclassified"],
        [[label, int(row["total"]), f"{row['mean']:.4f}", f"{row['median']:.4f}", f"{row['max']:.4f}", int(row["misclassified"])]
         for label, row in per_label.iterrows()])}

## 4. Top-K Weakest Samples (per label)
{_table(["Label", "object_name", "input_path", "siamese_score", "weakness", "misclassified?"], top_rows)}

## 5. Visual Spot Check ({len(spot_rows)} samples)
{_table(["Label", "object_name", "siamese_score", "weakness", "Test Image", "Verdict"], spot_rows)}

Verdicts are pending: the card pack's executor model is text-only and cannot view images.
An operator must classify each image (mislabeled / edge case / data quality / systematic).

## 6. Per-Label Breakdown
```
{breakdown}
```

## 7. Recommended Actions
1. **Relabel** — review the section 5 images; relabel any the operator tags `mislabeled`.
2. **Augment** — `kpi_gaps.parquet` ({len(samples)} samples x {lightings} lightings = {len(gaps)} filepaths) is the routing input for the next iteration.
3. **Threshold action** — {"KPI is met at the chosen threshold" if recall >= min_recall else "retrain with the mined data and re-run gap analysis"}.
4. **Systematic failures** — flag any section 5 image the operator tags `systematic` for model review.
"""
    (out / "RCA_Report.md").write_text(report)
    return {"unreachable": False, "report": str(out / "RCA_Report.md"),
            "gaps": str(out / "kpi_gaps.parquet"), "threshold": threshold,
            "target_defects": target_defects}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument("--rca-dir", type=Path, required=True, help="Timestamped gap_analysis results_dir")
    parser.add_argument("--min-recall", type=float, default=1.0, help="min_recall passed to gap_analysis")
    args = parser.parse_args()
    try:
        summary = prepare(args.results_dir, args.workspace, args.iter_label, args.rca_dir, args.min_recall)
    except (OSError, ValueError, KeyError) as exc:
        print(f"prepare_card_rca: {exc}", file=sys.stderr)
        return 2
    (args.rca_dir / "rca_target_defects.txt").write_text(
        "".join(label + "\n" for label in summary["target_defects"])
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
