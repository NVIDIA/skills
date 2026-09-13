#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a readable Auto-FL progress plot from the campaign ledger.

Usage:
    ``python plot_progress.py [RESULTS] [--output PATH] [options]``
Arguments:
    ``RESULTS`` is the campaign TSV ledger; options select the output PNG,
    metric label, label limits, and y-axis behavior.
Output:
    Atomically writes the PNG and prints baseline, best, and delta values.
Exit codes:
    0 when plotting succeeds; argparse uses 2 for invalid CLI syntax and data
    or dependency failures return a nonzero process status.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


class PlotDependencyError(RuntimeError):
    """Raised when the rich plotting dependency is unavailable."""


class NoScoredResultsError(ValueError):
    """Raised when a campaign has no scored rows to plot yet."""


@dataclass
class ProgressRecord:
    index: int
    status: str
    name: str
    score: Optional[float]
    runtime_seconds: float
    description: str
    kind: str = ""
    family: str = ""
    literature_event_id: str = ""


def load_campaign_guard():
    """Load the sibling campaign guard, the shared owner of the ledger score contract."""
    module_name = "nvflare_autofl_skill_campaign_guard"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    module_path = Path(__file__).resolve().with_name("campaign_guard.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load campaign guard from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def parse_score(value: Any) -> Optional[float]:
    return load_campaign_guard().parse_score(value)


def parse_runtime(value: Any) -> float:
    seconds = load_campaign_guard().parse_score(value)
    return max(0.0, seconds) if seconds is not None else 0.0


def format_runtime(seconds: float) -> str:
    if seconds <= 0:
        return ""
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def truncate(text: Any, limit: int) -> str:
    compact = " ".join(str(text or "").strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def compact_label(text: Any, limit: int) -> str:
    compact = " ".join(str(text or "").strip().split())
    replacements = {
        "server_momentum": "smom",
        "server_lr": "slr",
        "clip_grad_norm": "clip",
        "eta_min_factor": "eta_min",
        "label_smoothing": "ls",
    }
    for old, new in replacements.items():
        compact = compact.replace(old, new)
    for marker in [" (", ";"]:
        if marker in compact:
            compact = compact.split(marker, 1)[0]
    return truncate(compact, limit)


def record_label(record: ProgressRecord, limit: int) -> str:
    return compact_label(record.name or record.description, limit)


def resolve_candidate_kind(kind: Any, changed_files: Any) -> str:
    """Resolve the candidate kind, falling back to the legacy changed_files column when kind is absent (None)."""
    if kind is not None:
        return str(kind).strip().lower()
    changed = str(changed_files or "").strip().lower()
    if changed and changed != "none":
        return "source_edit"
    return "argument_only"


def load_results(path: Path) -> List[ProgressRecord]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for index, row in enumerate(csv.DictReader(f, delimiter="\t")):
            records.append(
                ProgressRecord(
                    index=index,
                    status=(row.get("status", "") or "").strip().lower(),
                    name=row.get("name", "") or row.get("candidate", "") or row.get("commit", ""),
                    score=parse_score(row.get("score", "")),
                    runtime_seconds=parse_runtime(row.get("runtime_seconds", "")),
                    description=row.get("diff_summary", "") or row.get("description", ""),
                    kind=resolve_candidate_kind(row.get("candidate_kind"), row.get("changed_files", "")),
                    family=(row.get("algorithm_family", "") or "").strip(),
                    literature_event_id=(row.get("literature_event_id", "") or "").strip(),
                )
            )
    return records


def normalize_records(records: Iterable[Any]) -> List[ProgressRecord]:
    normalized = []
    for index, record in enumerate(records):
        kind = getattr(record, "candidate_kind", None)
        if kind is None:
            kind = getattr(record, "kind", None)
        normalized.append(
            ProgressRecord(
                index=index,
                status=str(getattr(record, "status", "") or "").strip().lower(),
                name=str(getattr(record, "name", "") or ""),
                score=parse_score(getattr(record, "score", None)),
                runtime_seconds=parse_runtime(getattr(record, "runtime_seconds", 0.0)),
                description=str(getattr(record, "diff_summary", "") or getattr(record, "description", "") or ""),
                kind=resolve_candidate_kind(kind, getattr(record, "changed_files", "")),
                family=str(getattr(record, "algorithm_family", "") or getattr(record, "family", "") or "").strip(),
                literature_event_id=str(getattr(record, "literature_event_id", "") or "").strip(),
            )
        )
    return normalized


_CAMPAIGN_GUARD = load_campaign_guard()
better = _CAMPAIGN_GUARD.better
improvement_over_baseline = _CAMPAIGN_GUARD.improvement_over_baseline


def cumulative_best(values: Sequence[float], mode: str = "max") -> List[float]:
    incumbent = None
    result = []
    for value in values:
        if better(value, incumbent, mode):
            incumbent = value
        result.append(incumbent)
    return result


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight


def default_y_limits(
    scores: Sequence[float], baseline: float, mode: str = "max", full_y_range: bool = False
) -> Tuple[float, float]:
    score_min = min(scores)
    score_max = max(scores)
    if full_y_range:
        span = max(score_max - score_min, 0.01)
        return score_min - max(0.01, span * 0.08), score_max + max(0.01, span * 0.16)

    if mode == "max":
        useful_min = min(baseline, percentile(scores, 0.20))
        useful_span = max(score_max - useful_min, 0.01)
        lower = useful_min - max(0.01, useful_span * 0.20)
        upper = score_max + max(0.015, useful_span * 0.35)
        if score_min >= lower:
            full_span = max(score_max - score_min, 0.01)
            lower = score_min - max(0.01, full_span * 0.08)
        return lower, upper

    useful_max = max(baseline, percentile(scores, 0.80))
    useful_span = max(useful_max - score_min, 0.01)
    lower = score_min - max(0.015, useful_span * 0.35)
    upper = useful_max + max(0.01, useful_span * 0.20)
    if score_max <= upper:
        full_span = max(score_max - score_min, 0.01)
        upper = score_max + max(0.01, full_span * 0.08)
    return lower, upper


def select_observed_milestones(
    valid: Sequence[ProgressRecord], max_labels: int, mode: str = "max"
) -> List[Tuple[float, ProgressRecord]]:
    if max_labels <= 0:
        return []
    incumbent = None
    milestones = []
    final_running_best = None
    for record in valid:
        if record.score is None or not better(record.score, incumbent, mode):
            continue
        delta = 0.0 if incumbent is None else abs(record.score - incumbent)
        final_running_best = (delta, record)
        if incumbent is not None:
            milestones.append((delta, record))
        incumbent = record.score

    if final_running_best is None:
        return []
    if not milestones:
        return [final_running_best]

    final_index = final_running_best[1].index
    non_final = [item for item in milestones if item[1].index != final_index]
    selected = sorted(non_final, key=lambda item: item[0], reverse=True)[: max_labels - 1]
    selected_indices = {record.index for _, record in selected}
    selected_indices.add(final_index)
    return [(delta, record) for delta, record in milestones if record.index in selected_indices]


def select_literature_labels(records: Sequence[ProgressRecord], max_labels: int) -> List[ProgressRecord]:
    if max_labels <= 0 or not records:
        return []
    if len(records) <= max_labels:
        return list(records)
    latest = records[-1]
    selected = {latest.index}
    for record in sorted(records, key=lambda item: item.runtime_seconds, reverse=True):
        if len(selected) >= max_labels:
            break
        selected.add(record.index)
    return [record for record in records if record.index in selected]


def label_placement(
    label_number: int,
    record: ProgressRecord,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> Tuple[Tuple[int, int], str, str]:
    x_span = max(x_limits[1] - x_limits[0], 1.0)
    y_span = max(y_limits[1] - y_limits[0], 1e-9)
    x_fraction = (record.index - x_limits[0]) / x_span
    score = record.score if record.score is not None else y_limits[0]
    y_fraction = (score - y_limits[0]) / y_span
    near_right = x_fraction > 0.72
    near_top = y_fraction > 0.78
    x_offset = -10 if near_right else 10
    y_base = -12 if near_top else 12
    y_step = (label_number % 3) * 8
    return (
        (x_offset, y_base - y_step if near_top else y_base + y_step),
        "right" if near_right else "left",
        "top" if near_top else "bottom",
    )


def plot_progress(
    records: Iterable[Any],
    output: Path,
    metric_label: str,
    mode: str = "max",
    max_labels: int = 6,
    max_literature_labels: int = 4,
    full_y_range: bool = False,
) -> Tuple[float, float]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotDependencyError("matplotlib is required for the rich Auto-FL progress plot") from exc

    rows = normalize_records(records)
    valid = [record for record in rows if record.score is not None and record.status != "crash"]
    literature_rows = [record for record in rows if record.status == "literature"]
    if not valid:
        raise NoScoredResultsError("No non-crash rows with numeric scores found")

    baseline_row = next((record for record in valid if record.status == "baseline"), valid[0])
    baseline = baseline_row.score
    best_row = valid[0]
    for record in valid[1:]:
        if better(record.score, best_row.score, mode):
            best_row = record
    best_score = best_row.score

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    styles = {
        "discard": {"color": "#cccccc", "size": 18, "alpha": 0.55, "label": "Discarded"},
        "candidate": {"color": "#3498db", "size": 32, "alpha": 0.75, "label": "Candidate"},
        "keep": {"color": "#2ecc71", "size": 72, "alpha": 0.95, "label": "Kept"},
    }
    groups = {
        "discard": [record for record in valid if record.status == "discard"],
        "candidate": [record for record in valid if record.status == "candidate"],
        "keep": [record for record in valid if record.status in {"baseline", "keep"}],
    }
    literature_edge = "#8e44ad"
    for status, group in groups.items():
        if not group:
            continue
        style = styles[status]
        base_edge = "black" if status == "keep" else "none"
        base_width = 0.5 if status == "keep" else 0.0
        source_edits = [record for record in group if record.kind == "source_edit"]
        argument_only = [record for record in group if record.kind != "source_edit"]
        for subgroup, marker, label_suffix in ((argument_only, "o", ""), (source_edits, "D", " (source edit)")):
            if not subgroup:
                continue
            ax.scatter(
                [record.index for record in subgroup],
                [record.score for record in subgroup],
                c=style["color"],
                s=style["size"],
                alpha=style["alpha"],
                marker=marker,
                zorder=4 if status == "keep" else 2,
                label=style["label"] + label_suffix,
                edgecolors=[literature_edge if record.literature_event_id else base_edge for record in subgroup],
                linewidths=[0.9 if record.literature_event_id else base_width for record in subgroup],
            )
    if any(record.literature_event_id for group in groups.values() for record in group):
        ax.scatter(
            [], [], facecolors="none", edgecolors=literature_edge, linewidths=0.9, s=40, label="Literature-linked"
        )

    known_statuses = {"baseline", "keep", "discard", "candidate", "crash", "literature"}
    other = [record for record in valid if record.status not in known_statuses]
    if other:
        ax.scatter(
            [record.index for record in other],
            [record.score for record in other],
            c="#9b59b6",
            s=28,
            alpha=0.65,
            zorder=2,
            label="Other",
        )

    observed_scores = [record.score for record in valid]
    ax.step(
        [record.index for record in valid],
        cumulative_best(observed_scores, mode),
        where="post",
        color="#27ae60",
        linewidth=2.2,
        alpha=0.75,
        zorder=3,
        label="Running best observed",
    )

    runtime_rows = [record for record in rows if record.runtime_seconds > 0 and record.status != "literature"]
    total_runtime = sum(record.runtime_seconds for record in runtime_rows)
    average_runtime = total_runtime / len(runtime_rows) if runtime_rows else 0.0
    literature_runtime = sum(record.runtime_seconds for record in literature_rows)
    runtime_title = f", {format_runtime(total_runtime)} total" if total_runtime else ""
    if average_runtime:
        runtime_title += f", {format_runtime(average_runtime)} avg/candidate"
    if literature_rows:
        runtime_title += f", {len(literature_rows)} lit"
        if literature_runtime:
            runtime_title += f" ({format_runtime(literature_runtime)})"

    direction = "higher" if mode == "max" else "lower"
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel(f"{metric_label} ({direction} is better)", fontsize=12)
    ax.set_title(
        f"Auto-FL Progress ({metric_label}): {len(rows)} rows, {len(valid)} scored, "
        f"{sum(record.status == 'keep' for record in rows)} kept, "
        f"{sum(record.status == 'candidate' for record in rows)} candidate, "
        f"{sum(record.status == 'discard' for record in rows)} discarded, "
        f"{sum(record.status == 'crash' for record in rows)} crash{runtime_title}",
        fontsize=14,
    )
    ax.axhline(
        baseline,
        color="#7f8c8d",
        linewidth=1.2,
        alpha=0.5,
        linestyle="--",
        label="Baseline",
    )
    ax.grid(True, alpha=0.2)

    y_limits = default_y_limits(observed_scores, baseline, mode, full_y_range=full_y_range)
    ax.set_ylim(*y_limits)
    ax.set_xlim(-0.5, max(len(rows) - 0.5, max(record.index for record in valid) + 0.5))

    if literature_rows:
        event_color = "#8e44ad"
        event_y = y_limits[1] - max(y_limits[1] - y_limits[0], 1e-9) * 0.035
        for record in literature_rows:
            ax.axvline(record.index, color=event_color, linestyle=":", linewidth=1.1, alpha=0.45, zorder=1)
        ax.scatter(
            [record.index for record in literature_rows],
            [event_y for _ in literature_rows],
            marker="v",
            c=event_color,
            s=50,
            alpha=0.85,
            zorder=5,
            label="Literature review",
            edgecolors="white",
            linewidths=0.4,
        )
        event_x_limits = ax.get_xlim()
        event_x_span = max(event_x_limits[1] - event_x_limits[0], 1.0)
        for record in select_literature_labels(literature_rows, max_literature_labels):
            runtime = format_runtime(record.runtime_seconds)
            near_right = (record.index - event_x_limits[0]) / event_x_span > 0.88
            annotation = ax.annotate(
                f"lit #{record.index}{f' {runtime}' if runtime else ''}: {compact_label(record.description, 30)}",
                (record.index, event_y),
                textcoords="offset points",
                xytext=(-4, -4) if near_right else (4, -4),
                fontsize=7.0,
                color=event_color,
                alpha=0.9,
                rotation=90,
                ha="right" if near_right else "left",
                va="top",
                annotation_clip=True,
            )
            annotation.set_clip_on(True)

    for label_number, (_, record) in enumerate(select_observed_milestones(valid, max_labels, mode)):
        offset, horizontal_align, vertical_align = label_placement(label_number, record, ax.get_xlim(), ax.get_ylim())
        milestone_text = f"#{record.index} {record.score:.4f}: {record_label(record, 28)}"
        if record.literature_event_id:
            milestone_text += f" [{record.family or record.literature_event_id}]"
        annotation = ax.annotate(
            milestone_text,
            (record.index, record.score),
            textcoords="offset points",
            xytext=offset,
            fontsize=8.0,
            color="#1a7a3a",
            alpha=0.9,
            ha=horizontal_align,
            va=vertical_align,
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-",
                "color": "#1a7a3a",
                "alpha": 0.35,
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 4,
            },
        )
        annotation.set_clip_on(True)

    summary_lines = [
        f"Baseline: {baseline:.6f}",
        f"Best: {best_score:.6f}",
        f"Improvement: {improvement_over_baseline(baseline, best_score, mode):+.6f}",
    ]
    if total_runtime:
        summary_lines.append(f"Runtime: {format_runtime(total_runtime)}")
    if average_runtime:
        summary_lines.append(f"Avg/candidate: {format_runtime(average_runtime)}")
    if literature_rows:
        literature_summary = f"Lit reviews: {len(literature_rows)}"
        if literature_runtime:
            literature_summary += f" ({format_runtime(literature_runtime)})"
        summary_lines.append(literature_summary)
    summary_lines.append(f"Best run: #{best_row.index} {record_label(best_row, 36)}")
    ax.text(
        0.015,
        0.985,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.9,
        },
    )

    clipped = sum(score < y_limits[0] or score > y_limits[1] for score in observed_scores)
    if clipped:
        ax.text(
            0.99,
            0.015,
            f"{clipped} outlier{'s' if clipped != 1 else ''} outside displayed range",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )

    ax.legend(loc="best", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.stem}.", suffix=".png", delete=False
        ) as f:
            temp_path = Path(f.name)
        plt.tight_layout()
        fig.savefig(temp_path, dpi=150, facecolor="white", transparent=False)
        os.replace(temp_path, output)
    finally:
        plt.close(fig)
        if temp_path and temp_path.exists():
            temp_path.unlink()
    return baseline, best_score


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="results.tsv", help="path to the Auto-FL TSV ledger")
    parser.add_argument("--output", default="progress.png", help="output PNG path")
    parser.add_argument("--metric", default="score", help="metric label shown in the plot")
    parser.add_argument("--mode", choices=["max", "min"])
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument("--max-literature-labels", type=int, default=4)
    parser.add_argument("--full-y-range", action="store_true")
    args = parser.parse_args(argv)

    results_path = Path(args.path).resolve()
    mode = args.mode
    if mode is None:
        try:
            mode = load_campaign_guard().campaign_mode_for_results(results_path)
        except ValueError as exc:
            parser.error(str(exc))
    records = load_results(results_path)
    baseline, best = plot_progress(
        records,
        Path(args.output),
        args.metric,
        mode=mode,
        max_labels=args.max_labels,
        max_literature_labels=args.max_literature_labels,
        full_y_range=args.full_y_range,
    )
    print(f"Saved {args.output}")
    print(f"baseline={baseline:.6f}")
    print(f"best={best:.6f}")
    print(f"improvement={improvement_over_baseline(baseline, best, mode):+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
