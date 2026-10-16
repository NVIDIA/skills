#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select DEFT AOI probes or KPI-best RT-DETR checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metric_rows(paths: list[Path]) -> tuple[list[tuple[int, float]], list[int]]:
    result = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        latest_epoch = None
        pending_untagged = None
        for line in path.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_epoch = None
            if row.get("epoch") is not None:
                try:
                    row_epoch = int(row["epoch"])
                    latest_epoch = row_epoch
                except (TypeError, ValueError):
                    pass
            kpi = row.get("kpi") if isinstance(row.get("kpi"), dict) else {}
            score = kpi.get("val_mAP50", kpi.get("mAP50"))
            epoch = kpi.get("epoch")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_value):
                continue
            if epoch is not None:
                try:
                    epoch = int(epoch)
                except (TypeError, ValueError):
                    continue
            elif row_epoch is not None:
                epoch = row_epoch

            if epoch is not None:
                # TAO can emit an untagged evaluation KPI immediately before
                # repeating that value on the next epoch-tagged event.
                if pending_untagged is not None:
                    if pending_untagged[1] != score_value:
                        result.append(pending_untagged)
                    pending_untagged = None
                result.append((int(epoch), score_value))
            elif latest_epoch is not None:
                if pending_untagged is not None:
                    result.append(pending_untagged)
                pending_untagged = (latest_epoch, score_value)
        if pending_untagged is not None:
            result.append(pending_untagged)
    if not result:
        raise ValueError("status contains no finite KPI val_mAP50 rows")
    # A resumed run appends to the same status.json, so passing every phase
    # repeats pre-crash epochs. Count each epoch once; the latest row wins.
    by_epoch: dict[int, float] = {}
    duplicates = set()
    for epoch, score_value in result:
        if epoch in by_epoch:
            duplicates.add(epoch)
        by_epoch[epoch] = score_value
    return list(by_epoch.items()), sorted(duplicates)


def _metrics(paths: list[Path]) -> list[tuple[int, float]]:
    return _metric_rows(paths)[0]


def _set(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    node = value
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = replacement


def probes(manifest_path: Path, template_path: Path, statuses: list[Path],
           output: Path, history_path: Path | None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if len(statuses) != 3:
        raise ValueError("probe selection requires exactly three status files")
    manifest = json.loads(manifest_path.read_text())
    if len(manifest.get("probes", [])) != 3:
        raise ValueError("training manifest does not declare three probes")
    scored = []
    for probe, status in zip(manifest["probes"], statuses, strict=True):
        epoch, score = max(_metrics([status]), key=lambda row: (row[1], -row[0]))
        scored.append({**probe, "best_epoch": epoch, "best_kpi_mAP50": score})
    winner = max(scored, key=lambda row: (row["best_kpi_mAP50"], -row["index"]))
    spec = yaml.safe_load(template_path.read_text())
    for key, value in winner["overrides"].items():
        _set(spec, key, value)
    output.mkdir(parents=True)
    train_path, winner_path = output / "train.yaml", output / "probe_winner.json"
    history_output, incumbent = output / "history.json", output / "incumbent.json"
    if any(path.exists() for path in (train_path, winner_path, history_output, incumbent)):
        raise FileExistsError("refusing to overwrite probe selection outputs")
    train_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    history = json.loads(history_path.read_text()) if history_path else []
    if not isinstance(history, list):
        raise ValueError("history must be a JSON list")
    history.append({"iteration": int(manifest["iteration"]),
                    "train_size": int(manifest["train_size"])})
    _json(history_output, history)
    _json(incumbent, winner["overrides"])
    report = {"status": "COMPLETE", "winner": winner, "scores": scored,
              "train_spec": str(train_path.resolve())}
    _json(winner_path, report)
    return report


def checkpoint(policy_path: Path, spec_path: Path, statuses: list[Path], checkpoints: Path,
               planned_epochs: int, extension_applied: bool, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if planned_epochs < 1:
        raise ValueError("planned_epochs must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = yaml.safe_load(policy_path.read_text())
    rows, duplicate_epochs = _metric_rows(statuses)
    best_epoch, score = max(rows, key=lambda row: (row[1], -row[0]))
    selected = checkpoints / f"model_epoch_{best_epoch:03d}.pth"
    if not selected.is_file():
        raise FileNotFoundError(f"KPI-best checkpoint is missing: {selected}")
    training = policy["training"]
    late = best_epoch >= planned_epochs - int(training["late_best_window"])
    report = {"status": "COMPLETE", "action": "select", "best_epoch": best_epoch,
              "best_kpi_mAP50": score, "selected_checkpoint": str(selected.resolve()),
              "planned_epochs": planned_epochs, "extension_applied": extension_applied,
              "status_files": [str(path.resolve()) for path in statuses],
              "epochs_covered": sorted(epoch for epoch, _ in rows),
              "duplicate_epochs": duplicate_epochs}
    if late and not extension_applied:
        terminal = checkpoints / f"model_epoch_{planned_epochs - 1:03d}.pth"
        if not terminal.is_file():
            raise FileNotFoundError(f"terminal checkpoint for extension is missing: {terminal}")
        extended = planned_epochs + int(training["extension_epochs"])
        spec = yaml.safe_load(spec_path.read_text())
        spec["train"]["num_epochs"] = extended
        spec["train"].pop("pretrained_model_path", None)
        spec["train"]["resume_training_checkpoint_path"] = str(terminal.resolve())
        extension_path = output.with_name("extension.yaml")
        if extension_path.exists():
            raise FileExistsError(extension_path)
        extension_path.write_text(yaml.safe_dump(spec, sort_keys=False))
        report.update(action="extend", extended_num_epochs=extended,
                      resume_checkpoint=str(terminal.resolve()),
                      extension_spec=str(extension_path.resolve()))
    _json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probes")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--main-template", type=Path, required=True)
    probe.add_argument("--status", type=Path, action="append", required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--history", type=Path)
    select = sub.add_parser("checkpoint")
    select.add_argument("--policy", type=Path, required=True)
    select.add_argument("--train-spec", type=Path, required=True)
    select.add_argument("--status", type=Path, action="append", required=True)
    select.add_argument("--checkpoint-dir", type=Path, required=True)
    select.add_argument("--planned-epochs", type=int, required=True)
    select.add_argument("--extension-applied", action="store_true")
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = (probes(args.manifest.resolve(), args.main_template.resolve(), args.status,
                     args.output_dir.resolve(), args.history.resolve() if args.history else None)
              if args.command == "probes" else
              checkpoint(args.policy.resolve(), args.train_spec.resolve(), args.status,
                         args.checkpoint_dir.resolve(), args.planned_epochs,
                         args.extension_applied, args.output.resolve()))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
