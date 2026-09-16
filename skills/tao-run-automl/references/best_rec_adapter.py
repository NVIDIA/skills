#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit best_rec.json from an AutoMLRunner result — the read side of the
AutoML -> DEFT warm-start seam.

AutoMLRunner (the separate nvidia-tao-automl wheel) is unchanged; this adapter
serializes its winning recommendation into the best_rec.json contract
(skills/core/tao-artifacts/references/best_rec.schema.json), which the DEFT
pipeline reads to warm-start.

The load-bearing transform is BUDGET STRIPPING: ASHA/hyperband mutate the fidelity
key (num_epochs) per rung, so the winning spec carries a *rung* budget. Deep-merging
that into DEFT's hand-authored baseline spec would silently overwrite the real
epoch count — so budget keys are moved OUT of `specs` into a separate
`observed_budget`. And `metric_name` + `direction` are required: a bare score is
meaningless to `_pick_best` downstream. Output is validated against the schema so a
malformed hand-off fails loudly here, not after Phase-2 GPU-hours.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

SCHEMA_PATH = (Path(__file__).resolve().parents[3]
               / "core/tao-artifacts/references/best_rec.schema.json")
DEFAULT_BUDGET_KEYS = ("num_epochs", "train.num_epochs", "train.max_epochs", "epochs")


def _pop_nested(spec: dict, dotted: str):
    parts = dotted.split(".")
    d = spec
    for p in parts[:-1]:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    if isinstance(d, dict) and parts[-1] in d:
        return d.pop(parts[-1])
    return None


def build_best_rec(*, experiment_id: str, metric_name: str, direction: str,
                   best_rec_id, best_score, best_specs: dict, checkpoint_uri: str,
                   all_recs=None, budget_keys=DEFAULT_BUDGET_KEYS,
                   checkpoint_epoch=None, checkpoint_step=None) -> dict:
    if not metric_name:
        raise ValueError("metric_name is required (a bare score is meaningless downstream)")
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")

    specs = copy.deepcopy(best_specs)
    observed_budget: dict = {}
    for key in budget_keys:
        val = _pop_nested(specs, key)
        if val is not None:
            observed_budget[key.split(".")[-1]] = val

    best = {
        "rec_id": str(best_rec_id),
        "score": float(best_score),
        "specs": specs,
        "observed_budget": observed_budget,
        "checkpoint_uri": checkpoint_uri,
    }
    if checkpoint_epoch is not None:
        best["checkpoint_epoch"] = int(checkpoint_epoch)
    if checkpoint_step is not None:
        best["checkpoint_step"] = int(checkpoint_step)

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "metric_name": metric_name,
        "direction": direction,
        "best": best,
        "all_recs": list(all_recs or []),
    }


def validate(rec: dict) -> None:
    """Raise jsonschema.ValidationError if rec violates best_rec.schema.json."""
    import jsonschema
    jsonschema.validate(rec, json.loads(SCHEMA_PATH.read_text()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="JSON of the extracted runner fields; stdin if omitted")
    p.add_argument("-o", "--out", help="write best_rec.json here (else stdout)")
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args(argv)

    raw = json.loads(Path(args.input).read_text() if args.input and args.input != "-" else sys.stdin.read())
    rec = build_best_rec(**raw)
    if not args.no_validate:
        validate(rec)
    out = json.dumps(rec, indent=2)
    if args.out:
        Path(args.out).write_text(out + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
