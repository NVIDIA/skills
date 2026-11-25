#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the run's AP50 gates into a file `gap_analysis` can be given.

``init_deft_state.py`` accepts per-class AP50 thresholds, validates them, and stores
them at ``config.ap50_thresholds``. The gap spec needs them under ``weak_thresholds``,
in ``{class: {ap50: value}}`` shape, and ``apply_spec_overrides.py`` can only set a
nested block from a file. This writes that file.

The shipped asset leaves ``weak_thresholds`` as ``???``, so a build that skips this
step fails rather than gating on values the caller never chose. That matters beyond
the gate itself: the weak set decides the mining budget, so a substituted threshold
changes which images the iteration mines, not just which are reported weak.

Reads the state the run already froze rather than taking the thresholds again on the
command line: a second source is a second thing to disagree with deft_state.json.

Only the run's target classes are written. ``gap_analysis`` marks an image weak when any
listed class is under its gate, so a gate for a class the run does not target would
enlarge the weak set and with it the mining budget.

Inputs:  --results-dir (reads config.ap50_thresholds and config.target_classes), --out
Output:  a YAML file with a single top-level ``weak_thresholds`` key, one entry per
         target class

Exits 1 when the state has no thresholds or no target classes, when a target class has
no gate, or when a target's gate is outside [0, 1].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deft_stages import read_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="The run's results dir; config.ap50_thresholds is read from "
                             "its deft_state.json.")
    parser.add_argument("--out", required=True,
                        help="Where to write the weak_thresholds file, for "
                             "apply_spec_overrides.py --set-from-file.")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        state = read_state(args.results_dir)
        config = state.get("config")
        config = config if isinstance(config, dict) else {}

        thresholds = config.get("ap50_thresholds")
        if not isinstance(thresholds, dict) or not thresholds:
            raise ValueError(
                f"{args.results_dir}: config.ap50_thresholds is empty or missing. init "
                f"records it from --ap50-thresholds-json, so a run without it was not "
                f"initialised by init_deft_state.py")

        targets = config.get("target_classes") or []
        if isinstance(targets, str):
            targets = [c.strip() for c in targets.split(",") if c.strip()]

        if not targets:
            raise ValueError(
                f"{args.results_dir}: config.target_classes is empty. The gates written "
                f"here are scoped to it, so there is nothing to write")

        # A target with no gate is scored against gap_analysis's default_ap50_threshold
        # of 0.0, which can never mark an image weak, so that class can never be mined
        # for. init already refuses this; re-checked here because the file is what the
        # stage actually reads.
        ungated = [c for c in targets if c not in thresholds]
        if ungated:
            raise ValueError(
                f"target class(es) {ungated} have no AP50 gate in config.ap50_thresholds "
                f"{sorted(thresholds)}; they would be gated at 0.0 and never mined for")

        for name in targets:
            value = thresholds[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"ap50_thresholds[{name!r}] is {value!r}, not a number")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"ap50_thresholds[{name!r}] is {value}, outside [0, 1]")

        # Target classes only, in the order the run declares them. A class listed in
        # config.ap50_thresholds but not targeted must not reach this file: gap_analysis
        # marks an image weak when *any* listed class is under its gate, so a stray gate
        # would enlarge the weak set and with it the mining budget -- spending the
        # iteration's images on a class the run is not training for. init warns that
        # extra thresholds "will be ignored"; dropping them here is what makes that true.
        block = {"weak_thresholds": {c: {"ap50": float(thresholds[c])} for c in targets}}

        untargeted = [c for c in thresholds if c not in targets]
        if untargeted:
            print(f"note: {sorted(untargeted)} have gates in config.ap50_thresholds but "
                  f"are not target classes; omitted from {args.out}", file=sys.stderr)

        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(block, sort_keys=False), encoding="utf-8")

        if args.report_json:
            Path(args.report_json).expanduser().resolve().write_text(
                json.dumps({"out": str(out), "weak_thresholds": block["weak_thresholds"]},
                           indent=2) + "\n", encoding="utf-8")

        for name in targets:
            print(f"  {name}: {float(thresholds[name])}")
        print(f"wrote {out}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"prepare_thresholds_for_gap_analysis: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
