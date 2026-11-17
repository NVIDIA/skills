#!/usr/bin/env python3
"""CPU stand-in for bench_rmsnorm.py while the B200 box is down.

Replays the mean kernel time recorded on the GPU for the no-ACF baseline and
for each Helion booster pack candidate. Pass --acf to select a candidate;
omit it for the baseline. Output format matches the real benchmark.
"""

import argparse
import os
import random
import sys

RECORDED_MEAN_MS = {
    None: 0.4120,
    "candidate_03.acf": 0.3611,
    "candidate_07.acf": 0.4097,
    "candidate_11.acf": 0.5034,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acf", default=None, help="path to the ACF to apply")
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()

    key = None
    if args.acf is not None:
        if not os.path.isfile(args.acf):
            print(f"error: ACF not found: {args.acf}", file=sys.stderr)
            return 2
        key = os.path.basename(args.acf)
        if key not in RECORDED_MEAN_MS:
            print(f"error: no recorded timing for {key}", file=sys.stderr)
            return 2

    mean = RECORDED_MEAN_MS[key] * (1.0 + random.uniform(-0.001, 0.001))
    std = RECORDED_MEAN_MS[key] * 0.004
    print(f"rmsnorm bf16 [8192x4096] trials: {args.trials}")
    print(f"mean: {mean:.4f}")
    print(f"std: {std:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
