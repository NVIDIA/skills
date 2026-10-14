#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for callers of the former Python readiness helper.

Usage:
    python3 scripts/check_parabricks_runtime.py [readiness options]
Arguments:
    Forwarded to and validated by the Bash helper; run with --help for options.
Output:
    The Bash helper's text or JSON report. Container probes remain opt-in.
Exit codes:
    0: report produced, including any readiness gaps.
    2: invalid arguments or missing helper dependencies.
    Other Bash exit statuses are preserved; signal exits use 128 + signal.
"""

from pathlib import Path
import signal
import subprocess
import sys

SIGNAL_EXIT_OFFSET = 128


def main():
    script = Path(__file__).resolve().with_suffix(".sh")
    try:
        helper = subprocess.Popen(["bash", str(script), *sys.argv[1:]], shell=False)
    except OSError as exc:
        print("Cannot start the Bash readiness helper: {}".format(exc), file=sys.stderr)
        return 2

    def forward_signal(signum, _frame):
        helper.send_signal(signum)

    # Preserve cancellation through the compatibility entry point instead of
    # leaving an orphaned readiness check when a caller terminates this process.
    previous = {sig: signal.signal(sig, forward_signal)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        returncode = helper.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return returncode if returncode >= 0 else SIGNAL_EXIT_OFFSET - returncode


if __name__ == "__main__":
    sys.exit(main())
