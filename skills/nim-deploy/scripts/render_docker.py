#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the local (docker target) artifacts of a run through its runner, then advance the run.

The run's runner (input `runner`, else NIM_DEPLOY_RUNNER, else docker; see runners/__init__.py)
writes the files into the run folder. Prints one JSON object.
Exit codes: 0 ok, 2 bad input, 3 precondition failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_common as nd
import runners
from nd_common import RenderError


def main(argv: list[str] | None = None) -> None:
    help_text = getattr(runners.load(), "INPUT_HELP", "")
    parser = nd.render_parser(
        "Render the run's local artifacts and advance the run to validate.", help_text
    )
    args = parser.parse_args(argv)
    state, inputs = nd.open_run(args.run, ("docker",), args.preview)
    try:
        runner = runners.runner_of(state)
        paths = runner.render(inputs, nd.run_dir(state["run_id"]))
    except RenderError as exc:
        nd.reject_input(state, "docker_input", exc)
    except ValueError as exc:
        nd.reject_input(state, "bad_runner", exc)
    facts = {
        "paths": [str(path) for path in paths],
        "sha256": {str(path): nd.sha256_file(path) for path in paths},
        "container": runner.container_name(inputs),
        "url": runner.ready_url(inputs),
    }
    nd.close_run(state, paths, facts, args.preview)


if __name__ == "__main__":
    main()
