#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import shlex
import string
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
for module_name in ("init_deft_state", "metric_contract", "render_report"):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(SCRIPTS))

import init_deft_state  # noqa: E402


class ChangeNetInitStateContractTests(unittest.TestCase):
    def test_venv_python_symlink_survives_state_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            venv_bin = root / "venv/bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python3").symlink_to(sys.executable)
            venv_python = venv_bin / "python"
            venv_python.symlink_to("python3")
            results = root / "results"

            rc = init_deft_state.main(
                [
                    "--results-dir",
                    str(results),
                    "--workspace",
                    str(root / "workspace"),
                    "--python-executable",
                    str(venv_python),
                    "--kpi-target",
                    "FAR <= 1% at recall=100%",
                    "--max-iterations",
                    "1",
                    "--num-gpus",
                    "1",
                    "--gpu-model",
                    "NVIDIA RTX PRO 6000 Blackwell (96 GB)",
                    "--num-epochs",
                    "1",
                    "--num-sdg",
                    "2",
                    "--project",
                    "nvpcb",
                    "--step",
                    "1",
                    "--train-container",
                    "example/train:1",
                    "--ag-container",
                    "example/anomalygen:1",
                ]
            )

            self.assertEqual(rc, 0)
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(
                state["execution_policy"]["python_executable"],
                str(venv_python),
            )
            self.assertNotEqual(str(venv_python), str(venv_python.resolve()))

    def test_card_00_init_command_matches_argparse(self) -> None:
        card = (SKILL_ROOT / "cards/00-init-baseline-train.md").read_text()
        line = next(l for l in card.splitlines() if "init_deft_state.py" in l and l.startswith("$DPY"))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            env = {
                "RD": str(root / "results"),
                "WS": str(root / "workspace"),
                "GPU_MODEL": "NVIDIA RTX PRO 6000 Blackwell, 97887 MiB",
                "TRAIN_IMG": "example/train:1",
                "DS_IMG": "example/anomalygen:1",
            }
            argv = [string.Template(a).substitute(env)
                    for a in shlex.split(line.split("init_deft_state.py", 1)[1])]

            rc = init_deft_state.main(argv)

            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            self.assertEqual(state["config"]["gpu_model"], env["GPU_MODEL"])


if __name__ == "__main__":
    unittest.main()
