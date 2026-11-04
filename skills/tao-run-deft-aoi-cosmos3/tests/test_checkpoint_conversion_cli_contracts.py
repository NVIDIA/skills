#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILLS_ROOT = SKILL_ROOT.parents[1]
MODEL_ROOT = SKILLS_ROOT / "models/tao-finetune-cosmos-reason"
CONVERTER = MODEL_ROOT / "scripts/prepare_cosmos3_vlm_checkpoint.py"
COSMOS_REASON = SKILL_ROOT / "references/cosmos-reason.md"
PREFLIGHT = SKILL_ROOT / "references/preflight.md"
FLAG_PATTERN = re.compile(r"--[a-z][a-z0-9-]*")
REQUIRED_PUBLIC_FLAGS = {
    "--backend",
    "--base-model-path-or-uri",
    "--output-path",
    "--cache-dir",
    "--runtime-image",
    "--runtime-image-digest",
}
STALE_CONVERTER_FLAGS = {
    "--checkpoint-path",
    "--framework-image",
    "--framework-image-digest",
    "--secrets-env",
    "--validate-with-image",
    "--vlm-model-name",
}


def cli_flags(script: pathlib.Path) -> set[str]:
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(FLAG_PATTERN.findall(result.stdout))


def documented_converter_command() -> str:
    lines = COSMOS_REASON.read_text(encoding="utf-8").splitlines()
    script_line = next(
        index
        for index, line in enumerate(lines)
        if "prepare_cosmos3_vlm_checkpoint.py" in line
    )
    start = script_line - 1
    end = script_line
    while lines[end].rstrip().endswith("\\"):
        end += 1
    return "\n".join(lines[start : end + 1])


class CheckpointConversionCliContractTests(unittest.TestCase):
    def test_documented_command_matches_converter_help(self) -> None:
        command = documented_converter_command()
        documented = set(FLAG_PATTERN.findall(command))
        supported = cli_flags(CONVERTER)

        self.assertIn("--backend cosmos-framework", command)
        self.assertLessEqual(REQUIRED_PUBLIC_FLAGS, documented)
        missing = documented - supported
        if missing:
            self.assertEqual(missing, {"--backend"})
            self.assertIn(
                "model skill predates PR 230",
                PREFLIGHT.read_text(encoding="utf-8"),
            )
        self.assertTrue(documented.isdisjoint(STALE_CONVERTER_FLAGS))

    def test_application_guidance_uses_only_owned_converter_and_init_flags(self) -> None:
        documents = [SKILL_ROOT / "SKILL.md", SKILL_ROOT / "eval.config"]
        documents.extend(sorted((SKILL_ROOT / "references").glob("*.md")))
        text = "\n".join(path.read_text(encoding="utf-8") for path in documents)

        stale_converter_flags = set(FLAG_PATTERN.findall(text)) & STALE_CONVERTER_FLAGS
        self.assertEqual(stale_converter_flags, {"--framework-image-digest"})

    def test_preflight_points_to_existing_model_section(self) -> None:
        preflight = PREFLIGHT.read_text(encoding="utf-8")
        model_skill = (MODEL_ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertIn("`Nano checkpoint model-type choice`", preflight)
        self.assertIn("## Nano checkpoint model-type choice", model_skill)


if __name__ == "__main__":
    unittest.main()
