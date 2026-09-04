# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


class TestSkillModelIntake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill = (SKILL_ROOT / "SKILL.md").read_text()
        cls.acquire = (SKILL_ROOT / "references" / "model-acquire.md").read_text()
        cls.choice = cls.skill.split(
            "## Model choice — always offer two options", 1
        )[1].split("## Pipeline Overview", 1)[0]

    def test_exactly_two_ordered_model_choices_are_offered(self):
        self.assertEqual(
            re.findall(r"^### ([12])\. ", self.choice, flags=re.MULTILINE),
            ["1", "2"],
        )

    def test_default_is_the_validated_rtdetr_model(self):
        self.assertIn("Default model (recommended)", self.choice)
        self.assertIn("PekingU/rtdetr_r50vd", self.choice)

    def test_custom_choice_accepts_hf_or_versioned_ngc(self):
        self.assertIn("Hugging Face model ID", self.choice)
        self.assertIn("NVIDIA NGC catalog model URL including its version", self.choice)
        self.assertIn("rejects classification, segmentation", self.choice)

    def test_model_acquire_runbook_repeats_the_choice_gate(self):
        intake = self.acquire.split("## Intake — choose the model", 1)[1].split(
            "## MANDATORY", 1
        )[0]
        self.assertIn("Default model (recommended)", intake)
        self.assertIn("Custom object-detection model", intake)
        self.assertIn('INPUT="PekingU/rtdetr_r50vd"', intake)

    def test_dry_run_has_no_side_effects(self):
        normalized = " ".join(self.choice.split())
        for phrase in (
            "without browsing",
            "downloading",
            "launching Docker",
            "writing files",
            "starting processes",
        ):
            self.assertIn(phrase, normalized)


if __name__ == "__main__":
    unittest.main()
