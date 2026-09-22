#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import patch_eval_image_cap  # noqa: E402


FRAMEWORK_EVALUATOR = """\
from cosmos_rl.framework import checkpoints
from cosmos_rl.framework import runtime

def evaluate(config):
    return runtime.evaluate(config)
"""


class EvalImageCapCompatibilityTests(unittest.TestCase):
    def test_absent_cap_is_source_driven_for_current_and_future_tags(self) -> None:
        images = (
            "nvcr.io/nvstaging/tao/cosmos-rl:"
            "enhanced-hooks-custom-loggers-20260816-direct-cache-fast-v7",
            "registry.example:5000/tao/cosmos-rl:future-build-20270101-v2",
            "registry.example/tao/cosmos-rl@sha256:" + "a" * 64,
        )
        for image in images:
            with self.subTest(image=image), mock.patch.object(
                patch_eval_image_cap,
                "read_from_image",
                return_value=FRAMEWORK_EVALUATOR,
            ), contextlib.redirect_stdout(io.StringIO()) as stdout:
                result = patch_eval_image_cap.main(
                    ["--image", image, "--probe", "--images", "2"]
                )
            self.assertEqual(result, 0)
            self.assertIn("classification=cap_absent", stdout.getvalue())
            self.assertIn("cap_in_image=none", stdout.getvalue())
            self.assertIn("patch_needed=False", stdout.getvalue())

    def test_absent_cap_writes_explicit_no_patch_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output_dir = root / "patch"
            summary = root / "summary.json"
            with mock.patch.object(
                patch_eval_image_cap,
                "read_from_image",
                return_value=FRAMEWORK_EVALUATOR,
            ), contextlib.redirect_stdout(io.StringIO()) as stdout:
                result = patch_eval_image_cap.main(
                    [
                        "--image",
                        "example/cosmos-rl:future-tag",
                        "--output-dir",
                        str(output_dir),
                        "--summary",
                        str(summary),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertFalse((output_dir / "base.py").exists())
            payload = json.loads(summary.read_text())
            self.assertEqual(payload["classification"], "cap_absent")
            self.assertIsNone(payload["cap_in_image"])
            self.assertFalse(payload["patch_needed"])
            self.assertIsNone(payload["mount_argument"])
            self.assertIn("no patch needed", stdout.getvalue())

    def test_changed_vllm_shape_fails_with_unknown_verdict(self) -> None:
        changed = """\
from vllm import LLM

engine = LLM(model=checkpoint, limit_mm_per_prompt=limits_from_config())
"""
        with self.assertRaisesRegex(
            ValueError,
            "classification=unknown.*verify the new source shape",
        ):
            patch_eval_image_cap.classify_cap(changed, 2)


if __name__ == "__main__":
    unittest.main()
