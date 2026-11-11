#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline integration checks; run with the product Airflow Python environment.

Set CDF_ORCHESTRATION_ROOT when this skill is not inside the checkout under test.
"""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("CDF_ORCHESTRATION_ROOT") or SKILL.parents[1])
SCRIPT = SKILL / "scripts/payload.py"
spec = importlib.util.spec_from_file_location("dig_skill_payload", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def payload(mode="full"):
    data = {
        "api_version": "1.0",
        "pipeline_mode": mode,
        "output_directory": "s3://example-bucket/output",
        "external_services": False,
    }
    if mode != "defect":
        data.update(
            input_scene="s3://example-bucket/scene.zip",
            golden_generation={"configurations": [{"id": "g1"}, {"id": "g2"}]},
        )
    if mode in {"full", "defect"}:
        data["defect_generation"] = {
            "defaults": {
                "anomaly_input": {"class_by_component_type": {"ic": "IC"}},
                "anomaly_generation": {
                    "per_defect_counts": {"IC+bridge": 4, "IC+missing": 2},
                    "staging": {
                        "fetch": {
                            "input": ["s3://example-bucket/assets/"]
                            + (["s3://example-bucket/clean/"] if mode == "defect" else []),
                            "recipe": "s3://example-bucket/recipe.yaml",
                            "checkpoint": "s3://example-bucket/model.pt",
                        }
                    },
                },
            },
            "configurations": [
                {"id": "bridge", "anomaly_generation": {"per_defect_counts": {"IC+bridge": 1}}},
                {"id": "missing", "anomaly_generation": {"per_defect_counts": {"IC+missing": 3}}},
            ],
        }
    return data


class PayloadTests(unittest.TestCase):
    def test_all_modes_compile_and_scope_output_paths(self):
        for mode, counts in {
            "simulation": (2, 0),
            "sim2real": (2, 0),
            "full": (2, 4),
            "defect": (0, 2),
        }.items():
            with self.subTest(mode=mode):
                plan = helper.build_plan(payload(mode), REPO, "test-run")
                self.assertEqual((plan["simulation_jobs"], plan["anomalygen_jobs"]), counts)
                self.assertEqual(plan["output_directory"], "s3://example-bucket/output/test-run")
                if mode == "defect":
                    self.assertIsNone(plan["defects"][0]["golden_id"])
                    self.assertTrue(
                        plan["defects"][0]["output_directory"].endswith("/defects/bridge")
                    )

    def test_variant_counts_replace_defaults_without_mutating_input(self):
        data = payload()
        original = copy.deepcopy(data)
        plan = helper.build_plan(data, REPO)
        self.assertEqual(plan["defects"][0]["per_defect_counts"], {"IC+bridge": 1})
        self.assertEqual(plan["defects"][1]["per_defect_counts"], {"IC+missing": 3})
        self.assertEqual(data, original)

    def test_defect_needs_no_scene_mapping_or_image_edit(self):
        data = payload("defect")
        data["external_services"] = True
        del data["defect_generation"]["defaults"]["anomaly_input"]
        self.assertEqual(helper.build_plan(data, REPO)["anomalygen_jobs"], 2)

    def test_missing_explicit_staging_cannot_fall_back_to_dev_assets(self):
        data = payload("defect")
        del data["defect_generation"]["defaults"]["anomaly_generation"]["staging"]
        with self.assertRaisesRegex(helper.PayloadError, "staging.fetch.input"):
            helper.build_plan(data, REPO)

    def test_external_placeholder_is_rejected(self):
        data = payload("sim2real")
        data["external_services"] = True
        # Newer payload models reject the missing URL themselves; older ones leave it to the checker.
        with self.assertRaisesRegex(Exception, "image_edit_service_url"):
            helper.build_plan(data, REPO)
        data["cosmos"] = {"image_edit_service_url": "https://image-edit.example.com/v1"}
        self.assertEqual(helper.build_plan(data, REPO)["pipeline_mode"], "sim2real")

    def test_invalid_component_handoff_is_rejected_by_compiler(self):
        data = payload("sim2real")
        data["golden_generation"]["defaults"] = {"simulation_output": {"image": "../escape.png"}}
        with self.assertRaises(Exception):
            helper.build_plan(data, REPO)
        data = payload()
        del data["defect_generation"]["defaults"]["anomaly_input"]
        with self.assertRaises(Exception):
            helper.build_plan(data, REPO)

    def test_unused_override_keys_are_rejected(self):
        cases = {
            "image_genaration": lambda d: d["golden_generation"]
            .setdefault("defaults", {})
            .update(image_genaration={"capture": {}}),
            "anomaly_generation": lambda d: d["golden_generation"]["configurations"][0].update(
                anomaly_generation={"num_sdg": 2}
            ),
            "vlm_service_url": lambda d: d.update(
                cosmos={"vlm_service_url": "https://vlm.example.com/v1"}
            ),
            "replica": lambda d: d.update(service_lifecycle={"image_edit_service": {"replica": 4}}),
        }
        for key, mutate in cases.items():
            with self.subTest(key=key):
                data = payload()
                mutate(data)
                with self.assertRaisesRegex(helper.PayloadError, key):
                    helper.build_plan(data, REPO)

    def test_workflow_managed_keys_are_rejected(self):
        cases = {
            "input_scene": (
                "golden_generation",
                "image_generation",
                {"scene": {"usd": "s3://example-bucket/other.zip"}},
            ),
            "per image": ("golden_generation", "augmentation", {"data": []}),
            "image_edit entry": (
                "golden_generation",
                "augmentation",
                {"endpoints": [{"id": "vlm", "role": "vlm"}]},
            ),
            "staging.fetch.checkpoint": (
                "defect_generation",
                "anomaly_generation",
                {"checkpoint": "s3://example-bucket/m.pt"},
            ),
            "must keep": (
                "defect_generation",
                "anomaly_generation",
                {"stages": ["prepare", "amg", "generate", "export"]},
            ),
        }
        for message, (collection, component, override) in cases.items():
            with self.subTest(message=message):
                data = payload()
                data[collection]["configurations"][0][component] = override
                with self.assertRaisesRegex(helper.PayloadError, message):
                    helper.build_plan(data, REPO)

    def test_supported_component_overrides_compile(self):
        data = payload()
        data["golden_generation"]["configurations"][0]["augmentation"] = {
            "captioning": {
                "llm": {"text": "Render this PCB component as a soft-lit inspection photograph."}
            },
            "augmentation": {"parameters": {"extra_body": {"seed": 7}}},
            "endpoints": [
                {
                    "id": "image_edit",
                    "role": "image_edit",
                    "adapter": "openai.chat.completions",
                    "timeout": 600,
                }
            ],
        }
        data["defect_generation"]["defaults"]["anomaly_generation"]["base_seed"] = 3
        data["service_lifecycle"] = {"image_edit_service": {"replicas": 2}}
        plan = helper.build_plan(data, REPO)
        self.assertEqual(plan["anomalygen_jobs"], 4)
        self.assertEqual(
            plan["defects"][0]["staging_fetch"]["recipe"], "s3://example-bucket/recipe.yaml"
        )

    def test_pose_defect_override_compiles_without_defects(self):
        data = payload("sim2real")
        selector = [{"defects": ["tombstone"], "padding": 0.5}]
        data["golden_generation"]["configurations"] = [
            {
                "id": "tombstone",
                "image_generation": {
                    "operations": {"tombstone": {"ratio": 1.0, "angle": [70, 90]}},
                    "annotations": {"operation_labels": "defect"},
                    "capture": {
                        "passes": {
                            "component_focus": {"roi_selection": selector},
                            "component_focus_roi": {"roi_selection": selector},
                        }
                    },
                },
            }
        ]
        plan = helper.build_plan(data, REPO, "test-run")
        self.assertEqual((plan["simulation_jobs"], plan["anomalygen_jobs"]), (1, 0))
        self.assertEqual(plan["goldens"][0]["golden_id"], "tombstone")

    def test_invalid_collection_is_rejected_by_model(self):
        data = payload("simulation")
        data["golden_generation"]["configurations"] = [{"id": "same"}, {"id": "same"}]
        with self.assertRaises(Exception):
            helper.build_plan(data, REPO)

    def test_cli_json_and_error_exit_without_echoing_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "payload.json"
            command = [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(REPO),
                "--payload",
                str(path),
            ]
            path.write_text(json.dumps(payload("simulation")))
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["simulation_jobs"], 2)
            invalid = payload("simulation")
            invalid["unexpected"] = "sensitive-test-marker"
            path.write_text(json.dumps(invalid))
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("sensitive-test-marker", result.stdout + result.stderr)
            self.assertIn("unexpected", result.stderr)


if __name__ == "__main__":
    unittest.main()
