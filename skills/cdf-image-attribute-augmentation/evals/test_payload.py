#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the bundled payload normalizer and results summarizer.

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
PAYLOAD_SCRIPT = SKILL / "scripts/payload.py"
SUMMARY_SCRIPT = SKILL / "scripts/summarize_results.py"
EXAMPLE_PAYLOADS = sorted((REPO / "payloads/dev/image_attribute_augmentation_dag").glob("*.json"))
DISTRIBUTION_ASSET = SKILL / "assets/variable-distribution.json"
sys.path.insert(0, str(SKILL / "scripts"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_module("iaa_skill_payload", PAYLOAD_SCRIPT)
summary = load_module("iaa_skill_summary", SUMMARY_SCRIPT)


def internal_payload(**cosmos):
    return {
        "input_path": "s3://example-bucket/input",
        "output_directory": "s3://example-bucket/output",
        "external_services": False,
        "cosmos": cosmos,
    }


class NormalizePayloadTest(unittest.TestCase):
    def test_example_payloads_round_trip(self):
        self.assertTrue(EXAMPLE_PAYLOADS, "no checked-in IAA payloads found")
        for path in EXAMPLE_PAYLOADS:
            with self.subTest(payload=path.name):
                raw = json.loads(path.read_text(encoding="utf-8"))
                normalized = helper.normalize_payload(copy.deepcopy(raw))
                self.assertEqual(helper.normalize_payload(copy.deepcopy(normalized)), normalized)
                for field in ("checker_overrides", "cosmos_evaluator_service_url"):
                    if field in raw["cosmos"]:
                        self.assertEqual(normalized["cosmos"][field], raw["cosmos"][field])

    def test_normalized_examples_keep_dag_meaning(self):
        sys.path.insert(0, str(REPO / "airflow"))
        try:
            from dags.workflows.image_attribute_augmentation_dag.models.payload import (
                ImageAttributeAugmentationDagPayloadConfig,
            )
        except ImportError as exc:
            self.skipTest(f"DAG payload model is not importable: {exc}")
        finally:
            sys.path.pop(0)

        def dag_view(payload):
            config = ImageAttributeAugmentationDagPayloadConfig.model_validate(
                copy.deepcopy(payload)
            ).model_dump()
            if not config["external_services"]:
                # Internal mode replaces every service URL from service-startup XCom.
                for section in ("cosmos", helper.ATTRIBUTE_SEARCH_KEY):
                    for field in list(config[section]):
                        if (
                            field.endswith("_service_url")
                            and field != "cosmos_evaluator_service_url"
                        ):
                            config[section].pop(field)
            return config

        for path in EXAMPLE_PAYLOADS:
            with self.subTest(payload=path.name):
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    dag_view(helper.normalize_payload(copy.deepcopy(raw))), dag_view(raw)
                )

    def test_omitted_models_use_dag_defaults(self):
        normalized = helper.normalize_payload(internal_payload())
        self.assertEqual(normalized["cosmos"]["vlm_model"], helper.DEFAULT_MODELS["vlm"])
        self.assertEqual(
            normalized["cosmos"]["image_edit_model"], helper.DEFAULT_MODELS["image_edit"]
        )
        self.assertEqual(
            normalized[helper.ATTRIBUTE_SEARCH_KEY]["llm_model"], helper.DEFAULT_MODELS["llm"]
        )

    def test_evaluator_settings_are_preserved(self):
        normalized = helper.normalize_payload(
            internal_payload(
                checker_overrides={"metropolis.attribute_verification": False},
                cosmos_evaluator_service_url="http://arbitrator.example:8080",
            )
        )
        self.assertEqual(
            normalized["cosmos"]["checker_overrides"],
            {"metropolis.attribute_verification": False},
        )
        self.assertEqual(
            normalized["cosmos"]["cosmos_evaluator_service_url"],
            "http://arbitrator.example:8080",
        )

    def test_evaluator_defaults_match_dag(self):
        normalized = helper.normalize_payload(internal_payload())
        self.assertEqual(normalized["cosmos"]["checker_overrides"], {})
        self.assertIsNone(normalized["cosmos"]["cosmos_evaluator_service_url"])

    def test_invalid_checker_overrides_are_rejected(self):
        cases = {
            "unsupported checker": {"metropolis.hallucination": True},
            "non-boolean value": {"metropolis.attribute_verification": "true"},
            "non-object": ["metropolis.attribute_verification"],
        }
        for label, overrides in cases.items():
            with self.subTest(label), self.assertRaises(helper.PayloadError):
                helper.normalize_payload(internal_payload(checker_overrides=overrides))

    def test_unknown_fields_are_rejected(self):
        cases = {
            "top level": ({"extra": 1}, "extra"),
            "cosmos": ({"cosmos": {"num_augmentations": 2}}, "cosmos.num_augmentations"),
            "attribute search": (
                {helper.ATTRIBUTE_SEARCH_KEY: {"threshold": 0.5}},
                f"{helper.ATTRIBUTE_SEARCH_KEY}.threshold",
            ),
            "llm lifecycle": (
                {"service_lifecycle": {"llm_service": {"replicas": 1}}},
                "service_lifecycle.llm_service",
            ),
            "config file": (
                {helper.ATTRIBUTE_SEARCH_KEY: {"config_file": "s3://example-bucket/c.yaml"}},
                f"{helper.ATTRIBUTE_SEARCH_KEY}.config_file",
            ),
        }
        for label, (extra, field_path) in cases.items():
            payload = internal_payload()
            for key, value in extra.items():
                if isinstance(value, dict) and isinstance(payload.get(key), dict):
                    payload[key].update(value)
                else:
                    payload[key] = value
            with self.subTest(label):
                with self.assertRaisesRegex(helper.PayloadError, field_path.replace(".", r"\.")):
                    helper.normalize_payload(payload)

    def test_bundled_distribution_is_accepted(self):
        distribution = json.loads(DISTRIBUTION_ASSET.read_text(encoding="utf-8"))
        normalized = helper.normalize_payload(internal_payload(variable_distribution=distribution))
        self.assertIn(
            "shoe_color", normalized["cosmos"]["variable_distribution"]["conditional_variables"]
        )

    def test_omitted_distribution_uses_defaults(self):
        normalized = helper.normalize_payload(internal_payload())
        self.assertEqual(
            normalized["cosmos"]["variable_distribution"], helper.DEFAULT_VARIABLE_DISTRIBUTION
        )

    def test_partial_distribution_is_rejected(self):
        distribution = json.loads(DISTRIBUTION_ASSET.read_text(encoding="utf-8"))
        distribution["conditional_variables"].pop("shoe_color")
        with self.assertRaisesRegex(helper.PayloadError, "missing: shoe_color"):
            helper.normalize_payload(internal_payload(variable_distribution=distribution))

    def test_lookup_variable_does_not_satisfy_required_attribute(self):
        distribution = copy.deepcopy(helper.DEFAULT_VARIABLE_DISTRIBUTION)
        distribution["variables"]["shoe_color"] = {"white": {"white": 1.0}}
        with self.assertRaisesRegex(helper.PayloadError, "missing: shoe_color"):
            helper.normalize_payload(internal_payload(variable_distribution=distribution))


class RenderPayloadTest(unittest.TestCase):
    def test_render_includes_evaluator_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "payload.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PAYLOAD_SCRIPT),
                    "render",
                    "--input-path",
                    "s3://example-bucket/input",
                    "--output-directory",
                    "s3://example-bucket/output",
                    "--service-mode",
                    "internal",
                    "--cosmos-evaluator-url",
                    "http://arbitrator.example:8080",
                    "--checker-override",
                    "metropolis.attribute_verification=false",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
            )
            cosmos = json.loads(output.read_text(encoding="utf-8"))["cosmos"]
        self.assertEqual(cosmos["checker_overrides"], {"metropolis.attribute_verification": False})
        self.assertEqual(cosmos["cosmos_evaluator_service_url"], "http://arbitrator.example:8080")

    def test_render_rejects_malformed_checker_override(self):
        with self.assertRaises(helper.PayloadError):
            helper._parse_checker_override_args(["metropolis.attribute_verification=off"])


class SummarizeResultsTest(unittest.TestCase):
    def write_manifest(self, root, entries, skipped=None):
        dataset = root / "augmented_dataset"
        dataset.mkdir()
        (dataset / "augmented_data.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_ids": len(entries),
                        "total_scenes": len(entries),
                        "skipped_augmentations": skipped or [],
                    },
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )

    def entry(self, key, attributes):
        return {
            "person_key": key,
            "source_person_key": key.split("_aug")[0],
            "selected_attributes": attributes,
        }

    def test_complete_entries_have_no_quality_warnings(self):
        attributes = {name: "black" for name in summary.REQUIRED_ATTRIBUTES}
        with tempfile.TemporaryDirectory() as tmp:
            self.write_manifest(Path(tmp), [self.entry("p1_aug0", attributes)])
            result = summary.summarize(Path(tmp))
        self.assertEqual(result["quality_warnings"], [])

    def test_empty_and_incomplete_attributes_are_flagged(self):
        partial = {name: "black" for name in summary.REQUIRED_ATTRIBUTES if name != "shoe color"}
        with tempfile.TemporaryDirectory() as tmp:
            self.write_manifest(
                Path(tmp),
                [self.entry("p1_aug0", {}), self.entry("p2_aug0", partial)],
                skipped=[{"person_key": "p3"}],
            )
            result = summary.summarize(Path(tmp))
        self.assertEqual(result["entries_without_attributes"], ["p1_aug0"])
        self.assertEqual(result["entries_with_incomplete_attributes"], {"p2_aug0": ["shoe color"]})
        self.assertEqual(len(result["quality_warnings"]), 3)

    def write_sidecar(self, dataset, key):
        sidecars = dataset / key / "sidecars"
        sidecars.mkdir(parents=True)
        (sidecars / "augmented_data.json").write_text(json.dumps({"entries": []}), encoding="utf-8")

    def test_canonical_manifest_is_selected_over_scene_sidecars(self):
        attributes = {name: "black" for name in summary.REQUIRED_ATTRIBUTES}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root, [self.entry("p1_aug0", attributes)])
            self.write_sidecar(root / "augmented_dataset", "p1_aug0")
            from_run_dir = summary.summarize(root)
            from_dataset_dir = summary.summarize(root / "augmented_dataset")
        self.assertEqual(from_run_dir["manifest"], "augmented_dataset/augmented_data.json")
        self.assertEqual(from_dataset_dir["manifest"], "augmented_data.json")
        self.assertEqual(from_dataset_dir["entries"], 1)

    def test_sidecars_without_canonical_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "augmented_dataset"
            self.write_sidecar(dataset, "p1_aug0")
            with self.assertRaisesRegex(summary.SummaryError, "canonical"):
                summary.summarize(Path(tmp))


if __name__ == "__main__":
    unittest.main()
