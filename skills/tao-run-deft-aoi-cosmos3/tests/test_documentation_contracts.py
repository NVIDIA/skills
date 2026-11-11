#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for campaign-validated Cosmos3 DEFT documentation contracts."""

from __future__ import annotations

import pathlib
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]


def words(path: pathlib.Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").replace("\\\n", " ").split())


class DocumentationContractTests(unittest.TestCase):
    def test_install_layout_and_python_toml_compatibility(self) -> None:
        skill = words(SKILL_ROOT / "SKILL.md")
        preflight = words(SKILL_ROOT / "references/preflight.md")
        for text in (skill, preflight):
            for entry in ("skills/", "scripts/", "templates/", "versions.yaml"):
                self.assertIn(entry, text)
            self.assertIn("TAO_SKILL_BANK_PATH", text)
            self.assertIn("Any install that ships only the skill folders", text)
            self.assertNotIn(".claude", text)
            self.assertIn("scripts/resolve_tao_model.py", text)
        self.assertIn("Python 3.11+ `tomllib`", skill)
        self.assertIn("Python 3.10 with the `tomli` fallback", skill)
        self.assertIn("Blocker recovery (Python 3.10)", preflight)
        self.assertIn("skill-local `.venv`", preflight)
        self.assertIn("staged-set requirement, not an install", preflight)

    def test_preflight_uses_owned_image_resolvers(self) -> None:
        path = SKILL_ROOT / "references/preflight.md"
        source = path.read_text(encoding="utf-8")
        text = words(path)
        self.assertIn(
            'COSMOS_MODEL_ID="${COSMOS_MODEL_ID:-nvidia/Cosmos3-Nano}"',
            source,
        )
        self.assertIn("scripts/resolve_tao_image.py", text)
        self.assertIn("--action train", text)
        self.assertIn("--backend cosmos-framework", text)
        self.assertEqual(source.count("--backend cosmos-rl"), 1)
        self.assertIn("fail-closed negative signature", text)
        self.assertIn("model skill predates PR 230", text)
        for key in (
            "images.tao_toolkit.data_services",
            "images.metropolis_sdg.paidf_anomalygen",
        ):
            self.assertIn(key, source)
        self.assertGreaterEqual(source.count("resolve_versions_key.py"), 2)
        self.assertFalse(
            any(line.lstrip().startswith("docker run") for line in source.splitlines())
        )

    def test_shared_planner_uses_real_resolve_interface(self) -> None:
        for relative in ("SKILL.md", "references/preflight.md"):
            text = words(SKILL_ROOT / relative)
            self.assertIn("cosmos_workflow.py", text)
            self.assertIn("resolve", text)
            self.assertIn("--backend cosmos-framework", text)
            self.assertIn("--workload training", text)
            self.assertIn("resolve_tao_model.py", text)

    def test_main_checkpoint_preparation_contract_feeds_framework(self) -> None:
        skill = words(SKILL_ROOT / "SKILL.md")
        preflight = words(SKILL_ROOT / "references/preflight.md")
        data_layout = words(SKILL_ROOT / "references/data-layout.md")
        cosmos_reason = words(SKILL_ROOT / "references/cosmos-reason.md")
        pipeline = words(SKILL_ROOT / "references/pipeline-and-state.md")
        card = words(SKILL_ROOT / "skill-card.md")
        state = (SKILL_ROOT / "references/deft_state.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("PR 230 model helper `--backend`", skill)
        for text in (skill, preflight, data_layout, cosmos_reason):
            self.assertIn("prepare_cosmos3_vlm_checkpoint.py", text)
            self.assertIn("Qwen3-VL", text)
        self.assertIn("Qwen3-VL safetensors PTM", card)
        self.assertIn("Cosmos3-Nano-VLM", state)
        self.assertIn("--runtime-image", cosmos_reason)
        self.assertIn("--backend cosmos-framework", cosmos_reason)
        self.assertIn("cosmos-framework-train", cosmos_reason)
        self.assertIn(
            "The helper's `--runtime-image` and `--runtime-image-digest` must "
            "identify the model skill's resolved Framework image. The same "
            "immutable image runs checkpoint preparation, Train, Evaluate, "
            "and Inference",
            cosmos_reason,
        )
        for text in (skill, preflight, data_layout, cosmos_reason):
            self.assertNotIn("preparation-only", text)
            self.assertNotIn("Cosmos-RL backend image", text)
        self.assertIn(
            "After launch approval and before baseline evaluation", skill
        )
        self.assertIn(
            "selected canonical ID as source-model lineage, but do not pass "
            "the native online checkpoint directly to Framework",
            skill,
        )
        self.assertIn(
            "to convert the selected reasoner into a Qwen3-VL safetensors PTM, "
            "or validate and reuse an existing prepared output",
            skill,
        )
        self.assertIn(
            "iteration Evaluate sets `model.config_file` to that `config.yaml`, "
            "never the input SFT TOML",
            skill,
        )
        self.assertIn(
            "`model.config_file`: Train's saved Hydra `config.yaml` beside "
            "that DCP, never the input SFT TOML",
            cosmos_reason,
        )
        self.assertNotIn("saved Train TOML", cosmos_reason)
        self.assertNotIn("saved training TOML", pipeline)
        self.assertIn("Train's saved Hydra `config.yaml`", pipeline)
        self.assertNotIn("saved TOML and DCP", skill)
        self.assertIn(
            "The model being trained is still the selected Cosmos Reason 3 "
            "reasoner", skill
        )
        self.assertIn(
            "the Qwen3-VL PTM is only the on-disk format Framework consumes",
            skill,
        )
        self.assertIn(
            "Nano may use the helper's packaged Qwen3-VL default", skill
        )
        self.assertIn(
            "Edge and Super require a variant-specific, validated VLM base; "
            "never reuse Nano's conversion arguments",
            skill,
        )
        self.assertIn(
            "prove the selected Framework image can load the prepared PTM and "
            "train the selected variant",
            preflight,
        )
        self.assertIn(
            "the id is the locator, the reasoner is the model", preflight
        )
        for model_id in (
            "nvidia/Cosmos3-Nano",
            "nvidia/Cosmos3-Edge",
            "nvidia/Cosmos3-Super",
        ):
            self.assertIn(model_id, preflight)
        self.assertIn("report insufficient resources", preflight)
        self.assertIn("obtain an explicit variant choice", preflight)
        self.assertIn("never silently change or fall back", preflight)
        self.assertIn("prepared checkpoint host/compute-frame paths", preflight)
        self.assertIn("the variant-matched VLM base", preflight)
        self.assertIn(
            "After approval, prepare and validate the Qwen3-VL checkpoint",
            preflight,
        )

    def test_attention_split_is_normative_not_an_experiment_log(self) -> None:
        paths = (
            "SKILL.md",
            "references/preflight.md",
            "references/cosmos-reason.md",
            "eval.config",
        )
        texts = [words(SKILL_ROOT / path) for path in paths]
        for text in texts:
            self.assertNotIn("experiment E2", text)
            self.assertNotIn("E2 avoids", text)
            self.assertNotIn("0.2727", text)
            self.assertNotIn("0.9545", text)
        self.assertIn(
            "Train keeps `model.attn_implementation=\"cosmos\"` because `sdpa` "
            "collapses Framework LoRA SFT to the majority label on this image",
            texts[0],
        )
        self.assertIn(
            "Evaluate uses `sdpa` because its Hugging Face loader rejects "
            "`cosmos`",
            texts[0],
        )

    def test_training_sources_and_rcca_actions_remain_complete(self) -> None:
        skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        scripts = words(SKILL_ROOT / "references/scripts-and-agents.md")
        rcca = (SKILL_ROOT / "references/RCCA_REPORT_TEMPLATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("mine real image pairs from Proxy gaps", skill)
        self.assertIn("generate AnomalyGen synthetic NG", skill)
        self.assertIn("Enforce two-image, exact bare OK/NG ShareGPT records", scripts)
        self.assertIn("Monotonic bare training-data merge", scripts)
        self.assertIn("| Defect type | Count | Share |", rcca)
        self.assertIn("mining targets and SDG defect/count targets", rcca)

    def test_validation_report_and_split_summary_remain_distinct(self) -> None:
        for relative in ("references/aoi-annotation.md", "references/preflight.md"):
            text = (SKILL_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("validation_report.json", text)
            self.assertIn("split_contract_summary.json", text)
            self.assertIn("validate_sharegpt.py", text)
            self.assertIn("validate_split_contract.py", text)
            self.assertIn("require_files=true", text)
            self.assertIn("unique_target_images", text)
        preflight = (SKILL_ROOT / "references/preflight.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Commit only `validation_report.json`", preflight)

    def test_required_state_train_and_split_flags_are_documented(self) -> None:
        skill = words(SKILL_ROOT / "SKILL.md")
        preflight = words(SKILL_ROOT / "references/preflight.md")
        pipeline = words(SKILL_ROOT / "references/pipeline-and-state.md")
        annotation = words(SKILL_ROOT / "references/aoi-annotation.md")
        for flag in (
            "--base-model-path",
            "--framework-container",
            "--framework-image-digest",
        ):
            self.assertIn(flag, skill)
            self.assertIn(flag, preflight)
        self.assertIn("--framework-config", skill)
        self.assertIn("--framework-config", pipeline)
        for text in (preflight, pipeline, annotation):
            self.assertIn("--synthetic", text)
            self.assertIn("RESULTS_DIR/manifests/benchmark_manifest.json", text)
        self.assertIn(
            '--workspace "$WORKSPACE" --train "$TRAIN_JSON" '
            '--synthetic "$SYNTHETIC_JSON" '
            '--manifest "$RESULTS_DIR/manifests/benchmark_manifest.json"',
            preflight,
        )
        self.assertIn(
            '--train "$RESULTS_DIR/$LABEL/assemble/train_iter_${ITERATION}.json" '
            '--manifest "$RESULTS_DIR/manifests/benchmark_manifest.json"',
            annotation,
        )
        self.assertIn("`version == 6`, non-empty `final_artifacts`", skill)

    def test_fail_closed_stale_spec_and_credential_contracts(self) -> None:
        skill = words(SKILL_ROOT / "SKILL.md")
        preflight = words(SKILL_ROOT / "references/preflight.md")
        airgap = words(SKILL_ROOT / "references/air-gap.md")
        scripts = words(SKILL_ROOT / "references/scripts-and-agents.md")
        for text in (skill, preflight):
            self.assertIn("stale", text)
            self.assertIn("render_cfw_sft.py", text)
            self.assertIn("render_cfw_evaluate.py", text)
        self.assertIn("the harness reports restricted networking", airgap)
        self.assertIn("AnomalyGen images", airgap)
        self.assertIn("Guardrail safety model", airgap)
        self.assertIn("both `HF_TOKEN` and its legacy alias", airgap)
        self.assertIn("Check credential presence only", skill)
        self.assertIn("Check credential presence only", scripts)


if __name__ == "__main__":
    unittest.main()
