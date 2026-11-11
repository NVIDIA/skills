#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_cfw_evaluate  # noqa: E402
import render_cfw_sft  # noqa: E402
import submit_cfw_evaluate  # noqa: E402
import submit_cfw_inference  # noqa: E402
import submit_cfw_train  # noqa: E402


class FrameworkActionBackendTests(unittest.TestCase):
    def test_sft_profiles_use_direct_hf_then_direct_dcp_resume(self) -> None:
        for profile, expected_steps in (("smoke", 5), ("full", 500)):
            rendered = render_cfw_sft.render_profile(
                profile,
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name=f"iter1-{profile}",
            )
            parsed = render_cfw_sft.tomllib.loads(rendered)
            render_cfw_sft.validate_config(parsed, profile=profile)
            self.assertEqual(parsed["trainer"]["max_iter"], expected_steps)
            self.assertEqual(parsed["checkpoint"]["load_path"], "???")
            self.assertEqual(parsed["custom"]["images_per_record"], 2)
            self.assertEqual(parsed["model"]["attn_implementation"], "cosmos")
            self.assertNotIn("__TAO_CR3_", rendered)

        resumed = render_cfw_sft.tomllib.loads(
            render_cfw_sft.render_profile(
                "full",
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name="iter2-full",
                resume_checkpoint="/tao/iter1/train/checkpoints/iter_000000500",
            )
        )
        self.assertEqual(
            resumed["checkpoint"]["load_path"],
            "/tao/iter1/train/checkpoints/iter_000000500",
        )
        self.assertEqual(resumed["checkpoint"]["keys_to_skip_loading"], [])

        ten_epoch = render_cfw_sft.tomllib.loads(
            render_cfw_sft.render_profile(
                "full",
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name="iter1-full",
                num_epochs=10,
                train_records=37,
            )
        )
        self.assertEqual(ten_epoch["trainer"]["max_iter"], 370)
        self.assertEqual(ten_epoch["checkpoint"]["save_iter"], 37)
        self.assertEqual(ten_epoch["custom"]["num_epochs"], 10)

    def test_e2_attention_contract_splits_train_from_evaluate(self) -> None:
        # Controlled E2: same iter1 data/image scored 0.9545 with cosmos
        # training attention, while sdpa collapsed to all-NG and scored 0.2727.
        for profile in ("smoke", "full"):
            train = render_cfw_sft.tomllib.loads(
                render_cfw_sft.render_profile(
                    profile,
                    model_path="/tao/model",
                    annotation_path="/tao/data/train.json",
                    media_root="/tao/media",
                    run_name=f"e2-{profile}",
                )
            )
            self.assertEqual(train["model"]["attn_implementation"], "cosmos")

        evaluate = render_cfw_evaluate.build_config(
            annotation_path="/tao/data/benchmark.json",
            media_root="/tao/media",
            model_path="/tao/model",
            results_dir="/tao/results/evaluate_benchmark",
        )
        self.assertEqual(evaluate["model"]["attn_implementation"], "sdpa")

    def test_train_submit_invokes_native_entrypoint(self) -> None:
        config = render_cfw_sft.tomllib.loads(
            render_cfw_sft.render_profile(
                "smoke",
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name="iter1-smoke",
            )
        )
        command = submit_cfw_train.build_docker_argv(
            immutable_image="example/framework@sha256:" + "b" * 64,
            job_id="docker-train-test",
            config_host=pathlib.Path("/host/train.toml"),
            model_host=pathlib.Path("/host/model"),
            annotation_host=pathlib.Path("/host/train.json"),
            media_host=pathlib.Path("/host/media"),
            results_dir=pathlib.Path("/host/results"),
            adapter_host=pathlib.Path("/host/cfw_cr3_aoi_adapter.py"),
            config=config,
            gpus="all",
            identity_mounts=[],
            extra_mounts=[],
            resume_mount=None,
            offline=True,
        )
        self.assertIn("/workspace/.venv/bin/cosmos-framework-train", command)
        self.assertIn("--sft-toml=/tao/config/train.toml", command)
        self.assertIn("--user", command)
        self.assertIn("WORLD_SIZE=1", command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("example/framework@sha256:", " ".join(command))

    def test_submit_user_environment_uses_safe_fallback(self) -> None:
        cases = (
            ({"USER": "user-name", "LOGNAME": "login-name"}, "user-name"),
            ({"LOGNAME": "login-name"}, "login-name"),
            ({}, "tao"),
        )
        for environment, expected in cases:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertEqual(submit_cfw_train._username(), expected)
                    self.assertEqual(submit_cfw_evaluate._username(), expected)

    def test_stale_packaged_specs_fail_closed_before_submit(self) -> None:
        stale_evaluate = {
            "results_dir": "/tao-workspace/results/evaluate",
            "dataset": {
                "annotation_path": "/tao-workspace/annotations/benchmark.json",
                "media_dir": "/tao-workspace",
            },
            "model": {
                "model_name": "/tao-workspace/models/base",
                "tokenizer_model_name": "qwen2.5-vl-7b",  # gitleaks:allow (model name, not a credential)
            },
            "generation": {"mode": "generation"},
        }
        with self.assertRaisesRegex(
            ValueError,
            r"stale or foreign Framework evaluate spec: .*render_cfw_evaluate\.py",
        ):
            submit_cfw_evaluate.validate_config(
                stale_evaluate, pathlib.Path("/tao-workspace")
            )

        stale_train = {
            "job": {"task": "vlm", "experiment": "tao_cr3_aoi"},
            "model": {"precision": "bfloat16", "attn_implementation": "cosmos"},
            "custom": {
                "annotation_path": "/tao-workspace/annotations/train.json",
                "media_root": "/tao-workspace",
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            r"stale or foreign Framework train spec: .*render_cfw_sft\.py",
        ):
            submit_cfw_train.validate_submission_config(stale_train)

    def test_existing_target_dcp_is_refused_before_train_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary) / "iter1/train"
            dcp = results / "checkpoints/iter_000000005"
            dcp.mkdir(parents=True)
            (dcp / ".metadata").write_bytes(b"dcp")

            with self.assertRaisesRegex(
                ValueError,
                r"Framework would resume from the capped checkpoint, run zero "
                r"optimizer steps, and abort with zero valid training labels",
            ):
                submit_cfw_train.validate_existing_dcp_output(
                    results,
                    resume_checkpoint=None,
                    allow_existing_dcp=False,
                )

    def test_existing_target_dcp_allows_exact_resume_or_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary) / "iter1/train"
            dcp = results / "checkpoints/iter_000000005"
            dcp.mkdir(parents=True)
            (dcp / ".metadata").write_bytes(b"dcp")

            submit_cfw_train.validate_existing_dcp_output(
                results,
                resume_checkpoint=dcp,
                allow_existing_dcp=False,
            )
            submit_cfw_train.validate_existing_dcp_output(
                results,
                resume_checkpoint=None,
                allow_existing_dcp=True,
            )

    def test_evaluate_render_and_submit_are_single_gpu_safe(self) -> None:
        config = render_cfw_evaluate.build_config(
            annotation_path="/workspace/annotations/benchmark.json",
            media_root="/workspace",
            model_path="/workspace/models/base",
            results_dir="/workspace/results/iter1/evaluate_benchmark",
        )
        submit_cfw_evaluate._require_renderer_contract(config)
        rendered = render_cfw_evaluate.dump_toml(config)
        parsed = render_cfw_evaluate.tomllib.loads(rendered)
        self.assertEqual(parsed["num_gpus"], 1)
        self.assertEqual(parsed["vision"]["video_decoder"], "torchcodec-cuda-on-demand")
        self.assertEqual(parsed["vision"]["dataloader_num_workers"], 1)
        self.assertTrue(parsed["vision"]["dataloader_persistent_workers"])
        self.assertEqual(parsed["model"]["attn_implementation"], "sdpa")
        self.assertFalse(parsed["model"]["enable_lora"])
        self.assertEqual(parsed["generation"]["max_tokens"], 4)

        command = submit_cfw_evaluate.build_docker_argv(
            immutable_image="example/framework@sha256:" + "c" * 64,
            job_id="docker-eval-test",
            config_host=pathlib.Path("/workspace/specs/evaluate.toml"),
            workspace=pathlib.Path("/workspace"),
            results_dir=pathlib.Path("/workspace/results/iter1/evaluate_benchmark"),
            writable_export_parent=pathlib.Path("/workspace/results/iter1/train"),
            gpus="all",
            offline=False,
        )
        self.assertIn("/workspace/.venv/bin/cosmos-framework-evaluate", command)
        self.assertIn("--config", command)
        self.assertIn(
            "type=bind,src=/workspace/results/iter1/train,dst=/workspace/results/iter1/train",
            command,
        )

    def test_inference_submit_uses_native_entrypoint_and_dcp_handoff(self) -> None:
        command = submit_cfw_inference.build_docker_argv(
            immutable_image="example/framework@sha256:" + "d" * 64,
            job_id="docker-infer-test",
            workspace=pathlib.Path("/workspace"),
            results_dir=pathlib.Path("/workspace/results/inference"),
            model_path=pathlib.Path("/workspace/results/iter1/train/checkpoint"),
            media=pathlib.Path("/workspace/images/part.png"),
            prompt="Return exactly OK or NG.",
            media_type="image",
            max_new_tokens=4,
            framework_config=pathlib.Path("/workspace/results/iter1/train/config.yaml"),
            action_model_dir=pathlib.Path("/workspace/results/iter1/train/action_model"),
            base_model_path=pathlib.Path("/workspace/models/base"),
            gpus="all",
            offline=False,
        )
        self.assertIn("/workspace/.venv/bin/cosmos-framework-inference", command)
        for flag in ("--model_path", "--config_file", "--export_dir", "--vit_checkpoint_path"):
            self.assertIn(flag, command)
        self.assertIn("--enable_lora", command)
        self.assertIn("false", command)

    def test_training_annotations_mount_both_ordered_images_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            media = root / "media"
            media.mkdir()
            image = media / "part.png"
            external = root / "golden/reference.png"
            external.parent.mkdir()
            image.write_bytes(b"image")
            external.write_bytes(b"golden")
            annotation = root / "train.json"
            annotation.write_text(
                json.dumps([{"images": [str(image), str(external)], "conversations": []}]),
                encoding="utf-8",
            )
            self.assertEqual(
                submit_cfw_train.annotation_identity_mounts(annotation, media),
                [
                    (media.resolve(), str(media.resolve()), True),
                    (external.parent.resolve(), str(external.parent.resolve()), True),
                ],
            )

    def test_interim_adapter_enforces_two_image_bare_contract(self) -> None:
        source = (SCRIPTS / "cfw_cr3_aoi_adapter.py").read_text(encoding="utf-8")
        self.assertIn("exactly two string image paths", source)
        self.assertIn('{"OK", "NG"}', source)
        self.assertIn("Remove it when the pinned image provides", source)


if __name__ == "__main__":
    unittest.main()
