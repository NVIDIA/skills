#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
for module_name in ("init_deft_state", "metric_contract", "render_report"):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(SCRIPTS))

import init_deft_state  # noqa: E402


class Cosmos3InitStateContractTests(unittest.TestCase):
    @staticmethod
    def _workspace(root: pathlib.Path) -> pathlib.Path:
        workspace = root / "workspace"
        (workspace / "annotations").mkdir(parents=True)
        (workspace / "specs").mkdir()
        for filename in (
            "proxy_kpi.json",
            "benchmark_kpi.json",
            "mining_pool.json",
        ):
            (workspace / "annotations" / filename).write_text(
                "[]\n", encoding="utf-8"
            )
        (workspace / "specs/train_spec.toml").write_text(
            "value = 1\n", encoding="utf-8"
        )
        (workspace / "specs/evaluate_spec.toml").write_text(
            "value = 1\n", encoding="utf-8"
        )
        model = workspace / "models/base"
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            '{"model_type": "qwen3_vl"}\n', encoding="utf-8"
        )
        (model / "model.safetensors").write_bytes(b"weights")
        (model / "tao_conversion_provenance.json").write_text(
            '{"schema_version": 1}\n', encoding="utf-8"
        )
        return workspace

    @staticmethod
    def _argv(
        root: pathlib.Path,
        workspace: pathlib.Path,
        *extra: str,
    ) -> list[str]:
        return [
            "--results-dir",
            str(root / "results"),
            "--workspace",
            str(workspace),
            "--platform",
            "docker",
            "--max-iterations",
            "1",
            "--num-epochs",
            "1",
            "--num-sdg",
            "20",
            "--num-gpus",
            "1",
            "--num-nodes",
            "1",
            "--gpu-model",
            "NVIDIA H100 80GB",
            "--base-model-path",
            str(workspace / "models/base"),
            "--framework-container",
            "example/framework:1",
            "--framework-image-digest",
            "sha256:" + "a" * 64,
            "--mining-container",
            "example/mining:1",
            "--anomalygen-container",
            "example/anomalygen:1",
            *extra,
        ]

    def test_valid_base_model_alias_is_accepted_and_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)

            rc = init_deft_state.main(
                self._argv(root, workspace, "--base-model", "EDGE")
            )

            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            self.assertEqual(state["config"]["base_model"], "nvidia/Cosmos3-Edge")

    def test_repository_digest_is_accepted_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            argv = self._argv(root, workspace)
            digest_index = argv.index("sha256:" + "a" * 64)
            immutable = "example/framework@sha256:" + "a" * 64
            argv[digest_index] = immutable

            self.assertEqual(init_deft_state.main(argv), 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            self.assertEqual(state["config"]["training"]["image_digest"], immutable)
            self.assertEqual(
                state["config"]["containers"]["cosmos_framework"],
                "example/framework:1",
            )

    def test_unknown_base_model_is_rejected_with_allowed_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(
                    self._argv(root, workspace, "--base-model", "bogus-model")
                )

            self.assertEqual(rc, 2)
            self.assertFalse((root / "results/deft_state.json").exists())
            message = stderr.getvalue()
            self.assertIn("unsupported --base-model 'bogus-model'", message)
            for model in init_deft_state.SUPPORTED_BASE_MODELS:
                self.assertIn(model, message)

    def test_native_omni_source_is_rejected_until_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            (workspace / "models/base/config.json").write_text(
                '{"model_type": "cosmos3_omni"}\n', encoding="utf-8"
            )
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(self._argv(root, workspace))

            self.assertEqual(rc, 2)
            self.assertFalse((root / "results/deft_state.json").exists())
            self.assertIn("prepare_cosmos3_vlm_checkpoint.py", stderr.getvalue())
            self.assertIn("model_type=qwen3_vl", stderr.getvalue())

    def test_unprovenanced_qwen_snapshot_is_not_a_prepared_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            (workspace / "models/base/tao_conversion_provenance.json").unlink()
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(self._argv(root, workspace))

            self.assertEqual(rc, 2)
            self.assertFalse((root / "results/deft_state.json").exists())
            self.assertIn("schema-v1 conversion provenance", stderr.getvalue())

    def test_venv_python_symlink_survives_state_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            venv_bin = root / "venv/bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python3").symlink_to(sys.executable)
            venv_python = venv_bin / "python"
            venv_python.symlink_to("python3")

            rc = init_deft_state.main(
                self._argv(
                    root,
                    workspace,
                    "--python-executable",
                    str(venv_python),
                )
            )

            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            self.assertEqual(
                state["execution_policy"]["python_executable"],
                str(venv_python),
            )
            self.assertNotEqual(str(venv_python), str(venv_python.resolve()))


if __name__ == "__main__":
    unittest.main()
