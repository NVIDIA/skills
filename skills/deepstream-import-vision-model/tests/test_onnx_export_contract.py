# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the SafeTensors -> ONNX export path.

These run without torch, a GPU, or any model download: they check the wrapper script's
CLI contract and the exporter's static behaviour (input-size resolution, architecture
gating), which is what actually breaks when the export path is refactored.

The real export is exercised in-container against the DeepStream image; see
references/model-acquire.md step 2b-iii.

Run:
    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
EXPORT_SH = SKILL_DIR / "scripts" / "model" / "safetensors-to-onnx.sh"
EXPORT_PY = SKILL_DIR / "scripts" / "model" / "safetensors_to_onnx.py"
REQUIREMENTS = SKILL_DIR / "scripts" / "requirements.txt"


def _load_exporter_module():
    """Import safetensors_to_onnx with torch/transformers stubbed.

    Those live only in the container venv. Stubs are installed into sys.modules just long
    enough for the import, then removed, so nothing leaks into other tests.
    """
    import importlib
    import sys
    import types

    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")

    class _Module:  # stand-in base class for DetectionWrapper
        def __init__(self, *args, **kwargs):
            pass

    nn_stub.Module = _Module
    torch_stub.nn = nn_stub
    torch_stub.Tensor = object

    tf_stub = types.ModuleType("transformers")
    tf_stub.AutoConfig = object
    tf_stub.AutoModelForObjectDetection = object

    injected = {"torch": torch_stub, "torch.nn": nn_stub, "transformers": tf_stub}
    saved = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    module_dir = str(EXPORT_PY.parent)
    sys.path.insert(0, module_dir)
    try:
        sys.modules.pop("safetensors_to_onnx", None)
        return importlib.import_module("safetensors_to_onnx")
    finally:
        sys.path.remove(module_dir)
        sys.modules.pop("safetensors_to_onnx", None)
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class TestExportWrapperScript(unittest.TestCase):
    def test_wrapper_exists_and_parses(self):
        self.assertTrue(EXPORT_SH.is_file(), f"missing {EXPORT_SH}")
        rc = subprocess.run(["bash", "-n", str(EXPORT_SH)], capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0, rc.stderr)

    def test_wrapper_rejects_missing_args(self):
        rc = subprocess.run(["bash", str(EXPORT_SH)], capture_output=True, text=True)
        self.assertNotEqual(rc.returncode, 0)
        self.assertIn("Usage:", rc.stdout + rc.stderr)

    def test_wrapper_errors_without_venv(self):
        """Without the container-built venv it must fail with bootstrap guidance, not a traceback."""
        with tempfile.TemporaryDirectory() as td:
            rc = subprocess.run(
                ["bash", str(EXPORT_SH), "some/model", f"{td}/out"],
                capture_output=True, text=True,
            )
            combined = rc.stdout + rc.stderr
            # Either the venv is genuinely absent (expected on a dev box) or it exists and the
            # export proceeds; only the absent case is asserted here.
            if rc.returncode != 0:
                self.assertIn("venv not found", combined)
                self.assertIn("setup.sh", combined)

    def test_wrapper_no_longer_invokes_optimum(self):
        text = EXPORT_SH.read_text()
        self.assertNotIn("optimum-cli export", text)
        self.assertIn("safetensors_to_onnx.py", text)


class TestExporterModule(unittest.TestCase):
    """Static checks on the exporter — no torch import required."""

    @classmethod
    def setUpClass(cls):
        cls.source = EXPORT_PY.read_text()
        cls.tree = ast.parse(cls.source)

    def test_module_parses(self):
        self.assertTrue(EXPORT_PY.is_file(), f"missing {EXPORT_PY}")

    def test_offers_both_export_backends(self):
        """Neither backend handles every architecture, so both must be present.

        dynamo copes with models the tracer chokes on; TorchScript is the one that keeps
        RT-DETR's batch dimension dynamic (dynamo specializes it to the trace batch).
        """
        fns = {n.name for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("export_dynamo", fns)
        self.assertIn("export_torchscript", fns)
        self.assertIn("dynamo=True", self.source)
        self.assertIn("dynamo=False", self.source)

    def test_each_backend_uses_its_own_dynamic_shape_parameter(self):
        """dynamic_shapes belongs to dynamo, dynamic_axes to TorchScript — mixing is a silent no-op."""
        self.assertIn("torch.export.Dim", self.source)
        self.assertIn("dynamic_shapes", self.source)
        self.assertIn("dynamic_axes", self.source)

    def test_falls_back_when_a_backend_specializes_batch(self):
        """A static batch dim must trigger the next backend, not be accepted."""
        self.assertIn("export_torchscript", self.source.split("strategies")[1])
        self.assertIn("static batch dimension", self.source)

    def test_opset_defaults_to_18(self):
        """Opset 17 fails the dynamo downgrade pass: 'No Adapter To Version 17 for Resize'."""
        self.assertRegex(self.source, r'"--opset".*?default=18')

    def test_declares_deepstream_output_contract(self):
        self.assertIn('"pixel_values"', self.source)
        self.assertIn('"logits"', self.source)
        self.assertIn('"pred_boxes"', self.source)

    def test_writes_model_onnx_filename(self):
        """Downstream steps copy models/<name>/onnx_export/model.onnx — keep that name."""
        self.assertIn('"model.onnx"', self.source)

    def test_verifies_dynamic_batch_rather_than_assuming(self):
        """A backend can silently bake in a static batch dim; it must be checked, not assumed."""
        fns = {n.name for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("verify", fns)
        self.assertIn("consolidate_external_data", fns)
        # verify() reports whether the batch axis survived so the caller can switch backends.
        self.assertIn("dim_param", self.source)

    def test_rejects_non_detection_architectures(self):
        fns = {n.name for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("assert_detection_architecture", fns)
        self.assertIn("ForObjectDetection", self.source)

    def test_input_size_resolution_handles_known_preprocessor_shapes(self):
        """Exercise resolve_input_size for real, importing the module with torch stubbed.

        The function is pure stdlib, but the module imports torch/transformers at top level,
        which are container-only. Stubbing sys.modules keeps this runnable on a dev box
        without resorting to dynamic code evaluation, which Tier-1 flags as AST1/AST8.
        """
        resolve = _load_exporter_module().resolve_input_size

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.assertEqual(resolve(str(d), 512), (512, 512), "explicit override wins")
            self.assertEqual(resolve(str(d), None), (640, 640), "no config -> 640 default")

            (d / "preprocessor_config.json").write_text(
                json.dumps({"size": {"height": 800, "width": 1333}}))
            self.assertEqual(resolve(str(d), None), (800, 1333), "height/width honoured")

            (d / "preprocessor_config.json").write_text(
                json.dumps({"size": {"shortest_edge": 800, "longest_edge": 1333}}))
            self.assertEqual(resolve(str(d), None), (800, 800), "shortest_edge -> square")

            (d / "preprocessor_config.json").write_text("{ not valid json")
            self.assertEqual(resolve(str(d), None), (640, 640), "malformed config falls back")


class TestExportDependencies(unittest.TestCase):
    def test_optimum_is_gone(self):
        self.assertNotIn("optimum[exporters]", REQUIREMENTS.read_text())

    def test_onnxscript_pinned_for_dynamo(self):
        """The dynamo ONNX backend needs onnxscript at runtime."""
        self.assertRegex(REQUIREMENTS.read_text(), r"(?m)^onnxscript==")

    def test_transformers_past_the_rce_fixes(self):
        """GHSA-29pf-2h5f-8g72 is fixed in 5.3.0 and GHSA-fgcw-684q-jj6r in 5.5.0."""
        import re
        m = re.search(r"(?m)^transformers==(\d+)\.(\d+)\.(\d+)", REQUIREMENTS.read_text())
        self.assertIsNotNone(m, "transformers must be pinned")
        major, minor = int(m.group(1)), int(m.group(2))
        self.assertGreaterEqual((major, minor), (5, 5),
                                "transformers must be >= 5.5.0 to clear both HIGH RCE advisories")


if __name__ == "__main__":
    unittest.main()
