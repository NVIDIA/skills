# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jsonschema
import nibabel as nib
import numpy as np

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "scripts" / "export_evidence_pack.py"
SCHEMA = json.loads((SKILL_DIR / "validators" / "output_schema.json").read_text())
spec = importlib.util.spec_from_file_location("export_evidence_pack", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_result(path: Path, image: Path | None = None, mask: Path | None = None) -> None:
    output: dict = {
        "samples": [
            {
                "image_path": str(image or "/private/patient/image.nii.gz"),
                "label_path": str(mask or "/private/patient/mask.nii.gz"),
                "image_hu_min": -1000.0,
                "image_hu_max": 500.0,
                "image_hu_mean": -250.0,
                "image_hu_std": 125.0,
            }
        ],
        "output_label_mapping": [
            {"anatomy": "lung tumor", "output_label_id": 1, "maisi_label_id": 23}
        ],
    }
    path.write_text(
        json.dumps(
            {
                "skill": "nv_generate_ct_rflow",
                "version": "0.1.0",
                "input": {
                    "random_seed": 7,
                    "output_size_requested": [8, 8, 8],
                    "prompt": "private prompt text",
                    "patient_id": "PRIVATE-123",
                    "config_infer_override_path": "/private/patient/config.json",
                },
                "output": output,
                "metrics": {"generation_time_s": 2.5, "ok": True},
                "invocation": {
                    "upstream_commit": "abcdef1234567",
                    "access_token": "never-log-me",
                    "model_inventory": {
                        "files": [
                            {
                                "path": "/models/rflow.pt",
                                "sha256": "a" * 64,
                            }
                        ]
                    },
                },
            }
        )
    )


def _write_nifti_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_data = np.linspace(-1000.0, 1000.0, 8**3, dtype=np.float32).reshape(8, 8, 8)
    mask_data = np.zeros((8, 8, 8), dtype=np.uint8)
    mask_data[2:6, 2:6, 2:6] = 1
    image = tmp_path / "synthetic_image.nii.gz"
    mask = tmp_path / "synthetic_mask.nii.gz"
    nib.save(nib.Nifti1Image(image_data, np.eye(4)), image)
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask)
    return image, mask


def test_pack_extracts_reproducibility_metrics_and_metadata() -> None:
    pack = SKILL_DIR / "fixtures" / "sample_pack"
    summary = mod.collect_summary(pack)

    assert summary["source"]["skill_id"] == "medagent.nv_generate_ct_rflow"
    assert summary["params"]["reproducibility.seed"] == "17"
    assert summary["params"]["model.checkpoint_0.sha256"].startswith("93f065")
    assert summary["metrics"]["quality.sample_0.hu_std"] == 327.4
    assert summary["metrics"]["tumor_volume_pct"] == 4.2
    assert summary["artifact_plan"]["metadata_count"] >= 5
    assert "mlflow.note.content" in summary["tags"]
    assert "medical_ai_skills.recipe_sha256" in summary["tags"]


def test_direct_result_redacts_paths_secrets_and_prompt(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)

    summary = mod.collect_summary(result)
    serialized = json.dumps(summary["logged_summary"])
    document = json.dumps(summary["_documents"])

    assert "/private/patient" not in serialized
    assert "/private/patient" not in document
    assert "never-log-me" not in serialized
    assert "never-log-me" not in document
    assert "private prompt text" not in serialized
    assert "PRIVATE-123" not in serialized
    assert summary["tags"]["medical_ai_skills.source_prompt_sha256"]
    assert summary["tags"]["medical_ai_skills.reproducibility_seed"] == "7"


def test_nifti_quality_and_artifact_plan(tmp_path: Path) -> None:
    image, mask = _write_nifti_pair(tmp_path)
    result = tmp_path / "result.json"
    _write_result(result, image, mask)

    summary = mod.collect_summary(result, artifact_policy="preview")

    assert summary["metrics"]["quality.sample_0.hu_min"] == -1000.0
    assert summary["metrics"]["quality.sample_0.hu_max"] == 1000.0
    assert summary["metrics"]["quality.sample_0.hu_std"] > 0
    assert summary["metrics"]["quality.sample_0.snr_abs_mean_over_std"] >= 0
    assert summary["metrics"]["quality.sample_0.mask_foreground_pct"] == 12.5
    assert summary["metrics"]["quality.sample_0.lung_tumor_volume_pct"] == 12.5
    assert summary["artifact_plan"]["preview_count"] == 1
    assert all(item["status"] == "not_selected" for item in summary["artifact_plan"]["items"])


def test_log_summary_logs_params_notes_metadata_preview_and_raw(tmp_path: Path) -> None:
    image, mask = _write_nifti_pair(tmp_path)
    result = tmp_path / "result.json"
    _write_result(result, image, mask)
    summary = mod.collect_summary(result, artifact_policy="all")

    class ActiveRun:
        info = type("Info", (), {"run_id": "fake-run"})()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeMlflow:
        def __init__(self):
            self.dicts = []
            self.artifacts = []
            self.images = []

        def set_tracking_uri(self, uri):
            self.uri = uri

        def set_experiment(self, name):
            self.experiment = name

        def start_run(self, run_name=None):
            self.run_name = run_name
            return ActiveRun()

        def set_tags(self, tags):
            self.tags = tags

        def log_params(self, params):
            self.params = params

        def log_metrics(self, metrics):
            self.metrics = metrics

        def log_dict(self, payload, path):
            self.dicts.append((path, payload))

        def log_image(self, image, artifact_file):
            self.images.append((artifact_file, image.shape))

        def log_artifact(self, path, artifact_path=None):
            self.artifacts.append((Path(path).name, artifact_path))

    fake = FakeMlflow()
    logged = mod.log_summary(
        summary,
        mode="local",
        tracking_uri="file:///tmp/mlruns",
        experiment_name="test",
        run_name="test-run",
        mlflow_module=fake,
    )

    assert logged["run_id"] == "fake-run"
    assert fake.params["reproducibility.seed"] == "7"
    assert fake.tags["mlflow.note.content"].startswith("Post-hoc MLflow")
    assert fake.images[0][0] == "medical_ai_skills/previews/sample_0.png"
    assert (image.name, "medical_ai_skills/raw") in fake.artifacts
    assert (mask.name, "medical_ai_skills/raw") in fake.artifacts
    assert fake.dicts[0][0] == "medical_ai_skills/evidence_summary.json"
    assert len(fake.dicts) >= 2


def test_main_defaults_to_schema_valid_dry_run(capsys) -> None:
    pack = SKILL_DIR / "fixtures" / "sample_pack"
    return_code = mod.main([str(pack)])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["status"] == "dry_run"
    assert payload["mode"] == "dry-run"
    assert payload["artifact_plan"]["policy"] == "metadata"
    assert payload["mlflow"]["run_id"] is None
    jsonschema.Draft202012Validator(SCHEMA).validate(payload)


def test_live_preview_requires_explicit_confirmation(tmp_path: Path, capsys) -> None:
    result = tmp_path / "result.json"
    _write_result(result)

    return_code = mod.main([str(result), "--mode", "local", "--artifact-policy", "preview"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 2
    assert payload["status"] == "failed"
    assert "--confirm-medical-artifact-upload" in payload["mlflow"]["error"]


def test_tracking_uri_honors_environment(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")

    assert mod._tracking_uri("local", None) == "https://mlflow.example.test"


def test_artifact_size_limit_is_reported(tmp_path: Path) -> None:
    image = tmp_path / "large_image.nii.gz"
    image.write_bytes(b"x" * 32)
    result = tmp_path / "result.json"
    _write_result(result, image, None)

    summary = mod.collect_summary(result, artifact_policy="all", max_artifact_mb=0.000001)

    image_item = next(
        item for item in summary["artifact_plan"]["items"] if item["name"] == image.name
    )
    assert image_item["status"] == "over_size_limit"
