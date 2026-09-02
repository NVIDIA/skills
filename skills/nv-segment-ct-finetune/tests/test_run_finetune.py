# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_finetune.py"
spec = importlib.util.spec_from_file_location("run_finetune", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_prepare_bundle_files_stages_train_configs_from_local_upstream(tmp_path, monkeypatch):
    bundle = tmp_path / "skill" / "bundle"
    upstream_configs = (
        tmp_path / ".workbench_data" / "upstreams" / "NV-Segment-CTMR" / "NV-Segment-CT" / "configs"
    )
    upstream_configs.mkdir(parents=True)
    for name in (
        "train.json",
        "train_continual.json",
        "multi_gpu_train.json",
        "evaluate.json",
    ):
        (upstream_configs / name).write_text(f'{{"name": "{name}"}}\n')
    (bundle / "configs").mkdir(parents=True)
    (bundle / "metadata.json").write_text("{}\n")
    (bundle / "vista3d_pretrained_model").mkdir(parents=True)
    (bundle / "vista3d_pretrained_model" / "model.pt").write_bytes(b"model")
    (bundle / "label_dict.json").write_text('{"lung tumor": 23}\n')

    monkeypatch.setattr(mod, "BUNDLE_DIR", bundle)
    monkeypatch.setattr(mod, "SKILL_DIR", tmp_path / "skill")
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "LABEL_DICT", bundle / "label_dict.json")
    notes = mod.prepare_bundle_files()

    for name in (
        "train.json",
        "train_continual.json",
        "multi_gpu_train.json",
        "evaluate.json",
    ):
        assert (bundle / "configs" / name).read_text() == f'{{"name": "{name}"}}\n'
    assert (bundle / "configs" / "metadata.json").is_file()
    assert (bundle / "models" / "model.pt").is_file()
    assert "restored configs/train.json from local upstream cache" in notes


def test_prepare_bundle_files_restores_drifted_train_configs(tmp_path, monkeypatch):
    bundle = tmp_path / "skill" / "bundle"
    upstream_configs = (
        tmp_path / ".workbench_data" / "upstreams" / "NV-Segment-CTMR" / "NV-Segment-CT" / "configs"
    )
    upstream_configs.mkdir(parents=True)
    for name in (
        "train.json",
        "train_continual.json",
        "multi_gpu_train.json",
        "evaluate.json",
    ):
        (upstream_configs / name).write_text(f'{{"canonical": "{name}"}}\n')
    (bundle / "configs").mkdir(parents=True)
    for name in (
        "train.json",
        "train_continual.json",
        "multi_gpu_train.json",
        "evaluate.json",
    ):
        (bundle / "configs" / name).write_text(f'{{"drifted": "{name}"}}\n')
    (bundle / "metadata.json").write_text("{}\n")
    (bundle / "vista3d_pretrained_model").mkdir(parents=True)
    (bundle / "vista3d_pretrained_model" / "model.pt").write_bytes(b"model")
    (bundle / "label_dict.json").write_text('{"lung tumor": 23}\n')

    monkeypatch.setattr(mod, "BUNDLE_DIR", bundle)
    monkeypatch.setattr(mod, "SKILL_DIR", tmp_path / "skill")
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "LABEL_DICT", bundle / "label_dict.json")
    notes = mod.prepare_bundle_files()

    assert (bundle / "configs" / "evaluate.json").read_text() == '{"canonical": "evaluate.json"}\n'
    assert "restored configs/evaluate.json from local upstream cache" in notes


def test_upstream_config_dirs_accept_explicit_local_checkout(tmp_path, monkeypatch):
    config_dir = (
        tmp_path / ".workbench_data" / "upstreams" / "NV-Segment-CTMR" / "NV-Segment-CT" / "configs"
    )
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)

    assert config_dir in mod._upstream_config_dirs()


def test_resolve_softmax_bundle_root_accepts_pinned_layout(tmp_path, monkeypatch):
    bundle_root = tmp_path / "NV-Segment-CT"
    (bundle_root / "configs").mkdir(parents=True)
    (bundle_root / "scripts").mkdir()
    for name in ("train_continual_softmax.json", "inference_softmax.json"):
        (bundle_root / "configs" / name).write_text("{}\n")
    (bundle_root / "scripts" / "vista3d_softmax.py").write_text("class Vista3dSoftmax: pass\n")

    monkeypatch.setenv("NV_SEGMENT_CT_ROOT", str(bundle_root))
    monkeypatch.delenv("NV_SEGMENT_CTMR_ROOT", raising=False)

    assert mod.resolve_softmax_bundle_root() == bundle_root


def test_build_override_defines_bundle_image_and_label_keys(tmp_path):
    override = mod.build_override(
        tmp_path / "dataset",
        tmp_path / "datalist.json",
        {"default": [[1, 3]]},
        [64, 64, 64],
        1.0,
        2,
        5e-5,
        tmp_path / "checkpoints",
        tmp_path / "val_during_train",
    )

    assert override["image_key"] == "image"
    assert override["label_key"] == "label"


def test_build_override_auto_seg_matches_task06_prompt_settings(tmp_path):
    override = mod.build_override(
        tmp_path / "dataset",
        tmp_path / "datalist.json",
        {"default": [[1, 23]]},
        [128, 128, 128],
        1.0,
        5,
        5e-5,
        tmp_path / "checkpoints",
        tmp_path / "val_during_train",
        auto_seg=True,
    )

    assert override["drop_label_prob"] == 0.0
    assert override["drop_point_prob"] == 1.0
    expected_spacing = tuple(float("1.5") for _ in range(3))
    assert override["resample_to_spacing"] == expected_spacing


def test_build_softmax_override_uses_fixed_channel_contract(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    monkeypatch.setattr(mod, "BUNDLE_DIR", bundle)

    override = mod.build_softmax_override(
        tmp_path / "dataset",
        tmp_path / "datalist.json",
        {"default": [[1, 3], [2, 13]]},
        [128, 128, 128],
        0.5,
        10,
        1e-4,
        tmp_path / "checkpoints",
        tmp_path / "validation",
    )

    assert override["label_mappings"] == {"default": [[1, 3], [2, 13]]}
    assert override["finetune_model_path"] == str(bundle / "models" / "model.pt")
    assert override["ckpt_dir"] == str(tmp_path / "checkpoints")
    assert override["patch_size"] == [128, 128, 128]
    assert "drop_label_prob" not in override
    assert "drop_point_prob" not in override


def test_validate_softmax_mapping_accepts_unique_positive_pairs():
    mod.validate_softmax_mapping({"default": [[1, 3], [2, 13]]})


def test_child_process_env_keeps_runtime_values_and_drops_credentials(monkeypatch):
    monkeypatch.setenv("PATH", "/trusted/bin")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("OPENAI_API_KEY", "not-forwarded")
    monkeypatch.setenv("DATABRICKS_TOKEN", "tracking-only")

    env = mod._child_process_env({"NVSEG_FINETUNE_IN_AUTO_VENV": "1"})

    assert env["PATH"] == "/trusted/bin"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["NVSEG_FINETUNE_IN_AUTO_VENV"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "DATABRICKS_TOKEN" not in env

    tracking_env = mod._child_process_env(extra_keys=mod._MLFLOW_CHILD_ENV_KEYS)
    assert tracking_env["DATABRICKS_TOKEN"] == "tracking-only"
    assert "OPENAI_API_KEY" not in tracking_env


@pytest.mark.parametrize(
    "mapping",
    [
        {"default": []},
        {"default": [[0, 3]]},
        {"default": [[1, 3], [1, 13]]},
        {"default": [[1, 3], [2, 3]]},
    ],
)
def test_validate_softmax_mapping_rejects_invalid_channels(mapping):
    with pytest.raises(mod.typer.BadParameter, match="softmax"):
        mod.validate_softmax_mapping(mapping)


def test_task06_fixture_selects_sanity_preset() -> None:
    assert mod._fixture_preset(Path("/data/Task06")) == "sanity"
    assert mod._fixture_preset(Path("/data/Task06_Lung")) == "sanity"
    assert mod._fixture_preset(Path("/data/spleen_micro")) == "smoke"


def test_sanity_dataset_prefers_explicit_paths(tmp_path):
    fixture = tmp_path / "Task06"
    explicit = tmp_path / "explicit_task06"
    fixture.mkdir()
    explicit.mkdir()

    assert mod._resolve_sanity_dataset(fixture, None) == fixture.resolve()
    assert mod._resolve_sanity_dataset(fixture, explicit) == explicit.resolve()


def test_ensure_smoke_dataset_materializes_missing_niftis(tmp_path):
    dataset = tmp_path / "spleen_micro"
    dataset.mkdir()
    datalist = dataset / "datalist.json"
    datalist.write_text("""
{
  "training": [
    {"image": "imagesTr/spleen_00.nii.gz", "label": "labelsTr/spleen_00.nii.gz", "fold": 0},
    {"image": "imagesTr/spleen_01.nii.gz", "label": "labelsTr/spleen_01.nii.gz", "fold": 1}
  ],
  "testing": []
}
""")

    smoke_dir, smoke_datalist, generated = mod.ensure_smoke_dataset(
        dataset, datalist, tmp_path / "run"
    )

    assert generated is True
    assert smoke_datalist == smoke_dir / "datalist.json"
    assert (smoke_dir / "imagesTr" / "spleen_00.nii.gz").is_file()
    assert (smoke_dir / "labelsTr" / "spleen_01.nii.gz").is_file()


def test_metric_compat_config_stack_skips_when_mean_dice_accepts_num_classes(
    monkeypatch,
):
    monkeypatch.setattr(mod, "_mean_dice_accepts_num_classes", lambda: True)

    assert mod.metric_compat_config_stack() == []


def test_metric_compat_config_stack_writes_only_when_needed(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    monkeypatch.setattr(mod, "BUNDLE_DIR", bundle)
    monkeypatch.setattr(mod, "_mean_dice_accepts_num_classes", lambda: False)

    stack = mod.metric_compat_config_stack()

    assert stack == ["configs/mean_dice_no_num_classes.json"]
    payload = (bundle / "configs" / "mean_dice_no_num_classes.json").read_text()
    assert '"num_classes"' not in payload


def test_sanity_reference_checks_fail_low_recovery_run():
    checks = mod.sanity_reference_checks(
        formal_pretrained=0.6258574724197388,
        formal_finetuned=0.6258574724197388,
        formal_improvement=0.0,
        training_start=0.6326,
        training_best=0.6326,
        training_improvement=0.0,
        best_checkpoint_changed=False,
        overall_rc=0,
    )

    assert checks["passed"] is False
    assert "formal_pretrained_val_dice_ok" in checks["failed_checks"]
    assert "formal_improvement_ok" in checks["failed_checks"]
    assert "training_best_val_dice_ok" in checks["failed_checks"]
    assert "best_checkpoint_changed_ok" in checks["failed_checks"]


def test_sanity_reference_checks_pass_dwf_reference_like_run():
    checks = mod.sanity_reference_checks(
        formal_pretrained=0.67,
        formal_finetuned=0.684,
        formal_improvement=0.014,
        training_start=0.676,
        training_best=0.691,
        training_improvement=0.015,
        best_checkpoint_changed=True,
        overall_rc=0,
    )

    assert checks["passed"] is True
    assert checks["failed_checks"] == []


def test_compare_checkpoint_weights_detects_reserialized_identical_weights(tmp_path):
    torch = pytest.importorskip("torch")
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"
    state = {"layer.weight": torch.ones(2, 2)}

    torch.save(state, reference)
    torch.save({"layer.weight": state["layer.weight"].clone()}, candidate)

    comparison = mod.compare_checkpoint_weights(reference, candidate)

    assert comparison["compared"] is True
    assert comparison["weights_identical"] is True
    assert comparison["differing_tensors"] == 0


def test_compare_checkpoint_weights_detects_changed_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"

    torch.save({"layer.weight": torch.ones(2, 2)}, reference)
    torch.save({"layer.weight": torch.zeros(2, 2)}, candidate)

    comparison = mod.compare_checkpoint_weights(reference, candidate)

    assert comparison["compared"] is True
    assert comparison["weights_identical"] is False
    assert comparison["differing_tensors"] == 1
    assert comparison["max_abs_diff"] == 1.0


def test_mlflow_tracking_uses_monai_builtin_without_training_overrides(tmp_path):
    args, metadata = mod.build_mlflow_tracking_args(
        tmp_path,
        tracking_uri=None,
        experiment_name="task06-demo",
        requested_run_name="sanity",
    )

    assert args[:2] == ["--tracking", "mlflow"]
    assert "--tracking_uri" in args
    assert "--experiment_name" in args
    assert "--run_name" in args
    assert args[args.index("--save_execute_config") + 1] == "False"
    assert not any("dataloader" in value or "patch_size" in value for value in args)
    assert metadata["tracking_uri"] == (tmp_path / "mlruns").resolve().as_uri()
    assert metadata["experiment_name"] == "task06-demo"
    assert metadata["run_name"] == "sanity"
    assert args[args.index("--run_name") + 1] == "sanity"


def test_mlflow_tracking_preserves_explicit_remote_uri(tmp_path):
    args, metadata = mod.build_mlflow_tracking_args(
        tmp_path,
        tracking_uri="databricks",
        experiment_name="/Shared/nvseg",
        requested_run_name=None,
    )

    assert args[args.index("--tracking_uri") + 1] == "databricks"
    assert metadata["tracking_uri"] == "databricks"
    assert "--run_name" not in args
    assert "run_name" not in metadata
