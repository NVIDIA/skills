# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "run_anomalygennext_amp.py"
SPEC = importlib.util.spec_from_file_location("run_anomalygennext_amp", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def checkpoint_root(repo: Path) -> Path:
    root = repo / "checkpoints"
    for model_id in MODULE.AMP_HF_REPOS:
        cache = root / f"hf/hub/models--{model_id.replace('/', '--')}"
        (cache / "blobs").mkdir(parents=True)
        (cache / "snapshots").mkdir()
    model = root / "hf/hub/models--nvidia--Cosmos3-Nano"
    (model / "blobs").mkdir(parents=True)
    snapshot = model / "snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    sam2 = root / "facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt"
    sam2.parent.mkdir(parents=True)
    sam2.write_bytes(b"weights")
    return root


def inputs(root: Path) -> dict:
    (root / "manifests").mkdir(parents=True)
    (root / "embeddings").mkdir()
    pd.DataFrame([
        {"filepath": "/clean/a.png", "pool_key": "texture_1",
         "anomaly_type_eligibility": "texture_1+crack", "embedding": [1.0, 0.0]},
        {"filepath": "/clean/b.png", "pool_key": "texture_1",
         "anomaly_type_eligibility": "texture_1+crack", "embedding": [0.8, 0.2]},
    ]).to_parquet(root / "embeddings" / "clean_embeddings.parquet")
    pd.DataFrame([
        {"filepath": "/defect/shared.png", "embedding": [1.0, 0.0]},
    ]).to_parquet(root / "embeddings" / "fn_embeddings.parquet")
    pd.DataFrame([
        {"filepath": "/defect/shared.png", "fn_id": "fn-1", "query_order": 0,
         "dataset_id": "d", "pool_key": "texture_1",
         "anomaly_type": "texture_1+crack", "od_category": "defect"},
        {"filepath": "/defect/shared.png", "fn_id": "fn-2", "query_order": 1,
         "dataset_id": "d", "pool_key": "texture_1",
         "anomaly_type": "texture_1+crack", "od_category": "defect"},
    ]).to_parquet(root / "manifests" / "selected_fn_queries.parquet")
    mask_rows = []
    for index, (fn, branch) in enumerate(
        (fn, branch)
        for fn in ("fn-1", "fn-2")
        for branch in sorted(MODULE.BRANCHES)
    ):
        path = root / "masks" / f"{fn}-{branch}.png"
        values = np.zeros((32, 32), dtype=np.uint8)
        values[4 + index:12 + index, 5:15] = 255
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(values).save(path)
        mask_rows.append({"fn_id": fn, "branch": branch, "mask_path": str(path)})
    pd.DataFrame(mask_rows).to_parquet(root / "manifests" / "mask_selection.parquet")
    return {"retrieval": {"metric": "cosine", "candidate_topn": 2,
                           "min_similarity": -1.0,
                           "prior_clean_exclusion_manifest": ""}}


def test_plan_preserves_pairs_for_same_source_image(tmp_path: Path) -> None:
    report = MODULE.plan(tmp_path, inputs(tmp_path))
    assert report == {"candidates": 4, "amp_rows": 8, "embedding_dim": 2}
    candidates = pd.read_parquet(tmp_path / "manifests" / "knn_candidates.parquet")
    assert candidates.fn_id.nunique() == 2
    assert candidates.groupby("fn_id").clean_filepath.nunique().eq(2).all()
    requests = json.loads((tmp_path / "amp" / "amp_samples.json").read_text())
    assert len({row["name"] for row in requests}) == 8


def test_plan_ranks_distinct_images_using_their_best_crop(tmp_path: Path) -> None:
    config = inputs(tmp_path)
    pd.DataFrame([
        {"filepath": "/clean/a.png", "pool_key": "texture_1",
         "anomaly_type_eligibility": "texture_1+crack", "embedding": [0.6, 0.8]},
        {"filepath": "/clean/a.png", "pool_key": "texture_1",
         "anomaly_type_eligibility": "texture_1+crack", "embedding": [1.0, 0.0]},
        {"filepath": "/clean/b.png", "pool_key": "texture_1",
         "anomaly_type_eligibility": "texture_1+crack", "embedding": [0.9, 0.1]},
        {"filepath": "/clean/c.png", "pool_key": "texture_1",
         "anomaly_type_eligibility": "texture_1+crack", "embedding": [0.8, 0.2]},
        {"filepath": "/clean/0-ineligible.png", "pool_key": "texture_1",
         "anomaly_type_eligibility": "texture_1+oil", "embedding": [1.0, 0.0]},
    ]).to_parquet(tmp_path / "embeddings" / "clean_embeddings.parquet")

    report = MODULE.plan(tmp_path, config)

    assert report == {"candidates": 4, "amp_rows": 8, "embedding_dim": 2}
    candidates = pd.read_parquet(tmp_path / "manifests" / "knn_candidates.parquet")
    assert candidates.groupby("fn_id").clean_filepath.nunique().eq(2).all()
    assert candidates.groupby("fn_id").clean_filepath.apply(list).tolist() == [
        ["/clean/a.png", "/clean/b.png"],
        ["/clean/a.png", "/clean/b.png"],
    ]
    assert "/clean/0-ineligible.png" not in set(candidates.clean_filepath)


def test_plan_rejects_zero_norm_embeddings(tmp_path: Path) -> None:
    config = inputs(tmp_path)
    frame = pd.read_parquet(tmp_path / "embeddings" / "fn_embeddings.parquet")
    frame["embedding"] = pd.Series([[0.0, 0.0]], dtype=object)
    frame.to_parquet(tmp_path / "embeddings" / "fn_embeddings.parquet")
    with pytest.raises(ValueError, match="zero-norm"):
        MODULE.plan(tmp_path, config)


def test_run_launches_native_amp_offline_and_publishes_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    frozen = tmp_path / "prepared_anomalygennext_inputs/filtering_config.yaml"
    frozen.parent.mkdir()
    pool = tmp_path / "pool"
    pool.mkdir()
    frozen.write_text(
        f"defect_spec: /input/defects.jsonl\npool_dataset_root: {pool}\n"
        "amp:\n"
        "  model_id: acme/custom-amp\n"
    )
    repo = tmp_path / "repo"
    checkpoints = checkpoint_root(repo)
    custom_model = checkpoints / "acme/custom-amp"
    custom_model.mkdir(parents=True)
    (custom_model / "config.json").write_text("{}\n")
    (custom_model / "model.safetensors").write_bytes(b"weights")
    published = tmp_path.parent / "persistent-output"
    monkeypatch.setattr(MODULE, "plan", lambda root, value: {"candidates": 1})

    def fake_run(command: list[str], *, check: bool, stdout: object,
                 env: dict[str, str]) -> None:
        assert check is True and stdout is sys.stderr
        assert command[1:3] == ["-m", "anomalygen.scripts.auto_mask_placement.roi_place"]
        assert env["HF_HOME"] == str(checkpoints / "hf")
        assert env["HF_HUB_CACHE"] == str(checkpoints / "hf/hub")
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert command[command.index("--model_id") + 1] == "acme/custom-amp"
        print("native AMP progress", file=stdout)
        amp = tmp_path / "amp"
        amp.mkdir()
        (amp / "testcase.jsonl").write_text(
            json.dumps({"mask_filename": str(amp / "mask.png")}) + "\n"
        )

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    report = MODULE.run(tmp_path, checkpoints, pool, published, repo)

    assert report["testcase"] == str(published.resolve() / "amp/testcase.jsonl")
    row = json.loads((tmp_path / "amp/testcase.jsonl").read_text())
    assert row["mask_filename"] == str(published.resolve() / "amp/mask.png")
    captured = capsys.readouterr()
    assert captured.out == "" and "native AMP progress" in captured.err


def test_checkpoint_root_requires_canonical_mount_and_amp_assets(tmp_path: Path) -> None:
    external = checkpoint_root(tmp_path / "external")
    with pytest.raises(ValueError, match="must be mounted"):
        MODULE._validate_checkpoint_root(external, tmp_path / "repo")

    repo = tmp_path / "repo"
    root = checkpoint_root(repo)
    snapshots = root / "hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
    snapshots.rmdir()
    with pytest.raises(FileNotFoundError, match="Qwen3-VL-8B-Instruct"):
        MODULE._validate_checkpoint_root(root, repo)
    snapshots.mkdir()
    (root / "facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt").unlink()
    with pytest.raises(FileNotFoundError, match="SAM2.1 checkpoint"):
        MODULE._validate_checkpoint_root(root, repo)


def test_checkpoint_root_validates_configured_amp_model(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = checkpoint_root(repo)

    with pytest.raises(FileNotFoundError, match="acme/custom-amp"):
        MODULE._validate_checkpoint_root(root, repo, "acme/custom-amp")


def test_checkpoint_root_accepts_complete_direct_local_amp_model(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = checkpoint_root(repo)
    cached = root / "hf/hub/models--nvidia--Cosmos3-Nano"
    for path in sorted(cached.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    cached.rmdir()
    local = root / "nvidia/Cosmos3-Nano"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}\n")
    (local / "model.safetensors").write_bytes(b"weights")

    MODULE._validate_checkpoint_root(root, repo, "nvidia/Cosmos3-Nano")


def test_checkpoint_root_rejects_processor_only_amp_model(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = checkpoint_root(repo)
    model = root / "hf/hub/models--nvidia--Cosmos3-Edge/snapshots/revision"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n")
    (model / "tokenizer.json").write_text("{}\n")

    with pytest.raises(FileNotFoundError, match="complete checkpoints/nvidia/Cosmos3-Edge"):
        MODULE._validate_checkpoint_root(root, repo, "nvidia/Cosmos3-Edge")


def test_run_amp_contract_mounts_complete_checkpoint_root() -> None:
    contract = yaml.safe_load(
        (SCRIPT.parents[1] / "references/skill_info.yaml").read_text()
    )["actions"]["run_amp"]
    inputs = contract["inputs"]
    assert inputs["checkpoint_root"]["container_path"] == (
        "/workspace/paidf-anomalygen/checkpoints"
    )
    assert "sam2_checkpoint" not in inputs
    assert contract["args"]["checkpoint_root"] == "--checkpoint-root {checkpoint_root}"


def test_run_rejects_pool_mount_that_differs_from_frozen_path(tmp_path: Path) -> None:
    frozen = tmp_path / "prepared_anomalygennext_inputs/filtering_config.yaml"
    frozen.parent.mkdir()
    expected, wrong = tmp_path / "pool", tmp_path / "wrong-pool"
    expected.mkdir()
    wrong.mkdir()
    frozen.write_text(f"pool_dataset_root: {expected}\n")
    repo = tmp_path / "repo"
    checkpoints = checkpoint_root(repo)

    with pytest.raises(ValueError, match="pool must be remounted.*expected.*received"):
        MODULE.run(tmp_path, checkpoints, wrong, repo=repo)


def test_plan_rejects_nonbinary_amp_mask(tmp_path: Path) -> None:
    config = inputs(tmp_path)
    masks = pd.read_parquet(tmp_path / "manifests/mask_selection.parquet")
    bad = Path(masks.iloc[0].mask_path)
    values = np.zeros((32, 32), dtype=np.uint8)
    values[4:12, 5:15] = 3
    Image.fromarray(values).save(bad)
    with pytest.raises(ValueError, match="exactly binary values"):
        MODULE.plan(tmp_path, config)


def test_plan_still_rejects_fn_without_exactly_two_mask_branches(tmp_path: Path) -> None:
    config = inputs(tmp_path)
    masks = pd.read_parquet(tmp_path / "manifests/mask_selection.parquet")
    masks = masks[~((masks.fn_id == "fn-1") & (masks.branch == "fn_mask"))]
    masks.to_parquet(tmp_path / "manifests/mask_selection.parquet", index=False)

    with pytest.raises(ValueError, match="needs exactly two mask branches"):
        MODULE.plan(tmp_path, config)
