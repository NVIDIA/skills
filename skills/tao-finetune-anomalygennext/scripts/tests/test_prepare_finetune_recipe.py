# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "prepare_finetune_recipe.py"


def _image(path: Path, mode: str = "RGB", value: int | tuple[int, ...] = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (8, 8), value).save(path)


def _fixture(root: Path) -> list[str]:
    dataset = root / "dataset"
    anomaly = dataset / "texture" / "anomaly_image" / "defect" / "a.png"
    mask = dataset / "texture" / "mask" / "defect" / "a_mask.png"
    clean = dataset / "texture" / "clean_image" / "clean.png"
    validation_mask = root / "validation_mask.png"
    anomaly.parent.mkdir(parents=True, exist_ok=True)
    clean.parent.mkdir(parents=True, exist_ok=True)
    anomaly.write_bytes(b"image")
    clean.write_bytes(b"image")
    _image(mask, "L")
    _image(validation_mask, "L")
    with Image.open(mask) as image:
        pixels = image.copy()
    pixels.putpixel((0, 0), 255)
    pixels.save(mask)
    with Image.open(validation_mask) as image:
        validation_pixels = image.copy()
    validation_pixels.putpixel((0, 0), 255)
    validation_pixels.save(validation_mask)
    (dataset / "defect_spec.jsonl").write_text(
        json.dumps({"defect_type": "texture+defect", "spatial_dependency": "text",
                    "roi_prompt_defect_location": "on the surface"}) + "\n"
    )
    validation = root / "validation.jsonl"
    validation.write_text("".join(json.dumps({
        "image_filename": str(dataset / "texture" / "clean_image" / "clean.png"),
        "mask_filename": str(validation_mask),
        "anomaly_type": "texture+defect"}) + "\n" for _ in range(3)))
    base = root / "Cosmos3-Nano"
    checkpoints = root / "checkpoints"
    nn = checkpoints / "facebook" / "dinov2-large"
    (base / "model").mkdir(parents=True)
    (base / "checkpoint.json").write_text("{}")
    (base / "model" / ".metadata").write_bytes(b"metadata")
    (base / "model" / "__0_0.distcp").write_bytes(b"weights")
    (checkpoints / "hf").mkdir(parents=True)
    nn.mkdir(parents=True)
    (nn / "config.json").write_text("{}")
    (nn / "model.safetensors").write_bytes(b"weights")
    vae = root / "Wan2.2_VAE.pth"
    vae.write_bytes(b"model")
    return ["--dataset-root", str(dataset), "--validation-testcase", str(validation),
            "--base-checkpoint", str(base), "--vae-path", str(vae),
            "--checkpoint-root", str(checkpoints), "--dataset-name", "fixture",
            "--output", str(root / "out" / "recipe.yaml")]


def test_prepare_preserves_custom_knobs_and_freezes_identities(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"anomaly_types": [["texture", "defect"]],
                                        "batch_size": 7, "custom_knob": "kept"}))
    result = subprocess.run([sys.executable, str(SCRIPT), *args,
                             "--recipe-template", str(template), "--max-iter", "1000"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    recipe = yaml.safe_load((tmp_path / "out" / "recipe.yaml").read_text())
    assert recipe["batch_size"] == 7 and recipe["custom_knob"] == "kept"
    assert recipe["dataset_name"] == "fixture"
    assert recipe["max_iter"] == recipe["save_iter"] == recipe["validation_iter"] == 1000
    assert recipe["run_validation_on_start"] is True
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "recipe.validation.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(Path(row["image_filename"]).is_absolute() for row in rows)
    metadata = json.loads((tmp_path / "out" / "recipe.metadata.json").read_text())
    assert metadata["anomaly_types"] == ["texture+defect"]
    assert metadata["base_checkpoint_format"] == "dcp"
    assert metadata["checkpoint_root"] == str((tmp_path / "checkpoints").resolve())


def test_prepare_rejects_undercovered_validation(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    validation = Path(args[args.index("--validation-testcase") + 1])
    validation.write_text(validation.read_text().splitlines()[0] + "\n")
    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert result.returncode != 0
    assert "validation coverage mismatch" in result.stderr


def test_prepare_rejects_incomplete_checkpoint_root(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    checkpoint_root = Path(args[args.index("--checkpoint-root") + 1])
    (checkpoint_root / "hf").rmdir()

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    assert "lacks required Qwen tokenizer assets under hf/" in result.stderr


@pytest.mark.parametrize("missing", ["checkpoint.json", "model/.metadata", "model/__0_0.distcp"])
def test_prepare_rejects_incomplete_dcp_base_checkpoint(tmp_path: Path, missing: str) -> None:
    args = _fixture(tmp_path)
    base = Path(args[args.index("--base-checkpoint") + 1])
    (base / missing).unlink()

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    assert "base checkpoint must be a DCP directory" in result.stderr
    assert "cosmos_framework.scripts.convert_model_to_dcp" in result.stderr
    assert "references/container-runtime.md" in result.stderr


@pytest.mark.parametrize("invalid_shard", ["directory", "broken_symlink"])
def test_prepare_rejects_non_file_dcp_shard(tmp_path: Path, invalid_shard: str) -> None:
    args = _fixture(tmp_path)
    base = Path(args[args.index("--base-checkpoint") + 1])
    shard = base / "model" / "__0_0.distcp"
    shard.unlink()
    if invalid_shard == "directory":
        shard.mkdir()
    else:
        shard.symlink_to(base / "missing.distcp")

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    assert "base checkpoint must be a DCP directory" in result.stderr


@pytest.mark.parametrize("suffix", [".bmp", ".tif", ".tiff", ".webp"])
def test_prepare_rejects_runtime_unsupported_image_extensions(
        tmp_path: Path, suffix: str) -> None:
    args = _fixture(tmp_path)
    anomaly = tmp_path / "dataset/texture/anomaly_image/defect/a.png"
    anomaly.rename(anomaly.with_suffix(suffix))

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    assert "unsupported AnomalyGenNext image extension" in result.stderr


@pytest.mark.parametrize("key", ["image_filename", "mask_filename"])
def test_prepare_rejects_runtime_unsupported_validation_extensions(
        tmp_path: Path, key: str) -> None:
    args = _fixture(tmp_path)
    validation = Path(args[args.index("--validation-testcase") + 1])
    rows = [json.loads(line) for line in validation.read_text().splitlines()]
    source = Path(rows[0][key])
    unsupported = source.with_suffix(".bmp")
    unsupported.write_bytes(source.read_bytes())
    for row in rows:
        row[key] = str(unsupported)
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    label = key.removesuffix("_filename")
    assert f"unsupported AnomalyGenNext {label} extension" in result.stderr


def test_prepare_rejects_nonbinary_training_masks(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    mask = tmp_path / "dataset/texture/mask/defect/a_mask.png"
    _image(mask, "L", 1)

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    assert "mask must contain exactly binary pixel values 0 and 255" in result.stderr


def test_prepare_rejects_nonbinary_validation_masks(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    _image(tmp_path / "validation_mask.png", "L", 1)

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    assert "mask must contain exactly binary pixel values 0 and 255" in result.stderr


@pytest.mark.parametrize("relative,value", [
    ("dataset/texture/mask/defect/a_mask.png", 0),
    ("validation_mask.png", 0),
    ("dataset/texture/mask/defect/a_mask.png", 255),
    ("validation_mask.png", 255),
])
def test_prepare_rejects_uniform_masks(tmp_path: Path, relative: str, value: int) -> None:
    args = _fixture(tmp_path)
    _image(tmp_path / relative, "L", value)

    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    assert result.returncode != 0
    assert "mask must contain exactly binary pixel values 0 and 255" in result.stderr
