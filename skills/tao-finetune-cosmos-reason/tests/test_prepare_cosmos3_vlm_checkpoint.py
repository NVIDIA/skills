# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "prepare_cosmos3_vlm_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_cosmos3_vlm_checkpoint", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def converter_args(source: str, architecture: str) -> SimpleNamespace:
    return SimpleNamespace(
        backend="cosmos-rl",
        base_model_path_or_uri=source,
        base_model_revision="a" * 40,
        vlm_architecture_model_path_or_uri=architecture,
        vlm_architecture_model_revision="b" * 40,
        runtime_image="example.invalid/cosmos-rl:test",
    )


def test_docker_downloads_uri_inputs_to_symlink_free_local_directories(
    tmp_path: Path,
) -> None:
    args = converter_args("owner/source", "owner/architecture")

    command = MODULE.command(args, tmp_path / "output/prepared", tmp_path / "cache")
    shell = command[-1]

    assert "local_dir=Path(os.environ['BASE_MODEL_LOCAL_DIR'])" in shell
    assert "local_dir=Path(os.environ['ARCHITECTURE_MODEL_LOCAL_DIR'])" in shell
    assert "cache_dir='/cache/huggingface'" not in shell
    assert any(
        value.startswith("BASE_MODEL_LOCAL_DIR=/cache/huggingface/materialized/")
        for value in command
    )
    assert any(
        value.startswith("ARCHITECTURE_MODEL_LOCAL_DIR=/cache/huggingface/materialized/")
        for value in command
    )
    assert "/tao-local-cache:rw,exec,size=32g,mode=1777" in command
    assert "HOME=/tao-local-cache/home" in command
    assert "XDG_CACHE_HOME=/tao-local-cache/xdg" in command
    assert "HF_HOME=/tao-local-cache/huggingface" in command
    assert "UV_CACHE_DIR=/tao-local-cache/uv" in command
    assert "mkdir -p \"$HOME\" \"$XDG_CACHE_HOME\" \"$HF_HOME\" \"$UV_CACHE_DIR\"" in shell
    assert 'ln -s "$uv_executable" /tao-local-cache/bin/uv' in shell
    assert 'export PATH="/tao-local-cache/bin:$python_bin:/usr/bin:/bin"' in shell


def test_in_container_download_materializes_the_immutable_snapshot(
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    expected = MODULE.materialized_snapshot_dir(
        tmp_path,
        "owner/model",
        revision,
    )
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(expected)) as download:
        resolved = MODULE.resolve_input("hf_model://owner/model", revision, tmp_path)

    assert resolved == str(expected)
    download.assert_called_once_with(
        "owner/model",
        revision=revision,
        local_dir=str(expected),
    )
