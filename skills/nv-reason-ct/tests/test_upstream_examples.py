# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live example_1 chest/abdomen checks against the upstream CLI.

Both paths use the same cached, reviewed model snapshot. No network, package
installation, synthetic fallback, or clinical scoring is part of these tests.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["example_1", "example_2", "example_3"])
def test_real_example_metadata(wrapper, upstream_checkout, name):
    import nibabel as nib

    volume = upstream_checkout / "examples" / f"{name}.nii.gz"
    info = wrapper._volume_info(volume)
    image = nib.load(volume)
    assert info.shape == image.shape
    assert info.spacing_mm == pytest.approx(image.header.get_zooms())
    assert info.sha256 == hashlib.sha256(volume.read_bytes()).hexdigest()


# Forward only the listed runtime/cache settings. Authentication variables and
# mock flags are excluded from the offline child processes.
LIVE_ENVIRONMENT_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_MODULES_CACHE",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
)


def _run_logged(command, output_dir, name):
    env = {
        key: value for key in LIVE_ENVIRONMENT_KEYS if (value := os.environ.get(key)) is not None
    }
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    (output_dir / f"{name}.stdout.log").write_text(result.stdout)
    (output_dir / f"{name}.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, f"{name}: see {output_dir} (exit {result.returncode})"
    return result.stdout


def test_logged_subprocess_preserves_isolation_without_credentials(tmp_path, monkeypatch):
    cache = str(tmp_path / "model-cache")
    monkeypatch.setenv("HF_HUB_CACHE", cache)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for key in (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "UNRELATED_SERVICE_TOKEN",
        "MOCK_NV_REASON_CT",
    ):
        monkeypatch.setenv(key, "unit-test-only")

    def check_environment(command, **kwargs):
        assert command == [sys.executable, "--version"]
        env = kwargs["env"]
        assert set(env) <= set(LIVE_ENVIRONMENT_KEYS) | {
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        }
        assert env["HF_HUB_CACHE"] == cache
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        return subprocess.CompletedProcess(command, 0, "checked\n", "")

    monkeypatch.setattr(subprocess, "run", check_environment)
    assert _run_logged([sys.executable, "--version"], tmp_path, "env") == "checked\n"
    assert (tmp_path / "env.stdout.log").read_text() == "checked\n"
    assert (tmp_path / "env.stderr.log").read_text() == ""


@pytest.mark.parametrize("region", ["chest", "abdomen"])
def test_live_example_matches_upstream(live_revision, upstream_checkout, tmp_path, region):
    from huggingface_hub import hf_hub_download

    snapshot = Path(
        hf_hub_download(
            "nvidia/NV-Reason-CT",
            "config.json",
            revision=live_revision,
            local_files_only=True,
            token=False,
        )
    ).parent
    assert snapshot.name == live_revision
    volume = upstream_checkout / "examples" / "example_1.nii.gz"
    assert volume.is_file(), "Download the upstream example with Git LFS first"
    prompt = f"write a structured {'abdominal' if region == 'abdomen' else region} CT report"
    wrapper_output = _run_logged(
        [
            sys.executable,
            str(SKILL_DIR / "scripts" / "run_nv_reason_ct.py"),
            str(volume),
            "--trust-model-code",
            "--anatomy-region",
            region,
            "--prompt",
            prompt,
            "--thinking",
            "--max-new-tokens",
            "2048",
            "--local-files-only",
            "--model-id",
            "nvidia/NV-Reason-CT",
            "--revision",
            live_revision,
            "--out-dir",
            str(tmp_path / "artifacts"),
        ],
        tmp_path,
        "wrapper",
    )
    payload = json.loads(wrapper_output)
    schema = json.loads((SKILL_DIR / "validators" / "output_schema.json").read_text())
    jsonschema.validate(payload, schema)
    assert payload["runtime"]["mode"] == "hf_transformers"
    assert payload["runtime"]["mock"] is False
    assert payload["runtime"]["resolved_revision"] == live_revision
    assert payload["runtime"]["generated_tokens"] > 0
    assert payload["runtime"]["truncated_by_max_new_tokens"] is False
    assert payload["input"]["anatomy_region"] == region
    assert payload["input"]["enable_thinking"] is True
    assert payload["input"]["prompt"] == prompt
    assert payload["input"]["volume"]["sha256"] == hashlib.sha256(volume.read_bytes()).hexdigest()

    # Safe-path mode avoids mistaking upstream's accelerate/ config directory
    # for the optional Python package. Do not patch the upstream implementation.
    reference = _run_logged(
        [
            sys.executable,
            "-P",
            str(upstream_checkout / "inference.py"),
            str(volume),
            "--model",
            str(snapshot),
            "--region",
            region,
            "--prompt",
            prompt,
            "--max-new-tokens",
            "2048",
        ],
        tmp_path,
        "upstream",
    )
    assert (
        payload["output"]["response_text"].strip() == reference.strip()
    ), f"Wrapper and upstream responses differ; retain both outputs in {tmp_path}"
