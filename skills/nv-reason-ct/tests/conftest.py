# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in upstream example and live checks; never download or install assets."""

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def pytest_addoption(parser):
    group = parser.getgroup("nv-reason-ct")
    group.addoption(
        "--nv-reason-ct-upstream",
        help="Authorized NV-Reason-CT checkout with examples downloaded via Git LFS.",
    )
    group.addoption(
        "--nv-reason-ct-live",
        action="store_true",
        help="Run offline CUDA inference and compare with the upstream CLI.",
    )
    group.addoption(
        "--nv-reason-ct-trust-model-code",
        action="store_true",
        help="Authorize unsandboxed custom model code in both live inference paths.",
    )
    group.addoption(
        "--nv-reason-ct-revision",
        help="Reviewed, already cached immutable Hugging Face commit for live tests.",
    )


@pytest.fixture
def wrapper(monkeypatch):
    monkeypatch.delenv("NV_REASON_CT_MODEL", raising=False)
    monkeypatch.delenv("NV_REASON_CT_REVISION", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_nv_reason_ct.py"
    spec = importlib.util.spec_from_file_location("nv_reason_ct_test_wrapper", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def volume_double(wrapper, tmp_path, monkeypatch):
    """Opaque bytes plus a NiBabel loader double, NOT a generated NIfTI scan."""
    volume = tmp_path / "ct.nii.gz"
    volume.write_bytes(b"unit-test input identity; not NIfTI data")
    image = SimpleNamespace(
        shape=(4, 5, 6),
        header=SimpleNamespace(get_zooms=lambda: (1.0, 1.5, 2.0)),
        affine=np.eye(4),
        get_data_dtype=lambda: np.dtype("int16"),
    )

    def load(path):
        assert Path(path) == volume
        return image

    monkeypatch.setattr(wrapper, "_nifti_modules", lambda: (SimpleNamespace(load=load), np))
    return volume, image


@pytest.fixture(scope="session")
def upstream_checkout(pytestconfig):
    # Root-level collection can reach this nested conftest after CLI parsing.
    value = pytestconfig.getoption("--nv-reason-ct-upstream", default=None)
    if not value:
        if pytestconfig.getoption("--nv-reason-ct-live", default=False):
            pytest.fail("Provide --nv-reason-ct-upstream with an authorized checkout")
        pytest.skip("Real-volume checks require --nv-reason-ct-upstream")
    checkout = Path(value).expanduser().resolve()
    if not (checkout / "inference.py").is_file():
        pytest.fail(f"Missing upstream inference.py in {checkout}")
    return checkout


@pytest.fixture(scope="session")
def live_revision(pytestconfig):
    if not pytestconfig.getoption("--nv-reason-ct-live", default=False):
        pytest.skip("Live model inference requires --nv-reason-ct-live")
    if not pytestconfig.getoption("--nv-reason-ct-trust-model-code", default=False):
        pytest.fail("Live tests require explicit --nv-reason-ct-trust-model-code consent")
    revision = pytestconfig.getoption("--nv-reason-ct-revision", default=None) or ""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        pytest.fail("Live tests require a reviewed immutable --nv-reason-ct-revision")
    return revision
