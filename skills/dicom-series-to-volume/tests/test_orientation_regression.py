# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for DICOM-affine orientation handling."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "series_to_volume.py"
SPEC = importlib.util.spec_from_file_location("series_to_volume", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extreme_oblique_affine_reorients_to_ras_idempotently() -> None:
    """Preserve NiBabel's column-strength regression through DICOM geometry."""

    # Public, synthetic affine from nibabel/nibabel#1449 and PR #1450. Its
    # first and third input axes both align most strongly with the superior
    # axis, exposing the pre-5.4 order-dependent labeling behavior.
    expected_ras_affine = np.array(
        [
            [2.08759499, 0.0770245194, 0.112271041, 0.0],
            [-1.04219818, 0.158019245, -0.0534135476, 0.0],
            [-2.33361936, -0.00166752085, 0.124289364, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_lps = lps_to_ras @ expected_ras_affine

    row_vector = affine_lps[:3, 0]
    column_vector = affine_lps[:3, 1]
    slice_vector = affine_lps[:3, 2]
    first = SimpleNamespace(
        ImageOrientationPatient=np.concatenate(
            [row_vector / np.linalg.norm(row_vector), column_vector / np.linalg.norm(column_vector)]
        ),
        PixelSpacing=[np.linalg.norm(column_vector), np.linalg.norm(row_vector)],
        ImagePositionPatient=[0.0, 0.0, 0.0],
    )
    last = SimpleNamespace(ImagePositionPatient=slice_vector)

    affine, _spacing = MODULE._affine_from_dicom(first, last, n_slices=2)
    np.testing.assert_allclose(affine, expected_ras_affine)
    assert nib.aff2axcodes(affine) == ("I", "A", "R")

    image = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), affine)
    canonical = nib.as_closest_canonical(image)
    assert canonical.shape == (4, 3, 2)
    assert nib.aff2axcodes(canonical.affine) == ("R", "A", "S")

    canonical_again = nib.as_closest_canonical(canonical)
    assert canonical_again.shape == canonical.shape
    assert nib.aff2axcodes(canonical_again.affine) == ("R", "A", "S")
