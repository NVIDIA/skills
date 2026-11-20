#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generate synthetic DICOM fixtures for dicom_series_preflight.

  clean_no_phi/  — canonical pass (LPS CT, no populated PHI tags)
  clean_axial/   — warn demo (same geometry, PHI tags populated)
  flipped_lr/    — fail demo (LR-flipped IOP)

Each series has 32 slices of a 64 by 64 synthetic CT phantom.
"""

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def _phantom() -> np.ndarray:
    z, y, x = np.ogrid[:32, :64, :64]
    volume = np.full((32, 64, 64), -1000, dtype="<i2")
    body = (x - 32) ** 2 + (y - 32) ** 2 < (64 * 0.45) ** 2
    volume[(z >= 2) & (z < 30) & body] = 40
    volume[(z - 16) ** 2 + (y - 35) ** 2 + (x - 41) ** 2 < 25] = 60
    return volume


def write_series(folder: Path, volume: np.ndarray, x_axis: int, synthetic_tags: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    study_uid, series_uid = generate_uid(), generate_uid()
    for index, pixels in enumerate(volume):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = folder / f"slice_{index:03d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.StudyDate = "20260509"
        ds.StudyDescription = "Synthetic CT fixture for orientation gate demo"
        ds.AccessionNumber = "ACC-FIX-001"
        ds.SeriesDescription = "Synthetic axial CT"
        ds.SeriesNumber = 1
        ds.Modality = "CT"
        ds.BodyPartExamined = "ABDOMEN"
        if synthetic_tags:
            ds.PatientName = "SYNTHETIC^FIXTURE"
            ds.PatientID = "SYNTH-001"
            ds.PatientBirthDate = "19000101"
            ds.PatientSex = "O"
        ds.InstanceNumber = index + 1
        ds.ImageOrientationPatient = [x_axis, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, index * 2 * x_axis]
        ds.PixelSpacing = [1, 1]
        ds.SliceThickness = 2
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = pixels.tobytes()
        ds.save_as(path, enforce_file_format=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().out_dir
    phantom = _phantom()
    write_series(root / "clean_axial", phantom, 1, True)
    write_series(root / "flipped_lr", phantom, -1, True)
    write_series(root / "clean_no_phi", phantom, 1, False)
    print(f"Wrote three synthetic series under {root}")


if __name__ == "__main__":
    main()
