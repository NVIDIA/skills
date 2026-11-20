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

"""Write a small synthetic CT/MR input bundle in single-file NIfTI-1 format."""

import gzip
import json
import struct
import sys
from array import array
from pathlib import Path


def write_volume(path: Path, center: int, background: int, peak: int) -> None:
    header = bytearray(352)  # NIfTI-1 header followed by its four-byte extension flag.
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, 64, 64, 64, 1, 1, 1, 1)
    struct.pack_into("<hh", header, 70, 4, 16)  # signed 16-bit voxels
    struct.pack_into("<8f", header, 76, 1, 1.5, 1.5, 2, 1, 1, 1, 1)
    struct.pack_into("<ff", header, 108, 352, 1)  # data offset and intensity scale
    header[123] = 2  # millimetres
    struct.pack_into("<h", header, 254, 1)  # scanner-coordinate sform
    struct.pack_into("<12f", header, 280, 1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 2, 0)
    header[344:348] = b"n+1\0"

    voxels = array("h")
    for z in range(64):
        for y in range(64):
            for x in range(64):
                radius = (x - center) ** 2 + (y - 32) ** 2 + (z - 32) ** 2
                voxels.append(peak - radius // 12 if radius < 24**2 else background)
    if sys.byteorder != "little":
        voxels.byteswap()
    with path.open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
            compressed.write(header)
            compressed.write(voxels.tobytes())


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    datalist = {}
    for split, center in (("training", 28), ("validation", 34)):
        entries = []
        for modality, background, peak in (("ct", -1000, 80), ("mri", 0, 400)):
            name = f"{split}_{modality}.nii.gz"
            write_volume(root / name, center, background, peak)
            entries.append({"image": name, "modality": modality})
        datalist[split] = entries
    (root / "vae_datalist.json").write_text(json.dumps(datalist, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
