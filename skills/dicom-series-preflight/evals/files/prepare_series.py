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

"""Write three tiny Explicit VR Little Endian DICOM series using only Python."""

import struct
import sys
import uuid
from pathlib import Path


def uid(name: str) -> str:
    return f"2.25.{uuid.uuid5(uuid.NAMESPACE_URL, 'medical-AI-skills/test/' + name).int}"


def element(group: int, tag: int, vr: str, value: str | bytes) -> bytes:
    data = value.encode("ascii") if isinstance(value, str) else value
    if len(data) % 2:
        data += b"\0" if vr == "UI" else b" "
    prefix = struct.pack("<HH2s", group, tag, vr.encode("ascii"))
    length = (
        struct.pack("<HI", 0, len(data)) if vr in ("OB", "OW") else struct.pack("<H", len(data))
    )
    return prefix + length + data


def main(root: Path) -> None:
    ct_storage = "1.2.840.10008.5.1.4.1.1.2"
    for name, x_axis, subject in (
        ("series-a", -1, ""),
        ("series-b", 1, ""),
        ("series-c", 1, "TEST^SUBJECT"),
    ):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            instance = uid(f"{name}/{index}")
            meta = b"".join(
                (
                    element(2, 1, "OB", b"\0\1"),
                    element(2, 2, "UI", ct_storage),
                    element(2, 3, "UI", instance),
                    element(2, 0x10, "UI", "1.2.840.10008.1.2.1"),
                    element(2, 0x12, "UI", uid("writer")),
                )
            )
            fields = [
                element(8, 0x16, "UI", ct_storage),
                element(8, 0x18, "UI", instance),
                element(8, 0x60, "CS", "CT"),
                element(0x10, 0x10, "PN", subject),
                element(0x18, 0x50, "DS", "2"),
                element(0x20, 0x0D, "UI", uid(name + "/study")),
                element(0x20, 0x0E, "UI", uid(name + "/series")),
                element(0x20, 0x13, "IS", str(index + 1)),
                element(0x20, 0x32, "DS", f"0\\0\\{index * 2 * x_axis}"),
                element(0x20, 0x37, "DS", str(x_axis) + r"\0\0\0\1\0"),
                element(0x28, 2, "US", struct.pack("<H", 1)),
                element(0x28, 4, "CS", "MONOCHROME2"),
                element(0x28, 0x10, "US", struct.pack("<H", 8)),
                element(0x28, 0x11, "US", struct.pack("<H", 8)),
                element(0x28, 0x30, "DS", r"1\1"),
            ]
            for tag, value in ((0x100, 16), (0x101, 16), (0x102, 15), (0x103, 1)):
                fields.append(element(0x28, tag, "US", struct.pack("<H", value)))
            fields.append(element(0x7FE0, 0x10, "OW", bytes(8 * 8 * 2)))
            prefix = bytes(128) + b"DICM" + element(2, 0, "UL", struct.pack("<I", len(meta)))
            (folder / f"slice_{index}.dcm").write_bytes(prefix + meta + b"".join(fields))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
