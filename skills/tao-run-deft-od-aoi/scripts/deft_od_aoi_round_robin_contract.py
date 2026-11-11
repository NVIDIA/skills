#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared shape contract for round-robin admission signatures.

The discrete cosine transform (DCT) summarizes an image as spatial-frequency
coefficients. Each index row retains a 12-by-12 block for the whole image, the
same-sized block for its defect crop, and normalized ``center_x``, ``center_y``,
``width``, and ``height`` values: 144 + 144 + 4 = 292 features.
"""

DCT_SIZE = 32
DCT_KEEP = 12
BOX_POSITION_FEATURES = 4
ADMISSION_INDEX_WIDTH = 2 * DCT_KEEP * DCT_KEEP + BOX_POSITION_FEATURES
