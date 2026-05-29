# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Target reference for eval 3:
# 7-day global forecast of z500 and msl, 48GB GPU constraint.
# Model should be one that fits in 48GB (e.g. FCN, Pangu, FuXi).

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
from earth2studio.run import deterministic

# 1. Initialize model (must fit in 48GB VRAM)
model = FCN.load_model(FCN.load_default_package())

# 2. Initialize data source
data = GFS()

# 3. Initialize IO backend
io = ZarrBackend("output_eval3.zarr")

# 4. Subselect output variables
output_coords = OrderedDict(
    {
        "variable": np.array(["z500", "msl"]),
    }
)

# 5. Run deterministic forecast
# FCN has a 6-hour time step, 7 days = 168h / 6h = 28 steps
io = deterministic(
    time=["2024-01-15T00:00:00"],
    nsteps=28,
    prognostic=model,
    data=data,
    io=io,
    output_coords=output_coords,
    device=torch.device("cuda"),
)

print("Forecast complete.")
