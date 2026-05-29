# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Target reference for eval 2:
# Pangu + GFS, 5-day forecast, save z500 and t850 to Zarr.

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Pangu
from earth2studio.run import deterministic

# 1. Initialize Pangu model
model = Pangu.load_model(Pangu.load_default_package())

# 2. Initialize GFS data source
data = GFS()

# 3. Initialize Zarr IO backend
io = ZarrBackend("output_eval2.zarr")

# 4. Subselect output variables (only z500 and t850)
output_coords = OrderedDict(
    {
        "variable": np.array(["z500", "t850"]),
    }
)

# 5. Run deterministic forecast
# Pangu has a 6-hour time step, 5 days = 120h / 6h = 20 steps
io = deterministic(
    time=["2024-06-01T00:00:00"],
    nsteps=20,
    prognostic=model,
    data=data,
    io=io,
    output_coords=output_coords,
    device=torch.device("cuda"),
)

print("Forecast complete.")
