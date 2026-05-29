# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Target reference for eval 6:
# 30-day subseasonal forecast of t2m and sst using DLESyM.

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLESyM
from earth2studio.run import deterministic

# 1. Initialize DLESyM model (S2S/CM class, suited for subseasonal timescales)
model = DLESyM.load_model(DLESyM.load_default_package())

# 2. Initialize ARCO data source (ERA5, no auth required)
data = ARCO()

# 3. Initialize Zarr IO backend
io = ZarrBackend("output_eval6.zarr")

# 4. Subselect output variables
output_coords = OrderedDict(
    {
        "variable": np.array(["t2m", "sst"]),
    }
)

# 5. Run deterministic forecast
# DLESyM has a 6-hour time step, 30 days = 720h / 6h = 120 steps
io = deterministic(
    time=["2024-01-15T00:00:00"],
    nsteps=120,
    prognostic=model,
    data=data,
    io=io,
    output_coords=output_coords,
    device=torch.device("cuda"),
)

print("Forecast complete.")
