# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Target reference for eval 4:
# GraphCast + ARCO, 10-day forecast, all output variables, Zarr.

import torch

from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import GraphCastOperational
from earth2studio.run import deterministic

# 1. Initialize GraphCast model
model = GraphCastOperational.load_model(GraphCastOperational.load_default_package())

# 2. Initialize ARCO data source (ERA5, no auth required)
data = ARCO()

# 3. Initialize Zarr IO backend
io = ZarrBackend("output_eval4.zarr")

# 4. Run deterministic forecast (no output_coords — save all variables)
# GraphCast has a 6-hour time step, 10 days = 240h / 6h = 40 steps
io = deterministic(
    time=["2023-09-01T00:00:00"],
    nsteps=40,
    prognostic=model,
    data=data,
    io=io,
    device=torch.device("cuda"),
)

print("Forecast complete.")
