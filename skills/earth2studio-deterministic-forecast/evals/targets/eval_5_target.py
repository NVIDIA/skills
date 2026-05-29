# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Target reference for eval 5:
# 24-hour precipitation forecast over North America.
# Regional/NWC model (e.g. StormCast) + HRRR data source.

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.run import deterministic

# 1. Initialize regional model (NWC class, NA region)
model = StormCast.load_model(StormCast.load_default_package())

# 2. Initialize HRRR data source (high-res North America)
data = HRRR()

# 3. Initialize IO backend
io = ZarrBackend("output_eval5.zarr")

# 4. Subselect output — precipitation variable(s)
output_coords = OrderedDict(
    {
        "variable": np.array(["tp"]),
    }
)

# 5. Run deterministic forecast
# StormCast has a 1-hour time step, 24 hours = 24 steps
io = deterministic(
    time=["2024-03-15T00:00:00"],
    nsteps=24,
    prognostic=model,
    data=data,
    io=io,
    output_coords=output_coords,
    device=torch.device("cuda"),
)

print("Forecast complete.")
