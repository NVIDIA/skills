# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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

"""Wind speed diagnostic model that computes wind speed from u/v components.

This is a simple diagnostic model that does not require any checkpoints.
It computes wind speed at 10m (ws10m) from u10m and v10m components using
the formula: ws10m = sqrt(u10m^2 + v10m^2).
"""

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

INPUT_VARIABLES = ["u10m", "v10m"]
OUTPUT_VARIABLES = ["ws10m"]


class WindSpeed(torch.nn.Module):
    """Wind speed diagnostic that computes ws10m from u10m and v10m.

    A simple diagnostic model that computes wind speed at 10m from the
    u and v wind components. No checkpoints or external weights needed.

    Note
    ----
    Wind speed is computed as: ws10m = sqrt(u10m^2 + v10m^2)
    """

    def __init__(self) -> None:
        super().__init__()

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates with ws10m variable.
        """
        target_input_coords = self.input_coords()

        # Validate dimensions
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)

        # Validate coordinate values
        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(OUTPUT_VARIABLES)
        return output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Compute wind speed from u/v components.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, 2, lat, lon) where
            variable dimension contains [u10m, v10m].
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Wind speed tensor (batch, 1, lat, lon) and output coordinates.
        """
        output_coords = self.output_coords(coords)

        with torch.no_grad():
            # Extract u and v components
            # x shape: (batch, 2, lat, lon) where variable=["u10m", "v10m"]
            u = x[:, 0:1, :, :]
            v = x[:, 1:2, :, :]

            # Compute wind speed
            ws = torch.sqrt(u**2 + v**2)

        return ws, output_coords
