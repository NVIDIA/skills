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

from datetime import datetime

import numpy as np
import xarray as xr

from earth2studio.data.utils import prep_data_inputs
from earth2studio.utils.type import TimeArray, VariableArray


class RandomGaussian:
    """A randomly generated normally distributed data source on a 1-degree global grid.

    Generates random Gaussian data on a 181x360 lat/lon grid. Primarily useful
    for testing and development.

    Parameters
    ----------
    seed : int | None, optional
        Random seed for reproducibility, by default None
    """

    def __init__(self, seed: int | None = None):
        self.lat = np.linspace(-90.0, 90.0, 181)
        self.lon = np.linspace(0.0, 360.0, 360, endpoint=False)
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve random Gaussian data on a 1-degree global grid.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Random data array with dims [time, variable, lat, lon]
        """
        time, variable = prep_data_inputs(time, variable)

        shape = [len(time), len(variable), len(self.lat), len(self.lon)]
        coords = {
            "time": time,
            "variable": variable,
            "lat": self.lat,
            "lon": self.lon,
        }

        da = xr.DataArray(
            data=self.rng.standard_normal(shape),
            dims=list(coords),
            coords=coords,
        )

        return da
