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

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from earth2studio.data.utils import prep_forecast_inputs
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray


class RandomForecast:
    """A randomly generated uniform [0, 1] forecast data source on a 2.5-degree grid.

    Generates random uniform forecast data on a 73x144 lat/lon grid. Primarily
    useful for testing and development.

    Parameters
    ----------
    seed : int | None, optional
        Random seed for reproducibility, by default None
    """

    def __init__(self, seed: int | None = None):
        self.lat = np.linspace(-90.0, 90.0, 73)
        self.lon = np.linspace(0.0, 360.0, 144, endpoint=False)
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve random uniform forecast data on a 2.5-degree global grid.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Initialization timestamps to return data for.
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Random forecast data array with dims [time, lead_time, variable, lat, lon]
        """
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)

        shape = [
            len(time),
            len(lead_time),
            len(variable),
            len(self.lat),
            len(self.lon),
        ]
        coords = {
            "time": time,
            "lead_time": lead_time,
            "variable": variable,
            "lat": self.lat,
            "lon": self.lon,
        }

        da = xr.DataArray(
            data=self.rng.uniform(0.0, 1.0, size=shape),
            dims=list(coords),
            coords=coords,
        )

        return da
