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
import pandas as pd
import pyarrow as pa

from earth2studio.data.utils import prep_data_inputs
from earth2studio.utils.type import FieldArray, TimeArray, VariableArray


class RandomStations:
    """A randomly generated weather station observation data source.

    Generates random observations at random locations for specified times and
    variables. Produces 50 stations per time/variable combination with random
    lat, lon, elevation, and observation values.

    Parameters
    ----------
    n_obs : int, optional
        Number of random observations per time/variable, by default 50
    seed : int | None, optional
        Random seed for reproducibility, by default None
    """

    SOURCE_ID = "earth2studio.data.RandomStations"
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns")),
            pa.field("lat", pa.float32()),
            pa.field("lon", pa.float32()),
            pa.field("elevation", pa.float32()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(self, n_obs: int = 50, seed: int | None = None):
        self.n_obs = n_obs
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | FieldArray | None = None,
    ) -> pd.DataFrame:
        """Retrieve random station observation DataFrame.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.
        fields : str | list[str] | pa.Schema | FieldArray | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            Random station observation DataFrame
        """
        time, variable = prep_data_inputs(time, variable)

        data = []
        for t in time:
            t_dt = pd.to_datetime(t)
            for v in variable:
                for _ in range(self.n_obs):
                    row = {
                        "time": t_dt,
                        "lat": self.rng.uniform(-90.0, 90.0),
                        "lon": self.rng.uniform(0.0, 360.0),
                        "elevation": self.rng.uniform(0.0, 5000.0),
                        "observation": self.rng.standard_normal(),
                        "variable": v,
                    }
                    data.append(row)

        df = pd.DataFrame(data)
        df.attrs["source"] = self.SOURCE_ID

        # Filter to requested fields if specified
        if fields is not None:
            if isinstance(fields, str):
                fields = [fields]
            if isinstance(fields, pa.Schema):
                field_names = fields.names
            else:
                field_names = list(fields)
            df = df[[col for col in field_names if col in df.columns]]

        return df
