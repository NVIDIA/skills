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
import pytest

from earth2studio.data.random_forecast import RandomForecast


@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1),
        [
            datetime(year=2024, month=1, day=1, hour=6),
            datetime(year=2024, month=6, day=15, hour=12),
        ],
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=6),
        [timedelta(hours=6), timedelta(hours=12), timedelta(hours=24)],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["t2m", "msl"]])
def test_random_forecast_call(time, lead_time, variable):
    ds = RandomForecast(seed=42)
    data = ds(time, lead_time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]
    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    assert data.shape[0] == len(time)
    assert data.shape[1] == len(lead_time)
    assert data.shape[2] == len(variable)
    assert data.shape[3] == 73
    assert data.shape[4] == 144
    assert list(data.dims) == ["time", "lead_time", "variable", "lat", "lon"]
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


def test_random_forecast_grid():
    ds = RandomForecast()
    assert ds.lat[0] == -90.0
    assert ds.lat[-1] == 90.0
    assert len(ds.lat) == 73
    assert ds.lon[0] == 0.0
    assert len(ds.lon) == 144


def test_random_forecast_uniform_range():
    ds = RandomForecast(seed=99)
    time = datetime(2024, 1, 1)
    lead_time = timedelta(hours=6)
    variable = "t2m"

    data = ds(time, lead_time, variable)
    assert data.values.min() >= 0.0
    assert data.values.max() <= 1.0


def test_random_forecast_reproducibility():
    ds1 = RandomForecast(seed=123)
    ds2 = RandomForecast(seed=123)
    time = datetime(2024, 1, 1)
    lead_time = timedelta(hours=6)
    variable = "t2m"

    data1 = ds1(time, lead_time, variable)
    data2 = ds2(time, lead_time, variable)
    assert np.array_equal(data1.values, data2.values)


def test_random_forecast_different_seeds():
    ds1 = RandomForecast(seed=1)
    ds2 = RandomForecast(seed=2)
    time = datetime(2024, 1, 1)
    lead_time = timedelta(hours=6)
    variable = "t2m"

    data1 = ds1(time, lead_time, variable)
    data2 = ds2(time, lead_time, variable)
    assert not np.array_equal(data1.values, data2.values)
