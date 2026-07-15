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
import pytest

from earth2studio.data.random_gaussian import RandomGaussian


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
@pytest.mark.parametrize("variable", ["t2m", ["t2m", "msl"]])
def test_random_gaussian_call(time, variable):
    ds = RandomGaussian(seed=42)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape[0] == len(time)
    assert data.shape[1] == len(variable)
    assert data.shape[2] == 181
    assert data.shape[3] == 360
    assert list(data.dims) == ["time", "variable", "lat", "lon"]
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


def test_random_gaussian_grid():
    ds = RandomGaussian()
    assert ds.lat[0] == -90.0
    assert ds.lat[-1] == 90.0
    assert len(ds.lat) == 181
    assert ds.lon[0] == 0.0
    assert len(ds.lon) == 360


def test_random_gaussian_reproducibility():
    ds1 = RandomGaussian(seed=123)
    ds2 = RandomGaussian(seed=123)
    time = datetime(2024, 1, 1)
    variable = "t2m"

    data1 = ds1(time, variable)
    data2 = ds2(time, variable)
    assert np.array_equal(data1.values, data2.values)


def test_random_gaussian_different_seeds():
    ds1 = RandomGaussian(seed=1)
    ds2 = RandomGaussian(seed=2)
    time = datetime(2024, 1, 1)
    variable = "t2m"

    data1 = ds1(time, variable)
    data2 = ds2(time, variable)
    assert not np.array_equal(data1.values, data2.values)
