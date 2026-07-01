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

import pyarrow as pa
import pytest

from earth2studio.data.random_stations import RandomStations


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
def test_random_stations_call(time, variable):
    ds = RandomStations(n_obs=50, seed=42)
    df = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    expected_rows = len(time) * len(variable) * 50
    assert len(df) == expected_rows
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert "elevation" in df.columns
    assert "time" in df.columns


def test_random_stations_schema():
    ds = RandomStations()
    assert hasattr(ds, "SCHEMA")
    assert isinstance(ds.SCHEMA, pa.Schema)
    assert "time" in ds.SCHEMA.names
    assert "lat" in ds.SCHEMA.names
    assert "lon" in ds.SCHEMA.names
    assert "elevation" in ds.SCHEMA.names
    assert "observation" in ds.SCHEMA.names
    assert "variable" in ds.SCHEMA.names


def test_random_stations_n_obs():
    ds = RandomStations(n_obs=10, seed=42)
    time = datetime(2024, 1, 1)
    variable = "t2m"
    df = ds(time, variable)
    assert len(df) == 10


def test_random_stations_fields_filter():
    ds = RandomStations(n_obs=50, seed=42)
    time = datetime(2024, 1, 1)
    variable = "t2m"

    # Filter to subset of fields
    df = ds(time, variable, fields=["lat", "lon", "observation"])
    assert list(df.columns) == ["lat", "lon", "observation"]


def test_random_stations_fields_schema_filter():
    ds = RandomStations(n_obs=50, seed=42)
    time = datetime(2024, 1, 1)
    variable = "t2m"

    # Filter using pyarrow schema
    subset_schema = pa.schema(
        [
            pa.field("time", pa.timestamp("ns")),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )
    df = ds(time, variable, fields=subset_schema)
    assert list(df.columns) == ["time", "observation", "variable"]


def test_random_stations_reproducibility():
    ds1 = RandomStations(n_obs=10, seed=123)
    ds2 = RandomStations(n_obs=10, seed=123)
    time = datetime(2024, 1, 1)
    variable = "t2m"

    df1 = ds1(time, variable)
    df2 = ds2(time, variable)
    assert df1.equals(df2)


def test_random_stations_coordinate_ranges():
    ds = RandomStations(n_obs=100, seed=42)
    time = datetime(2024, 1, 1)
    variable = "t2m"
    df = ds(time, variable)

    assert df["lat"].min() >= -90.0
    assert df["lat"].max() <= 90.0
    assert df["lon"].min() >= 0.0
    assert df["lon"].max() <= 360.0
    assert df["elevation"].min() >= 0.0
    assert df["elevation"].max() <= 5000.0
