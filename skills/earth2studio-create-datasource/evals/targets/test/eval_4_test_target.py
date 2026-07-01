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

import pathlib
import shutil
from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr

from earth2studio.data.phoo import PhooAnalysis


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
        [
            datetime(year=2024, month=1, day=1, hour=6),
            datetime(year=2024, month=6, day=15, hour=12),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["t2m", "msl"]])
def test_phoo_fetch(time, variable):
    ds = PhooAnalysis(cache=False)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape[0] == len(time)
    assert data.shape[1] == len(variable)
    assert data.shape[2] == 181
    assert data.shape[3] == 360
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize("cache", [True, False])
def test_phoo_cache(cache):
    time = datetime(year=2024, month=1, day=1, hour=0)
    variable = ["t2m"]

    ds = PhooAnalysis(cache=cache)
    data = ds(time, variable)

    assert data.shape == (1, 1, 181, 360)
    assert pathlib.Path(ds.cache).is_dir() == cache

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
def test_phoo_call_mock(tmp_path):
    """Exercise the full __call__ path with mocked S3 fetch (no network)."""
    fake_data = np.full((181, 360), 273.15, dtype=np.float32)

    # Create a fake NetCDF file in tmp_path
    fake_ds = xr.Dataset({"temperature": (["lat", "lon"], fake_data)})
    fake_nc_path = tmp_path / "fake.nc"
    fake_ds.to_netcdf(fake_nc_path)

    async def _fake_fetch_remote_file(*args, **kwargs):
        return str(fake_nc_path)

    ds = PhooAnalysis(cache=True, verbose=False)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(
            PhooAnalysis, "_fetch_remote_file", side_effect=_fake_fetch_remote_file
        ),
    ):
        ds.fs = object()  # type: ignore[assignment]
        data = ds(datetime(2024, 1, 1, 0), ["t2m", "msl"])

    assert data.shape == (1, 2, 181, 360)
    assert list(data.dims) == ["time", "variable", "lat", "lon"]
    np.testing.assert_allclose(data.sel(variable="t2m").values[0], fake_data)
    np.testing.assert_allclose(data.sel(variable="msl").values[0], fake_data)

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
def test_phoo_exceptions():
    # Invalid time (not 6-hour aligned)
    ds = PhooAnalysis(cache=False)
    with pytest.raises(ValueError):
        ds(datetime(2024, 1, 1, 3), "t2m")

    # Invalid time (before MIN_DATE)
    with pytest.raises(ValueError):
        ds(datetime(2019, 1, 1, 0), "t2m")

    # Invalid variable (not in lexicon)
    with pytest.raises(KeyError):
        from earth2studio.lexicon import PhooLexicon

        PhooLexicon["invalid_variable_xyz"]


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2019, month=12, day=31),
        datetime(year=2024, month=1, day=1, hour=3),
        datetime(year=2024, month=1, day=1, hour=0, minute=30),
    ],
)
def test_phoo_available(time):
    assert not PhooAnalysis.available(time)


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
        datetime(year=2024, month=6, day=15, hour=12),
        np.datetime64("2024-01-01T06:00"),
    ],
)
def test_phoo_available_valid(time):
    assert PhooAnalysis.available(time)
