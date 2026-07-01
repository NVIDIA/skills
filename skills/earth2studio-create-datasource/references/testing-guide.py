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

"""Test reference skeleton for new Earth2Studio data sources.

This file is a TEMPLATE. Copy it to test/data/test_<source_name>.py,
rename all ``source`` prefixes, and fill in every ``# FILL:`` comment.

Canonical test order (must match this sequence):
1. test_source_fetch         — slow, xfail, real network
2. test_source_cache         — slow, xfail, cache toggle
3. test_source_call_mock     — REQUIRED, no network, no xfail
4. test_source_exceptions    — error paths
5. test_source_available     — available() classmethod
6. test_source_validate_time — (optional) _validate_time internals

Rules:
- No docstrings on test functions (project convention)
- Use strategic parameterization — minimize combos, maximize coverage
- Target 90%+ line coverage with --slow
- Run: pytest test/data/test_<source_name>.py -v

Automatic Test Skipping:
------------------------
If your data source requires optional dependencies (e.g., special decoders,
API clients), register your test file in test/conftest.py's _TEST_DEPENDENCIES
mapping. This ensures pytest skips the file during collection when dependencies
are missing, preventing ImportError before tests even run.

Example in test/conftest.py:

    _TEST_DEPENDENCIES: dict[str, list[str]] = {
        # ...existing entries...
        "test/data/test_newsource.py": ["data"],           # use pyproject.toml group
        "test/data/test_special.py": ["special_decoder"],  # or individual package
    }

This replaces the need for pytest.importorskip() at module level for skipping
the entire file. Use pytest.importorskip() only when you need to skip specific
tests within a file that otherwise runs without optional deps.

See real examples:
- Gridded ForecastSource: test/data/test_cfs.py
- DataFrame DataFrameSource: test/data/test_nnja.py
- HTTP DataFrame: test/data/test_gdas.py
"""

from __future__ import annotations

import pathlib
import shutil
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

# FILL: Import your source class
# from earth2studio.data import SourceName

# FILL: If source has optional deps, guard with importorskip:
# pytest.importorskip("special_decoder", reason="special_decoder not installed")

# FILL: Pick a known-valid time for network tests (inside source's valid range)
_TEST_TIME = datetime(year=2024, month=6, day=1, hour=0)


# ======================================================================
# 1. Network fetch test (slow, xfail)
# ======================================================================


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        _TEST_TIME,  # FILL: single datetime
        [_TEST_TIME, _TEST_TIME + timedelta(hours=6)],  # FILL: list of datetimes
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["t2m", "msl"]])  # FILL: real variables
def test_source_fetch(time, variable):
    # FILL: Instantiate your source
    # ds = SourceName(cache=False)
    # data = ds(time, variable)
    # FILL: For ForecastSource, add lead_time arg:
    # data = ds(time, timedelta(hours=6), variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    # FILL: Assert shape matches expectations
    # DataSource: (time, variable, lat, lon)
    # assert data.shape == (len(time), len(variable), EXPECTED_LAT, EXPECTED_LON)
    # ForecastSource: (time, lead_time, variable, lat, lon)
    # assert data.shape == (len(time), 1, len(variable), EXPECTED_LAT, EXPECTED_LON)

    # assert not np.isnan(data.values).any()
    # assert np.array_equal(data.coords["variable"].values, np.array(variable))
    pass


# ======================================================================
# 2. Cache test (slow, xfail)
# ======================================================================


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_source_cache(cache):
    # FILL: Instantiate and fetch
    # ds = SourceName(cache=cache)
    # data = ds(_TEST_TIME, ["t2m"])
    #
    # assert not np.isnan(data.values).any()
    # assert pathlib.Path(ds.cache).is_dir() == cache
    #
    # # Second fetch exercises cache path
    # data2 = ds(_TEST_TIME, ["t2m"])
    # assert data2.shape == data.shape
    #
    # try:
    #     shutil.rmtree(ds.cache)
    # except FileNotFoundError:
    #     pass
    pass


# ======================================================================
# 3. Mock test — REQUIRED (no network, no xfail, no slow)
# ======================================================================


@pytest.mark.timeout(15)
def test_source_call_mock(tmp_path, monkeypatch):
    # FILL: Set cache dir to tmp_path
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # FILL: Create fake data that your fetch_array/decode would return
    # For gridded sources:
    fake_grid = np.full((181, 360), 273.15, dtype=np.float32)  # FILL: grid dims

    # FILL: Define async mocks for the I/O layer
    async def _fake_fetch_array(task):
        return fake_grid

    # FILL: Instantiate source
    # ds = SourceName(cache=True)

    # FILL: Patch the async methods that do real I/O
    # Strategy: patch _async_init (skip fs creation) + fetch_array (skip network)
    #
    # with (
    #     patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
    #     patch.object(ds, "fetch_array", side_effect=_fake_fetch_array),
    # ):
    #     ds.fs = object()  # type: ignore[assignment]  # truthy sentinel
    #     data = ds(_TEST_TIME, ["t2m", "msl"])
    #
    # # Verify output shape and content
    # assert data.shape == (1, 2, 181, 360)  # FILL: expected shape
    # np.testing.assert_allclose(data.sel(variable="t2m").values[0], fake_grid)
    pass


# ======================================================================
# 3b. Mock test for DataFrame sources
# ======================================================================


@pytest.mark.timeout(15)
def test_source_call_mock_dataframe(tmp_path, monkeypatch):
    # FILL: Only include this if your source returns pd.DataFrame
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # FILL: Create mock DataFrame matching your source's SCHEMA
    # mock_df = pd.DataFrame({
    #     "time": pd.to_datetime(["2024-06-01 00:30:00", "2024-06-01 01:00:00"]),
    #     "lat": np.array([40.0, 35.0], dtype=np.float32),
    #     "lon": np.array([250.0, 280.0], dtype=np.float32),
    #     "observation": np.array([273.15, 288.0], dtype=np.float32),
    #     "variable": ["t", "t"],
    #     # FILL: Add all SCHEMA columns
    # })

    # FILL: Mock the async fetch path
    # with patch("earth2studio.data.<module>._sync_async") as mock_sync:
    #     mock_sync.return_value = mock_df
    #     ds = SourceName(cache=False)
    #     df = ds(_TEST_TIME, ["t"])
    #
    # assert isinstance(df, pd.DataFrame)
    # assert list(df.columns) == ds.SCHEMA.names
    # assert not df.empty
    # assert set(df["variable"].unique()) == {"t"}
    pass


# ======================================================================
# 4. Exception / error handling tests
# ======================================================================


@pytest.mark.timeout(10)
def test_source_exceptions():
    # FILL: Test invalid variable via lexicon
    # with pytest.raises(KeyError):
    #     from earth2studio.lexicon import SourceNameLexicon
    #     SourceNameLexicon["nonexistent_variable"]

    # FILL: Test invalid time raises ValueError from _validate_time
    # ds = SourceName(cache=False)
    # with pytest.raises(ValueError):
    #     ds(datetime(1800, 1, 1), ["t2m"])  # before MIN_DATE

    # FILL: Test misaligned time (e.g. odd hour for 6h interval)
    # with pytest.raises(ValueError):
    #     ds(datetime(2024, 1, 1, 13), ["t2m"])

    # FILL: For DataFrame sources, test invalid fields
    # with pytest.raises(KeyError):
    #     SourceName.resolve_fields(["nonexistent_field"])
    pass


# ======================================================================
# 5. Available classmethod test
# ======================================================================


@pytest.mark.timeout(5)
def test_source_available():
    # FILL: Valid time should return True
    # assert SourceName.available(datetime(2024, 6, 1, 0)) is True

    # FILL: Time before MIN_DATE should return False
    # assert SourceName.available(datetime(1800, 1, 1)) is False

    # FILL: Misaligned time should return False
    # assert SourceName.available(datetime(2024, 1, 1, 13)) is False

    # FILL: Test with np.datetime64 input
    # assert SourceName.available(np.datetime64("2024-06-01T00:00")) is True
    pass


# ======================================================================
# 6. (Optional) Validate time internals
# ======================================================================


@pytest.mark.timeout(5)
def test_source_validate_time():
    # FILL: Only include if _validate_time has non-trivial logic worth testing
    # directly. Otherwise the available() test covers it.

    # FILL: Valid times pass without exception
    # SourceName._validate_time([datetime(2024, 1, 1, 0)])
    # SourceName._validate_time([datetime(2024, 1, 1, 6)])

    # FILL: Invalid times raise ValueError
    # with pytest.raises(ValueError):
    #     SourceName._validate_time([datetime(2024, 1, 1, 1)])  # off-grid
    # with pytest.raises(ValueError):
    #     SourceName._validate_time([datetime(1900, 1, 1, 0)])  # too early
    pass
