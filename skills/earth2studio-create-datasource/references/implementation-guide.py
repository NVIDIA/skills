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

"""Implementation reference skeleton for new Earth2Studio data sources.

This file is a TEMPLATE. Copy it, rename the class, and fill in every
section marked with ``# FILL:``.  Each ``# FILL:`` comment describes
exactly what to replace.

See real examples:
- Gridded DataSource / ForecastSource: earth2studio/data/cfs.py
- DataFrame DataFrameSource: earth2studio/data/nnja.py
- HTTP-based: earth2studio/data/gdas.py

Conventions:
- File lives at earth2studio/data/<source_name>.py
- Class name: PascalCase (e.g. GFS, ARCO, CFS_FX, HimawariAHI)
- File name: lowercase with underscores (e.g. gfs.py, cfs.py)
"""

from __future__ import annotations

import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import s3fs
import xarray as xr
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
    # FILL: Use prep_forecast_inputs instead if ForecastSource
    # prep_forecast_inputs,
)

# FILL: Import your lexicon class
# from earth2studio.lexicon import SourceNameLexicon
# ---------------------------------------------------------------------------
# Optional dependency imports (only if you need packages in the `data` extra)
# ---------------------------------------------------------------------------
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

from earth2studio.utils.type import TimeArray, VariableArray

try:
    pass
    # FILL: Import optional packages here, e.g.:
    # import special_decoder
except ImportError:
    OptionalDependencyFailure("data")
    # FILL: Assign None fallbacks:
    # special_decoder = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FILL: Bucket/URL, path prefix, grid dimensions, time constraints
SOURCE_BUCKET = "my-bucket-name"
SOURCE_PREFIX = "path/to/data"
MIN_DATE = datetime(2020, 1, 1)
# FILL: Time step interval in seconds (e.g. 6h = 21600)
INTERVAL_SECONDS = 21600


# ---------------------------------------------------------------------------
# Async task dataclass
# ---------------------------------------------------------------------------


@dataclass
class SourceNameAsyncTask:
    """A single async fetch unit.

    Each task represents one (time, variable) pair to be fetched in parallel.
    """

    # FILL: Index into the output array/dataframe where this result goes
    data_array_indices: tuple[int, ...]
    # FILL: Full URI to the remote file/object
    remote_uri: str
    # FILL: Transform to apply after reading (from lexicon get_item)
    modifier: Callable


# ---------------------------------------------------------------------------
# Data source class
# ---------------------------------------------------------------------------


@check_optional_dependencies()
class SourceName:
    """Short description of the data source.

    Extended description: what data this provides, spatial/temporal resolution,
    grid type, etc.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation, by default 600.
    async_workers : int, optional
        Max concurrent async fetch tasks, by default 16.
    retries : int, optional
        Number of retry attempts per failed fetch with exponential backoff,
        by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.

    Note
    ----
    FILL: Reference URL for the data store / documentation.
    """

    # FILL: If this is a DataFrameSource, define SCHEMA here:
    # SCHEMA = pa.schema([
    #     E2STUDIO_SCHEMA.field("time"),
    #     E2STUDIO_SCHEMA.field("lat"),
    #     E2STUDIO_SCHEMA.field("lon"),
    #     E2STUDIO_SCHEMA.field("observation"),
    #     E2STUDIO_SCHEMA.field("variable"),
    # ])

    def __init__(
        self,
        # FILL: Add source-specific params first (e.g. member, satellite, etc.)
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ) -> None:
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        self.fs: s3fs.S3FileSystem | None = None

    # ------------------------------------------------------------------
    # Async initialization
    # ------------------------------------------------------------------
    async def _async_init(self) -> None:
        """Initialize the async filesystem connection."""
        # FILL: Choose the right filesystem for your backend:
        # S3 (anonymous):
        self.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={},
            asynchronous=True,
            skip_instance_cache=True,
        )
        # FILL: Or HTTP:
        # from fsspec.implementations.http import HTTPFileSystem
        # self.fs = HTTPFileSystem(asynchronous=True)
        # FILL: Or GCS:
        # import gcsfs
        # self.fs = gcsfs.GCSFileSystem(token="anon", asynchronous=True)

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        # FILL: ForecastSource adds: lead_time: timedelta | list[timedelta] | LeadTimeArray,
        # FILL: DataFrameSource adds: fields: str | list[str] | pa.Schema | None = None,
    ) -> xr.DataArray:
        """Retrieve data for given times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Date times to fetch.
        variable : str | list[str] | VariableArray
            Variable names from the source's lexicon.

        Returns
        -------
        xr.DataArray
            Fetched data array with dims (time, variable, lat, lon).
        """
        # FILL: For ForecastSource return shape is (time, lead_time, variable, lat, lon)
        try:
            xr_array = _sync_async(
                self.fetch, time, variable, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)
        return xr_array

    # ------------------------------------------------------------------
    # Async fetch
    # ------------------------------------------------------------------
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Date times to fetch.
        variable : str | list[str] | VariableArray
            Variable names from the source's lexicon.

        Returns
        -------
        xr.DataArray
            Data array with dims (time, variable, lat, lon).
        """
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        # FILL: For ForecastSource use:
        # time_list, lead_list, variable_list = prep_forecast_inputs(
        #     time, lead_time, variable
        # )

        self._validate_time(time_list)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # FILL: Create output array with correct shape and coordinates
        # For DataSource:
        xr_array = xr.DataArray(
            data=np.empty(
                (len(time_list), len(variable_list), 181, 360),  # FILL: grid dims
                dtype=np.float32,
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time_list,
                "variable": variable_list,
                # FILL: lat/lon coordinate arrays
                "lat": np.linspace(90, -90, 181),
                "lon": np.linspace(0, 359, 360),
            },
        )

        tasks = self._create_tasks(time_list, variable_list)

        # Use managed_session for guaranteed cleanup of the async connection
        async with managed_session(self.fs):
            coros = [self.fetch_wrapper(task, xr_array=xr_array) for task in tasks]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                task_timeout=60.0,
                desc="Fetching SourceName data",  # FILL: source name
                verbose=(not self._verbose),
            )

        return xr_array

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------
    def _create_tasks(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> list[SourceNameAsyncTask]:
        """Build list of async tasks for all (time, variable) pairs."""
        tasks = []
        for i, t in enumerate(time_list):
            for j, v in enumerate(variable_list):
                # FILL: Import your lexicon class at the top
                # try:
                #     remote_key, modifier = SourceNameLexicon[v]
                # except KeyError:
                #     logger.warning(f"Variable {v} not in lexicon, skipping")
                #     continue
                remote_key = v  # FILL: replace with lexicon lookup
                modifier: Callable = lambda x: x  # FILL: from lexicon

                # FILL: Build the remote URI from time + remote_key
                uri = f"s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}/{t:%Y%m%d}/{remote_key}.nc"

                tasks.append(
                    SourceNameAsyncTask(
                        data_array_indices=(i, j),
                        remote_uri=uri,
                        modifier=modifier,
                    )
                )
        return tasks

    # ------------------------------------------------------------------
    # Fetch wrapper (retry + write into output)
    # ------------------------------------------------------------------
    async def fetch_wrapper(
        self, task: SourceNameAsyncTask, xr_array: xr.DataArray
    ) -> None:
        """Fetch a single task with retry and write result into the array."""
        out = await async_retry(
            self.fetch_array,
            task,
            retries=self._retries,
            backoff=1.0,
            task_timeout=60.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        i, j = task.data_array_indices
        xr_array[i, j] = task.modifier(out)

    # ------------------------------------------------------------------
    # Fetch single array (pure async I/O)
    # ------------------------------------------------------------------
    async def fetch_array(self, task: SourceNameAsyncTask) -> np.ndarray:
        """Download and decode a single data chunk.

        FILL: This is where the actual I/O happens. Use pure async calls:
        - await self.fs._cat_file(uri) for full file
        - await self.fs._cat_file(uri, start=offset, end=offset+length)
          for byte-range reads
        Then decode the bytes into a numpy array.
        """
        if self.fs is None:
            raise ValueError("Filesystem not initialized")

        # FILL: Pure async read — never use blocking I/O here
        data_bytes = await self.fs._cat_file(task.remote_uri)

        # FILL: Decode bytes into numpy array
        # Example for NetCDF: write to cache, open with netCDF4/h5py
        # Example for GRIB: write to cache, open with pygrib
        # Example for zarr: use async zarr store
        array = np.frombuffer(data_bytes, dtype=np.float32).reshape(181, 360)

        return array

    # ------------------------------------------------------------------
    # Time validation
    # ------------------------------------------------------------------
    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that requested times are within range and on-grid.

        FILL: Adjust the interval and date range for your source.
        """
        for t in times:
            if (t - datetime(1900, 1, 1)).total_seconds() % INTERVAL_SECONDS != 0:
                raise ValueError(
                    f"Requested {t} must align to "
                    f"{INTERVAL_SECONDS // 3600}h intervals"
                )
            if t < MIN_DATE:
                raise ValueError(f"Requested {t} is before minimum date {MIN_DATE}")

    # ------------------------------------------------------------------
    # Cache property
    # ------------------------------------------------------------------
    @property
    def cache(self) -> str:
        """Local cache directory for downloaded data."""
        # FILL: Replace "source_name" with your source's short id
        cache_location = os.path.join(datasource_cache_root(), "source_name")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(cache_location, f"tmp_{self._tmp_cache_hash}")
        return cache_location

    # ------------------------------------------------------------------
    # Available classmethod
    # ------------------------------------------------------------------
    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is fetchable.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check.

        Returns
        -------
        bool
            If date time is available.
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True


# ===========================================================================
# LEXICON TEMPLATE (put in earth2studio/lexicon/<source_name>.py)
# ===========================================================================
#
# from collections.abc import Callable
# import numpy as np
# from earth2studio.lexicon.base import LexiconType
#
#
# class SourceNameLexicon(metaclass=LexiconType):
#     """Lexicon for SourceName data source.
#
#     Note
#     ----
#     Variable documentation: FILL: URL
#     """
#
#     VOCAB = {
#         # FILL: Map Earth2Studio variable names -> remote keys
#         # Surface variables: descriptive abbreviation
#         "t2m": "2m_temperature",
#         "u10m": "10m_u_component_of_wind",
#         # Pressure-level: {short_name}{level_hPa}
#         "t500": "temperature::500",
#         "z500": "geopotential::500",
#     }
#
#     @classmethod
#     def get_item(cls, val: str) -> tuple[str, Callable]:
#         """Return remote key and modifier function for a variable.
#
#         Parameters
#         ----------
#         val : str
#             Earth2Studio variable name.
#
#         Returns
#         -------
#         tuple[str, Callable]
#             Remote key string and post-processing function.
#         """
#         remote_key = cls.VOCAB[val]
#         # FILL: Add unit conversions or transforms as needed, e.g.:
#         # if val.startswith("z"):
#         #     return remote_key, lambda x: x * 9.81  # geopotential -> height
#         return remote_key, lambda x: x


# ===========================================================================
# REGISTRATION CHECKLIST
# ===========================================================================
#
# 1. earth2studio/data/__init__.py
#    Add in alphabetical order:
#      from .source_name import SourceName
#
# 2. earth2studio/lexicon/__init__.py
#    Add in alphabetical order:
#      from .source_name import SourceNameLexicon
#
# 3. docs/modules/datasources_analysis.rst (or _forecast.rst, _dataframe.rst)
#    Add class to the .. autosummary:: directive in alphabetical order.
#
# 4. CHANGELOG.md
#    Under the current unreleased version, add:
#      ### Added
#      - Added SourceName DataSource for <description> (`SourceName`)
