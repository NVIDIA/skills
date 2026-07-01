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

import hashlib
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
)
from earth2studio.lexicon import PhooLexicon
from earth2studio.utils.type import TimeArray, VariableArray


@dataclass
class PhooAsyncTask:
    """Async task for a single variable/time NetCDF file fetch."""

    time_index: int
    variable_index: int
    s3_uri: str
    modifier: Callable


class PhooAnalysis:
    """Phoo global analysis data source on a 1-degree lat/lon grid.

    Fetches analysis data from the ``s3://phoo-weather-data`` S3 bucket. Data
    is stored as individual NetCDF files per variable and time step on a
    181x360 global grid at 6-hour intervals from 2020-01-01 to present.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in seconds after which download will be cancelled, by default 600
    async_workers : int, optional
        Maximum number of concurrent download tasks, by default 16
    retries : int, optional
        Number of retry attempts for failed downloads, by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://phoo-weather-data.example.com/docs

    Badges
    ------
    region:global dataclass:analysis product:temp product:atmos
    """

    PHOO_BUCKET = "phoo-weather-data"
    PHOO_LAT = np.linspace(-90.0, 90.0, 181)
    PHOO_LON = np.linspace(0.0, 360.0, 360, endpoint=False)
    MIN_DATE = datetime(2020, 1, 1)

    def __init__(
        self,
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

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem.

        Note
        ----
        Async fsspec expects initialization inside of the execution loop.
        """
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=True, skip_instance_cache=True
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve Phoo analysis data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.
            Must be in the Phoo lexicon.

        Returns
        -------
        xr.DataArray
            Phoo weather data array with dims [time, variable, lat, lon]
        """
        try:
            xr_array = _sync_async(
                self.fetch, time, variable, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.
            Must be in the Phoo lexicon.

        Returns
        -------
        xr.DataArray
            Phoo weather data array with dims [time, variable, lat, lon]
        """
        if self.fs is None:
            await self._async_init()

        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Pre-allocate output array
        xr_array = xr.DataArray(
            data=np.empty(
                (len(time), len(variable), len(self.PHOO_LAT), len(self.PHOO_LON))
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.PHOO_LAT,
                "lon": self.PHOO_LON,
            },
        )

        async_tasks = self._create_tasks(time, variable)

        async with managed_session(self.fs):
            coros = [self.fetch_wrapper(task, xr_array) for task in async_tasks]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                task_timeout=60.0,
                desc="Fetching Phoo data",
                verbose=(not self._verbose),
            )

        return xr_array

    def _create_tasks(
        self, time: list[datetime], variable: list[str]
    ) -> list[PhooAsyncTask]:
        """Create download tasks for each time/variable combination.

        Parameters
        ----------
        time : list[datetime]
            Timestamps to be downloaded (UTC).
        variable : list[str]
            List of variables to be downloaded.

        Returns
        -------
        list[PhooAsyncTask]
            List of async download tasks.
        """
        tasks: list[PhooAsyncTask] = []
        for i, t in enumerate(time):
            for k, v in enumerate(variable):
                try:
                    phoo_name, modifier = PhooLexicon[v]
                except KeyError as e:
                    raise KeyError(f"Variable '{v}' not found in PhooLexicon") from e

                s3_uri = self._netcdf_uri(t, phoo_name)
                tasks.append(
                    PhooAsyncTask(
                        time_index=i,
                        variable_index=k,
                        s3_uri=s3_uri,
                        modifier=modifier,
                    )
                )
        return tasks

    async def fetch_wrapper(self, task: PhooAsyncTask, xr_array: xr.DataArray) -> None:
        """Wrapper to download and insert array data into the output."""
        out = await async_retry(
            self.fetch_array,
            task,
            retries=self._retries,
            backoff=1.0,
            task_timeout=120.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        xr_array[task.time_index, task.variable_index] = task.modifier(out)

    async def fetch_array(self, task: PhooAsyncTask) -> np.ndarray:
        """Fetch a single NetCDF variable array from S3.

        Parameters
        ----------
        task : PhooAsyncTask
            Async task containing S3 URI and modifier.

        Returns
        -------
        np.ndarray
            2D array of shape (181, 360).
        """
        logger.debug(f"Fetching Phoo file: {task.s3_uri}")
        cache_path = await self._fetch_remote_file(task.s3_uri)
        with xr.open_dataset(cache_path) as ds:
            # Assume single variable in file, extract first data variable
            var_name = list(ds.data_vars)[0]
            values = ds[var_name].values
        return values

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for Phoo data source.

        Parameters
        ----------
        times : list[datetime]
            List of date times to validate.
        """
        for time in times:
            if time.hour % 6 != 0 or time.minute != 0 or time.second != 0:
                raise ValueError(
                    f"Requested datetime {time} must align to a 6-hour cycle "
                    f"(00, 06, 12, 18z)."
                )
            if time < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {time} is earlier than "
                    f"PhooAnalysis.MIN_DATE ({cls.MIN_DATE.isoformat()})."
                )

    async def _fetch_remote_file(self, path: str) -> str:
        """Download a remote file to local cache using async S3.

        Parameters
        ----------
        path : str
            S3 URI to download.

        Returns
        -------
        str
            Local cache path of the downloaded file.
        """
        if self.fs is None:
            raise ValueError("File system is not initialized")

        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            data = await self.fs._cat_file(path)
            with open(cache_path, "wb") as fh:
                fh.write(data)

        return cache_path

    def _netcdf_uri(self, time: datetime, variable: str) -> str:
        """Generate the S3 URI for a given time and variable.

        Parameters
        ----------
        time : datetime
            Timestamp.
        variable : str
            Remote variable name.

        Returns
        -------
        str
            S3 URI path.
        """
        return (
            f"s3://{self.PHOO_BUCKET}/analysis/"
            f"{time.year:04d}/{time.month:02d}/{time.day:02d}/"
            f"{time.hour:02d}z/{variable}.nc"
        )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "phoo")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_phoo_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check.

        Returns
        -------
        bool
            If date time is available based on offline validation.
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
