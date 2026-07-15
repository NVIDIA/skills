################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################

import asyncio
import logging
import os
import sys
import threading
import time
from concurrent.futures import Future
from enum import Enum
from typing import Any, Coroutine, Dict, Optional

from . import ds_logger

logger = ds_logger.get_logger(__name__)

# TimeMeasure should support with statement
# TimeMeasure should support elapsed_time method
# TimeMeasure should support reset method
# TimeMeasure should support get_elapsed_time method


class TimeMeasure:
    def __init__(self, text: str = ""):
        self._text = text
        self._lock = threading.Lock()
        self._execute_done_time = None
        self._1st_execute_done_time = None
        self.reset()

    def __enter__(self):
        """Support for context manager (with statement)"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager (with statement)"""
        return False

    @property
    def elapsed_time(self):
        """Get elapsed time since start or last reset"""
        return time.time() - self._start_time

    @property
    def start_time(self):
        return self._start_time

    def now(self):
        return time.time()

    def reset(self):
        """Reset the timer to current time"""
        self._start_time = time.time()
        self.last_execute_time = self._start_time

    def log_elapsed_time(self, message):
        logger.info(f"{self._text} {message} in {self.elapsed_time:.3f} seconds")

    def update_execute_time(self):
        with self._lock:
            self._execute_done_time = time.time()
            if self._1st_execute_done_time is None:
                self._1st_execute_done_time = self._execute_done_time

    @property
    def total_execute_time(self):
        """Get execution time since start or last reset"""
        with self._lock:
            if self._execute_done_time is None:
                return None
            return self._execute_done_time - self._start_time

    @property
    def first_execute_time(self):
        """Get execution time since start or last reset"""
        with self._lock:
            if self._1st_execute_done_time is None:
                return None
            return self._1st_execute_done_time - self._start_time


class SafeThreadEventLoop:
    def __init__(self):
        self._event_loop = asyncio.new_event_loop()

        def exception_handler(loop, context):
            """Handle exceptions in the event loop."""
            exception = context.get("exception")
            message = context.get("message", "Unhandled exception in event loop")
            logger.error(f"Event loop exception: {message}", exc_info=exception)

        def loop_thread():
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.set_exception_handler(exception_handler)
            self._event_loop.run_forever()

        self._thread = threading.Thread(target=loop_thread, daemon=True)
        self._thread.start()

    def __del__(self):
        self.close()

    def run_coroutine_threadsafe(self, coroutine: Coroutine) -> Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self._event_loop)

    @property
    def event_loop(self):
        return self._event_loop

    def close(self):
        if self._thread is not None and self._thread.is_alive():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._thread.join()
            self._event_loop = None
            self._thread = None


def get_media_info_gst(uri_or_file: str, username="", password=""):
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstPbutils", "1.0")
    from gi.repository import Gst, GstPbutils  # noqa: E402

    Gst.init(None)

    uri_or_file = str(uri_or_file)

    if uri_or_file.startswith("rtsp://") or uri_or_file.startswith("file://"):
        uri = uri_or_file
    else:
        uri = "file://" + os.path.abspath(str(uri_or_file))

    def select_stream(source, idx, caps):
        if "audio" in caps.to_string():
            return False
        return True

    def source_setup(discoverer, source):
        if uri.startswith("rtsp://"):
            source.connect("select-stream", select_stream)
            source.set_property("timeout", 1000000)
            if username and password:
                source.set_property("user-id", username)
                source.set_property("user-pw", password)

    discoverer = GstPbutils.Discoverer()
    discoverer.connect("source-setup", source_setup)

    try:
        file_info = discoverer.discover_uri(uri)
    except gi.repository.GLib.GError as e:
        logger.exception(f"Unsupported file type - {uri} Error:{e}")
        raise e
    for stream_info in file_info.get_stream_list():
        if isinstance(stream_info, GstPbutils.DiscovererVideoInfo):
            video_duration_sec = float(file_info.get_duration()) / 1e9
            video_codec = str(GstPbutils.pb_utils_get_codec_description(stream_info.get_caps()))
            video_width = int(stream_info.get_width())
            video_height = int(stream_info.get_height())
            video_fps = float(stream_info.get_framerate_num() / stream_info.get_framerate_denom())
            is_image = bool(stream_info.is_image())
            return video_duration_sec, video_fps, video_width, video_height
    logger.error(f"uri path: {uri} is not a video file")
    raise Exception(f"uri path: {uri} is not a video file")
