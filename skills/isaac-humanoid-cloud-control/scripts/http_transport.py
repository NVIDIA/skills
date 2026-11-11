# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP(S)-only transport for the viewer and Mission Control helpers.

Use open_http(url_or_request, timeout=seconds) as a context manager. Returns an
HTTPResponse; raises HTTPError for non-2xx responses and URLError for connection
failures. Redirects are rejected, so requests cannot switch destination or scheme.
"""

from contextlib import contextmanager
import http.client
import io
import math
import urllib.error
import urllib.parse
import urllib.request


@contextmanager
def open_http(target, *, timeout):
    """Open one validated HTTP(S) request, without redirects or proxy discovery."""
    request = target if isinstance(target, urllib.request.Request) else urllib.request.Request(target)
    url = request.full_url
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment or any(ord(c) <= 32 or ord(c) == 127 for c in url)):
        raise ValueError("Expected an HTTP(S) URL without credentials, fragment, or whitespace")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("HTTP timeout must be positive and finite")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, port=parsed.port, timeout=timeout)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    try:
        try:
            connection.request(request.get_method(), path, body=request.data,
                               headers=dict(request.header_items()))
            response = connection.getresponse()
        except TimeoutError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise urllib.error.URLError(exc) from exc
        if not 200 <= response.status < 300:
            raise urllib.error.HTTPError(url, response.status, response.reason,
                                         response.headers, io.BytesIO(response.read()))
        yield response
    finally:
        connection.close()
