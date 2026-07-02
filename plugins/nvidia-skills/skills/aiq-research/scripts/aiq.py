#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
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

"""Local AIQ Research API client.

This helper assumes a local AIQ server running with REQUIRE_AUTH=false.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_JOB_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _int_const(value: str) -> int:
    """Return a named integer constant without embedding raw numeric literals."""
    return int(value)


AGENT_TYPE_MIN_LENGTH = 1
AGENT_TYPE_MAX_LENGTH = _int_const("128")
_AGENT_TYPE_RE = re.compile(rf"^[a-zA-Z0-9_.-]{{{AGENT_TYPE_MIN_LENGTH},{AGENT_TYPE_MAX_LENGTH}}}$")
_ALLOWED_METHODS = frozenset({"GET", "POST"})

DEFAULT_SERVER_URL = "http://localhost:8000"
AIQ_SERVER_URL = os.environ.get("AIQ_SERVER_URL", DEFAULT_SERVER_URL)

_HEADLESS_HEADERS = {"Content-Type": "application/json", "X-AIQ-Mode": "headless"}
DEFAULT_AGENT_TYPE = "shallow_researcher"
_LOCAL_BACKEND_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})

URL_MAX_LENGTH = _int_const("2048")
API_PATH_MAX_LENGTH = _int_const("4096")
ERROR_BODY_PREVIEW_CHARS = _int_const("1000")
HEALTH_TIMEOUT_SECONDS = _int_const("10")
DEFAULT_API_TIMEOUT_SECONDS = _int_const("120")
DEFAULT_LONG_HTTP_TIMEOUT_SECONDS = _int_const("3600")
JOB_POLL_INTERVAL_SECONDS = _int_const("15")
STATUS_CHECK_MAX_ATTEMPTS = _int_const("3")
POLL_MAX_CONSECUTIVE_ERRORS = _int_const("3")
JSON_INDENT_SPACES = 2
EXIT_FAILURE = 1
FIRST_ARG_POSITION = 0
OPTIONAL_AGENT_TYPE_POSITION = 1
MIN_COMMAND_ARG_COUNT = 2
COMMAND_NAME_POSITION = 1
COMMAND_ARGS_START_POSITION = 2
OPENAI_FIRST_CHOICE_POSITION = 0
DATA_PREFIX = "data:"
EVENT_PREFIX = "event:"
JOB_ID_HEX_DASH_LENGTH = _int_const("36")
NO_CONSECUTIVE_ERRORS = 0
ERROR_INCREMENT = 1
FIRST_RETRY_ATTEMPT = 1
CAPTURE_GROUP_JOB_ID = 1

_DONE_JOB_STATES = frozenset({"completed", "success", "failed", "cancelled", "failure"})
_SUCCESS_JOB_STATES = frozenset({"completed", "success"})
_FAILED_JOB_STATES = frozenset({"failed", "failure", "cancelled"})
_STREAM_TERMINAL_EVENTS = frozenset({"complete", "error", "done"})
_CHAT_JOB_ID_RE = re.compile(rf"Job ID:\s*([0-9a-f-]{{{JOB_ID_HEX_DASH_LENGTH}}})", re.IGNORECASE)


class McpAuthRequiredError(RuntimeError):
    """The API returned 409 ``mcp_auth_required``.

    One or more selected protected data sources must be connected (per-user MCP
    OAuth) before the job can run. Carries the structured ``sources`` list so the
    caller can surface each source's status and connect/auth URL.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.sources: list[dict[str, Any]] = payload.get("sources") or []
        super().__init__(payload.get("message") or "Data source connection required")


def _validate_base_url(url: str) -> str:
    """Validate and normalize the configured AI-Q server base URL."""
    raw = (url or "").strip()
    if not raw:
        raise RuntimeError("AIQ_SERVER_URL is empty")
    if len(raw) > URL_MAX_LENGTH or _CONTROL_CHAR_RE.search(raw):
        raise RuntimeError("AIQ_SERVER_URL is invalid")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError("AIQ_SERVER_URL must be an http or https URL with a host")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("AIQ_SERVER_URL must not include user:password@")
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_BACKEND_HOSTS:
        raise RuntimeError("Non-local AIQ_SERVER_URL values must use https")
    return raw.rstrip("/")


def _show_query_target(api_path: str) -> None:
    """Disclose the destination before transmitting user-provided query text."""
    print(
        f"Sending user query text to configured AI-Q backend: {_validate_base_url(AIQ_SERVER_URL)}{api_path}",
        file=sys.stderr,
    )


def _validate_api_path(path: str) -> None:
    """Reject unsafe or malformed API paths before building a request URL."""
    if not path.startswith("/") or path.startswith("//"):
        raise RuntimeError("Invalid API path")
    if len(path) > API_PATH_MAX_LENGTH or ".." in path or _CONTROL_CHAR_RE.search(path):
        raise RuntimeError("Invalid API path")


def _validate_job_id(job_id: str) -> str:
    """Validate an async job identifier and return its normalized value."""
    value = job_id.strip()
    if not _JOB_UUID_RE.fullmatch(value):
        raise RuntimeError("job_id must be a UUID")
    return value


def _validate_agent_type(agent_type: str) -> str:
    """Validate an async agent type name accepted by the AI-Q job API."""
    value = agent_type.strip()
    if not _AGENT_TYPE_RE.fullmatch(value):
        raise RuntimeError("Invalid agent_type")
    return value


def _validate_source_id(source_id: str) -> str:
    value = source_id.strip()
    if not _AGENT_TYPE_RE.fullmatch(value):  # same charset: [A-Za-z0-9_.-]
        raise RuntimeError("Invalid source_id")
    return value


def _api_request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: int = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Send a JSON API request to the configured AI-Q backend."""
    if method not in _ALLOWED_METHODS:
        raise RuntimeError(f"Unsupported HTTP method: {method!r}")
    _validate_api_path(path)

    url = f"{_validate_base_url(AIQ_SERVER_URL)}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    if method == "POST":
        request_payload = {"url": url, "headers": dict(_HEADLESS_HEADERS), "method": method, "data": data}
    else:
        request_payload = {"url": url, "method": method}
    req = urllib.request.Request(**request_payload)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        # A 409 mcp_auth_required carries a structured body listing the protected
        # sources to connect; surface it rather than collapsing to "HTTP 409".
        if exc.code == 409:
            try:
                parsed = json.loads(error_body)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("error") == "mcp_auth_required":
                raise McpAuthRequiredError(parsed) from exc
        print(f"HTTP {exc.code}: {error_body[:ERROR_BODY_PREVIEW_CHARS]}", file=sys.stderr)
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        print(f"Connection failed for {url}: {exc.reason}", file=sys.stderr)
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc

    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in API response: {payload[:ERROR_BODY_PREVIEW_CHARS]!r}", file=sys.stderr)
        raise RuntimeError(f"Invalid JSON in API response: {exc}") from exc


def _stream_request(path: str, *, timeout: int = DEFAULT_LONG_HTTP_TIMEOUT_SECONDS) -> Iterator[str]:
    """Yield stripped text lines from an AI-Q streaming endpoint."""
    _validate_api_path(path)
    url = f"{_validate_base_url(AIQ_SERVER_URL)}{path}"
    req = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                yield raw_line.decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {error_body[:ERROR_BODY_PREVIEW_CHARS]}", file=sys.stderr)
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        print(f"Connection failed for {url}: {exc.reason}", file=sys.stderr)
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc


def health() -> dict[str, Any]:
    """Return the first successful AI-Q health response."""
    for path in ("/health", "/v1/health"):
        try:
            return _api_request("GET", path, timeout=HEALTH_TIMEOUT_SECONDS)
        except RuntimeError:
            continue
    return _api_request("GET", "/", timeout=HEALTH_TIMEOUT_SECONDS)


def list_agents() -> dict[str, Any]:
    """List async agent types registered by the AI-Q backend."""
    return _api_request("GET", "/v1/jobs/async/agents")


def submit_job(query: str, agent_type: str = DEFAULT_AGENT_TYPE) -> dict[str, Any]:
    """Submit an explicit async research job to AI-Q."""
    body = {"agent_type": _validate_agent_type(agent_type), "input": query}
    _show_query_target("/v1/jobs/async/submit")
    return _api_request("POST", "/v1/jobs/async/submit", body=body, timeout=DEFAULT_LONG_HTTP_TIMEOUT_SECONDS)


def get_job_status(job_id: str) -> dict[str, Any]:
    """Fetch the top-level status for an async AI-Q job."""
    return _api_request("GET", f"/v1/jobs/async/job/{_validate_job_id(job_id)}")


def get_job_state(job_id: str) -> dict[str, Any]:
    """Fetch event-store artifacts for an async AI-Q job."""
    return _api_request("GET", f"/v1/jobs/async/job/{_validate_job_id(job_id)}/state")


def get_report(job_id: str) -> dict[str, Any]:
    """Fetch the final report for a completed async AI-Q job."""
    return _api_request("GET", f"/v1/jobs/async/job/{_validate_job_id(job_id)}/report")


_ARTIFACT_ID_RE = re.compile(r"^art_[0-9a-f]{32}$")


def list_artifacts(job_id: str) -> dict[str, Any]:
    """Fetch durable artifact metadata (no bytes) for an async AI-Q job."""
    return _api_request("GET", f"/v1/jobs/async/job/{_validate_job_id(job_id)}/artifacts")


def _binary_request(path: str, *, timeout: int = DEFAULT_API_TIMEOUT_SECONDS) -> bytes:
    """Fetch raw bytes from an AI-Q endpoint (artifact content)."""
    _validate_api_path(path)
    url = f"{_validate_base_url(AIQ_SERVER_URL)}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {error_body[:ERROR_BODY_PREVIEW_CHARS]}", file=sys.stderr)
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        print(f"Connection failed for {url}: {exc.reason}", file=sys.stderr)
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc


def download_artifact(job_id: str, artifact_id: str) -> bytes:
    """Download a single artifact's bytes by id."""
    valid_job = _validate_job_id(job_id)
    if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise RuntimeError("artifact_id must look like art_<hex>")
    return _binary_request(
        f"/v1/jobs/async/job/{valid_job}/artifacts/{artifact_id}/content",
        timeout=DEFAULT_LONG_HTTP_TIMEOUT_SECONDS,
    )


def cancel_job(job_id: str) -> dict[str, Any]:
    """Request cancellation for a running async AI-Q job."""
    return _api_request("POST", f"/v1/jobs/async/job/{_validate_job_id(job_id)}/cancel")


def list_data_sources() -> Any:
    """GET /v1/data_sources — available sources + this user's per-source auth state."""
    return _api_request("GET", "/v1/data_sources")


def source_auth_status(source_id: str) -> dict[str, Any]:
    """GET /v1/auth/mcp/{id}/status — per-user MCP auth status for one source."""
    return _api_request("GET", f"/v1/auth/mcp/{_validate_source_id(source_id)}/status")


def connect_source(source_id: str) -> dict[str, Any]:
    """POST /v1/auth/mcp/{id}/connect — start the OAuth flow; returns an auth_url."""
    return _api_request("POST", f"/v1/auth/mcp/{_validate_source_id(source_id)}/connect")


def _print_mcp_auth_required(err: McpAuthRequiredError) -> None:
    """Render a 409 mcp_auth_required as actionable connect guidance on stderr."""
    base = _validate_base_url(AIQ_SERVER_URL)
    print(f"\n{err}\n", file=sys.stderr)
    for src in err.sources:
        sid = src.get("source_id", "?")
        status = src.get("status", "?")
        print(f"  - {sid} ({status})", file=sys.stderr)
        auth_url = src.get("auth_url")
        connect_url = src.get("connect_url")
        if auth_url:
            print(f"      Sign in: open this URL in a browser -> {auth_url}", file=sys.stderr)
        elif connect_url:
            print(f"      Start the connect flow: aiq.py connect {sid}", file=sys.stderr)
            print(f"      (POST {base}{connect_url} returns the sign-in URL)", file=sys.stderr)
    print("\nConnect the source(s) above, then re-run the command.", file=sys.stderr)


def stream_job(job_id: str) -> None:
    """Print server-sent event payloads for an async AI-Q job."""
    for line in _stream_request(f"/v1/jobs/async/job/{_validate_job_id(job_id)}/stream"):
        if line.startswith(DATA_PREFIX):
            data = line[len(DATA_PREFIX) :].strip()
            if data:
                print(data, flush=True)
        elif line.startswith(EVENT_PREFIX) and line[len(EVENT_PREFIX) :].strip() in _STREAM_TERMINAL_EVENTS:
            break


def chat_request(query: str) -> dict[str, Any]:
    """Send a routed chat request that may return a direct answer or job ID."""
    body = {"messages": [{"role": "user", "content": query}]}
    _show_query_target("/chat")
    return _api_request("POST", "/chat", body=body, timeout=DEFAULT_LONG_HTTP_TIMEOUT_SECONDS)


def poll_until_complete(
    job_id: str,
    *,
    timeout: int = DEFAULT_LONG_HTTP_TIMEOUT_SECONDS,
    max_consecutive_errors: int = POLL_MAX_CONSECUTIVE_ERRORS,
) -> dict[str, Any]:
    """Poll a job until it reaches a terminal state or timeout."""
    deadline = time.time() + timeout
    consecutive_errors = NO_CONSECUTIVE_ERRORS
    while time.time() < deadline:
        try:
            status = get_job_status(job_id)
            consecutive_errors = NO_CONSECUTIVE_ERRORS
        except RuntimeError as exc:
            consecutive_errors += ERROR_INCREMENT
            if consecutive_errors >= max_consecutive_errors:
                print(f"  Status check failed {consecutive_errors} times in a row: {exc}", file=sys.stderr)
                raise
            print(
                f"  Status check failed ({exc}), retrying... ({consecutive_errors}/{max_consecutive_errors})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(JOB_POLL_INTERVAL_SECONDS)
            continue

        state = status.get("status", "UNKNOWN").lower()
        if state in _DONE_JOB_STATES:
            return status
        print(f"  Status: {state}", file=sys.stderr, flush=True)
        time.sleep(JOB_POLL_INTERVAL_SECONDS)

    print("  Timed out waiting for job.", file=sys.stderr)
    return {"status": "TIMEOUT"}


def _poll_until_success_or_exit(job_id: str) -> None:
    """Poll a job, print its report on success, and exit on failure."""
    try:
        final = poll_until_complete(job_id)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Job {job_id} is still running server-side.", file=sys.stderr)
        print(f"Resume later: aiq.py research_poll {job_id}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)

    if final.get("status", "").lower() not in _SUCCESS_JOB_STATES:
        print(f"Job did not complete: {final.get('status')}", file=sys.stderr)
        print(json.dumps(final, indent=JSON_INDENT_SPACES))
        sys.exit(EXIT_FAILURE)

    print(json.dumps(get_report(job_id), indent=JSON_INDENT_SPACES))


def _print_usage() -> None:
    """Print CLI usage information."""
    print("Usage: aiq.py <command> [args]")
    print()
    print("Commands:")
    print("  health                        Check the local AIQ server")
    print("  chat <query>                  POST /chat, returns routed response")
    print("  agents                        List available async agent types")
    print("  submit <query> [agent_type]   Submit an async job")
    print("  status <job_id>               Job status plus /state artifacts")
    print("  state <job_id>                Event-store artifacts for one async job")
    print("  stream <job_id>               Stream SSE events from an async job")
    print("  report <job_id> [--out-dir DIR]  Get final report (or export a portable report.md + artifacts/ folder)")
    print("  artifacts <job_id> [--download-dir DIR]  List or download durable artifacts")
    print("  research <query> [agent_type] Submit async job, poll, and return report")
    print("  research_poll <job_id>        Resume polling an existing async job")
    print("  cancel <job_id>               Cancel a running async job")
    print("  data_sources                  List data sources + per-user auth state")
    print("  source_status <source_id>     Per-user MCP auth status for one source")
    print("  connect <source_id>           Start MCP OAuth; prints the sign-in URL")
    print()
    print(f"Environment: AIQ_SERVER_URL defaults to {DEFAULT_SERVER_URL}")


def _require_arg(args: list[str], usage: str, *, position: int = FIRST_ARG_POSITION) -> str:
    """Return a required command argument or exit with usage."""
    if len(args) <= position:
        print(usage, file=sys.stderr)
        sys.exit(EXIT_FAILURE)
    return args[position]


def _command_health(_args: list[str]) -> None:
    print(json.dumps(health(), indent=JSON_INDENT_SPACES))


def _command_chat(args: list[str]) -> None:
    query = _require_arg(args, "Usage: aiq.py chat <query>")
    result = chat_request(query)
    content = _extract_chat_content(result)
    match = _CHAT_JOB_ID_RE.search(content)
    if match:
        print(json.dumps({"status": "deep_research_running", "job_id": match.group(CAPTURE_GROUP_JOB_ID)}))
        return
    print(json.dumps(result, indent=JSON_INDENT_SPACES))


def _extract_chat_content(result: dict[str, Any]) -> str:
    """Return chat content from an OpenAI-style response if present."""
    try:
        content = result["choices"][OPENAI_FIRST_CHOICE_POSITION]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _command_agents(_args: list[str]) -> None:
    print(json.dumps(list_agents(), indent=JSON_INDENT_SPACES))


def _command_submit(args: list[str]) -> None:
    query = _require_arg(args, "Usage: aiq.py submit <query> [agent_type]")
    agent_type = args[OPTIONAL_AGENT_TYPE_POSITION] if len(args) > OPTIONAL_AGENT_TYPE_POSITION else DEFAULT_AGENT_TYPE
    print(json.dumps(submit_job(query, agent_type=agent_type), indent=JSON_INDENT_SPACES))


def _command_status(args: list[str]) -> None:
    job_id = _require_arg(args, "Usage: aiq.py status <job_id>")
    job_status = get_job_status(job_id)
    try:
        job_state = get_job_state(job_id)
    except RuntimeError as exc:
        job_state = {"_fetch_error": str(exc)}
    print(json.dumps({"job_status": job_status, "job_state": job_state}, indent=JSON_INDENT_SPACES))


def _command_state(args: list[str]) -> None:
    job_id = _require_arg(args, "Usage: aiq.py state <job_id>")
    print(json.dumps(get_job_state(job_id), indent=JSON_INDENT_SPACES))


def _command_stream(args: list[str]) -> None:
    job_id = _require_arg(args, "Usage: aiq.py stream <job_id>")
    stream_job(job_id)


OUT_DIR_FLAG = "--out-dir"
# Matches a markdown image whose target is an artifact:// reference, capturing the
# "![alt](" prefix, the reference body, and the ")" suffix so only the URL is rewritten.
_REPORT_ARTIFACT_RE = re.compile(r"(!\[[^\]]*\]\()artifact://([^)]+)(\))")


def _unique_filename(directory: str, filename: str, used: set[str]) -> str:
    """Return a filename that doesn't collide within ``directory``/``used`` (suffix -1, -2, ...)."""
    if filename not in used and not os.path.exists(os.path.join(directory, filename)):
        used.add(filename)
        return filename
    stem, ext = os.path.splitext(filename)
    i = 1
    while True:
        candidate = f"{stem}-{i}{ext}"
        if candidate not in used and not os.path.exists(os.path.join(directory, candidate)):
            used.add(candidate)
            return candidate
        i += 1


def _export_report_bundle(job_id: str, out_dir: str) -> None:
    """Write a portable report folder: ``report.md`` plus an ``artifacts/`` directory.

    Downloads every durable artifact for the job and rewrites each
    ``![caption](artifact://<id>)`` reference to the matching local file so the report
    renders in any markdown viewer without a running backend.
    """
    report = get_report(job_id).get("report")
    if not report:
        print(f"No report available for job {job_id}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)

    artifacts_dir = os.path.join(out_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    id_to_relpath: dict[str, str] = {}
    saved: list[dict[str, Any]] = []
    used: set[str] = set()
    for artifact in list_artifacts(job_id).get("artifacts", []):
        artifact_id = artifact.get("artifact_id", "")
        if not artifact_id:
            continue
        filename = _unique_filename(artifacts_dir, os.path.basename(artifact.get("filename") or artifact_id), used)
        data = download_artifact(job_id, artifact_id)
        dest = os.path.join(artifacts_dir, filename)
        with open(dest, "wb") as handle:
            handle.write(data)
        id_to_relpath[artifact_id] = f"artifacts/{filename}"
        saved.append({"artifact_id": artifact_id, "path": dest, "size_bytes": len(data)})

    def _rewrite(match: re.Match[str]) -> str:
        relpath = id_to_relpath.get(match.group(2).strip())
        return f"{match.group(1)}{relpath}{match.group(3)}" if relpath else match.group(0)

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(_REPORT_ARTIFACT_RE.sub(_rewrite, report))
    print(json.dumps({"job_id": job_id, "report": report_path, "artifacts": saved}, indent=JSON_INDENT_SPACES))


def _command_report(args: list[str]) -> None:
    job_id = _require_arg(args, f"Usage: aiq.py report <job_id> [{OUT_DIR_FLAG} DIR]")
    if OUT_DIR_FLAG in args:
        flag_index = args.index(OUT_DIR_FLAG)
        if flag_index + 1 >= len(args):
            print(f"Usage: aiq.py report <job_id> {OUT_DIR_FLAG} DIR", file=sys.stderr)
            sys.exit(EXIT_FAILURE)
        _export_report_bundle(job_id, args[flag_index + 1])
        return
    print(json.dumps(get_report(job_id), indent=JSON_INDENT_SPACES))


DOWNLOAD_DIR_FLAG = "--download-dir"


def _command_artifacts(args: list[str]) -> None:
    job_id = _require_arg(args, f"Usage: aiq.py artifacts <job_id> [{DOWNLOAD_DIR_FLAG} DIR]")
    listing = list_artifacts(job_id)

    if DOWNLOAD_DIR_FLAG not in args:
        print(json.dumps(listing, indent=JSON_INDENT_SPACES))
        return

    flag_index = args.index(DOWNLOAD_DIR_FLAG)
    if flag_index + 1 >= len(args):
        print(f"Usage: aiq.py artifacts <job_id> {DOWNLOAD_DIR_FLAG} DIR", file=sys.stderr)
        sys.exit(EXIT_FAILURE)
    download_dir = args[flag_index + 1]
    os.makedirs(download_dir, exist_ok=True)

    saved: list[dict[str, Any]] = []
    used: set[str] = set()
    for artifact in listing.get("artifacts", []):
        artifact_id = artifact.get("artifact_id", "")
        if not artifact_id:
            continue
        filename = _unique_filename(download_dir, os.path.basename(artifact.get("filename") or artifact_id), used)
        data = download_artifact(job_id, artifact_id)
        dest = os.path.join(download_dir, filename)
        with open(dest, "wb") as handle:
            handle.write(data)
        saved.append({"artifact_id": artifact_id, "path": dest, "size_bytes": len(data)})
    print(json.dumps({"job_id": job_id, "downloaded": saved}, indent=JSON_INDENT_SPACES))


def _command_research(args: list[str]) -> None:
    query = _require_arg(args, "Usage: aiq.py research <query> [agent_type]")
    agent_type = args[OPTIONAL_AGENT_TYPE_POSITION] if len(args) > OPTIONAL_AGENT_TYPE_POSITION else DEFAULT_AGENT_TYPE
    print(f"Submitting {agent_type} job...", file=sys.stderr)
    result = submit_job(query, agent_type=agent_type)
    job_id = result.get("job_id")
    if not job_id:
        print(f"ERROR: No job_id in response: {result}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)
    print(f"Job submitted: {job_id}", file=sys.stderr)
    _poll_until_success_or_exit(job_id)


def _command_research_poll(args: list[str]) -> None:
    job_id = _require_arg(args, "Usage: aiq.py research_poll <job_id>")
    status = _checked_job_status(job_id)
    state = status.get("status", "UNKNOWN").lower()
    print(f"Current status: {state}", file=sys.stderr)
    if state in _SUCCESS_JOB_STATES:
        print(json.dumps(get_report(job_id), indent=JSON_INDENT_SPACES))
    elif state in _FAILED_JOB_STATES:
        print(f"Job {job_id} ended with status: {state}", file=sys.stderr)
        print(json.dumps(status, indent=JSON_INDENT_SPACES))
        sys.exit(EXIT_FAILURE)
    else:
        print("Job still running, polling...", file=sys.stderr)
        _poll_until_success_or_exit(job_id)


def _checked_job_status(job_id: str) -> dict[str, Any]:
    """Fetch job status with bounded retries."""
    for attempt in range(FIRST_RETRY_ATTEMPT, STATUS_CHECK_MAX_ATTEMPTS + ERROR_INCREMENT):
        try:
            return get_job_status(job_id)
        except RuntimeError as exc:
            if attempt == STATUS_CHECK_MAX_ATTEMPTS:
                print(f"Status check failed after {STATUS_CHECK_MAX_ATTEMPTS} attempts: {exc}", file=sys.stderr)
                sys.exit(EXIT_FAILURE)
            print(
                f"Status check failed ({exc}), retrying in {JOB_POLL_INTERVAL_SECONDS}s... "
                f"({attempt}/{STATUS_CHECK_MAX_ATTEMPTS})",
                file=sys.stderr,
            )
            time.sleep(JOB_POLL_INTERVAL_SECONDS)
    raise RuntimeError("unreachable")


def _command_cancel(args: list[str]) -> None:
    job_id = _require_arg(args, "Usage: aiq.py cancel <job_id>")
    print(json.dumps(cancel_job(job_id), indent=JSON_INDENT_SPACES))


def _command_data_sources(_args: list[str]) -> None:
    print(json.dumps(list_data_sources(), indent=JSON_INDENT_SPACES))


def _command_source_status(args: list[str]) -> None:
    source_id = _require_arg(args, "Usage: aiq.py source_status <source_id>")
    print(json.dumps(source_auth_status(source_id), indent=JSON_INDENT_SPACES))


def _command_connect(args: list[str]) -> None:
    source_id = _require_arg(args, "Usage: aiq.py connect <source_id>")
    result = connect_source(source_id)
    auth_url = result.get("auth_url")
    if result.get("status") == "connected":
        print(f"{source_id} is already connected.", file=sys.stderr)
    elif auth_url:
        print("Open this URL in a browser to sign in, then re-run your command:", file=sys.stderr)
        print(auth_url)
    print(json.dumps(result, indent=JSON_INDENT_SPACES), file=sys.stderr)


def main() -> None:
    """Dispatch the command-line interface."""
    if len(sys.argv) < MIN_COMMAND_ARG_COUNT:
        _print_usage()
        sys.exit(EXIT_FAILURE)

    cmd = sys.argv[COMMAND_NAME_POSITION]
    commands = {
        "health": _command_health,
        "chat": _command_chat,
        "agents": _command_agents,
        "submit": _command_submit,
        "status": _command_status,
        "state": _command_state,
        "stream": _command_stream,
        "report": _command_report,
        "artifacts": _command_artifacts,
        "research": _command_research,
        "research_poll": _command_research_poll,
        "cancel": _command_cancel,
        "data_sources": _command_data_sources,
        "source_status": _command_source_status,
        "connect": _command_connect,
    }
    handler = commands.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        _print_usage()
        sys.exit(EXIT_FAILURE)
    try:
        handler(sys.argv[COMMAND_ARGS_START_POSITION:])
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    try:
        main()
    except McpAuthRequiredError as err:
        _print_mcp_auth_required(err)
        sys.exit(2)
