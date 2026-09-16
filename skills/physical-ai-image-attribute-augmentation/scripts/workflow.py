#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Operate an SDG workflow through the Workflow API without exposing secrets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from payload import PayloadError, normalize_payload

TERMINAL_STATUSES = {"success", "failed", "stopped", "unknown"}


class WorkflowError(RuntimeError):
    """A safe-to-display API or local workflow error."""


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise WorkflowError(f"{name} is required")
    return value


def api_context() -> tuple[str, str]:
    endpoint = required_env("WEBSERVER_ENDPOINT").rstrip("/")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WorkflowError("WEBSERVER_ENDPOINT must be an http:// or https:// URL")
    return endpoint, required_env("NGC_API_KEY")


def check_health(endpoint: str, api_key: str) -> None:
    result = request_json(f"{endpoint}/health", api_key, timeout=15)
    if result.get("status") != "healthy":
        raise WorkflowError("SDG Workflow API health check did not report healthy")


def request_json(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 180,
    retries: int = 0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise WorkflowError(
            f"Unexpected URL scheme: {parsed.scheme!r} — only http/https are permitted"
        )
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise WorkflowError("API returned a non-object JSON response")
                return decoded
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")[:2000]
            raise WorkflowError(
                f"API returned HTTP {exc.code}: {response_body}"
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise WorkflowError(
                    f"API request failed after {retries + 1} attempt(s): {exc}"
                ) from exc
            time.sleep(min(10 * (attempt + 1), 30))
        except json.JSONDecodeError as exc:
            raise WorkflowError("API returned invalid JSON") from exc
    raise AssertionError("unreachable")


def start_workflow(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    workflow_name: str,
    compute: str,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({"workflow_name": workflow_name, "compute": compute})
    result = request_json(
        f"{endpoint}/workflow?{query}",
        api_key,
        method="POST",
        payload=payload,
        retries=2,
    )
    if not result.get("workflow_id"):
        raise WorkflowError("API response did not include workflow_id")
    return result


def get_status(endpoint: str, api_key: str, workflow_id: str) -> dict[str, Any]:
    return request_json(
        f"{endpoint}/workflow/{urllib.parse.quote(workflow_id, safe='')}", api_key
    )


def get_results(endpoint: str, api_key: str, workflow_id: str) -> dict[str, Any]:
    return request_json(
        f"{endpoint}/workflow/{urllib.parse.quote(workflow_id, safe='')}/results",
        api_key,
    )


def safe_destination(output_dir: Path, relative_path: str) -> Path:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WorkflowError(f"API returned an unsafe result path: {relative_path!r}")
    root = output_dir.resolve()
    destination = (root / Path(*path.parts)).resolve()
    if destination != root and root not in destination.parents:
        raise WorkflowError(f"Result path escapes output directory: {relative_path!r}")
    return destination


def download_results(results: dict[str, Any], output_dir: Path) -> int:
    files = results.get("files")
    if not isinstance(files, list):
        raise WorkflowError("Results response has no files list")
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise WorkflowError("Results response contains an invalid file entry")
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            raise WorkflowError(f"Result {entry['path']!r} has no download URL")
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme not in ("https", "http"):
            raise WorkflowError(
                f"Download URL has unexpected scheme: {parsed_url.scheme!r}"
            )
        destination = safe_destination(output_dir, entry["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        try:
            with (
                urllib.request.urlopen(url, timeout=300) as response,  # noqa: S310
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            temporary.replace(destination)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise WorkflowError(f"Failed to download result {entry['path']!r}") from exc
        print(f"Downloaded: {entry['path']}")
        downloaded += 1
    return downloaded


IMAGE_ATTRIBUTE_AUGMENTATION_WORKFLOW = "image_attribute_augmentation_dag"


def load_start_payload(payload_path: Path, workflow_name: str) -> dict[str, Any]:
    """Load a start payload, enforcing the bundled contract for the augmentation DAGs."""
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkflowError("Payload must be a JSON object")
    if (
        workflow_name == IMAGE_ATTRIBUTE_AUGMENTATION_WORKFLOW
        or workflow_name.startswith(f"{IMAGE_ATTRIBUTE_AUGMENTATION_WORKFLOW}_")
    ):
        try:
            return normalize_payload(payload)
        except PayloadError as exc:
            raise WorkflowError(
                f"Image Attribute Augmentation payload validation failed: {exc}"
            ) from exc
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SDG workflow operations through the SDG Workflow API."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "preflight", help="Validate API credentials and upload configuration presence."
    )
    start = commands.add_parser("start", help="Start a workflow and print its ID.")
    start.add_argument("--payload", required=True)
    start.add_argument("--workflow-name", required=True)
    start.add_argument(
        "--compute", required=True, choices=("nvcf", "k8s", "kubernetes", "osmo")
    )
    status = commands.add_parser("status", help="Query workflow status once.")
    status.add_argument("--workflow-id", required=True)
    results = commands.add_parser(
        "results", help="List result metadata without signed URLs."
    )
    results.add_argument("--workflow-id", required=True)
    download = commands.add_parser(
        "download", help="Download results with traversal protection."
    )
    download.add_argument("--workflow-id", required=True)
    download.add_argument("--output-dir", required=True)
    stop = commands.add_parser("stop", help="Stop an active workflow.")
    stop.add_argument("--workflow-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        endpoint, api_key = api_context()
        if args.command == "preflight":
            check_health(endpoint, api_key)
            print("API configuration: ready")
            print("API health: healthy")
            print(
                "Upload configuration: ready"
                if os.environ.get("UPLOAD_DESTINATION", "").strip()
                else "Upload configuration: not set (only required for local dataset upload)"
            )
            return 0
        if args.command == "start":
            payload = load_start_payload(Path(args.payload), args.workflow_name)
            result = start_workflow(
                endpoint, api_key, payload, args.workflow_name, args.compute
            )
            print(f"Workflow ID: {result['workflow_id']}")
            print(f"Status: {result.get('status', 'unknown')}")
            return 0
        if args.command == "status":
            result = get_status(endpoint, api_key, args.workflow_id)
            status_value = str(result.get("status", "unknown"))
            print(f"Workflow ID: {args.workflow_id}")
            print(f"Status: {status_value}")
            return 1 if status_value in {"failed", "stopped", "unknown"} else 0
        if args.command == "results":
            result = get_results(endpoint, api_key, args.workflow_id)
            safe_result = {
                "workflow_id": result.get("workflow_id", args.workflow_id),
                "file_count": result.get("file_count", len(result.get("files", []))),
                "total_size": result.get("total_size"),
                "files": [
                    {key: entry.get(key) for key in ("path", "size", "last_modified")}
                    for entry in result.get("files", [])
                    if isinstance(entry, dict)
                ],
            }
            print(json.dumps(safe_result, indent=2))
            return 0
        if args.command == "download":
            count = download_results(
                get_results(endpoint, api_key, args.workflow_id), Path(args.output_dir)
            )
            print(f"Downloaded {count} file(s) to {args.output_dir}")
            return 0
        result = request_json(
            f"{endpoint}/workflow/{urllib.parse.quote(args.workflow_id, safe='')}",
            api_key,
            method="DELETE",
        )
        print(f"Workflow ID: {result.get('workflow_id', args.workflow_id)}")
        print(f"Status: {result.get('status', 'unknown')}")
        return 0
    except (OSError, json.JSONDecodeError, WorkflowError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
