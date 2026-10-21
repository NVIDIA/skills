#!/usr/bin/env python3
"""Submit a public RFD4-Proteina NIM request, or prepare/extract it offline.

Usage:
    python scripts/design.py --request request.json --output-dir new-run --dry-run
    Omit --dry-run only when submission is requested.

Arguments:
    --request: required JSON file with complete inline mmCIF inputs.
    --output-dir: required new artifact directory; existing paths are refused.
    --mode: hosted (default) or local. Hosted uses NVIDIA_API_KEY, or NGC_API_KEY.
    --base-url: overrides RFD4_NIM_URL for local; defaults to localhost:8000.
    --timeout: positive socket timeout in seconds (default 600).
    --dry-run: save and preflight offline. --response-json: extract a saved response
    offline. These two options are mutually exclusive. Use --help for all defaults.

Output:
    JSON status on stdout and errors on stderr. Saves request.json and transport.json;
    after submission/extraction, also preserves response.raw, parsed response.json,
    returned mmCIF structures, sequences.fasta, and summary.json when available.

Exit codes:
    0: prepared offline or extracted a response passing basic checks; inspect status.
    1: request, transport, response, validation, or filesystem failure.
    2: argparse command-line usage error.
    3: HTTP 202 pending; preserve diagnostics without automatically resubmitting.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HOSTED_URL = "https://health.api.nvidia.com/v1/biology/ipd/rfd4-proteina/generative"
API_PATH = "/biology/ipd/rfd4-proteina/generative"
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 600
NVCF_POLL_SECONDS = 300
MAX_INLINE_FILENAME_LENGTH = 255
MAX_SEED = 2**63 - 1
MAX_NUM_STEPS = 1000
MAX_SEARCH_CHECKPOINTS = 21
TASKS = {
    "generation", "motif_scaffolding", "enzyme_design",
    "small_molecule_binder", "antibody_design", "partial_diffusion", "inverse_folding",
}


class ClientError(ValueError):
    pass


def load_json(data: bytes | str):
    def reject_constant(value):
        raise ClientError(f"Non-finite JSON value: {value}")
    return json.loads(data, parse_constant=reject_constant)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message: str) -> None:
    if not condition:
        raise ClientError(message)


def is_mmcif(content) -> bool:
    if not isinstance(content, str):
        return False
    first = next((s.strip() for s in content.splitlines() if s.strip() and not s.lstrip().startswith("#")), "")
    return first.startswith("data_")


def integer(value, name: str, lower=1, upper=None) -> None:
    require(type(value) is int and value >= lower and (upper is None or value <= upper),
            f"{name} must be an integer >= {lower}" + (f" and <= {upper}" if upper is not None else ""))


def validate_request(body: dict) -> None:
    """Catch common wire mistakes; the deployed DTO and pipeline remain authoritative."""
    require(isinstance(body, dict), "Request must be a JSON object")
    unknown = set(body) - {"task", "input", "sampling", "partial", "search", "metrics", "output"}
    require(not unknown, f"Unknown top-level fields: {sorted(unknown)}; this is not the RFdiffusion contigs API")
    require(isinstance(body.get("task"), str) and body["task"] in TASKS,
            "Choose a supported RFD4 task (see references/parameters.md)")
    inp = body.get("input")
    require(isinstance(inp, dict), "input must be an object")
    dialect = inp.get("format")
    require(isinstance(dialect, str) and dialect in {"annotated_cif", "condition_spec"},
            "input.format must be annotated_cif or condition_spec")
    files = [inp.get("file")] if dialect == "annotated_cif" else inp.get("files")
    require(isinstance(files, list) and files, "Attach at least one inline mmCIF file")
    names = set()
    for item in files:
        require(isinstance(item, dict), "Each inline file must be an object with filename and content")
        name = item.get("filename")
        require(isinstance(name, str) and 0 < len(name) <= MAX_INLINE_FILENAME_LENGTH and name not in {".", ".."}
                and not any(c in name for c in ("/", "\\", "\x00")), "filename must be a bare name, not a path")
        require(name.lower().endswith((".cif", ".mmcif")), "Use a .cif or .mmcif filename; renaming a PDB does not convert it")
        require(name not in names, f"Duplicate attached filename: {name}")
        require(is_mmcif(item.get("content")), f"{name}: content must contain actual mmCIF starting with data_")
        names.add(name)
    if dialect == "condition_spec":
        spec = inp.get("spec")
        require(isinstance(spec, dict), "input.spec must be a parsed JSON object, not YAML text")
        require(isinstance(spec.get("input"), str) and spec["input"] in names,
                "spec.input must name an attached file")

        def check_references(node):
            if isinstance(node, dict):
                for value in node.values():
                    check_references(value)
            elif isinstance(node, list):
                for value in node:
                    check_references(value)
            elif isinstance(node, str) and node.lower().endswith((".cif", ".mmcif", ".pdb", ".ent", ".cif.gz", ".pdb.gz", ".ent.gz")):
                require(node in names, f"Spec references unattached structure: {node}")
        check_references(spec)
        conditions = spec.get("conditions", {})
        require(isinstance(conditions, dict), "spec.conditions must be an object")
    sampling = body.get("sampling", {})
    require(isinstance(sampling, dict), "sampling must be an object")
    count = sampling.get("num_replicates", 1)
    integer(count, "sampling.num_replicates")
    if sampling.get("batch_size") is not None:
        integer(sampling["batch_size"], "sampling.batch_size")
    if sampling.get("seed") is not None:
        integer(sampling["seed"], "sampling.seed", 0, MAX_SEED)
    if sampling.get("num_steps") is not None:
        integer(sampling["num_steps"], "sampling.num_steps", upper=MAX_NUM_STEPS)
    if body["task"] == "partial_diffusion":
        require(isinstance(body.get("partial"), dict) and body["partial"], "partial_diffusion requires a non-empty partial block")
    else:
        require(body.get("partial") is None, "partial is only valid for partial_diffusion")
    search = body.get("search")
    if search is not None:
        require(isinstance(search, dict), "search must be an object")
        tags = search.get("reward_metrics_tags")
        require(isinstance(tags, list) and tags and all(isinstance(tag, str) and tag for tag in tags),
                "Search requires non-empty reward_metrics_tags")
        weights = search.get("weights")
        require(isinstance(weights, dict) and weights
                and all(isinstance(key, str) and key and type(value) in (int, float) and math.isfinite(value)
                        for key, value in weights.items()),
                "Search requires metric.output_key weights with finite numeric values")
        points = search.get("step_checkpoints")
        require(isinstance(points, list) and 2 <= len(points) <= MAX_SEARCH_CHECKPOINTS
                and all(type(x) is int for x in points) and points[0] == 0
                and all(a < b for a, b in zip(points, points[1:])), "Search checkpoints must increase strictly from 0")
        require(sampling.get("num_steps") is not None or sampling.get("recycle_strategy") is not None
                or sampling.get("recycle_schedule") is not None, "Search must pin its schedule")
    for name in ("metrics", "output"):
        require(isinstance(body.get(name, {}), dict), f"{name} must be an object")


def endpoint(mode: str, base_url: str | None) -> str:
    require(mode in {"hosted", "local"}, "Choose hosted or local mode")
    if mode == "hosted":
        require(base_url is None, "--base-url applies only to --mode local")
        return HOSTED_URL
    base = base_url or os.environ.get("RFD4_NIM_URL") or DEFAULT_LOCAL_BASE_URL
    parsed = urllib.parse.urlsplit(base)
    require(parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username
            and not parsed.password and not parsed.query and not parsed.fragment,
            "Local base URL must be http(s)://host:port without credentials, query, or fragment")
    require(parsed.path in {"", "/"}, "RFD4_NIM_URL/--base-url is a base URL, without the inference path")
    return base.rstrip("/") + API_PATH


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_headers(mode: str) -> dict[str, str]:
    """Load hosted credentials only for a live hosted submission, never for local."""
    require(mode in {"hosted", "local"}, "Choose hosted or local mode")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if mode == "hosted":
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
        require(isinstance(key, str) and bool(key.strip()),
                "Set NVIDIA_API_KEY (or NGC_API_KEY) to your NVIDIA API key for hosted inference")
        key = key.strip()
        # Reject unsafe header bytes with a constant error, never one quoting the key.
        require(all(33 <= ord(char) <= 126 for char in key),
                "The API key must contain printable ASCII characters without whitespace")
        headers["Authorization"] = f"Bearer {key}"
        headers["NVCF-POLL-SECONDS"] = str(NVCF_POLL_SECONDS)
    return headers


def post(url: str, payload: bytes, mode: str, timeout: float):
    request = urllib.request.Request(url, data=payload, headers=request_headers(mode), method="POST")
    opener = urllib.request.build_opener(NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        selected = {name: response.headers[name] for name in
                    ("Content-Type", "NVCF-REQID", "X-Request-ID", "Retry-After") if name in response.headers}
        return response.status, selected, response.read()


def extract_result(body: dict, result: dict, output_dir: Path) -> dict:
    require(isinstance(result, dict) and "error" not in result, "Response is not a successful design object")
    designs = result.get("designs")
    require(isinstance(designs, list) and designs, "Response has no designs; a scaffold/example response is not a result")
    errors, notices, records, fasta = [], [], [], []
    expected_task = body["task"]
    returned_task = result.get("task")
    if returned_task != expected_task:
        errors.append(f"Response task {result.get('task')!r} does not match {body['task']!r}")
    searching = body.get("search") is not None
    expected_count = body.get("sampling", {}).get("num_replicates", 1)
    if not searching and len(designs) != expected_count:
        errors.append(f"Expected {expected_count} plain-sampling designs, received {len(designs)}")
    if searching and not isinstance(result.get("search"), dict):
        errors.append("Search response is missing the search report")
    for key in ("model", "runtime", "schedule"):
        if not isinstance(result.get(key), dict):
            errors.append(f"Response lacks {key} provenance")
    for i, design in enumerate(designs):
        require(isinstance(design, dict), f"designs[{i}] is not an object")
        label = f"design_{i:04d}"
        record = {k: v for k, v in design.items() if k not in {"structure", "sequence"}}
        if not isinstance(design.get("design_id"), str) or not design["design_id"]:
            errors.append(f"{label}: missing design_id")
        diagnostics = design.get("diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get("all_coordinates_finite") is not True:
            errors.append(f"{label}: finite coordinates were not confirmed")
        origin = design.get("origin")
        valid_origins = {"search_final", "search_intermediate"} if searching else {"sample"}
        if not isinstance(origin, str) or origin not in valid_origins:
            errors.append(f"{label}: unexpected origin {origin!r}")
        structure = design.get("structure")
        if structure is not None:
            if isinstance(structure, dict) and structure.get("format") == "mmcif" and is_mmcif(structure.get("content")):
                name = label + ".cif"
                (output_dir / name).write_text(structure["content"], encoding="utf-8")
                record["structure_file"] = name
                record["suggested_filename"] = structure.get("filename")
            else:
                errors.append(f"{label}: malformed mmCIF structure")
        elif body.get("output", {}).get("structure_format", "mmcif") != "none":
            errors.append(f"{label}: requested structure is missing")
        sequence = design.get("sequence")
        if sequence is not None:
            if isinstance(sequence, str) and re.fullmatch(r"[A-Z]+", sequence):
                fasta.append(f">{label}\n{sequence}\n")
                record["fasta_id"] = label
                record["sequence_length"] = len(sequence)
            else:
                errors.append(f"{label}: sequence is not non-empty uppercase one-letter text")
        elif body.get("output", {}).get("include_sequence", True):
            errors.append(f"{label}: requested sequence is missing")
        if body.get("metrics", {}).get("tags") and not design.get("metrics"):
            notices.append(f"{label}: metrics were requested but returned empty")
        records.append(record)
    if fasta:
        (output_dir / "sequences.fasta").write_text("".join(fasta), encoding="utf-8")
    summary = {k: result.get(k) for k in
               ("request_id", "task", "duration_ms", "model", "runtime", "schedule", "search", "warnings", "aggregated_metrics")}
    summary.update({"status": "failed_validation" if errors else "completed", "design_count": len(designs),
                    "origin_counts": dict(Counter(str(d.get("origin")) for d in designs)),
                    "designs": records, "validation_errors": errors, "client_notices": notices})
    return summary


def run(args) -> int:
    payload = args.request.read_bytes()
    body = load_json(payload)
    validate_request(body)
    url = endpoint(args.mode, args.base_url)
    require(math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be positive and finite")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "request.json").write_bytes(payload)
    transport = {"started_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "url": url,
                 "request_sha256": hashlib.sha256(payload).hexdigest(), "timeout_seconds": args.timeout,
                 "source": "saved_response" if args.response_json else "network", "status": "prepared"}
    save_json(output_dir / "transport.json", transport)
    if args.dry_run:
        transport.update(source="dry_run", status="prepared_not_submitted")
        save_json(output_dir / "transport.json", transport)
        print(json.dumps({"status": transport["status"], "request": str(output_dir / "request.json"), "url": url}))
        return 0
    started = time.monotonic()
    try:
        if args.response_json:
            status, headers, raw = None, {}, args.response_json.read_bytes()
        else:
            status, headers, raw = post(url, payload, args.mode, args.timeout)
        (output_dir / "response.raw").write_bytes(raw)
        transport.update(http_status=status, response_headers=headers, wall_seconds=time.monotonic() - started,
                         response_sha256=hashlib.sha256(raw).hexdigest())
        if status == HTTPStatus.ACCEPTED:
            transport["status"] = "pending"
            save_json(output_dir / "transport.json", transport)
            print(json.dumps({"status": "pending", "output_dir": str(output_dir), "response_headers": headers,
                              "message": "202 is not completion. Preserve the request ID; no automatic POST retry or guessed polling URL."}))
            return 3
        if status is not None and status != HTTPStatus.OK:
            hint = (" Verify the NVIDIA API key and access to the selected service; "
                    "registry access does not establish hosted API access."
                    if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN} else "")
            raise ClientError(f"HTTP {status}; inspect response.raw. No automatic resubmission.{hint}")
        result = load_json(raw)
        save_json(output_dir / "response.json", result)
        summary = extract_result(body, result, output_dir)
        summary["transport_source"] = transport["source"]
        save_json(output_dir / "summary.json", summary)
        transport["status"] = summary["status"]
        save_json(output_dir / "transport.json", transport)
        print(json.dumps({"status": summary["status"], "design_count": summary["design_count"],
                          "summary": str(output_dir / "summary.json"), "validation_errors": summary["validation_errors"],
                          "warnings": summary["warnings"], "client_notices": summary["client_notices"]}))
        return 1 if summary["validation_errors"] else 0
    except (OSError, ValueError, urllib.error.URLError) as exc:
        transport.update(status="failed", error=str(exc), wall_seconds=time.monotonic() - started)
        save_json(output_dir / "transport.json", transport)
        raise ClientError(f"{exc} Artifacts: {output_dir}") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request", type=Path, required=True, help="Complete JSON request, including inline mmCIF")
    parser.add_argument("--mode", choices=("hosted", "local"), default="hosted")
    parser.add_argument("--base-url", help=f"Local base URL; default RFD4_NIM_URL or {DEFAULT_LOCAL_BASE_URL}")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are never overwritten")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help=f"HTTP socket timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})")
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--dry-run", action="store_true", help="Preflight and save request without network access")
    offline.add_argument("--response-json", type=Path, help="Extract a saved response offline; does not submit or prove a live run")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
