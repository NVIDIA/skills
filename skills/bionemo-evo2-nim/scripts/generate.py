#!/usr/bin/env python3
"""Execute one Evo 2 generation request and save validated, reproducible outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import groupby
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import requests


HOSTED_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"


def clean_dna(value: str) -> str:
    sequence = "".join(value.upper().split())
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("Provide a nonempty A/C/G/T sequence; ambiguity codes need an explicit modeling choice")
    return sequence


def number(value: object, label: str, minimum: float, maximum: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} is outside the allowed range")
    return float(value)


def validate_result(result: object, num_tokens: int) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from Evo 2")
    sequence = result.get("sequence")
    if not isinstance(sequence, str) or not sequence or set(sequence.upper()) - set("ACGT"):
        raise ValueError("Response sequence must be nonempty A/C/G/T DNA")
    if len(sequence) != num_tokens:
        raise ValueError(f"Requested {num_tokens} new bases, received {len(sequence)}; inspect the raw response")
    probs = result.get("sampled_probs")
    if not isinstance(probs, list) or len(probs) != len(sequence):
        raise ValueError("sampled_probs must contain one probability per generated base")
    for value in probs:
        number(value, "sampled_probs", 0, 1)
    number(result.get("elapsed_ms"), "elapsed_ms", 0)
    timings = result.get("elapsed_ms_per_token")
    if timings is not None:
        if not isinstance(timings, list) or len(timings) != len(sequence):
            raise ValueError("elapsed_ms_per_token must contain one timing per generated base")
        for value in timings:
            number(value, "elapsed_ms_per_token", 0)
    dna = sequence.upper()
    return {
        "generated_bases": len(sequence),
        "gc_fraction": (dna.count("G") + dna.count("C")) / len(dna),
        "ambiguous_base_fraction": 0.0,
        "longest_homopolymer": max(sum(1 for _ in group) for _, group in groupby(dna)),
        "sampled_probs": {
            "count": len(probs), "min": min(probs), "max": max(probs),
            "mean": sum(probs) / len(probs), "valid": True,
        },
        "elapsed_ms": result["elapsed_ms"],
        "per_token_timing_available": timings is not None,
    }


def generate(args: argparse.Namespace) -> dict:
    sequence = clean_dna(args.sequence)
    if args.num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    number(args.temperature, "temperature", 0, 1.3)
    number(args.top_k, "top_k", 0, 6)
    number(args.top_p, "top_p", 0, 1)
    number(args.timeout, "timeout", 1)
    mode = args.mode or os.getenv("NIM_API_MODE") or ("local" if os.getenv("EVO2_NIM_URL") else "hosted")
    if mode not in {"hosted", "local"}:
        raise ValueError("NIM_API_MODE must be hosted or local")
    headers = {"Content-Type": "application/json"}
    if mode == "hosted":
        key = os.getenv("NGC_API_KEY")
        if not key:
            raise ValueError("Set NGC_API_KEY in the environment for hosted Evo 2")
        headers["Authorization"] = f"Bearer {key}"
        url = HOSTED_URL
    else:
        base = os.getenv("EVO2_NIM_URL", "http://localhost:8000").rstrip("/")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("EVO2_NIM_URL must be an HTTP(S) service URL without embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("EVO2_NIM_URL must not include a query or fragment")
        url = f"{base}/biology/arc/evo2/generate"

    payload = {
        "sequence": sequence, "num_tokens": args.num_tokens,
        "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
        "random_seed": args.seed, "enable_sampled_probs": True,
        "enable_elapsed_ms_per_token": True,
    }
    output = args.output_dir.resolve()
    paths = {name: output / filename for name, filename in {
        "request": "request.json", "response": "response.json",
        "raw_response": "response.raw",
        "fasta": "generated.fasta", "metrics": "metrics.json",
    }.items()}
    # Directory creation atomically reserves every artifact path for this run.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError("Output directory already exists; choose a new --output-dir for this request") from exc
    paths["request"].write_text(json.dumps(payload, indent=2) + "\n")
    # Keep the endpoint and request visible without ever printing the credential.
    print(json.dumps({
        "event": "request", "method": "POST", "url": url, "payload": payload,
        "authentication": "Authorization: Bearer from NGC_API_KEY" if mode == "hosted" else "none",
    }), flush=True)
    start = time.monotonic()
    response = requests.post(url, headers=headers, json=payload, timeout=(10, args.timeout), allow_redirects=False)
    wall_ms = round((time.monotonic() - start) * 1000)
    # Keep the exact body even when HTTP status, JSON parsing, or validation fails.
    paths["raw_response"].write_bytes(response.content)
    if response.status_code != 200:
        raise RuntimeError(f"Evo 2 returned HTTP {response.status_code}; generation was not completed")
    result = response.json()
    # The parsed JSON supplements the raw body; never manufacture missing fields.
    paths["response"].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    metrics = validate_result(result, args.num_tokens)
    fasta = f">evo2_generated seed={args.seed}\n{result['sequence']}\n"
    paths["fasta"].write_text(fasta)
    # Verify persisted content before presenting the run as complete.
    if json.loads(paths["response"].read_text()) != result or paths["fasta"].read_text() != fasta:
        raise RuntimeError("Saved artifacts do not match the response")
    metrics.update({
        "status": "completed", "http_status": response.status_code, "mode": mode,
        "endpoint": url, "wall_ms": wall_ms, "random_seed": args.seed,
        "artifacts": {name: str(path) for name, path in paths.items()},
    })
    paths["metrics"].write_text(json.dumps(metrics, indent=2) + "\n")
    summary = dict(metrics)
    sequence = result["sequence"]
    summary["sequence" if len(sequence) <= 256 else "sequence_preview"] = sequence[:256]
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="DNA prompt; case and whitespace are normalized")
    parser.add_argument("--mode", choices=["hosted", "local"], help="Explicit mode; otherwise use NIM_API_MODE/EVO2_NIM_URL")
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1, help="Development reproducibility seed")
    parser.add_argument("--timeout", type=float, default=180, help="Response read timeout in seconds")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory reserved for this request; must not already exist")
    args = parser.parse_args()
    try:
        generate(args)
    except requests.RequestException as exc:
        # Exception strings can include headers or URLs; report only the exception type.
        print(f"Evo 2 request failed ({type(exc).__name__}); no successful generation to report", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
