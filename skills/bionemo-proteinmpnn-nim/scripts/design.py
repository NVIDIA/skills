#!/usr/bin/env python3
"""Execute one ProteinMPNN request and save sequences with response-derived scores."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

import requests


HOSTED_URL = "https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")
DESIGN_HEADER = re.compile(r"(?:^|,)\s*(?:T|sample|seq)\s*=")
NATIVE_HEADER = re.compile(r"(?:^|[\s,|])(?:native|wt|wild[-_ ]type)(?:$|[\s,|])", re.I)


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def parse_fasta(text: object) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Response mfasta must be a nonempty string")
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if not line[1:].strip():
                raise ValueError("FASTA header must not be empty")
            records.append({"header": line[1:].strip(), "sequence": ""})
        elif not records:
            raise ValueError("FASTA sequence appears before its header")
        else:
            records[-1]["sequence"] += line
    if not records:
        raise ValueError("Response contains no FASTA records")
    for record in records:
        # ProteinMPNN separates chains with '/'; retain that representation.
        if any(not chain or set(chain.upper()) - AMINO_ACIDS for chain in record["sequence"].split("/")):
            raise ValueError("FASTA contains an empty chain or invalid amino-acid sequence")
    return records


def design_results(result: object, expected_count: int) -> dict:
    """Align scores without assigning the first design's score to a native row."""
    if not isinstance(result, dict):
        raise ValueError("ProteinMPNN must return a JSON object")
    records = parse_fasta(result.get("mfasta"))
    first_header = records[0]["header"]
    first_is_native = bool(NATIVE_HEADER.search(first_header)) or (
        not DESIGN_HEADER.search(first_header)
        and (
            bool(re.search(r"(?:^|,)\s*(?:fixed_chains|designed_chains)\s*=", first_header))
            or (len(records) == expected_count + 1
                and all(DESIGN_HEADER.search(row["header"]) for row in records[1:]))
        )
    )
    native_count = int(first_is_native)
    designs = records[native_count:]
    if len(designs) != expected_count or any(NATIVE_HEADER.search(row["header"]) for row in designs):
        raise ValueError(f"Expected {expected_count} designed sequences, plus an optional leading native/WT record")

    scores = result.get("scores")
    score_source = "response.scores"
    if scores is None:
        # Some responses carry scores in FASTA headers instead of a JSON array.
        scores = []
        for row in designs:
            match = re.search(r"(?:^|,)\s*score\s*=\s*([^,\s]+)", row["header"])
            if match is None:
                raise ValueError("Missing designed-sequence scores in both scores and FASTA headers")
            scores.append(float(match.group(1)))
        score_source = "mfasta.header.score"
    elif not isinstance(scores, list):
        raise ValueError("Response scores must be an array")
    elif native_count and len(scores) == len(records):
        # When the array includes the native record, remove its corresponding score.
        scores = scores[1:]
    if len(scores) != len(designs):
        raise ValueError("Score count does not match the designed FASTA records; inspect response.json")
    for index, (row, score) in enumerate(zip(designs, scores), start=1):
        finite_number(score, "Designed-sequence score")
        row.update({"design_index": index, "score": score})
    return {
        "generated_count": len(designs), "native_count": native_count,
        "score_source": score_source, "scores": scores, "sequences": designs,
    }


def design(args: argparse.Namespace) -> dict:
    if not 1 <= args.num_sequences <= 100:
        raise ValueError("--num-sequences must be between 1 and 100")
    if not 0 <= finite_number(args.temperature, "Temperature") <= 1:
        raise ValueError("--temperature must be between 0 and 1")
    if finite_number(args.timeout, "Timeout") <= 0:
        raise ValueError("--timeout must be positive")
    if args.omit_aas and any(aa not in AMINO_ACIDS - {"X"} for aa in args.omit_aas):
        raise ValueError("--omit-aas must contain standard one-letter amino-acid codes")
    pdb = args.pdb.resolve()
    pdb_content = pdb.read_text(encoding="utf-8")
    if not any(line.startswith("ATOM  ") for line in pdb_content.splitlines()):
        raise ValueError("Input PDB must contain ATOM records")

    mode = args.mode or os.getenv("NIM_API_MODE") or ("local" if os.getenv("PROTEINMPNN_NIM_URL") else None)
    headers = {"Content-Type": "application/json"}
    if mode == "hosted":
        key = os.getenv("NGC_API_KEY")
        if not key:
            raise ValueError("Set NGC_API_KEY in the environment for hosted ProteinMPNN")
        headers["Authorization"] = f"Bearer {key}"
        url = HOSTED_URL
    elif mode == "local":
        base = os.getenv("PROTEINMPNN_NIM_URL", "http://localhost:8000").rstrip("/")
        parsed = urlsplit(base)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("PROTEINMPNN_NIM_URL must be an HTTP(S) base URL without credentials, query, or fragment")
        url = f"{base}/biology/ipd/proteinmpnn/predict"
    else:
        raise ValueError("Choose --mode hosted or local, or set NIM_API_MODE")

    payload = {
        "input_pdb": pdb_content, "num_seq_per_target": args.num_sequences,
        "sampling_temp": [args.temperature], "use_soluble_model": args.soluble,
        "ca_only": args.ca_only,
    }
    for name, value in (("input_pdb_chains", args.chains), ("omit_AAs", args.omit_aas), ("random_seed", args.seed)):
        if value is not None:
            payload[name] = value

    output = args.output_dir.resolve()
    # One atomic reservation prevents concurrent runs from sharing artifacts.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError("Choose a new --output-dir; this directory already exists") from exc
    paths = {name: output / filename for name, filename in {
        "request": "request.json", "raw_response": "response.raw", "response": "response.json",
        "fasta": "designed_sequences.fa", "summary": "summary.json",
    }.items()}
    paths["request"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Print metadata, never credentials or a potentially large input structure.
    print(json.dumps({"event": "request", "method": "POST", "endpoint": url, "mode": mode,
                      "input_pdb_path": str(pdb), "num_seq_per_target": args.num_sequences,
                      "sampling_temp": [args.temperature], "request_path": str(paths["request"])}), flush=True)
    response = requests.post(url, headers=headers, json=payload, timeout=(10, args.timeout), allow_redirects=False)
    paths["raw_response"].write_bytes(response.content)
    if response.status_code != 200:
        raise RuntimeError(f"ProteinMPNN returned HTTP {response.status_code}; inspect response.raw; design is not complete")
    result = response.json()
    paths["response"].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = design_results(result, args.num_sequences)
    # Preserve the complete returned FASTA, including any native reference header.
    paths["fasta"].write_bytes(result["mfasta"].encode("utf-8"))
    summary.update({"status": "completed", "http_status": response.status_code,
                    "mode": mode, "endpoint": url, "input_pdb_path": str(pdb),
                    "artifacts": {name: str(path) for name, path in paths.items()}})
    paths["summary"].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if paths["fasta"].read_bytes() != result["mfasta"].encode("utf-8") or json.loads(paths["summary"].read_text()) != summary:
        raise RuntimeError("Saved artifacts do not match the response")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="Use the user's supplied PDB path")
    parser.add_argument("--mode", choices=["hosted", "local"], help="Explicit mode overrides environment configuration")
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1, help="One temperature per request")
    parser.add_argument("--chains", nargs="+", help="Chains to design; omit to design all chains")
    parser.add_argument("--omit-aas", nargs="+", help="One-letter amino-acid codes to exclude")
    parser.add_argument("--soluble", action="store_true")
    parser.add_argument("--ca-only", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=300, help="Response read timeout in seconds")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory reserved for this run")
    args = parser.parse_args()
    try:
        design(args)
    except requests.RequestException as exc:
        print(f"ProteinMPNN request failed ({type(exc).__name__}); no completed design to report", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
