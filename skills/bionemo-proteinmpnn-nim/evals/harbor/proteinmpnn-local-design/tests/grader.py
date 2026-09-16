#!/usr/bin/env python3
"""Deterministic verifier for the local ProteinMPNN NIM Harbor task."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

WORKSPACE = Path(os.environ.get("HARBOR_WORKSPACE", "/workspace"))
OUTPUT_DIR = WORKSPACE / "output"
TRAJECTORY_JSON = Path(os.environ.get("HARBOR_ATIF_PATH", "/logs/agent/trajectory.json"))
REWARD_JSON = Path(os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
REWARD_TXT = Path(os.environ.get("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt"))
BASE_URL = os.environ.get("PROTEINMPNN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{BASE_URL}/biology/ipd/proteinmpnn/predict"
HEALTH_URL = f"{BASE_URL}/v1/health/ready"

EXPECTED_COUNT = 3
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWYX")
WEIGHTS = {
    "artifact_contract": 0.15,
    "request_contract": 0.20,
    "response_contract": 0.15,
    "fasta_contract": 0.15,
    "sequence_validity": 0.10,
    "execution_evidence": 0.10,
    "local_nim_replay": 0.15,
}


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object"
    return value, None


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence).upper()))
            header = line[1:].strip()
            sequence = []
        elif header is not None:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence).upper()))
    return records


def designed_records(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if len(records) == EXPECTED_COUNT:
        return records
    if len(records) == EXPECTED_COUNT + 1:
        return records[1:]
    return []


def finite_numbers(value: Any, expected: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected:
        return None
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return numbers if all(math.isfinite(item) for item in numbers) else None


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("NIM returned a non-object JSON response")
    return parsed


def close_scores(left: list[float] | None, right: list[float] | None) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(left, right))


def grade() -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics = {name: 0.0 for name in WEIGHTS}
    details: dict[str, dict[str, Any]] = {}

    request_json, request_error = load_json(OUTPUT_DIR / "request.json")
    response_json, response_error = load_json(OUTPUT_DIR / "response.json")
    summary_json, summary_error = load_json(OUTPUT_DIR / "summary.json")
    fasta_path = OUTPUT_DIR / "designed_sequences.fa"
    fasta_text = fasta_path.read_text(encoding="utf-8") if fasta_path.is_file() else ""

    artifact_errors = [error for error in (request_error, response_error, summary_error) if error]
    if not fasta_path.is_file():
        artifact_errors.append("missing designed_sequences.fa")
    metrics["artifact_contract"] = (4 - len(artifact_errors)) / 4
    details["artifact_contract"] = {
        "score": metrics["artifact_contract"],
        "reason": "all four output artifacts are present and parseable" if not artifact_errors else "; ".join(artifact_errors),
    }

    request_checks: list[tuple[str, bool]] = []
    if request_json is not None:
        input_pdb = request_json.get("input_pdb")
        request_checks = [
            ("input_pdb contains ATOM records", isinstance(input_pdb, str) and input_pdb.count("ATOM") >= 20),
            ("num_seq_per_target is 3", request_json.get("num_seq_per_target") == EXPECTED_COUNT),
            ("random_seed is 1", request_json.get("random_seed") == 1),
            ("sampling_temp is [0.1]", request_json.get("sampling_temp") == [0.1]),
            ("use_soluble_model is false", request_json.get("use_soluble_model") is False),
            ("ca_only is false", request_json.get("ca_only") is False),
        ]
    passed_request = sum(passed for _, passed in request_checks)
    metrics["request_contract"] = passed_request / 6
    failed_request = [name for name, passed in request_checks if not passed]
    details["request_contract"] = {
        "score": metrics["request_contract"],
        "reason": "request matches all required ProteinMPNN parameters" if not failed_request and request_checks else "; ".join(failed_request or ["request unavailable"]),
    }

    response_fasta = response_json.get("mfasta") if response_json else None
    response_records = parse_fasta(response_fasta) if isinstance(response_fasta, str) else []
    designs = designed_records(response_records)
    saved_scores = finite_numbers(response_json.get("scores"), EXPECTED_COUNT) if response_json else None
    response_checks = [
        ("mfasta is non-empty", bool(response_records)),
        ("mfasta contains exactly 3 designs, plus at most one native row", bool(designs)),
        ("scores contains 3 finite values", saved_scores is not None),
    ]
    metrics["response_contract"] = sum(passed for _, passed in response_checks) / len(response_checks)
    details["response_contract"] = {
        "score": metrics["response_contract"],
        "reason": "; ".join(name for name, passed in response_checks if not passed) or "response has valid Multi-FASTA and score fields",
    }

    fasta_matches = isinstance(response_fasta, str) and fasta_text.strip() == response_fasta.strip()
    summary_count = summary_json.get("generated_count") if summary_json else None
    summary_sequences = summary_json.get("sequences") if summary_json else None
    summary_matches = summary_count == EXPECTED_COUNT and isinstance(summary_sequences, list) and len(summary_sequences) == EXPECTED_COUNT
    metrics["fasta_contract"] = (float(fasta_matches) + float(summary_matches)) / 2
    details["fasta_contract"] = {
        "score": metrics["fasta_contract"],
        "reason": "saved FASTA and summary agree with the response" if metrics["fasta_contract"] == 1.0 else "saved FASTA or summary does not agree with the response",
    }

    valid_sequences = bool(designs) and all(
        sequence and set(sequence) <= VALID_AAS for _, sequence in designs
    )
    consistent_lengths = bool(designs) and len({len(sequence) for _, sequence in designs}) == 1
    metrics["sequence_validity"] = (float(valid_sequences) + float(consistent_lengths)) / 2
    details["sequence_validity"] = {
        "score": metrics["sequence_validity"],
        "reason": "three non-empty amino-acid sequences have consistent lengths" if metrics["sequence_validity"] == 1.0 else "designed sequences are empty, invalid, or length-inconsistent",
    }

    trajectory_text = TRAJECTORY_JSON.read_text(encoding="utf-8", errors="replace") if TRAJECTORY_JSON.is_file() else ""
    used_local_endpoint = "localhost:8000/biology/ipd/proteinmpnn/predict" in trajectory_text or "127.0.0.1:8000/biology/ipd/proteinmpnn/predict" in trajectory_text
    used_hosted_endpoint = "health.api.nvidia.com/v1/biology/ipd/proteinmpnn" in trajectory_text
    metrics["execution_evidence"] = 1.0 if used_local_endpoint and not used_hosted_endpoint else 0.0
    details["execution_evidence"] = {
        "score": metrics["execution_evidence"],
        "reason": "trajectory shows the local endpoint and no hosted endpoint" if metrics["execution_evidence"] else "trajectory does not prove exclusive use of the local endpoint",
    }

    replay_reason = "saved request unavailable"
    if request_json is not None:
        try:
            health = http_json(HEALTH_URL, timeout=15.0)
            replay = http_json(PREDICT_URL, request_json, timeout=300.0)
            replay_records = designed_records(parse_fasta(str(replay.get("mfasta", ""))))
            replay_scores = finite_numbers(replay.get("scores"), EXPECTED_COUNT)
            exact_sequences = bool(designs) and [seq for _, seq in designs] == [seq for _, seq in replay_records]
            exact_scores = close_scores(saved_scores, replay_scores)
            if exact_sequences and exact_scores:
                metrics["local_nim_replay"] = 1.0
                replay_reason = "local NIM replay reproduced the saved sequences and scores"
            elif replay_records and saved_scores is not None and replay_scores is not None:
                metrics["local_nim_replay"] = 0.6
                replay_reason = "local NIM replay succeeded but was not byte-for-byte deterministic"
            else:
                metrics["local_nim_replay"] = 0.25
                replay_reason = "local NIM replay succeeded but its response contract did not match"
            if health.get("status") not in (None, "ready"):
                metrics["local_nim_replay"] = min(metrics["local_nim_replay"], 0.5)
                replay_reason += f"; health status was {health.get('status')!r}"
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
            replay_reason = f"local NIM replay failed: {type(exc).__name__}: {exc}"
    details["local_nim_replay"] = {
        "score": metrics["local_nim_replay"],
        "reason": replay_reason,
    }

    return metrics, details


def write_reward(metrics: dict[str, float], details: dict[str, dict[str, Any]]) -> None:
    overall = round(sum(metrics[name] * weight for name, weight in WEIGHTS.items()), 4)
    reward = {
        "overall": overall,
        "custom_metrics": {name: round(score, 4) for name, score in metrics.items()},
        "details": details,
    }
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps(reward, indent=2, sort_keys=True), encoding="utf-8")
    REWARD_TXT.write_text(f"{overall:.4f}\n", encoding="utf-8")


def main() -> None:
    try:
        metrics, details = grade()
    except Exception as exc:  # Always satisfy Harbor's reward-file contract.
        metrics = {name: 0.0 for name in WEIGHTS}
        details = {
            "grader_error": {
                "score": 0.0,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        }
    write_reward(metrics, details)


if __name__ == "__main__":
    main()
