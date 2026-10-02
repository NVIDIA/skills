#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade deterministic NVA workflow contracts from a Harbor ATIF trajectory."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

WELCOME = (
    "Welcome to the NVIDIA Nemotron Voice Agent (NVA for short). We will customize NVA for "
    "creating ambient healthcare agents.\n"
    "Now I will make a fresh clone of the Nemotron Voice Agent repository, and this will be "
    "the directory we work out of. Where would you like me to clone the repo to? Please provide "
    "a path.\n"
    "If you already have a clone of the repository somewhere, please point me to the path."
)

WELCOME_CASES = {
    "nva-ambient-explicit-welcome",
    "nva-ambient-implicit-patient-intake",
    "nva-ambient-context-path-confirmation",
}

TARGET_SKILL = "ambient-healthcare-agent-with-nemotron-voice-agent"
NVA_ENV_PATH = "/workspace/nva-clinic/.env"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_message_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _message_text(value["content"])
    return ""


def _agent_evidence(trajectory: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    messages: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        return "", []
    for step in steps:
        if not isinstance(step, dict) or step.get("source") not in {"agent", "assistant"}:
            continue
        message = _message_text(step.get("message")).strip()
        if message:
            messages.append(message)
        calls = step.get("tool_calls")
        if isinstance(calls, list):
            tool_calls.extend(call for call in calls if isinstance(call, dict))
    return (messages[-1] if messages else ""), tool_calls


def _contains_all(text: str, required: list[str]) -> bool:
    lowered = text.lower()
    return all(item.lower() in lowered for item in required)


def _is_target_skill_loader(call: dict[str, Any]) -> bool:
    """Return whether a tool call only loads the skill under evaluation."""
    name = call.get("function_name") or call.get("name")
    arguments = call.get("arguments") or call.get("input")
    return (
        isinstance(name, str)
        and name.lower() == "skill"
        and isinstance(arguments, dict)
        and arguments.get("skill") == TARGET_SKILL
    )


def grade(entry: dict[str, Any], trajectory: dict[str, Any]) -> tuple[float, str]:
    """Return the case-specific contract score and a human-readable reason."""
    case_id = str(entry.get("id", ""))
    final_message, tool_calls = _agent_evidence(trajectory)
    operational_tool_calls = [call for call in tool_calls if not _is_target_skill_loader(call)]
    normalized = final_message.replace("\r\n", "\n").strip()
    tool_blob = json.dumps(tool_calls, sort_keys=True).lower()

    if case_id in WELCOME_CASES:
        exact = normalized == WELCOME
        idle = not operational_tool_calls
        score = (float(exact) + float(idle)) / 2.0
        return score, f"exact_welcome={exact}; no_operational_tool_calls={idle}"

    if case_id == "nva-ambient-invalid-repository-gate":
        reports_invalid = _contains_all(normalized, ["nva", "path"]) and any(
            word in normalized.lower() for word in ("invalid", "not a usable", "missing")
        )
        stopped = not any(token in tool_blob for token in ("docker", "apply_generic_agent_template"))
        score = (float(reports_invalid) + float(stopped)) / 2.0
        return score, f"reports_invalid_path={reports_invalid}; stopped_before_gates={stopped}"

    if case_id == "nva-ambient-credential-hard-gate":
        reports_failure = "nvidia_api_key" in normalized.lower() and any(
            word in normalized.lower() for word in ("fail", "missing", "required")
        )
        stopped = (
            "run_expected_conversation" not in tool_blob
            and re.search(r"docker(?:-compose|\s+compose).*?\bup\b", tool_blob) is None
        )
        score = (float(reports_failure) + float(stopped)) / 2.0
        return score, f"reports_credential_failure={reports_failure}; no_app_start_or_live_test={stopped}"

    if case_id == "nva-ambient-scenario-disclosure":
        disclosed = _contains_all(
            normalized,
            [
                "What type of voice agent application would you like to create?",
                "appointment-making",
                "patient-intake",
                "customize your own use case",
                "NVIDIA Llama 3.3 Nemotron Super 49B",
                "https://integrate.api.nvidia.com/v1",
                "Jordan Patel",
                "1979-09-24",
                "Maya Chen",
                "1988-04-12",
            ],
        )
        idle = not operational_tool_calls
        score = (float(disclosed) + float(idle)) / 2.0
        return score, f"complete_endpoint_data_disclosure={disclosed}; no_operational_tool_calls={idle}"

    if case_id == "nva-ambient-public-endpoint-env-disclosure":
        choices_and_path = _contains_all(
            normalized,
            [
                "public",
                "local nim",
                "existing nim",
                "mixed",
                NVA_ENV_PATH,
                "NVIDIA_API_KEY",
            ],
        )
        key_instruction = any(word in normalized.lower() for word in ("fill", "set", "enter"))
        key_purpose = any(word in normalized.lower() for word in ("authenticate", "access", "utilize"))
        no_secret_request = "paste" not in normalized.lower() or "do not paste" in normalized.lower()
        idle = not operational_tool_calls
        score = sum((choices_and_path, key_instruction, key_purpose, no_secret_request, idle)) / 5.0
        return score, (
            f"choices_and_path={choices_and_path}; key_instruction={key_instruction}; "
            f"key_purpose={key_purpose}; no_secret_request={no_secret_request}; "
            f"no_operational_tool_calls={idle}"
        )

    if case_id == "nva-ambient-negative-ordinary-deploy":
        no_welcome = WELCOME not in normalized
        routes_to_nva = _contains_all(normalized, ["nva"]) and any(
            word in normalized.lower() for word in ("deploy", "configure")
        )
        score = (float(no_welcome) + float(routes_to_nva)) / 2.0
        return score, f"target_not_activated={no_welcome}; routes_to_nva={routes_to_nva}"

    if case_id == "nva-ambient-negative-text-only-agent":
        no_welcome = WELCOME not in normalized
        text_only = "text-only" in normalized.lower() or "text based" in normalized.lower()
        idle = not operational_tool_calls
        preserves_boundary = text_only and idle
        score = (float(no_welcome) + float(preserves_boundary)) / 2.0
        return score, (
            f"target_not_activated={no_welcome}; "
            f"preserves_text_only_boundary={preserves_boundary}"
        )

    return 0.0, f"unknown eval case id: {case_id}"


def main() -> int:
    trajectory_path = Path(os.getenv("HARBOR_ATIF_PATH", "/logs/agent/trajectory.json"))
    entry_path = Path(os.getenv("HARBOR_ENTRY_JSON", "/tests/entry.json"))
    reward_json = Path(os.getenv("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
    reward_txt = Path(os.getenv("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt"))

    entry = _load_json(entry_path)
    trajectory = _load_json(trajectory_path)
    score, reason = grade(entry, trajectory)
    payload = {
        "overall": score,
        "custom_metrics": {"nva_workflow_contract": score},
        "details": {"nva_workflow_contract": {"score": score, "reason": reason}},
    }
    reward_json.parent.mkdir(parents=True, exist_ok=True)
    reward_txt.parent.mkdir(parents=True, exist_ok=True)
    reward_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    reward_txt.write_text(f"{score:.6f}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
