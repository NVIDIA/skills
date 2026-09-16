#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply a scenario template to NVA's src/examples/generic example."""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from nva_generic_defaults import (
    GENERIC_EXAMPLE_KEY,
    check_generic_compatibility,
    format_compatibility,
)

IMPORT_BLOCK = """# BEGIN ambient agent generic imports
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties

from examples.generic.ambient_agent_tools import TOOL_RESULT_FUNCTIONS as AMBIENT_AGENT_TOOL_RESULT_FUNCTIONS

# END ambient agent generic imports
"""

PATIENT_INTAKE_HANDLER_IMPORT_BLOCK = """# BEGIN ambient agent generic imports
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties

from examples.generic.ambient_agent_tools import TOOL_RESULT_FUNCTIONS as AMBIENT_AGENT_TOOL_RESULT_FUNCTIONS
from examples.generic.patient_intake_speech_guard import PatientIntakeToolResponseFrame

# END ambient agent generic imports
"""

HANDLER_BLOCK = """# BEGIN ambient agent generic handlers
_AMBIENT_AGENT_DIRECT_RESPONSE_TOOLS = {\"review_patient_intake\", \"record_patient_intake\"}


async def _emit_ambient_agent_response(llm, text: str) -> None:
    \"\"\"Send a tool-authored confirmation through the normal text and TTS path.\"\"\"
    await llm.push_frame(LLMFullResponseStartFrame())
    await llm.push_frame(LLMTextFrame(text=text))
    await llm.push_frame(LLMFullResponseEndFrame())


def _build_ambient_agent_tool_handler(tool_name: str):
    async def _handler(params: FunctionCallParams) -> None:
        result = AMBIENT_AGENT_TOOL_RESULT_FUNCTIONS[tool_name](params.arguments or {})
        response_text = str(result.get(\"response_text\") or \"\").strip()
        if (
            tool_name in _AMBIENT_AGENT_DIRECT_RESPONSE_TOOLS
            and result.get(\"status\") in {\"review_required\", \"confirmation_required\", \"saved\"}
            and response_text
        ):
            await _emit_ambient_agent_response(params.llm, response_text)
            await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
            return
        await params.result_callback(result)

    _handler.__name__ = f"handle_{tool_name}"
    return _handler


# END ambient agent generic handlers
"""

PATIENT_INTAKE_HANDLER_BLOCK = """# BEGIN ambient agent generic handlers
_AMBIENT_AGENT_DIRECT_RESPONSE_TOOLS = {\"review_patient_intake\", \"record_patient_intake\"}


async def _emit_ambient_agent_response(llm, text: str) -> None:
    \"\"\"Send a tool-authored confirmation through the normal text and TTS path.\"\"\"
    await llm.push_frame(LLMFullResponseStartFrame())
    await llm.push_frame(PatientIntakeToolResponseFrame(text=text))
    await llm.push_frame(LLMFullResponseEndFrame())


def _build_ambient_agent_tool_handler(tool_name: str):
    async def _handler(params: FunctionCallParams) -> None:
        result = AMBIENT_AGENT_TOOL_RESULT_FUNCTIONS[tool_name](params.arguments or {})
        response_text = str(result.get(\"response_text\") or \"\").strip()
        if (
            tool_name in _AMBIENT_AGENT_DIRECT_RESPONSE_TOOLS
            and result.get(\"status\") in {\"review_required\", \"confirmation_required\", \"saved\"}
            and response_text
        ):
            await _emit_ambient_agent_response(params.llm, response_text)
            await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
            return
        await params.result_callback(result)

    _handler.__name__ = f\"handle_{tool_name}\"
    return _handler


def build_patient_intake_guarded_handler(tool_name: str, handler, conversation_state):
    \"\"\"Block review/save calls until every field is evidenced in conversation history.\"\"\"

    async def _guarded_handler(params: FunctionCallParams) -> None:
        missing_fields = conversation_state.missing_fields()
        if tool_name in _AMBIENT_AGENT_DIRECT_RESPONSE_TOOLS and missing_fields:
            response_text = conversation_state.collection_response()
            result = {
                \"status\": \"needs_more_information\",
                \"missing_fields\": missing_fields,
                \"response_text\": response_text,
                \"note\": \"tool call blocked because required fields were not present in conversation history\",
            }
            await _emit_ambient_agent_response(params.llm, response_text)
            await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
            return
        await handler(params)

    _guarded_handler.__name__ = f\"guarded_{getattr(handler, '__name__', tool_name)}\"
    return _guarded_handler


# END ambient agent generic handlers
"""

REGISTRY_BLOCK = """    # BEGIN ambient agent generic registry
    **{name: _build_ambient_agent_tool_handler(name) for name in AMBIENT_AGENT_TOOL_RESULT_FUNCTIONS},
    # END ambient agent generic registry
"""

DATETIME_IMPORT_BLOCK = """# BEGIN ambient agent datetime imports
from datetime import datetime
from os import getenv
from zoneinfo import ZoneInfo

# END ambient agent datetime imports
"""

DATETIME_HELPER_BLOCK = """# BEGIN ambient agent datetime context
def _ambient_agent_add_current_datetime_context(prompt: str) -> str:
    timezone_name = getenv("NVA_HEALTHCARE_AGENT_TIMEZONE", "UTC")
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception:
        timezone_name = "UTC"
        timezone = ZoneInfo("UTC")

    now = datetime.now(timezone)
    today = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
    return (
        prompt.rstrip()
        + "\\n\\nCurrent datetime context:\\n"
        + f"- Current timezone: {timezone_name}\\n"
        + f"- Current datetime: {now.isoformat()}\\n"
        + f"- Today is {today}\\n"
        + "Use this context to resolve relative healthcare dates "
        + "such as today, tomorrow, next week, and next Monday. Preserve "
        + "absolute dates the user provides."
    )


# END ambient agent datetime context
"""

DATETIME_CALL_BLOCK = """    # BEGIN ambient agent current datetime context
    base_system_content = _ambient_agent_add_current_datetime_context(base_system_content)
    # END ambient agent current datetime context
"""

PATIENT_INTAKE_IMPORT_BLOCK = """# BEGIN ambient agent patient intake imports
from examples.generic.patient_intake_speech_guard import (
    PatientIntakeConversationState,
    PatientIntakeSpeechGuardProcessor,
    PatientIntakeTurnReminderProcessor,
    patient_intake_welcome_frames,
)
from examples.generic.tool_handlers import build_patient_intake_guarded_handler

# END ambient agent patient intake imports
"""

PATIENT_INTAKE_TEMPERATURE_BLOCK = """    # BEGIN ambient agent patient intake temperature override
    if prompt_key == \"patient_intake_healthcare\":
        llm_settings.temperature = 0.0
    # END ambient agent patient intake temperature override
"""

PATIENT_INTAKE_CONVERSATION_STATE_BLOCK = """    # BEGIN ambient agent patient intake conversation state
    patient_intake_conversation_state = (
        PatientIntakeConversationState() if prompt_key == \"patient_intake_healthcare\" else None
    )
    # END ambient agent patient intake conversation state
"""

PATIENT_INTAKE_TOOL_REGISTRATION_BLOCK = """    # BEGIN ambient agent patient intake tool registration
    if tools_enabled:
        for name in registered_tools:
            handler = TOOL_HANDLERS[name]
            if patient_intake_conversation_state:
                handler = build_patient_intake_guarded_handler(name, handler, patient_intake_conversation_state)
            llm.register_function(name, handler)
            logger.info(f\"Registered tool handler: {name}\")
    else:
        logger.info(f\"Tool calling disabled for prompt_key={prompt_key!r} (no tools_available in prompts.yaml)\")
    # END ambient agent patient intake tool registration
"""

PATIENT_INTAKE_USER_AGGREGATOR_BLOCK = """    # BEGIN ambient agent patient intake user aggregator
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_aggregator_params(
            welcome_enabled,
            interruptible_welcome=prompt_key == \"patient_intake_healthcare\",
        ),
    )
    # END ambient agent patient intake user aggregator
"""

PATIENT_INTAKE_PROCESSOR_SETUP_BLOCK = """    # BEGIN ambient agent patient intake processors
    patient_intake_turn_reminder = (
        PatientIntakeTurnReminderProcessor(patient_intake_conversation_state)
        if patient_intake_conversation_state
        else None
    )
    patient_intake_speech_guard = (
        PatientIntakeSpeechGuardProcessor(patient_intake_conversation_state)
        if patient_intake_conversation_state
        else None
    )
    # END ambient agent patient intake processors
"""

INTERRUPTIBLE_WELCOME_HELPER_BLOCK = """# BEGIN ambient agent interruptible welcome
IMMEDIATE_SPEECH_TIMEOUT_SECONDS = 0.0


def build_user_aggregator_params(
    welcome_enabled: bool,
    *,
    interruptible_welcome: bool = False,
) -> LLMUserAggregatorParams:
    \"\"\"Return user-turn configuration, optionally allowing welcome barge-in.\"\"\"
    mute_during_welcome = welcome_enabled and not interruptible_welcome
    if not parse_env_bool(\"USE_SILERO_VAD_TURN_DETECTION\", default=False):
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_mute_strategies=build_user_mute_strategies(mute_during_welcome),
            user_turn_strategies=UserTurnStrategies(stop=build_smart_turn_stop_strategies()),
        )

    stop_secs = parse_env_float(\"SILERO_VAD_STOP_SECS\", 0.5, min_value=0.0)
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=stop_secs)),
        user_mute_strategies=build_user_mute_strategies(mute_during_welcome),
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=IMMEDIATE_SPEECH_TIMEOUT_SECONDS)],
        ),
    )
# END ambient agent interruptible welcome
"""

PATIENT_INTAKE_PRE_LLM_BLOCK = """            # BEGIN ambient agent patient intake pre-LLM processor
            *([patient_intake_turn_reminder] if patient_intake_turn_reminder else []),
            # END ambient agent patient intake pre-LLM processor
"""

PATIENT_INTAKE_POST_LLM_BLOCK = """            # BEGIN ambient agent patient intake post-LLM processor
            *([patient_intake_speech_guard] if patient_intake_speech_guard else []),
            # END ambient agent patient intake post-LLM processor
"""

PATIENT_INTAKE_WELCOME_BLOCK = """        # BEGIN ambient agent patient intake welcome
        if prompt_key == \"patient_intake_healthcare\" and welcome_enabled:
            await task.queue_frames(patient_intake_welcome_frames())
            return
        # END ambient agent patient intake welcome
"""

APPOINTMENT_WELCOME_IMPORT_BLOCK = """# BEGIN ambient agent appointment welcome imports
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame

# END ambient agent appointment welcome imports
"""

APPOINTMENT_WELCOME_BLOCK = """        # BEGIN ambient agent appointment welcome
        if prompt_key == \"appointment_making_healthcare\" and welcome_enabled:
            await task.queue_frames(
                [
                    LLMFullResponseStartFrame(),
                    LLMTextFrame(
                        text=(
                            \"Hello and welcome to the appointment making agent. \"
                            \"Let's get started. First, could you please tell me what type \"
                            \"of appointment you're looking for?\"
                        )
                    ),
                    LLMFullResponseEndFrame(),
                ]
            )
            return
        # END ambient agent appointment welcome
"""

APPOINTMENT_HOST_DB_PATH = (
    Path("data") / "appointment-making" / "appointment_schedule.sqlite"
)
APPOINTMENT_CONTAINER_DB_PATH = (
    "/app/data/appointment-making/appointment_schedule.sqlite"
)
APPOINTMENT_VOLUME = "./data/appointment-making:/app/data/appointment-making"


def main() -> int:
    """Apply the selected generic-agent scenario artifacts to an NVA checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nva-root", required=True, type=Path)
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        help=(
            "Directory containing prompt.yaml, tools.yaml, tool-implementation.py, "
            "and optionally expected-conversation.yaml, patient-intake-speech-guard.py, "
            "and test-ambient-agent-tools.py."
        ),
    )
    parser.add_argument("--scenario-name", help="Optional label for command output.")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--tools-file", type=Path)
    parser.add_argument("--implementation-file", type=Path)
    parser.add_argument("--expected-file", type=Path)
    parser.add_argument("--pipeline-support-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--database-dir", type=Path)
    args = parser.parse_args()

    files = _resolve_files(args)
    generic_dir = args.nva_root / "src" / "examples" / "generic"
    patient_intake_support = bool(files.get("pipeline_support"))
    prompt_template = _validate_yaml(files["prompt"])
    appointment_welcome_support = (
        isinstance(prompt_template, dict)
        and "appointment_making_healthcare" in prompt_template
    )
    if patient_intake_support and appointment_welcome_support:
        raise RuntimeError(
            "The appointment and patient-intake pipeline overlays cannot be combined"
        )
    scenario = _scenario_label(args)
    if patient_intake_support:
        scenario = "patient-intake"
    elif appointment_welcome_support:
        scenario = "appointment-making"
    compatibility = check_generic_compatibility(
        args.nva_root,
        scenario=scenario,
    )
    if compatibility.get("errors"):
        print(
            "FAIL: NVA generic compatibility preflight failed. "
            "The target repo layout or defaults differ from what this skill can patch safely.",
            file=sys.stderr,
        )
        print(format_compatibility(compatibility), file=sys.stderr)
        return 2
    if compatibility.get("warnings"):
        print("NVA generic compatibility preflight PASS with warnings:")
        for warning in compatibility["warnings"]:
            print(f"  warning: {warning}")
    else:
        print("NVA generic compatibility preflight PASS.")

    prompts_target = generic_dir / "prompts.yaml"
    tools_target = generic_dir / "tools.yaml"
    handlers_target = generic_dir / "tool_handlers.py"
    implementation_target = generic_dir / "ambient_agent_tools.py"
    pipeline_support_target = generic_dir / "patient_intake_speech_guard.py"
    pipeline_target = generic_dir / "pipeline.py"
    shared_pipeline_utils_target = (
        args.nva_root / "src" / "examples" / "shared" / "pipeline_utils.py"
    )
    registry_target = args.nva_root / "examples_registry.yaml"

    prompt_keys = _merge_yaml_blocks(prompts_target, files["prompt"])
    tool_keys = _merge_yaml_blocks(tools_target, files["tools"])
    shutil.copyfile(files["implementation"], implementation_target)
    if files.get("pipeline_support"):
        shutil.copyfile(files["pipeline_support"], pipeline_support_target)
        _patch_shared_pipeline_utils(shared_pipeline_utils_target)
    _patch_tool_handlers(
        handlers_target,
        patient_intake_support=patient_intake_support,
    )
    _patch_pipeline(
        pipeline_target,
        patient_intake_support=patient_intake_support,
        appointment_welcome_support=appointment_welcome_support,
    )
    _set_generic_default_prompt(registry_target, prompt_keys[0])
    database_target = None
    appointment_database_path = None
    compose_override_path = None
    if files.get("database"):
        database_target = _copy_database_artifacts(files["database"], generic_dir)
        appointment_database_path = _prepare_host_appointment_database(args.nva_root)
        compose_override_path = _write_compose_override(args.nva_root)

    expected_target = generic_dir / "ambient_agent_expected_conversation.yaml"
    if files.get("expected"):
        shutil.copyfile(files["expected"], expected_target)

    test_target = None
    if files.get("test"):
        test_target = args.nva_root / "tests" / "unit" / "test_ambient_agent_tools.py"
        test_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files["test"], test_target)

    _validate_yaml(prompts_target)
    _validate_yaml(tools_target)
    _validate_yaml(registry_target)

    print("Applied generic NVA agent template.")
    print(f"  scenario: {_scenario_label(args)}")
    print(f"  prompt_keys: {', '.join(prompt_keys)}")
    print(f"  default_prompt: {prompt_keys[0]}")
    print(f"  tool_keys: {', '.join(tool_keys)}")
    print(f"  implementation: {implementation_target}")
    if files.get("pipeline_support"):
        print(f"  pipeline_support: {pipeline_support_target}")
        print(f"  shared_pipeline_utils: {shared_pipeline_utils_target}")
    print(f"  pipeline: {pipeline_target}")
    if database_target:
        print(f"  database_support: {database_target}")
    if appointment_database_path:
        print(f"  appointment_database: {appointment_database_path}")
    if compose_override_path:
        print(f"  compose_override: {compose_override_path}")
    if files.get("expected"):
        print(f"  expected_conversation: {expected_target}")
    if test_target:
        print(f"  unit_test: {test_target}")
    return 0


def _resolve_files(args: argparse.Namespace) -> dict[str, Path]:
    files = {
        "prompt": args.prompt_file,
        "tools": args.tools_file,
        "implementation": args.implementation_file,
        "expected": args.expected_file,
        "pipeline_support": args.pipeline_support_file,
        "test": args.test_file,
        "database": args.database_dir,
    }
    if args.scenario_dir:
        scenario_dir = args.scenario_dir.resolve()
        defaults = {
            "prompt": scenario_dir / "prompt.yaml",
            "tools": scenario_dir / "tools.yaml",
            "implementation": scenario_dir / "tool-implementation.py",
            "expected": scenario_dir / "expected-conversation.yaml",
            "pipeline_support": scenario_dir / "patient-intake-speech-guard.py",
            "test": scenario_dir / "test-ambient-agent-tools.py",
            "database": scenario_dir / "database",
        }
        files = {name: files[name] or default for name, default in defaults.items()}

    required = ("prompt", "tools", "implementation")
    missing_args = [
        f"--{name.replace('_', '-')}-file" for name in required if files[name] is None
    ]
    if missing_args:
        raise SystemExit(
            "Provide --scenario-dir or explicit artifact files for: "
            + ", ".join(missing_args)
        )

    resolved: dict[str, Path] = {}
    for name, path in files.items():
        if path is None:
            continue
        resolved_path = path.resolve()
        optional_file_args = {
            "expected": args.expected_file,
            "pipeline_support": args.pipeline_support_file,
            "test": args.test_file,
        }
        if (
            name in optional_file_args
            and not resolved_path.is_file()
            and optional_file_args[name] is None
        ):
            continue
        if name == "database":
            if resolved_path.is_dir():
                resolved[name] = resolved_path
                continue
            if args.database_dir is None:
                continue
            raise SystemExit(f"database directory does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise SystemExit(f"{name} file does not exist: {resolved_path}")
        resolved[name] = resolved_path
    return resolved


def _scenario_label(args: argparse.Namespace) -> str:
    if args.scenario_name:
        return args.scenario_name
    if args.scenario_dir:
        return args.scenario_dir.resolve().name
    return "custom"


def _merge_yaml_blocks(target: Path, source: Path) -> list[str]:
    source_text = source.read_text(encoding="utf-8").strip() + "\n"
    source_data = _validate_yaml_text(source_text, source)
    if not isinstance(source_data, dict) or not source_data:
        raise RuntimeError(
            f"{source} must contain one or more top-level YAML mapping entries"
        )

    target_text = target.read_text(encoding="utf-8")
    for key in source_data:
        block = _extract_top_level_block(source_text, str(key))
        target_text = _replace_or_append_top_level_block(target_text, str(key), block)
    target.write_text(target_text, encoding="utf-8")
    return [str(key) for key in source_data]


def _extract_top_level_block(text: str, key: str) -> str:
    lines = text.splitlines()
    start = _find_top_level_key(lines, key)
    if start is None:
        raise RuntimeError(f"Could not find top-level key {key!r}")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")) and line.rstrip().endswith(":"):
            end = index
            break
    return "\n".join(lines[start:end]).rstrip() + "\n"


def _replace_or_append_top_level_block(text: str, key: str, block: str) -> str:
    lines = text.rstrip().splitlines()
    start = _find_top_level_key(lines, key)
    if start is None:
        return text.rstrip() + "\n\n" + block

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")) and line.rstrip().endswith(":"):
            end = index
            break
    replacement = block.rstrip().splitlines()
    if end < len(lines):
        replacement.append("")
    new_lines = lines[:start] + replacement + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def _find_top_level_key(lines: list[str], key: str) -> int | None:
    prefix = f"{key}:"
    for index, line in enumerate(lines):
        if line == prefix:
            return index
    return None


def _patch_tool_handlers(path: Path, *, patient_intake_support: bool = False) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame\n",
        "",
    )
    text = text.replace(
        "from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties\n",
        "",
    )
    text = text.replace(
        "from pipecat.services.llm_service import FunctionCallParams\n", ""
    )
    text = _replace_marker_block(
        text,
        "# BEGIN ambient agent generic imports",
        "# END ambient agent generic imports",
        PATIENT_INTAKE_HANDLER_IMPORT_BLOCK if patient_intake_support else IMPORT_BLOCK,
        legacy_markers=[
            (
                "# BEGIN ambient healthcare generic imports",
                "# END ambient healthcare generic imports",
            ),
        ],
        insert_after="from loguru import logger\n",
    )
    text = _replace_marker_block(
        text,
        "# BEGIN ambient agent generic handlers",
        "# END ambient agent generic handlers",
        PATIENT_INTAKE_HANDLER_BLOCK if patient_intake_support else HANDLER_BLOCK,
        legacy_markers=[
            (
                "# BEGIN ambient healthcare generic handlers",
                "# END ambient healthcare generic handlers",
            ),
        ],
        insert_before=(
            "# ---------------------------------------------------------------------------\n"
            "# Registry\n"
            "# ---------------------------------------------------------------------------"
        ),
    )
    text = _replace_marker_block(
        text,
        "    # BEGIN ambient agent generic registry",
        "    # END ambient agent generic registry",
        REGISTRY_BLOCK,
        legacy_markers=[
            (
                "    # BEGIN ambient healthcare generic registry",
                "    # END ambient healthcare generic registry",
            ),
        ],
        insert_before="}\n",
    )
    path.write_text(text, encoding="utf-8")


def _patch_pipeline(
    path: Path,
    *,
    patient_intake_support: bool = False,
    appointment_welcome_support: bool = False,
) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_marker_or_after_imports(
        text,
        "# BEGIN ambient agent datetime imports",
        "# END ambient agent datetime imports",
        DATETIME_IMPORT_BLOCK,
    )
    text = _replace_marker_or_before_function(
        text,
        "# BEGIN ambient agent datetime context",
        "# END ambient agent datetime context",
        DATETIME_HELPER_BLOCK,
        function_name="bot",
    )
    text = _replace_marker_or_after_assignment(
        text,
        "    # BEGIN ambient agent current datetime context",
        "    # END ambient agent current datetime context",
        DATETIME_CALL_BLOCK,
        function_name="bot",
        target_names={"prompt_key", "base_system_content"},
    )
    if appointment_welcome_support:
        text = _replace_marker_or_after_imports(
            text,
            "# BEGIN ambient agent appointment welcome imports",
            "# END ambient agent appointment welcome imports",
            APPOINTMENT_WELCOME_IMPORT_BLOCK,
        )
        text = _patch_appointment_welcome(text)
    if patient_intake_support:
        text = _replace_marker_or_after_imports(
            text,
            "# BEGIN ambient agent patient intake imports",
            "# END ambient agent patient intake imports",
            PATIENT_INTAKE_IMPORT_BLOCK,
        )
        text = _replace_marker_or_before_assignment(
            text,
            "    # BEGIN ambient agent patient intake temperature override",
            "    # END ambient agent patient intake temperature override",
            PATIENT_INTAKE_TEMPERATURE_BLOCK,
            function_name="bot",
            target_names={"llm"},
        )
        text = _replace_marker_or_after_assignment(
            text,
            "    # BEGIN ambient agent patient intake conversation state",
            "    # END ambient agent patient intake conversation state",
            PATIENT_INTAKE_CONVERSATION_STATE_BLOCK,
            function_name="bot",
            target_names={"tools_enabled"},
        )
        text = _replace_marker_or_exact(
            text,
            "    # BEGIN ambient agent patient intake tool registration",
            "    # END ambient agent patient intake tool registration",
            PATIENT_INTAKE_TOOL_REGISTRATION_BLOCK,
            exact_candidates=[
                (
                    "    if tools_enabled:\n"
                    "        for name in registered_tools:\n"
                    "            llm.register_function(name, TOOL_HANDLERS[name])\n"
                    '            logger.info(f"Registered tool handler: {name}")\n'
                    "    else:\n"
                    '        logger.info(f"Tool calling disabled for prompt_key={prompt_key!r} '
                    '(no tools_available in prompts.yaml)")\n'
                ),
                (
                    "    if tools_enabled:\n"
                    "        for name in registered_tools:\n"
                    "            handler = TOOL_HANDLERS[name]\n"
                    "            if patient_intake_conversation_state:\n"
                    "                handler = build_patient_intake_guarded_handler("
                    "name, handler, patient_intake_conversation_state)\n"
                    "            llm.register_function(name, handler)\n"
                    '            logger.info(f"Registered tool handler: {name}")\n'
                    "    else:\n"
                    '        logger.info(f"Tool calling disabled for prompt_key={prompt_key!r} '
                    '(no tools_available in prompts.yaml)")\n'
                ),
            ],
        )
        text = _replace_marker_or_exact(
            text,
            "    # BEGIN ambient agent patient intake user aggregator",
            "    # END ambient agent patient intake user aggregator",
            PATIENT_INTAKE_USER_AGGREGATOR_BLOCK,
            exact_candidates=[
                (
                    "    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(\n"
                    "        context,\n"
                    "        user_params=build_user_aggregator_params(welcome_enabled),\n"
                    "    )\n"
                ),
                (
                    "    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(\n"
                    "        context,\n"
                    "        user_params=build_user_aggregator_params(\n"
                    "            welcome_enabled,\n"
                    '            interruptible_welcome=prompt_key == "patient_intake_healthcare",\n'
                    "        ),\n"
                    "    )\n"
                ),
            ],
        )
        text = _replace_marker_block(
            text,
            "    # BEGIN ambient agent patient intake processors",
            "    # END ambient agent patient intake processors",
            PATIENT_INTAKE_PROCESSOR_SETUP_BLOCK,
            insert_after=PATIENT_INTAKE_USER_AGGREGATOR_BLOCK,
        )
        text = _replace_marker_block(
            text,
            "            # BEGIN ambient agent patient intake pre-LLM processor",
            "            # END ambient agent patient intake pre-LLM processor",
            PATIENT_INTAKE_PRE_LLM_BLOCK,
            insert_after="            user_aggregator,\n",
        )
        text = _replace_marker_block(
            text,
            "            # BEGIN ambient agent patient intake post-LLM processor",
            "            # END ambient agent patient intake post-LLM processor",
            PATIENT_INTAKE_POST_LLM_BLOCK,
            insert_after="            llm,\n",
        )
        text = _patch_patient_intake_welcome(text)
    path.write_text(text, encoding="utf-8")


def _patch_shared_pipeline_utils(path: Path) -> None:
    """Add an opt-in interruptible welcome without changing other examples."""
    text = path.read_text(encoding="utf-8")
    legacy_immediate_timeout = "0.0"
    base_block = f"""def build_user_aggregator_params(welcome_enabled: bool) -> LLMUserAggregatorParams:
    \"\"\"Return user-turn configuration, defaulting to Pipecat smart turn.\"\"\"
    if not parse_env_bool(\"USE_SILERO_VAD_TURN_DETECTION\", default=False):
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_mute_strategies=build_user_mute_strategies(welcome_enabled),
            user_turn_strategies=UserTurnStrategies(stop=build_smart_turn_stop_strategies()),
        )

    stop_secs = parse_env_float(\"SILERO_VAD_STOP_SECS\", 0.5, min_value=0.0)
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=stop_secs)),
        user_mute_strategies=build_user_mute_strategies(welcome_enabled),
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout={legacy_immediate_timeout})],
        ),
    )
"""
    upgraded_block = INTERRUPTIBLE_WELCOME_HELPER_BLOCK.removeprefix(
        "# BEGIN ambient agent interruptible welcome\n"
    ).removesuffix("# END ambient agent interruptible welcome\n")
    text = _replace_marker_or_exact(
        text,
        "# BEGIN ambient agent interruptible welcome",
        "# END ambient agent interruptible welcome",
        INTERRUPTIBLE_WELCOME_HELPER_BLOCK,
        exact_candidates=[base_block, upgraded_block],
    )
    path.write_text(text, encoding="utf-8")


def _set_generic_default_prompt(path: Path, prompt_key: str) -> None:
    """Make the generic-assistant UI session default use the applied scenario."""
    lines = path.read_text(encoding="utf-8").splitlines()
    examples_indent: int | None = None
    generic_indent: int | None = None
    defaults_indent: int | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped == "examples:":
            examples_indent = indent
            generic_indent = None
            defaults_indent = None
            continue

        if examples_indent is None:
            continue

        if indent <= examples_indent and stripped:
            examples_indent = None
            generic_indent = None
            defaults_indent = None
            continue

        if stripped == f"{GENERIC_EXAMPLE_KEY}:" and indent > examples_indent:
            generic_indent = indent
            defaults_indent = None
            continue

        if generic_indent is None:
            continue

        if indent <= generic_indent and stripped:
            break

        if stripped == "defaults:" and indent > generic_indent:
            defaults_indent = indent
            continue

        if defaults_indent is None:
            continue

        if indent <= defaults_indent and stripped:
            defaults_indent = None
            continue

        if stripped.startswith("prompt:") and indent > defaults_indent:
            lines[index] = f"{line[:indent]}prompt: [{prompt_key}]"
            path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            return

    raise RuntimeError(
        f"Could not update examples.{GENERIC_EXAMPLE_KEY}.defaults.prompt in {path}"
    )


def _copy_database_artifacts(source: Path, generic_dir: Path) -> Path:
    """Install database support without deleting an existing directory."""
    source = source.resolve(strict=True)
    generic_dir = generic_dir.resolve(strict=True)
    target = generic_dir / "ambient_healthcare_appointment_database"
    resolved_target = target.resolve(strict=False)

    if target.is_symlink():
        raise RuntimeError(f"Refusing to replace symlinked database target: {target}")
    if resolved_target.parent != generic_dir:
        raise RuntimeError(
            f"Database target must be a direct child of the Generic example: {target}"
        )
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"Database target exists and is not a directory: {target}")
    if (
        source == resolved_target
        or source.is_relative_to(resolved_target)
        or resolved_target.is_relative_to(source)
    ):
        raise RuntimeError(
            f"Database source and target must not overlap: {source} -> {target}"
        )

    # Merge only the known bundled files. This preserves unrelated files from a
    # prior installation and makes repeated template application idempotent.
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def _prepare_host_appointment_database(nva_root: Path) -> Path:
    appointment_database_path = nva_root / APPOINTMENT_HOST_DB_PATH
    appointment_database_path.parent.mkdir(parents=True, exist_ok=True)

    old_db_path = os.environ.get("NVA_APPOINTMENT_DB_PATH")
    old_sys_path = list(sys.path)
    try:
        os.environ["NVA_APPOINTMENT_DB_PATH"] = str(appointment_database_path)
        sys.path.insert(0, str(nva_root / "src"))
        module_name = "examples.generic.ambient_healthcare_appointment_database.db"
        sys.modules.pop(module_name, None)
        sys.modules.pop(
            "examples.generic.ambient_healthcare_appointment_database.seed", None
        )
        db_module = importlib.import_module(module_name)
        db_module.init_db()
    finally:
        if old_db_path is None:
            os.environ.pop("NVA_APPOINTMENT_DB_PATH", None)
        else:
            os.environ["NVA_APPOINTMENT_DB_PATH"] = old_db_path
        sys.path[:] = old_sys_path
    return appointment_database_path


def _write_compose_override(nva_root: Path) -> Path:
    """Write the appointment data mount without editing NVA's docker-compose.yml."""
    path = nva_root / "docker-compose.override.yml"
    if path.exists():
        data = _validate_yaml(path)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise RuntimeError(
                f"{path} must contain a YAML mapping to merge appointment database settings"
            )
    else:
        data = {}

    services = data.setdefault("services", {})
    if not isinstance(services, dict):
        raise RuntimeError(f"{path} services entry must be a mapping")

    compose = _validate_yaml(nva_root / "docker-compose.yml")
    compose_services = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(compose_services, dict):
        raise RuntimeError(
            f"{nva_root / 'docker-compose.yml'} services entry must be a mapping"
        )
    generic_services = sorted(
        str(name)
        for name in compose_services
        if str(name).startswith("generic-assistant")
    )
    if not generic_services:
        raise RuntimeError(
            "docker-compose.yml has no generic-assistant services to override"
        )

    for service_name in generic_services:
        service = services.setdefault(service_name, {})
        if not isinstance(service, dict):
            raise RuntimeError(
                f"{path} services.{service_name} entry must be a mapping"
            )
        _merge_environment(
            service,
            {
                "NVA_APPOINTMENT_DB_PATH": APPOINTMENT_CONTAINER_DB_PATH,
            },
        )
        _merge_volume(service, APPOINTMENT_VOLUME)

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _merge_environment(service: dict[str, Any], additions: dict[str, str]) -> None:
    environment = service.get("environment")
    if environment is None:
        service["environment"] = dict(additions)
        return
    if isinstance(environment, dict):
        environment.update(additions)
        return
    if isinstance(environment, list):
        for key, value in additions.items():
            updated = False
            for index, item in enumerate(environment):
                if isinstance(item, str) and item.split("=", 1)[0] == key:
                    environment[index] = f"{key}={value}"
                    updated = True
                    break
            if not updated:
                environment.append(f"{key}={value}")
        return
    raise RuntimeError("Compose service environment must be a mapping or list")


def _merge_volume(service: dict[str, Any], volume: str) -> None:
    volumes = service.get("volumes")
    if volumes is None:
        service["volumes"] = [volume]
        return
    if not isinstance(volumes, list):
        raise RuntimeError("Compose service volumes must be a list")
    target = volume.split(":", 2)[1]
    for existing in volumes:
        if (
            isinstance(existing, str)
            and len(existing.split(":")) > 1
            and existing.split(":", 2)[1] == target
        ):
            return
        if isinstance(existing, dict) and existing.get("target") == target:
            return
    volumes.append(volume)


def _replace_marker_or_after_imports(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
) -> str:
    if start_marker in text:
        return _replace_marker_block(text, start_marker, end_marker, replacement)
    tree = _parse_python(text)
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if not imports:
        raise RuntimeError("Could not find a top-level Python import block")
    offset = _node_end_offset(text, imports[-1])
    managed_end = re.match(
        r"\n*# END ambient agent [^\n]* imports\n",
        text[offset:],
    )
    if managed_end:
        offset += managed_end.end()
    return _insert_python_block(text, offset, replacement)


def _replace_marker_or_before_function(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    *,
    function_name: str,
) -> str:
    if start_marker in text:
        return _replace_marker_block(text, start_marker, end_marker, replacement)
    function = _find_function(_parse_python(text), function_name)
    if function is None:
        raise RuntimeError(f"Could not find Python function {function_name!r}")
    return _insert_python_block(text, _node_start_offset(text, function), replacement)


def _replace_marker_or_after_assignment(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    *,
    function_name: str,
    target_names: set[str],
) -> str:
    if start_marker in text:
        return _replace_marker_block(text, start_marker, end_marker, replacement)
    function = _find_function(_parse_python(text), function_name)
    assignment = _find_assignment(function, target_names) if function else None
    if assignment is None:
        raise RuntimeError(
            f"Could not find assignment to {sorted(target_names)!r} in {function_name}()"
        )
    return _insert_python_block(text, _node_end_offset(text, assignment), replacement)


def _replace_marker_or_before_assignment(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    *,
    function_name: str,
    target_names: set[str],
) -> str:
    if start_marker in text:
        return _replace_marker_block(text, start_marker, end_marker, replacement)
    function = _find_function(_parse_python(text), function_name)
    assignment = _find_assignment(function, target_names) if function else None
    if assignment is None:
        raise RuntimeError(
            f"Could not find assignment to {sorted(target_names)!r} in {function_name}()"
        )
    return _insert_python_block(text, _node_start_offset(text, assignment), replacement)


def _patch_appointment_welcome(text: str) -> str:
    """Queue the fixed appointment greeting and skip the model-generated intro."""
    marker = "        # BEGIN ambient agent appointment welcome"
    end_marker = "        # END ambient agent appointment welcome"
    if marker in text:
        text = _replace_marker_block(
            text, marker, end_marker, APPOINTMENT_WELCOME_BLOCK
        )

    tree = _parse_python(text)
    bot = _find_function(tree, "bot")
    if bot is None:
        raise RuntimeError("Could not find bot() for appointment welcome setup")
    session_call = _find_call(bot, "register_session_start_handlers")
    start_handler = _find_function(bot, "_on_session_start", search_nested=True)
    if session_call is None or start_handler is None or not start_handler.body:
        raise RuntimeError(
            "Could not find the shared session-start hook for the appointment welcome"
        )
    if marker not in text:
        text = _insert_python_block(
            text,
            _node_end_offset(text, start_handler.body[-1]),
            APPOINTMENT_WELCOME_BLOCK,
        )

    tree = _parse_python(text)
    bot = _find_function(tree, "bot")
    session_call = _find_call(bot, "register_session_start_handlers") if bot else None
    keyword = (
        next(
            (item for item in session_call.keywords if item.arg == "welcome_enabled"),
            None,
        )
        if session_call
        else None
    )
    if keyword is None:
        raise RuntimeError(
            "register_session_start_handlers() has no welcome_enabled keyword to suppress "
            "the generated LLM welcome for appointment making"
        )
    return _replace_node_text(
        text,
        keyword.value,
        'welcome_enabled and prompt_key != "appointment_making_healthcare"',
    )


def _patch_patient_intake_welcome(text: str) -> str:
    marker = "        # BEGIN ambient agent patient intake welcome"
    end_marker = "        # END ambient agent patient intake welcome"
    if marker in text:
        text = _replace_marker_block(
            text, marker, end_marker, PATIENT_INTAKE_WELCOME_BLOCK
        )

    tree = _parse_python(text)
    bot = _find_function(tree, "bot")
    if bot is None:
        raise RuntimeError("Could not find bot() for patient-intake welcome setup")
    session_call = _find_call(bot, "register_session_start_handlers")
    if session_call is not None:
        if marker not in text:
            start_handler = _find_function(bot, "_on_session_start", search_nested=True)
            if start_handler is None or not start_handler.body:
                raise RuntimeError(
                    "Could not find _on_session_start() for the patient-intake welcome"
                )
            text = _insert_python_block(
                text,
                _node_end_offset(text, start_handler.body[-1]),
                PATIENT_INTAKE_WELCOME_BLOCK,
            )
        tree = _parse_python(text)
        bot = _find_function(tree, "bot")
        session_call = (
            _find_call(bot, "register_session_start_handlers") if bot else None
        )
        keyword = (
            next(
                (
                    item
                    for item in session_call.keywords
                    if item.arg == "welcome_enabled"
                ),
                None,
            )
            if session_call
            else None
        )
        if keyword is None:
            raise RuntimeError(
                "register_session_start_handlers() has no welcome_enabled keyword to suppress "
                "the generated LLM welcome for patient intake"
            )
        return _replace_node_text(
            text,
            keyword.value,
            'welcome_enabled and prompt_key != "patient_intake_healthcare"',
        )

    if marker in text:
        return text
    intro_statement = _find_statement_call(bot, "context.add_message")
    if intro_statement is None:
        raise RuntimeError(
            "Could not find an inline or shared session-start hook for the patient-intake welcome"
        )
    return _insert_python_block(
        text,
        _node_start_offset(text, intro_statement),
        PATIENT_INTAKE_WELCOME_BLOCK,
    )


def _parse_python(text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        raise RuntimeError(f"Could not parse target Python source: {exc}") from exc


def _find_function(
    tree: ast.AST,
    name: str,
    *,
    search_nested: bool = False,
) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    nodes = ast.walk(tree) if search_nested else getattr(tree, "body", [])
    for node in nodes:
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == name
        ):
            return node
    return None


def _find_assignment(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
    target_names: set[str],
) -> ast.Assign | ast.AnnAssign | None:
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {
                child.id
                for target in targets
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }
            if target_names <= names:
                return node
    return None


def _find_call(tree: ast.AST, name: str) -> ast.Call | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) == name:
            return node
    return None


def _find_statement_call(tree: ast.AST, name: str) -> ast.stmt | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and any(
            isinstance(child, ast.Call) and _call_name(child.func) == name
            for child in ast.walk(node)
        ):
            return node
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_start_offset(text: str, node: ast.AST) -> int:
    return _line_offsets(text)[node.lineno - 1]


def _node_end_offset(text: str, node: ast.AST) -> int:
    offsets = _line_offsets(text)
    end_line = getattr(node, "end_lineno", node.lineno)
    return offsets[end_line] if end_line < len(offsets) else len(text)


def _replace_node_text(text: str, node: ast.AST, replacement: str) -> str:
    offsets = _line_offsets(text)
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return text[:start] + replacement + text[end:]


def _insert_python_block(text: str, offset: int, replacement: str) -> str:
    prefix = text[:offset].rstrip("\n")
    suffix = text[offset:].lstrip("\n")
    return f"{prefix}\n\n{replacement.rstrip()}\n\n{suffix}"


def _replace_marker_or_exact(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    *,
    exact_candidates: list[str],
) -> str:
    """Replace a managed block or migrate one known unmarked source shape."""
    if start_marker in text:
        return _replace_marker_block(text, start_marker, end_marker, replacement)
    for candidate in exact_candidates:
        if candidate in text:
            return text.replace(candidate, replacement, 1)
    raise RuntimeError(f"Could not find {start_marker!r} or a compatible source block")


def _replace_marker_block(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    *,
    legacy_markers: list[tuple[str, str]] | None = None,
    insert_after: str | None = None,
    insert_before: str | None = None,
) -> str:
    for candidate_start, candidate_end in [
        (start_marker, end_marker),
        *(legacy_markers or []),
    ]:
        start = text.find(candidate_start)
        if start >= 0:
            end = text.find(candidate_end, start)
            if end < 0:
                raise RuntimeError(
                    f"Found {candidate_start!r} without {candidate_end!r}"
                )
            end = text.find("\n", end)
            if end < 0:
                end = len(text)
            else:
                end += 1
            return text[:start] + replacement + text[end:]

    if insert_after is not None:
        index = text.find(insert_after)
        if index < 0:
            raise RuntimeError(f"Could not find insertion point after {insert_after!r}")
        index += len(insert_after)
        suffix = text[index:].removeprefix("\n")
        return text[:index] + "\n" + replacement + suffix

    if insert_before is not None:
        index = text.rfind(insert_before)
        if index < 0:
            raise RuntimeError(
                f"Could not find insertion point before {insert_before!r}"
            )
        return text[:index] + replacement + text[index:]

    raise RuntimeError("insert_after or insert_before is required")


def _validate_yaml(path: Path) -> Any:
    return _validate_yaml_text(path.read_text(encoding="utf-8"), path)


def _validate_yaml_text(text: str, path: Path) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in {path}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
