#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect NVA generic example defaults without hard-coding NVA release values."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

GENERIC_EXAMPLE_KEY = "generic-assistant"
GENERIC_DIR_PARTS = ("src", "examples", "generic")
SERVICE_CATALOG_FILES = ("services.cloud.yaml", "services.local.yaml")


def inspect_generic_defaults(nva_root: Path) -> dict[str, Any]:
    """Return the resolved generic example defaults for a target NVA checkout."""
    nva_root = nva_root.resolve()
    registry = load_yaml_mapping(nva_root / "examples_registry.yaml")
    generic_dir = nva_root.joinpath(*GENERIC_DIR_PARTS)
    defaults = _generic_defaults(registry)
    available_service_keys = {
        service_kind: _available_service_keys(nva_root, service_kind)
        for service_kind in ("llm", "asr", "tts")
    }
    report: dict[str, Any] = {
        "nva_root": str(nva_root),
        "git": _git_identity(nva_root),
        "example_key": GENERIC_EXAMPLE_KEY,
        "registry_path": str(nva_root / "examples_registry.yaml"),
        "generic_dir": str(generic_dir),
        "defaults": {name: _coerce_key_list(value) for name, value in defaults.items()},
        "available_service_keys": available_service_keys,
        "available_llm_keys": available_service_keys["llm"],
        "available_asr_keys": available_service_keys["asr"],
        "available_tts_keys": available_service_keys["tts"],
    }

    prompt_defaults = report["defaults"].get("prompt") or []
    if prompt_defaults:
        report["default_prompt"] = prompt_defaults[0]

    for service_kind in ("llm", "asr", "tts"):
        keys = report["defaults"].get(service_kind) or []
        if not keys:
            continue
        resolved_services: list[dict[str, Any]] = []
        service_errors: list[str] = []
        for key in keys:
            try:
                resolved_services.append(
                    resolve_service_config(nva_root, service_kind, key)
                )
            except RuntimeError as exc:
                service_errors.append(f"{key}: {exc}")
        if resolved_services:
            report[f"default_{service_kind}"] = resolved_services[0]
            report[f"default_{service_kind}_services"] = resolved_services
        if service_errors:
            report[f"default_{service_kind}_error"] = service_errors[0]
            report[f"default_{service_kind}_errors"] = service_errors

    return report


def check_generic_compatibility(
    nva_root: Path,
    *,
    require_patch_points: bool = True,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Check that this skill can customize the target NVA generic example."""
    nva_root = nva_root.resolve()
    generic_dir = nva_root.joinpath(*GENERIC_DIR_PARTS)
    errors: list[str] = []
    warnings: list[str] = []

    required_files = [
        nva_root / "docker-compose.yml",
        nva_root / "examples_registry.yaml",
        generic_dir / "pipeline.py",
        generic_dir / "prompts.yaml",
        generic_dir / "tools.yaml",
        generic_dir / "tool_handlers.py",
        generic_dir.parent / "shared" / "pipeline_utils.py",
    ]
    for path in required_files:
        if not path.is_file():
            errors.append(f"Missing required file: {path}")
    if not _discover_nva_skill(nva_root, "deploy"):
        warnings.append(
            "Missing an NVA deploy skill under skills/*/SKILL.md or "
            ".agents/skills/*/SKILL.md"
        )

    catalog_paths = [
        generic_dir / name
        for name in SERVICE_CATALOG_FILES
        if (generic_dir / name).is_file()
    ]
    if not catalog_paths:
        errors.append(
            "Missing generic service catalogs: expected at least one of "
            + ", ".join(str(generic_dir / name) for name in SERVICE_CATALOG_FILES)
        )
    elif not (generic_dir / "services.cloud.yaml").is_file():
        warnings.append(
            "services.cloud.yaml is missing; the default public NVIDIA AI Endpoint path may not be available."
        )

    report: dict[str, Any] = {
        "nva_root": str(nva_root),
        "git": _git_identity(nva_root),
        "example_key": GENERIC_EXAMPLE_KEY,
        "errors": errors,
        "warnings": warnings,
    }

    try:
        defaults_report = inspect_generic_defaults(nva_root)
        report.update(defaults_report)
        if "default_prompt" not in defaults_report:
            errors.append(
                f"{GENERIC_EXAMPLE_KEY} has no default prompt in examples_registry.yaml."
            )
        prompt_key = defaults_report.get("default_prompt")
        if prompt_key and (generic_dir / "prompts.yaml").is_file():
            prompts = load_yaml_mapping(generic_dir / "prompts.yaml")
            prompt_entry = prompts.get(str(prompt_key))
            if (
                not isinstance(prompt_entry, dict)
                or not str(prompt_entry.get("content") or "").strip()
            ):
                errors.append(
                    f"Default prompt {prompt_key!r} was not found in {generic_dir / 'prompts.yaml'}."
                )
        for service_kind in ("llm", "asr", "tts"):
            service_errors = defaults_report.get(f"default_{service_kind}_errors")
            if isinstance(service_errors, list):
                errors.extend(str(error) for error in service_errors)
        for service_kind in ("llm", "asr", "tts"):
            if f"default_{service_kind}" not in defaults_report:
                errors.append(
                    f"{GENERIC_EXAMPLE_KEY} default {service_kind.upper()} could not be resolved."
                )
    except RuntimeError as exc:
        errors.append(str(exc))

    if require_patch_points:
        _check_patch_points(generic_dir, errors, scenario=scenario)

    return report


def resolve_llm_config(
    nva_root: Path,
    *,
    llm_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Resolve an LLM entry into the OpenAI-compatible config used by live tests."""
    nva_root = nva_root.resolve()
    if not llm_key:
        llm_key = default_service_key(nva_root, "llm")
    config = resolve_service_config(nva_root, "llm", llm_key)
    entry = dict(config["entry"])
    model_id = model or str(entry.get("model_id") or entry.get("model") or "")
    resolved_base_url = base_url or str(entry.get("base_url") or "")
    if not model_id:
        raise RuntimeError(
            f"LLM catalog key {llm_key!r} has no model_id or model field."
        )
    if not resolved_base_url:
        raise RuntimeError(f"LLM catalog key {llm_key!r} has no base_url field.")

    name = str(entry.get("name") or llm_key)
    if model:
        name = f"{name} with model override"
    return {
        "key": llm_key,
        "name": name,
        "model_id": model_id,
        "base_url": resolved_base_url,
        "source_file": config["source_file"],
        "source_section": config["source_section"],
        "extra_payload": extra_payload(entry.get("extra_params")),
    }


def resolve_service_config(
    nva_root: Path, service_kind: str, service_key: str
) -> dict[str, Any]:
    for candidate in iter_service_entries(nva_root, service_kind):
        if candidate["key"] == service_key:
            return candidate

    available = _available_service_keys(nva_root, service_kind)
    available_text = ", ".join(item["key"] for item in available) or "none"
    raise RuntimeError(
        f"Could not resolve {service_kind} key {service_key!r} in generic service catalogs. "
        f"Available {service_kind} keys: {available_text}."
    )


def default_service_key(nva_root: Path, service_kind: str) -> str:
    registry = load_yaml_mapping(nva_root / "examples_registry.yaml")
    defaults = _generic_defaults(registry)
    keys = _coerce_key_list(defaults.get(service_kind))
    if not keys:
        raise RuntimeError(
            f"examples.{GENERIC_EXAMPLE_KEY}.defaults.{service_kind} must define at least one key."
        )
    return keys[0]


def iter_service_entries(nva_root: Path, service_kind: str) -> list[dict[str, Any]]:
    generic_dir = nva_root.joinpath(*GENERIC_DIR_PARTS)
    entries: list[dict[str, Any]] = []
    for catalog_name in SERVICE_CATALOG_FILES:
        catalog_path = generic_dir / catalog_name
        if not catalog_path.is_file():
            continue
        catalog = load_yaml_mapping(catalog_path)
        catalog_label = catalog_name.removeprefix("services.").removesuffix(".yaml")
        entries.extend(
            _entries_from_catalog(catalog, service_kind, catalog_path, catalog_label)
        )
    return entries


def extra_payload(extra_params: Any) -> dict[str, Any]:
    if not extra_params:
        return {}
    if isinstance(extra_params, str):
        try:
            parsed = json.loads(extra_params)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid LLM extra_params JSON: {exc}") from exc
    elif isinstance(extra_params, dict):
        parsed = extra_params
    else:
        raise RuntimeError("LLM extra_params must be a JSON string or mapping.")

    payload: dict[str, Any] = {}
    extra_body = parsed.get("extra_body")
    if isinstance(extra_body, dict):
        payload.update(extra_body)
    for key, value in parsed.items():
        if key != "extra_body":
            payload[key] = value
    return payload


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"YAML file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML file must contain a mapping: {path}")
    return data


def format_inspection(report: dict[str, Any]) -> str:
    lines = [
        "NVA generic inspection",
        f"  nva_root: {report.get('nva_root')}",
    ]
    git = report.get("git") or {}
    if git.get("describe"):
        lines.append(f"  git: {git['describe']}")
    lines.append(f"  example: {report.get('example_key', GENERIC_EXAMPLE_KEY)}")
    if report.get("default_prompt"):
        lines.append(f"  default_prompt: {report['default_prompt']}")
    llm_services = report.get("default_llm_services")
    if isinstance(llm_services, list) and len(llm_services) > 1:
        lines.append("  default_llm_services:")
        for value in llm_services:
            lines.extend(_format_service("llm", value, indent="    "))
    else:
        value = report.get("default_llm")
        if isinstance(value, dict):
            lines.extend(_format_service("llm", value))
    error = report.get("default_llm_error")
    if error:
        lines.append(f"  default_llm: unresolved ({error})")
    for service_kind in ("asr", "tts"):
        value = report.get(f"default_{service_kind}")
        if isinstance(value, dict):
            lines.extend(_format_service(service_kind, value))
        error = report.get(f"default_{service_kind}_error")
        if error:
            lines.append(f"  default_{service_kind}: unresolved ({error})")
    for service_kind in ("llm", "asr", "tts"):
        available = report.get(f"available_{service_kind}_keys") or []
        if available:
            lines.append(f"  available_{service_kind}_keys:")
            for item in available:
                lines.append(
                    f"    - {item['key']} ({item['source_section']}; {item['source_file']})"
                )
    return "\n".join(lines)


def format_compatibility(report: dict[str, Any]) -> str:
    lines = [format_inspection(report)]
    warnings = report.get("warnings") or []
    errors = report.get("errors") or []
    if warnings:
        lines.append("  compatibility_warnings:")
        lines.extend(f"    - {warning}" for warning in warnings)
    if errors:
        lines.append("  compatibility: FAIL")
        lines.append("  compatibility_errors:")
        lines.extend(f"    - {error}" for error in errors)
    else:
        lines.append("  compatibility: PASS")
    return "\n".join(lines)


def _format_service(
    service_kind: str, config: dict[str, Any], *, indent: str = "  "
) -> list[str]:
    entry = config.get("entry") or {}
    lines = [
        f"{indent}default_{service_kind}: {config.get('key')}",
        f"{indent}  name: {entry.get('name') or config.get('key')}",
        f"{indent}  source: {config.get('source_section')} in {config.get('source_file')}",
    ]
    if service_kind == "llm":
        lines.append(
            f"{indent}  model: {entry.get('model_id') or entry.get('model') or ''}"
        )
        lines.append(f"{indent}  base_url: {entry.get('base_url') or ''}")
    else:
        if entry.get("model"):
            lines.append(f"{indent}  model: {entry.get('model')}")
        if entry.get("server"):
            lines.append(f"{indent}  server: {entry.get('server')}")
        if entry.get("voice_id"):
            lines.append(f"{indent}  voice_id: {entry.get('voice_id')}")
    return lines


def _generic_defaults(registry: dict[str, Any]) -> dict[str, Any]:
    examples = registry.get("examples")
    if not isinstance(examples, dict):
        raise RuntimeError("examples_registry.yaml must contain an examples mapping.")
    entry = examples.get(GENERIC_EXAMPLE_KEY)
    if not isinstance(entry, dict):
        available = ", ".join(str(key) for key in examples) or "none"
        raise RuntimeError(
            f"examples_registry.yaml does not define {GENERIC_EXAMPLE_KEY!r}. "
            f"Available examples: {available}."
        )
    defaults = entry.get("defaults")
    if not isinstance(defaults, dict):
        raise RuntimeError(
            f"examples.{GENERIC_EXAMPLE_KEY}.defaults must be a mapping."
        )
    return defaults


def _coerce_key_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _entries_from_catalog(
    catalog: dict[str, Any],
    service_kind: str,
    catalog_path: Path,
    catalog_label: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    top_level = catalog.get(service_kind)
    if isinstance(top_level, dict):
        entries.extend(
            _entry_records(top_level, service_kind, catalog_path, catalog_label)
        )
    for section_name, section in catalog.items():
        if section_name == service_kind or not isinstance(section, dict):
            continue
        nested = section.get(service_kind)
        if isinstance(nested, dict):
            entries.extend(
                _entry_records(
                    nested,
                    service_kind,
                    catalog_path,
                    f"{catalog_label}/{section_name}",
                )
            )
    return entries


def _entry_records(
    entries: dict[str, Any],
    service_kind: str,
    catalog_path: Path,
    source_section: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, entry in entries.items():
        if isinstance(entry, dict):
            records.append(
                {
                    "service_kind": service_kind,
                    "key": str(key),
                    "entry": entry,
                    "source_file": str(catalog_path),
                    "source_section": source_section,
                }
            )
    return records


def _available_service_keys(nva_root: Path, service_kind: str) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    keys: list[dict[str, str]] = []
    for item in iter_service_entries(nva_root, service_kind):
        record = (
            str(item["key"]),
            str(item["source_file"]),
            str(item["source_section"]),
        )
        if record in seen:
            continue
        seen.add(record)
        keys.append(
            {
                "key": str(item["key"]),
                "source_file": str(item["source_file"]),
                "source_section": str(item["source_section"]),
            }
        )
    return keys


def _git_identity(nva_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    describe = _git_output(nva_root, "describe", "--tags", "--always", "--dirty")
    if describe:
        result["describe"] = describe
    branch = _git_output(nva_root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        result["branch"] = branch
    commit = _git_output(nva_root, "rev-parse", "--short", "HEAD")
    if commit:
        result["commit"] = commit
    return result


def _git_output(nva_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(nva_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _check_patch_points(
    generic_dir: Path,
    errors: list[str],
    *,
    scenario: str | None = None,
) -> None:
    pipeline_path = generic_dir / "pipeline.py"
    if pipeline_path.is_file():
        pipeline_text = pipeline_path.read_text(encoding="utf-8")
        try:
            pipeline_tree = ast.parse(pipeline_text, filename=str(pipeline_path))
        except SyntaxError as exc:
            errors.append(f"Could not parse {pipeline_path}: {exc}")
            pipeline_tree = None

        if pipeline_tree is not None:
            bot = _find_function(pipeline_tree, "bot")
            if not any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                for node in pipeline_tree.body
            ):
                errors.append(
                    f"{pipeline_path} has no top-level import block for healthcare imports."
                )
            if bot is None:
                errors.append(
                    f"{pipeline_path} has no async bot function for healthcare setup."
                )
            elif not _has_assignment_targets(
                bot, {"prompt_key", "base_system_content"}
            ):
                errors.append(
                    f"{pipeline_path} does not assign prompt_key and base_system_content in bot(); "
                    "the skill cannot attach current datetime context safely."
                )

            if scenario == "patient-intake" and bot is not None:
                _check_patient_intake_pipeline_shape(
                    pipeline_path,
                    pipeline_text,
                    pipeline_tree,
                    bot,
                    errors,
                )
            if scenario == "appointment-making" and bot is not None:
                _check_appointment_welcome_pipeline_shape(
                    pipeline_path,
                    bot,
                    errors,
                )

    shared_pipeline_utils_path = generic_dir.parent / "shared" / "pipeline_utils.py"
    if scenario == "patient-intake" and shared_pipeline_utils_path.is_file():
        shared_pipeline_utils_text = shared_pipeline_utils_path.read_text(
            encoding="utf-8"
        )
        _check_marker_or_text(
            shared_pipeline_utils_text,
            errors,
            path=shared_pipeline_utils_path,
            marker="# BEGIN ambient agent interruptible welcome",
            required_text="def build_user_aggregator_params(",
            purpose="interruptible welcome helper insertion",
        )

    handlers_path = generic_dir / "tool_handlers.py"
    if handlers_path.is_file():
        handlers_text = handlers_path.read_text(encoding="utf-8")
        _check_marker_or_text(
            handlers_text,
            errors,
            path=handlers_path,
            marker="# BEGIN ambient agent generic imports",
            required_text="from pipecat.services.llm_service import FunctionCallParams\n",
            purpose="ambient tool import insertion",
        )
        _check_marker_or_text(
            handlers_text,
            errors,
            path=handlers_path,
            marker="# BEGIN ambient agent generic handlers",
            required_text=(
                "# ---------------------------------------------------------------------------\n"
                "# Registry\n"
                "# ---------------------------------------------------------------------------"
            ),
            purpose="ambient handler insertion",
        )
        _check_marker_or_text(
            handlers_text,
            errors,
            path=handlers_path,
            marker="    # BEGIN ambient agent generic registry",
            required_text="}\n",
            purpose="ambient registry insertion",
        )

    registry_path = generic_dir.parents[2] / "examples_registry.yaml"
    if registry_path.is_file() and not _can_update_generic_default_prompt(
        registry_path
    ):
        errors.append(
            f"Could not find examples.{GENERIC_EXAMPLE_KEY}.defaults.prompt in {registry_path}; "
            "the applier would not be able to make the healthcare prompt the browser default."
        )


def _check_appointment_welcome_pipeline_shape(
    path: Path,
    bot: ast.AsyncFunctionDef | ast.FunctionDef,
    errors: list[str],
) -> None:
    """Require the shared startup hook used by the fixed appointment greeting."""
    start_handler = _find_function(bot, "_on_session_start", search_nested=True)
    session_call = next(
        (
            node
            for node in ast.walk(bot)
            if isinstance(node, ast.Call)
            and _call_name(node.func) == "register_session_start_handlers"
        ),
        None,
    )
    if start_handler is None or not start_handler.body or session_call is None:
        errors.append(
            f"{path} has no supported shared session-start hook for the "
            "deterministic appointment welcome."
        )
        return
    if not any(keyword.arg == "welcome_enabled" for keyword in session_call.keywords):
        errors.append(
            f"{path} has no welcome_enabled setting on register_session_start_handlers(); "
            "the appointment welcome cannot suppress the model-generated intro."
        )


def _check_patient_intake_pipeline_shape(
    path: Path,
    text: str,
    tree: ast.Module,
    bot: ast.AsyncFunctionDef | ast.FunctionDef,
    errors: list[str],
) -> None:
    """Validate semantic patient-intake patch capabilities, not release text."""
    checks = [
        (
            _has_assignment_target(bot, "llm_settings"),
            "an llm_settings assignment",
        ),
        (
            _has_assignment_target(bot, "tools_enabled"),
            "a tools_enabled assignment",
        ),
        (
            _has_call(bot, "LLMContextAggregatorPair"),
            "an LLMContextAggregatorPair call",
        ),
        (
            _has_call(bot, "Pipeline"),
            "a Pipeline call",
        ),
    ]
    for passed, description in checks:
        if not passed:
            errors.append(f"{path} has no {description} required by patient intake.")

    managed_welcome = "# BEGIN ambient agent patient intake welcome" in text
    inline_welcome = _has_call(bot, "context.add_message")
    shared_welcome = (
        _has_call(bot, "register_session_start_handlers")
        and _find_function(tree, "_on_session_start", search_nested=True) is not None
    )
    if not (managed_welcome or inline_welcome or shared_welcome):
        errors.append(
            f"{path} has no supported inline or shared session-start hook for the "
            "deterministic patient-intake welcome."
        )


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


def _has_assignment_target(tree: ast.AST, target_name: str) -> bool:
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and target_name in _assignment_names(node)
        for node in ast.walk(tree)
    )


def _has_assignment_targets(tree: ast.AST, target_names: set[str]) -> bool:
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and target_names <= _assignment_names(node)
        for node in ast.walk(tree)
    )


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _has_call(tree: ast.AST, call_name: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _call_name(node.func) == call_name
        for node in ast.walk(tree)
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _discover_nva_skill(nva_root: Path, capability: str) -> list[Path]:
    """Find repo skills by frontmatter name, tolerating directory renames."""
    matches: list[Path] = []
    for base in (nva_root / "skills", nva_root / ".agents" / "skills"):
        for path in sorted(base.glob("*/SKILL.md")):
            name = _skill_frontmatter_name(path)
            if capability in name.lower().replace("_", "-").split("-"):
                matches.append(path)
    return matches


def _skill_frontmatter_name(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return path.parent.name
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return path.parent.name


def _check_marker_or_text(
    text: str,
    errors: list[str],
    *,
    path: Path,
    marker: str,
    required_text: str,
    purpose: str,
) -> None:
    if marker in text or required_text in text:
        return
    errors.append(f"{path} is missing the expected {purpose} point.")


def _can_update_generic_default_prompt(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    examples_indent: int | None = None
    generic_indent: int | None = None
    defaults_indent: int | None = None

    for line in lines:
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
            generic_indent = None
            defaults_indent = None
            continue
        if stripped == "defaults:" and indent > generic_indent:
            defaults_indent = indent
            continue
        if defaults_indent is None:
            continue
        if indent <= defaults_indent and stripped:
            defaults_indent = None
            continue
        if stripped.startswith("prompt:") and indent > defaults_indent:
            return True
    return False
