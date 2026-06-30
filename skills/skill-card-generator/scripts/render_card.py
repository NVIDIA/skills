# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
render_card.py — Render a skill card from a validated context JSON
using the Jinja template.

Usage:
  python3 render_card.py --context <context.json> \
                         --template <path/to/skill-card.md.j2> \
                         --out <output.md>

The template is in references/skill-card.md.j2. The agent does not
author layout — it only produces the context JSON. Rendering is
deterministic so two identical contexts always produce identical cards.
"""

import argparse
import json
import re
import sys
from pathlib import Path

IMPORT_ERROR_EXIT_CODE = int("2")
CONTEXT_VALIDATION_EXIT_CODE = int("3")
CATALOG_ERROR_EXIT_CODE = int("4")
MISSING = object()

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    print(
        "ERROR: jinja2 not installed. Install with:\n"
        "  pip install jinja2 --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(IMPORT_ERROR_EXIT_CODE)


# ─── Minimal context schema ───────────────────────────────────────────────
# Key: (type, required). required=True means missing key = error.
# Lists can be empty; strings can be "" but must be present.

SCHEMA = {
    "skill_name": (str, True),
    "skill_kind": (str, True),  # "Agent" or similar
    "description_sentence": (str, True),
    "usage_posture": (str, True),  # commercial | research_dev | demonstration
    "owner": (dict, True),  # {kind, verify?, verify_reason?, name?, card_link?}
    "license_identifier": ((str, type(None)), False),
    "license_verify": (bool, False),  # True → wrap rendered license in red VERIFY span
    "license_verify_reason": (str, False),  # short explanation, shown in HTML comment
    "use_case": (str, True),
    "deployment_geography": (str, True),
    "credential_requirements": (dict, False),  # optional for backward compatibility
    "references": (list, True),  # [{label, url}]
    "output": (dict, True),  # {types: [str], format, parameters, other_properties}
    "skill_version": (str, True),
    "evaluation": (dict, False),  # optional: evaluation details
}

VALID_USAGE = {"commercial", "research_dev", "demonstration"}
VALID_OWNER_KINDS = {"nvidia", "third_party"}
VALID_CREDENTIAL_STATUSES = {"yes", "no", "optional", "not specified"}
VALID_CREDENTIAL_TYPES = {
    "API key",
    "OAuth Token",
    "Cloud Credentials",
    "Service Account",
    "None",
}
SENSITIVE_CREDENTIAL_VALUE_PATTERNS = (
    re.compile(
        r"(?i)\b[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|"
        r"PRIVATE_KEY|CLIENT_SECRET|ACCESS_KEY)[A-Z0-9_]*\s*[:=]\s*"
        r"(?:[^\s\"'`]+|\"[^\"]*\"|'[^']*')"
    ),
    re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key|"
        r"access[_-]?key|private[_-]?key|client[_-]?secret)\b\s*[:=]\s*"
        r"(?:[^\s\"'`]+|\"[^\"]*\"|'[^']*')"
    ),
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)[?&](?:token|api_key|key|secret|password|access_token)="
        r"[^&\s)>\]\"'`]+"
    ),
    re.compile(
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
        r"\b(?:sk|hf|ghp|glpat|nvapi)-?[A-Za-z0-9_=-]{20,}\b|"
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
EVALUATION_STRING_FIELDS = ("agent", "tasks", "results_markdown")
EVALUATION_METRIC_GROUPS = ("dimensions", "signals")
TESTING_COMPLETED_FIELDS = (
    "agent_red_teaming",
    "network_security",
    "product_security",
)


def validate(ctx: dict) -> list[str]:
    errors = []
    _validate_schema(ctx, errors)
    _validate_usage(ctx, errors)
    _validate_owner(ctx, errors)
    _validate_credential_requirements(ctx, errors)
    _validate_output(ctx, errors)
    _validate_evaluation(ctx, errors)
    _validate_references(ctx, errors)
    return errors


def _validate_schema(ctx: dict, errors: list[str]) -> None:
    for key, (typ, required) in SCHEMA.items():
        if key not in ctx:
            if required:
                errors.append(f"missing required key: '{key}'")
            continue
        if not isinstance(ctx[key], typ):
            expected = _type_name(typ)
            errors.append(
                f"'{key}' should be {expected}, got {type(ctx[key]).__name__}"
            )


def _type_name(typ) -> str:
    if isinstance(typ, tuple):
        return " or ".join(t.__name__ for t in typ)
    return typ.__name__


def _validate_usage(ctx: dict, errors: list[str]) -> None:
    if "usage_posture" in ctx and ctx["usage_posture"] not in VALID_USAGE:
        errors.append(
            f"'usage_posture' must be one of {sorted(VALID_USAGE)}, got {ctx['usage_posture']!r}"
        )


def _validate_owner(ctx: dict, errors: list[str]) -> None:
    if "owner" in ctx and isinstance(ctx["owner"], dict):
        kind = ctx["owner"].get("kind")
        if kind not in VALID_OWNER_KINDS:
            errors.append(
                f"'owner.kind' must be one of {sorted(VALID_OWNER_KINDS)}, got {kind!r}"
            )
        if kind == "third_party":
            for k in ("name", "card_link"):
                if not ctx["owner"].get(k):
                    errors.append(
                        f"'owner.{k}' required when owner.kind == 'third_party'"
                    )


def _validate_credential_requirements(ctx: dict, errors: list[str]) -> None:
    credentials = ctx.get("credential_requirements")
    if credentials is None or not isinstance(credentials, dict):
        return

    for key in ("requires_api_key_or_credential", "credential_types"):
        if key not in credentials:
            errors.append(f"'credential_requirements.{key}' missing")

    status = credentials.get("requires_api_key_or_credential")
    if isinstance(status, str):
        if status not in VALID_CREDENTIAL_STATUSES:
            errors.append(
                "'credential_requirements.requires_api_key_or_credential' must be "
                f"one of {sorted(VALID_CREDENTIAL_STATUSES)}, got {status!r}"
            )
    elif status is not None:
        errors.append(
            "'credential_requirements.requires_api_key_or_credential' should be str, "
            "got "
            f"{type(status).__name__}"
        )

    _validate_string_list(
        "credential_requirements.credential_types", credentials.get("credential_types"), errors
    )

    types = credentials.get("credential_types")
    if isinstance(types, list):
        for idx, t in enumerate(types):
            if isinstance(t, str):
                valid = (
                    t in VALID_CREDENTIAL_TYPES
                    or (t.startswith("Other [") and t.endswith("]"))
                )
                if not valid:
                    errors.append(
                        f"'credential_requirements.credential_types[{idx}]' must be one of "
                        f"{sorted(VALID_CREDENTIAL_TYPES)} or 'Other [description]', got {t!r}"
                    )


def _validate_string_list(path: str, value: object, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"'{path}' should be list, got {type(value).__name__}")
        return
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(
                f"'{path}[{idx}]' should be str, got {type(item).__name__}"
            )


def _validate_credential_string_list_values(
    path: str, value: object, errors: list[str]
) -> None:
    if not isinstance(value, list):
        return
    for idx, item in enumerate(value):
        if isinstance(item, str):
            _validate_no_sensitive_credential_value(f"{path}[{idx}]", item, errors)


def _validate_no_sensitive_credential_value(
    path: str, value: str, errors: list[str]
) -> None:
    if any(pattern.search(value) for pattern in SENSITIVE_CREDENTIAL_VALUE_PATTERNS):
        errors.append(
            f"'{path}' must describe credential requirements without containing "
            "credential values, assignments, bearer tokens, private keys, or "
            "secret-like tokens"
        )


def _validate_output(ctx: dict, errors: list[str]) -> None:
    if "output" in ctx and isinstance(ctx["output"], dict):
        for k in ("types", "format", "parameters", "other_properties"):
            if k not in ctx["output"]:
                errors.append(f"'output.{k}' missing")


def _validate_evaluation(ctx: dict, errors: list[str]) -> None:
    evaluation = ctx.get("evaluation")
    if not isinstance(evaluation, dict):
        return
    _validate_evaluation_strings(evaluation, errors)
    _validate_evaluation_agents(evaluation, errors)
    _validate_evaluation_metrics(evaluation, errors)
    _validate_testing_completed(evaluation, errors)


def _validate_evaluation_strings(evaluation: dict, errors: list[str]) -> None:
    for key in EVALUATION_STRING_FIELDS:
        if key in evaluation and not isinstance(evaluation[key], str):
            errors.append(
                f"'evaluation.{key}' should be str, got "
                f"{type(evaluation[key]).__name__}"
            )


def _validate_evaluation_agents(evaluation: dict, errors: list[str]) -> None:
    agents = evaluation.get("agents", MISSING)
    if agents is MISSING:
        return
    if not isinstance(agents, list):
        errors.append(
            "'evaluation.agents' should be list, got " f"{type(agents).__name__}"
        )
        return
    for idx, agent in enumerate(agents):
        if not isinstance(agent, str):
            errors.append(
                f"'evaluation.agents[{idx}]' should be str, got "
                f"{type(agent).__name__}"
            )


def _validate_evaluation_metrics(evaluation: dict, errors: list[str]) -> None:
    metrics = evaluation.get("metrics", MISSING)
    if metrics is MISSING or isinstance(metrics, str):
        return
    if not isinstance(metrics, dict):
        errors.append(
            "'evaluation.metrics' should be str or dict, got "
            f"{type(metrics).__name__}"
        )
        return
    for group in EVALUATION_METRIC_GROUPS:
        entries = metrics.get(group, MISSING)
        if entries is not MISSING:
            _validate_metric_entries(group, entries, errors)


def _validate_metric_entries(group: str, entries: object, errors: list[str]) -> None:
    if not isinstance(entries, list):
        errors.append(
            f"'evaluation.metrics.{group}' should be list, got "
            f"{type(entries).__name__}"
        )
        return
    for idx, entry in enumerate(entries):
        _validate_metric_entry(group, idx, entry, errors)


def _validate_metric_entry(
    group: str, idx: int, entry: object, errors: list[str]
) -> None:
    if not isinstance(entry, dict):
        errors.append(
            f"'evaluation.metrics.{group}[{idx}]' should be dict, got "
            f"{type(entry).__name__}"
        )
        return
    for field in ("name", "description"):
        if field not in entry:
            errors.append(f"'evaluation.metrics.{group}[{idx}].{field}' missing")
        elif not isinstance(entry[field], str):
            errors.append(
                f"'evaluation.metrics.{group}[{idx}].{field}' should be str, got "
                f"{type(entry[field]).__name__}"
            )


def _validate_testing_completed(evaluation: dict, errors: list[str]) -> None:
    testing_completed = evaluation.get("testing_completed", MISSING)
    if testing_completed is MISSING:
        return
    if not isinstance(testing_completed, dict):
        errors.append(
            "'evaluation.testing_completed' should be dict, got "
            f"{type(testing_completed).__name__}"
        )
        return
    for key in TESTING_COMPLETED_FIELDS:
        if key not in testing_completed:
            errors.append(f"'evaluation.testing_completed.{key}' missing")
        elif not isinstance(testing_completed[key], bool):
            errors.append(
                f"'evaluation.testing_completed.{key}' should be bool, got "
                f"{type(testing_completed[key]).__name__}"
            )


def _validate_references(ctx: dict, errors: list[str]) -> None:
    for item in ctx.get("references", []):
        if not isinstance(item, dict) or "label" not in item or "url" not in item:
            errors.append("each 'references' item needs 'label' and 'url'")
            break


def _load_catalog(template_dir: Path, name: str) -> list:
    """Load a canned-entries catalog from references/catalog/<name>.json.

    Missing catalog file is tolerated (returns []) so the renderer still works
    for stripped-down skill directories, but the normal path is that both
    limitations.json and risks.json exist.
    """
    catalog_path = template_dir / "catalog" / f"{name}.json"
    if not catalog_path.exists():
        return []
    try:
        data = json.loads(catalog_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: catalog {catalog_path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(CATALOG_ERROR_EXIT_CODE)
    if not isinstance(data, list):
        print(f"ERROR: catalog {catalog_path} must be a JSON array", file=sys.stderr)
        sys.exit(CATALOG_ERROR_EXIT_CODE)
    return data


def _apply_marker_defaults(ctx: dict) -> None:
    """Ensure optional verify-marker fields exist so StrictUndefined doesn't bite."""
    ctx.setdefault("license_verify", False)
    ctx.setdefault("license_verify_reason", "")
    if isinstance(ctx.get("owner"), dict):
        ctx["owner"].setdefault("verify", False)
        ctx["owner"].setdefault("verify_reason", "")
    credentials = ctx.setdefault("credential_requirements", {})
    if isinstance(credentials, dict):
        credentials.setdefault("requires_api_key_or_credential", "not specified")
        credentials.setdefault("credential_types", [])


def render(context_path: Path, template_path: Path, out_path: Path) -> None:
    ctx = json.loads(context_path.read_text())
    errors = validate(ctx)
    if errors:
        print("Context validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(CONTEXT_VALIDATION_EXIT_CODE)

    _apply_marker_defaults(ctx)

    template_dir = template_path.parent

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    tmpl = env.get_template(template_path.name)
    rendered = tmpl.render(**ctx)
    out_path.write_text(rendered)
    print(f"Rendered card: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    render(args.context, args.template, args.out)


if __name__ == "__main__":
    main()
