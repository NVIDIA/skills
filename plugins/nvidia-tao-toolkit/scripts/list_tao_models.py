#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List packaged TAO model capabilities from shipped model metadata.

AutoML enablement is model-level metadata (`automl_enabled: true` in
skills/<network>/references/skill_info.yaml). Runnable AutoML support is then
gated by the exact packaged dataclass schema for the selected action.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# "python3 -I <script>" leaves the folder of the script off the import path, so
# add it for the sibling modules. A run as a script writes no __pycache__ here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    sys.dont_write_bytecode = True

try:
    import yaml
except ModuleNotFoundError:
    # One line and an exit code of its own instead of a traceback. "python3 -I" and
    # "python3 -s" leave out the user site, where pip installs for a user who is not
    # root, so the line names that case before it names the install command.
    _no_user_site = (
        " It runs without the user site (python3 -I or -s, or PYTHONNOUSERSITE), which hides "
        "a PyYAML that pip installed for the user: start the script without that option "
        "first. If PyYAML is still missing, install it with:"
    )
    print(f"error: this Python ({sys.executable}) cannot import PyYAML, which this helper "
          "script needs." + (_no_user_site if sys.flags.no_user_site else " Install it with:")
          + " python3 -m pip install pyyaml", file=sys.stderr)
    raise SystemExit(4)

from skill_layers import layer_file, layer_skill_dirs, skill_dir  # noqa: E402


# The bank this script ships in, so it works from a clone or any plugin install
# without setup. $TAO_SKILL_BANK_PATH overrides it when the caller means another bank.
DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH") or Path(__file__).resolve().parents[1]
)
SCHEMA_DIR_REL = Path("schemas")
TRAIN_SCHEMA_REL = SCHEMA_DIR_REL / "train.schema.json"
SUPPORT_RULE = (
    "AutoML is enabled at model level; runnable AutoML for an action also "
    "requires skills/<network>/schemas/<action>.schema.json to be packaged "
    "and valid."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "automl"),
        default="all",
        help="List all model actions or AutoML-capable models.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--action",
        default="",
        help="When --scope automl is used, filter support to one action.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_schema_manifest(skill_bank: Path) -> dict[str, Any]:
    """Load the packaged model action manifest."""
    return load_json(layer_file(skill_bank, "models", "schemas.manifest.json"))


def load_automl_manifest(skill_bank: Path) -> dict[str, Any]:
    """Load the packaged AutoML compatibility manifest if present."""
    path = layer_file(skill_bank, "models", "automl_support.json")
    if not path.exists():
        return {"supported": [], "unsupported": []}
    return load_json(path)


def parse_bool(value: Any, default: bool = True) -> bool:
    """Parse a bool-ish metadata value."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_scalar(value: str) -> Any:
    """Parse the small scalar subset used by references/skill_info.yaml."""
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    return text.strip("\"'")


def load_skill_info(skill_bank: Path, model: str) -> dict[str, Any]:
    """Load top-level model metadata from references/skill_info.yaml."""
    path = skill_dir(skill_bank, "models", model) / "references" / "skill_info.yaml"
    if not path.exists():
        return {}

    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith((" ", "\t", "#")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data


def load_backend_capabilities(skill_bank: Path, model: str) -> dict[str, Any]:
    """Load backend/action matrices declared by shared model frontends."""
    model_dir = skill_dir(skill_bank, "models", model)
    path = model_dir / "references" / "skill_info.yaml"
    if not path.exists():
        return {}
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("backend_contracts"), dict):
        return {}
    backends: dict[str, Any] = {}
    for name, declaration in parsed["backend_contracts"].items():
        if isinstance(declaration, str):
            relative = declaration
            declared_image = None
        elif isinstance(declaration, dict):
            relative = declaration.get("path")
            declared_image = declaration.get("container_image")
        else:
            raise ValueError(
                f"backend_contracts.{name} must be a relative YAML path or mapping"
            )
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"backend_contracts.{name}.path must be a relative YAML path")
        contract_path = model_dir / str(relative)
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError(f"{contract_path} must contain a YAML object")
        actions = contract.get("actions", {})
        backends[str(name)] = {
            "container_image": declared_image or contract.get("container_image"),
            "contract": str(relative),
            "actions": {
                str(action): {
                    "supported": bool(config.get("supported", False)) if isinstance(config, dict) else False,
                    "native": bool(
                        config.get(
                            "native",
                            bool(contract.get("native", False)) and bool(config.get("supported", False)),
                        )
                    )
                    if isinstance(config, dict)
                    else False,
                    "reason": config.get("reason", "") if isinstance(config, dict) else "",
                }
                for action, config in actions.items()
            }
            if isinstance(actions, dict)
            else {},
        }
    return {
        "selection_policy": parsed.get("backend_selection", {}),
        "implementations": backends,
    }


def skill_info_actions(skill_bank: Path, model: str) -> list[str]:
    """Read action names from a model skill_info.yaml without a YAML dependency."""
    path = skill_dir(skill_bank, "models", model) / "references" / "skill_info.yaml"
    if not path.exists():
        return []

    actions: list[str] = []
    in_actions = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("actions:"):
            in_actions = True
            continue
        if not in_actions:
            continue
        if line and not line.startswith((" ", "\t")):
            break
        if line.startswith("  ") and not line.startswith("    "):
            name = line.strip()
            if name.endswith(":"):
                actions.append(name[:-1])
    return sorted(actions)


def load_model_schema_manifest(skill_bank: Path, model: str) -> dict[str, Any]:
    """Load skills/<model>/schemas/manifest.json when shipped."""
    path = skill_dir(skill_bank, "models", model) / "schemas" / "manifest.json"
    if not path.exists():
        return {}
    return load_json(path)


def action_schema_rel(action: str) -> Path:
    """Return the relative schema path for an action."""
    return SCHEMA_DIR_REL / f"{action}.schema.json"


def action_schema_status(skill_bank: Path, model: str, action: str) -> tuple[bool, str]:
    """Return whether a model has a valid packaged action dataclass schema."""
    schema_rel = action_schema_rel(action)
    schema_path = skill_dir(skill_bank, "models", model) / schema_rel
    if not schema_path.exists():
        return False, f"{schema_rel.as_posix()} is not packaged"

    try:
        load_json(schema_path)
    except json.JSONDecodeError as exc:
        return False, f"{schema_rel.as_posix()} is invalid JSON: {exc.msg}"
    except OSError as exc:
        return False, f"{schema_rel.as_posix()} cannot be read: {exc}"
    except ValueError as exc:
        return False, str(exc)

    return True, f"{schema_rel.as_posix()} is packaged and valid"


def train_schema_status(skill_bank: Path, model: str) -> tuple[bool, str]:
    """Return whether a model has a valid packaged train dataclass schema."""
    return action_schema_status(skill_bank, model, "train")


def build_model_records(skill_bank: Path) -> list[dict[str, Any]]:
    """Build sorted model records from model-level metadata and schema manifests."""
    skill_bank = skill_bank.expanduser()
    global_manifest = load_schema_manifest(skill_bank)
    manifest_models = global_manifest.get("models", {})
    if not isinstance(manifest_models, dict):
        raise ValueError("catalog/schemas.manifest.json is missing a models object")

    model_names = set(manifest_models)
    model_names.update(
        item.name
        for item in layer_skill_dirs(skill_bank, "models")
        if (item / "references" / "skill_info.yaml").exists()
    )

    records: list[dict[str, Any]] = []
    for model in sorted(model_names):
        metadata = manifest_models.get(model, {})
        if not isinstance(metadata, dict):
            metadata = {}
        skill_info = load_skill_info(skill_bank, model)
        backend_capabilities = load_backend_capabilities(skill_bank, model)
        schema_manifest = load_model_schema_manifest(skill_bank, model)
        has_train_schema, train_schema_reason = train_schema_status(skill_bank, model)
        schema_actions = schema_manifest.get("actions", {})
        metadata_actions = metadata.get("actions", [])
        if not isinstance(metadata_actions, list):
            metadata_actions = []
        actions = set(metadata_actions)
        if isinstance(schema_actions, dict):
            actions.update(schema_actions)
        actions.update(skill_info_actions(skill_bank, model))
        actions = sorted(actions)
        failures = metadata.get("failures", {})
        if not isinstance(failures, dict):
            failures = {}
        if not failures:
            schema_failures = schema_manifest.get("failures", {})
            if isinstance(schema_failures, dict):
                failures = schema_failures

        action_schema_statuses = {}
        for action in actions:
            ok, reason = action_schema_status(skill_bank, model, action)
            action_schema_statuses[action] = {
                "schema": action_schema_rel(action).as_posix(),
                "valid": ok,
                "status": reason,
            }

        records.append(
            {
                "model": model,
                "network_arch": (
                    skill_info.get("network_arch")
                    or metadata.get("network_arch")
                    or schema_manifest.get("network_arch")
                    or model
                ),
                "actions": actions,
                "failures": failures,
                "automl_enabled": parse_bool(
                    skill_info.get("automl_enabled", metadata.get("automl_enabled")),
                    default=True,
                ),
                "automl_blocked_reason": skill_info.get("automl_blocked_reason", ""),
                "has_train_schema": has_train_schema,
                "train_schema": TRAIN_SCHEMA_REL.as_posix(),
                "train_schema_status": train_schema_reason,
                "action_schema_statuses": action_schema_statuses,
                "backends": backend_capabilities,
            }
        )
    return records


def build_all_models(skill_bank: Path) -> dict[str, Any]:
    """Return packaged model/action support."""
    return {
        "schema_version": 1,
        "source": "catalog/schemas.manifest.json",
        "models": build_model_records(skill_bank),
    }


def action_metadata(skill_bank: Path, model: str, action: str) -> dict[str, Any]:
    """Return AutoML parameter metadata for a model action."""
    schema_manifest = load_model_schema_manifest(skill_bank, model)
    actions = schema_manifest.get("actions", {})
    action_item = actions.get(action) if isinstance(actions, dict) else None
    if isinstance(action_item, dict):
        return {
            "spec_template": action_item.get("spec_template"),
            "automl_default_parameters": action_item.get("automl_default_parameters", []),
            "automl_disabled_parameters": action_item.get("automl_disabled_parameters", []),
        }

    schema_rel = action_schema_rel(action)
    schema_path = skill_dir(skill_bank, "models", model) / schema_rel
    if schema_path.exists():
        schema = load_json(schema_path)
        return {
            "spec_template": f"references/spec_template_{action}.yaml",
            "automl_default_parameters": schema.get("automl_default_parameters", []),
            "automl_disabled_parameters": schema.get("automl_disabled_parameters", []),
        }
    return {
        "spec_template": None,
        "automl_default_parameters": [],
        "automl_disabled_parameters": [],
    }


def build_automl_support(skill_bank: Path, action_filter: str = "") -> dict[str, Any]:
    """Return model-level AutoML support, validated per packaged action schema."""
    automl_manifest = load_automl_manifest(skill_bank)
    model_records = {item["model"]: item for item in build_model_records(skill_bank)}

    supported_manifest = {
        item.get("model"): item
        for item in automl_manifest.get("supported", [])
        if isinstance(item, dict) and item.get("model")
    }

    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for model in sorted(model_records):
        record = model_records[model]

        if not record["automl_enabled"]:
            unsupported.append(
                {
                    "model": model,
                    "network_arch": record["network_arch"],
                    "automl_enabled": False,
                    "reason": "automl_enabled is false in model metadata",
                    "train_schema_status": record["train_schema_status"],
                }
            )
            continue

        if record.get("automl_blocked_reason"):
            unsupported.append(
                {
                    "model": model,
                    "network_arch": record["network_arch"],
                    "automl_enabled": True,
                    "reason": record["automl_blocked_reason"],
                    "train_schema_status": record["train_schema_status"],
                }
            )
            continue

        candidate_actions = record["actions"]
        if action_filter:
            candidate_actions = [action for action in candidate_actions if action == action_filter]

        supported_actions = []
        unsupported_actions = []
        for action in candidate_actions:
            status = record["action_schema_statuses"].get(action, {})
            if status.get("valid"):
                manifest_item = supported_manifest.get(model, {})
                action_manifest = {}
                for item_action in manifest_item.get("supported_actions", []):
                    if isinstance(item_action, dict) and item_action.get("action") == action:
                        action_manifest = item_action
                        break
                metadata = action_metadata(skill_bank, model, action)
                supported_actions.append(
                    {
                        "action": action,
                        "schema": status.get("schema", action_schema_rel(action).as_posix()),
                        "schema_status": status.get("status", ""),
                        "spec_template": (
                            action_manifest.get("spec_template")
                            or metadata.get("spec_template")
                        ),
                        "automl_default_parameters": (
                            metadata.get("automl_default_parameters")
                            or action_manifest.get("automl_default_parameters", [])
                        ),
                    }
                )
            else:
                unsupported_actions.append(
                    {
                        "action": action,
                        "reason": status.get("status", f"{action_schema_rel(action).as_posix()} is not packaged"),
                    }
                )

        if supported_actions:
            manifest_item = supported_manifest.get(model, {})
            train_action = next(
                (item for item in supported_actions if item["action"] == "train"),
                {},
            )
            item = {
                "model": model,
                "network_arch": record["network_arch"],
                "automl_enabled": True,
                "train_schema": train_action.get("schema", TRAIN_SCHEMA_REL.as_posix()),
                "train_schema_status": record["train_schema_status"],
                "train_spec_template": (
                    manifest_item.get("train_spec_template")
                    or train_action_metadata(skill_bank, model).get("train_spec_template")
                ),
                "actions": record["actions"],
                "supported_actions": supported_actions,
                "unsupported_actions": unsupported_actions,
                "automl_default_parameters": (
                    train_action_metadata(skill_bank, model).get("automl_default_parameters")
                    or manifest_item.get("automl_default_parameters", [])
                ),
            }
            supported.append(item)
            continue

        if action_filter and not candidate_actions:
            reason = f"action {action_filter!r} is not declared for this model"
        elif unsupported_actions:
            reason = "; ".join(
                f"{item['action']}: {item['reason']}" for item in unsupported_actions
            )
        else:
            reason = record["train_schema_status"]
        unsupported.append(
            {
                "model": model,
                "network_arch": record["network_arch"],
                "automl_enabled": True,
                "reason": reason,
                "train_schema_status": record["train_schema_status"],
                "unsupported_actions": unsupported_actions,
            }
        )

    return {
        "schema_version": 1,
        "source": [
            "skills/<network>/references/skill_info.yaml",
            "skills/<network>/schemas/manifest.json",
            "skills/<network>/schemas/<action>.schema.json",
        ],
        "support_rule": SUPPORT_RULE,
        "action_filter": action_filter,
        "supported": supported,
        "unsupported": unsupported,
    }


def train_action_metadata(skill_bank: Path, model: str) -> dict[str, Any]:
    """Return AutoML parameter metadata for a model train action."""
    schema_manifest = load_model_schema_manifest(skill_bank, model)
    actions = schema_manifest.get("actions", {})
    train = actions.get("train") if isinstance(actions, dict) else None
    if isinstance(train, dict):
        return {
            "train_spec_template": train.get("spec_template"),
            "automl_default_parameters": train.get("automl_default_parameters", []),
        }

    train_schema = skill_dir(skill_bank, "models", model) / TRAIN_SCHEMA_REL
    if train_schema.exists():
        schema = load_json(train_schema)
        return {
            "train_spec_template": "references/spec_template_train.yaml",
            "automl_default_parameters": schema.get("automl_default_parameters", []),
        }
    return {"train_spec_template": None, "automl_default_parameters": []}


def action_text(actions: list[str]) -> str:
    """Format action names for compact text output."""
    return ", ".join(actions) if actions else "no packaged action schemas"


def format_all_models_text(data: dict[str, Any]) -> str:
    """Format packaged model/action support for a human."""
    lines = ["Packaged TAO models and action schemas:"]
    for item in data["models"]:
        train_schema = (
            "train schema valid" if item["has_train_schema"]
            else f"train schema {item['train_schema_status']}"
        )
        lines.append(
            "- {model} ({network_arch}): {actions}; {schema}".format(
                model=item["model"],
                network_arch=item["network_arch"],
                actions=action_text(item["actions"]),
                schema=train_schema,
            )
        )
        implementations = item.get("backends", {}).get("implementations", {})
        for backend, details in implementations.items():
            supported = [
                action for action, config in details.get("actions", {}).items() if config.get("supported")
            ]
            lines.append(f"  backend {backend}: {action_text(supported)}")
    return "\n".join(lines)


def format_automl_text(data: dict[str, Any]) -> str:
    """Format AutoML support for a human."""
    suffix = f" Action filter: {data['action_filter']}." if data.get("action_filter") else ""
    lines = [data["support_rule"] + suffix, "", "Supported AutoML models/actions:"]
    if data["supported"]:
        for item in data["supported"]:
            action_parts = []
            for action_item in item.get("supported_actions", []):
                params = action_item.get("automl_default_parameters", [])
                params_text = ", ".join(params) if params else "schema-defined defaults"
                action_parts.append(f"{action_item['action']} [{params_text}]")
            actions_text = "; ".join(action_parts) if action_parts else "train"
            lines.append(
                f"- {item['model']} ({item['network_arch']}): automl_enabled=true; "
                f"actions: {actions_text}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Not supported:")
    if data["unsupported"]:
        lines.extend(
            f"- {item['model']} ({item['network_arch']}): "
            f"automl_enabled={str(item.get('automl_enabled', False)).lower()}; {item['reason']}"
            for item in data["unsupported"]
        )
    else:
        lines.append("- None")
    return "\n".join(lines)


def main() -> int:
    """Run the model listing helper."""
    args = parse_args()
    skill_bank = args.skill_bank.expanduser()
    data = (
        build_automl_support(skill_bank, action_filter=args.action)
        if args.scope == "automl"
        else build_all_models(skill_bank)
    )

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    elif args.scope == "automl":
        print(format_automl_text(data))
    else:
        print(format_all_models_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
