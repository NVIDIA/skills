---
name: config-conventions
description: Configuration conventions for NeMo-RL. YAML is the single source of truth for defaults. Covers TypedDict usage, exemplar YAML updates, and forbidden default patterns.
when_to_use: Adding or modifying config fields; reviewing config changes; 'where do I set defaults', 'TypedDict pattern', 'exemplar YAML', 'forbidden default patterns', during code review of config files.
---

# Configuration Conventions

## Core Rule

**YAML is the single source of truth for defaults.** Do not set non-`None` defaults in code for configuration values. The loaded YAML (and any user overrides) must supply required values.

## Access Config Directly

For required attributes, write code like `policy_cfg["precision"]` and assume it is present. Do not introduce hidden defaults deep in the code.

## Express Optionality via TypedDict

Use `typing.NotRequired` to mark optional attributes. Optional attributes may be absent/`None`; code may check for their presence.

## Where Defaults Live

- Exemplar configs under `examples/configs/*.yaml` include documented defaults.
- Recipe YAMLs under `examples/configs/recipes/**/*.yaml` are runnable snapshots and may omit documentation.

## Documenting New Config Keys

When adding a new config key to a `TypedDict` subclass, document:
- The key's purpose
- Valid values/types
- Recommended default (if applicable)

Reflect the default in the exemplar YAMLs under `examples/configs/*.yaml`.

## Recipe YAMLs Must Set `defaults`

Recipe YAMLs under `examples/configs/recipes/**/*.yaml` must set `defaults: <exemplar>.yaml` to inherit from one of the exemplar configs in `examples/configs/*.yaml`. This keeps recipes minimal — they only override what differs from the exemplar.

If a recipe YAML does not have a `defaults` key, run:

```bash
uv run ./tools/config_cli.py minimize <recipe.yaml>
```

This will minimize the config and assign the appropriate `defaults` key.

## Accessing NotRequired Fields

When accessing a `NotRequired` field, use an `in` check or `.get(key)` / `.get(key, None)`. Never provide a non-`None` default — that hides behavior and defeats the purpose of making the field optional.

**Do:**
```python
# .get() with None (not a hidden default)
stop_properly_penalty_coef = cfg.get("stop_properly_penalty_coef", None)

# Truthiness check for optional booleans
if master_config.grpo.get("skip_reference_policy_logprobs_calculation"):
    ...

# Nested NotRequired: check presence at each level explicitly
if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
    ...
```

**Don't:**
```python
# Hidden boolean default — should come from YAML
disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)

# Hidden non-trivial default — caller has no idea True is the fallback
normalize_rewards = grpo_config.get("normalize_rewards", True)

# Chained .get() with hidden defaults at each level
megatron_enable = config.get("megatron_cfg", {}).get("enabled", False)
```

If a `NotRequired` field is absent, the code should handle that explicitly — not paper over it with a magic default.

## Forbidden Patterns

**Don't:**
```python
# Hidden default in code
precision = policy_cfg.get("precision", "bfloat16")

# Function parameter defaulting a config value
def build_policy(policy_cfg, precision: str = "bfloat16"):
    ...
```

**Do:**
```python
# Required attribute: expect it from YAML or user override
precision: str = policy_cfg["precision"]

# Optional attribute: check for presence
if "milestones" in scheduler_cfg:
    configure_milestones(scheduler_cfg["milestones"])
```

See also: @docs/design-docs/design-and-philosophy.md (TypedDict and Configuration Defaults section).
