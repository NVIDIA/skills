#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Generate metadata.json and skills.sh.json for the NVIDIA skills catalog.

Pipeline:
    discover skills/**/SKILL.md
        -> parse YAML frontmatter (multiline-block-scalar aware)
        -> map to product via components.d/*.yml
        -> classify against baseline metadata.json (added/removed/renamed/changed)
        -> carry forward valid existing metadata
        -> AI enrichment for missing fields (strict-JSON, taxonomy-bound)
        -> roll plugins.d/*.yml up into plugin taxonomy from bundled skills
        -> validate against metadata.schema.json + skills-sh.schema.json
        -> emit byte-stable metadata.json and skills.sh.json

Modes:
    --write   (default)  Regenerate and write outputs.
    --check              Regenerate in memory; fail with diff if checked-in
                         outputs are stale or schema-invalid. Used in PR CI.
    --no-ai              Disable AI enrichment; fail when enrichment is needed.
    --report-only        Print change classification only; do not write/validate.

Environment:
    INFERENCE_API_KEY    NVIDIA Inference API token (required when AI is used).
    INFERENCE_API_URL    Override endpoint. Defaults to NVIDIA Inference API.
    INFERENCE_MODEL      Required when AI enrichment runs. No default; a missing
                         value with enrichment-needed skills causes a hard fail.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import difflib
import json
import os
import posixpath
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable

import yaml  # PyYAML; already a workflow dependency.

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - install path documented in README
    print(
        "ERROR: jsonschema is required. Install with `pip install jsonschema`.",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Paths

# This script lives at .github/scripts/marketplace/generate-skill-metadata.py
# so REPO_ROOT is three levels up.
MARKETPLACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MARKETPLACE_DIR.parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
COMPONENTS_DIR = REPO_ROOT / "components.d"
PLUGINS_D_DIR = REPO_ROOT / "plugins.d"
# The built and mirrored plugin trees both live here. A module constant
# rather than REPO_ROOT / "plugins" at the call site, so tests can point the
# mirrored-plugin pass at a fixture without a repository on disk.
PLUGINS_DIR = REPO_ROOT / "plugins"

# All marketplace artifacts (generated outputs, schemas, the subdomain config,
# the exclusions list, and this generator script) live under
# .github/scripts/marketplace/. Canonical output layout follows upstream
# commits f83be81 + 6928ef1: metadata.json sits inside marketplace/;
# skills.sh.json stays at the repo root as the published index file.
METADATA_PATH = MARKETPLACE_DIR / "metadata.json"
SKILLS_SH_PATH = REPO_ROOT / "skills.sh.json"

SCHEMA_PATH = MARKETPLACE_DIR / "metadata.schema.json"
SKILLS_SH_SCHEMA_PATH = MARKETPLACE_DIR / "skills-sh.schema.json"
SUBDOMAINS_PATH = MARKETPLACE_DIR / "skills-subdomains.json"
EXCLUSIONS_PATH = MARKETPLACE_DIR / "metadata-exclusions.yaml"

MVP_FIELDS = (
    "product.primary",
    "classification.category.primary",
    "catalog.subdomain",
    "audience",
    "discovery.activity_tags",
)

# Taxonomy fields rolled up from a plugin's bundled skills.
#
# Only three, and deliberately so. `classification.category.primary` (Domain)
# is the one facet build.nvidia.com displays for plugins at launch;
# `product.primary` (Library) and `audience` are emitted but not displayed, so
# the frontend can switch either on later without a generator change or a
# regeneration. `catalog.subdomain` and `discovery.activity_tags` are omitted
# because neither has a defined meaning for a bundle.
PLUGIN_FIELDS = (
    "classification.category.primary",
    "product.primary",
    "audience",
)


# ---------------------------------------------------------------------------
# Data classes


@dataclasses.dataclass
class Skill:
    path: str  # repo-relative, e.g. "skills/cuopt-routing-api-python"
    name: str
    description: str
    frontmatter: dict
    skill_card: str | None = None  # raw skill-card.md text (optional)
    card_yaml: dict | None = None  # parsed card.yaml (optional)


@dataclasses.dataclass
class Classification:
    added: list[str]
    removed: list[str]
    renamed: dict[str, str]  # old_path -> new_path
    materially_changed: list[str]
    unchanged: list[str]
    excluded: list[str]


# ---------------------------------------------------------------------------
# Frontmatter parsing


_FRONTMATTER_RE = re.compile(
    # Capture the YAML body INCLUDING the final newline of the last value
    # so that block-scalar clip-mode (`description: >` / `description: |`)
    # preserves the single trailing \n YAML semantically appends. The closing
    # `---` line follows on its own line.
    r"\A---\r?\n(?P<body>.*?\n)---\s*?(\r?\n|\Z)",
    re.DOTALL,
)


def parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from a SKILL.md.

    Handles multiline block scalars (``description: |`` and ``description: >``)
    via PyYAML, exactly as the PRD requires. The capture intentionally retains
    the trailing newline that precedes the closing ``---`` so YAML's default
    clip-mode chomping behavior works correctly for end-of-document scalars.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    body = m.group("body")
    data = yaml.safe_load(body)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Inputs


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def taxonomy_from_schema(schema: dict) -> dict:
    """Extract per-field controlled values from the metadata schema.

    Returns ``{field_name: {"kind": "single"|"multi-csv", "values": [...]}}``.
    Single-value fields use ``enum``; multi-value comma-separated fields use
    ``pattern`` (a regex like ``^(?:a|b|c)(?:,(?:a|b|c))*$``).
    """
    md = schema["$defs"]["metadata"]["properties"]

    def from_pattern(pattern: str) -> list[str]:
        m = re.search(r"\^\(\?:([^)]+)\)", pattern)
        return m.group(1).split("|") if m else []

    out: dict[str, dict] = {}
    for field, prop in md.items():
        if "enum" in prop:
            out[field] = {"kind": "single", "values": list(prop["enum"])}
        elif "pattern" in prop:
            out[field] = {"kind": "multi-csv", "values": from_pattern(prop["pattern"])}
    return out


def load_exclusions() -> set[str]:
    if not EXCLUSIONS_PATH.exists():
        return set()
    data = load_yaml(EXCLUSIONS_PATH) or {}
    items = data.get("exclusions") or []
    out: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            out.add(name.strip())
    return out


def load_components(product_enum: list[str]) -> dict[str, dict]:
    """Build map: repo-relative skill path (without trailing slash) -> component dict.

    Source of truth is ``components.d/*.yml`` only — the synced product
    registries published by upstream sub-repos. Skills that have no
    ``components.d/`` entry (e.g. catalog-only skills staged via direct PR)
    fall through to AI enrichment for ``product.primary`` like any other
    skill missing a deterministic mapping.
    """
    mapping: dict[str, dict] = {}

    if not COMPONENTS_DIR.exists():
        return mapping

    for ymlf in sorted(COMPONENTS_DIR.glob("*.yml")):
        data = load_yaml(ymlf) or {}
        cname = data.get("name")
        crepo = data.get("repo")
        declared = {}
        if "product_primary" in data:
            validate_product_primary(data["product_primary"], product_enum, str(ymlf))
            declared["product_primary"] = data["product_primary"]
        for entry in data.get("skills") or []:
            if not isinstance(entry, dict):
                continue
            spath = entry.get("path") or ""
            if not isinstance(spath, str) or not spath:
                continue
            spath = spath.rstrip("/")
            # Key by the CATALOG path, not the source-repo path. The yml
            # `path` reflects the source repo's layout (which may nest skills
            # under product subdirs, e.g. TAO's skills/models/<name>/), while
            # lookups in build_skill_entry use the catalog-side skills/<dir>
            # path. catalog_dir is the invariant between the two.
            catalog_dir = entry.get("catalog_dir")
            if isinstance(catalog_dir, str) and catalog_dir.strip():
                key = f"skills/{catalog_dir.strip().rstrip('/')}"
            elif spath.startswith("skills/"):
                # Legacy entries without catalog_dir: source path is assumed
                # catalog-shaped (pre-AIQ-flat sweeps).
                key = spath
            else:
                # Source-side paths like .claude/skills/ with no catalog_dir
                # cannot be mapped to a catalog location.
                continue
            mapping[key] = {
                "component_name": cname,
                "component_repo": crepo,
                "components_file": ymlf.name,
                "catalog_dir": catalog_dir,
                **declared,
            }
    return mapping


# ---------------------------------------------------------------------------
# Discovery


def discover_skills(exclusions: set[str]) -> tuple[list[Skill], list[str]]:
    """Walk ``skills/**/SKILL.md`` and return (active_skills, excluded_skill_names).

    Each parent dir of a SKILL.md is one skill.
    """
    active: list[Skill] = []
    excluded: list[str] = []

    if not SKILLS_DIR.exists():
        return active, excluded

    for skill_md in sorted(SKILLS_DIR.rglob("SKILL.md")):
        skill_dir = skill_md.parent
        rel = skill_dir.relative_to(REPO_ROOT).as_posix()
        if not rel.startswith("skills/"):
            continue

        text = skill_md.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if not fm:
            raise GeneratorError(
                f"{rel}/SKILL.md: missing or unparseable YAML frontmatter."
            )
        name = fm.get("name")
        description = fm.get("description")
        if not isinstance(name, str) or not name.strip():
            raise GeneratorError(
                f"{rel}/SKILL.md: frontmatter `name` is missing or not a string."
            )
        if not isinstance(description, str) or not description.strip():
            raise GeneratorError(
                f"{rel}/SKILL.md: frontmatter `description` is missing or not a string."
            )

        if name in exclusions:
            excluded.append(name)
            continue

        # Optional companion files for AI context.
        skill_card_path = skill_dir / "skill-card.md"
        skill_card = (
            skill_card_path.read_text(encoding="utf-8")
            if skill_card_path.exists()
            else None
        )
        card_yaml_path = skill_dir / "card.yaml"
        card_yaml: dict | None = None
        if card_yaml_path.exists():
            try:
                loaded = load_yaml(card_yaml_path)
                if isinstance(loaded, dict):
                    card_yaml = loaded
            except yaml.YAMLError:
                card_yaml = None

        active.append(
            Skill(
                path=rel,
                # Preserve YAML's parsed values verbatim so the output matches
                # the authored frontmatter exactly. YAML already strips
                # block-scalar indentation; literal blocks keep a trailing \n.
                name=name,
                description=description,
                frontmatter=fm,
                skill_card=skill_card,
                card_yaml=card_yaml,
            )
        )

    return active, sorted(excluded)


# ---------------------------------------------------------------------------
# Classification (delta vs baseline)


def classify(
    current: list[Skill],
    baseline: dict | None,
    excluded_now: list[str],
) -> Classification:
    cur_paths = {s.path for s in current}
    cur_by_path = {s.path: s for s in current}

    base_skills = (baseline or {}).get("skills") or []
    base_paths = {b["path"] for b in base_skills if isinstance(b, dict) and "path" in b}
    base_by_path = {b["path"]: b for b in base_skills if isinstance(b, dict) and "path" in b}

    added = sorted(cur_paths - base_paths)
    removed_candidates = sorted(base_paths - cur_paths)

    # Detect simple renames by matching name+description across baseline-removed
    # and current-added. Both fields must match to avoid false positives when
    # two unrelated skills happen to share the same name.
    renamed: dict[str, str] = {}
    name_desc_to_added = {
        (cur_by_path[p].name, cur_by_path[p].description): p for p in added
    }
    for old in list(removed_candidates):
        old_entry = base_by_path[old]
        key = (old_entry.get("name"), old_entry.get("description"))
        if key in name_desc_to_added:
            new_path = name_desc_to_added[key]
            renamed[old] = new_path
            removed_candidates.remove(old)
            added.remove(new_path)

    # Material change = name/description differ between baseline and current.
    materially_changed: list[str] = []
    unchanged: list[str] = []
    for path in sorted(cur_paths & base_paths):
        cur = cur_by_path[path]
        base = base_by_path[path]
        if cur.name != base.get("name") or cur.description != base.get("description"):
            materially_changed.append(path)
        else:
            unchanged.append(path)

    return Classification(
        added=added,
        removed=removed_candidates,
        renamed=renamed,
        materially_changed=materially_changed,
        unchanged=unchanged,
        excluded=excluded_now,
    )


# ---------------------------------------------------------------------------
# Carry-forward + enrichment


def existing_valid_metadata(
    skill: Skill,
    baseline: dict | None,
    rename_map: dict[str, str],
    schema_validator: Draft202012Validator,
    authoritative: dict[str, str] | None = None,
) -> dict | None:
    """Return baseline metadata for ``skill`` if it is valid under the current schema.

    Honors renames (old_path -> new_path) by looking up the old path's entry.
    Apply authoritative values before validation so a renamed product enum
    does not discard the skill's other valid metadata or require AI.
    """
    base_skills = (baseline or {}).get("skills") or []
    base_by_path = {b["path"]: b for b in base_skills if isinstance(b, dict) and "path" in b}

    candidate_paths = [skill.path]
    inverse_renames = {new: old for old, new in rename_map.items()}
    if skill.path in inverse_renames:
        candidate_paths.append(inverse_renames[skill.path])

    for cp in candidate_paths:
        entry = base_by_path.get(cp)
        if not entry:
            continue
        md = entry.get("metadata") or {}
        if not isinstance(md, dict):
            continue
        md = {**md, **(authoritative or {})}
        # Validate just this skill entry shape against the schema's $defs/skill.
        candidate_entry = {
            "path": skill.path,
            "name": skill.name,
            "description": skill.description,
            "metadata": md,
        }
        try:
            schema_validator.validate({"skills": [candidate_entry]})
        except Exception:
            continue
        return dict(md)
    return None


def missing_required_fields(metadata: dict | None) -> list[str]:
    if not metadata:
        return list(MVP_FIELDS)
    return [k for k in MVP_FIELDS if not metadata.get(k)]


# ---------------------------------------------------------------------------
# AI enrichment client


class EnrichmentError(Exception):
    pass


def build_ai_client(allow_ai: bool):
    """Return an enrichment callable, or None if AI is disabled."""
    if not allow_ai:
        return None
    return _AIClient()


_RETRY_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 1  # doubles on each retry: 1s, 2s, 4s


class _AIClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("INFERENCE_API_KEY")
        self.api_url = os.environ.get(
            "INFERENCE_API_URL",
            "https://inference-api.nvidia.com/v1/chat/completions",
        )
        self.model = os.environ.get("INFERENCE_MODEL")

    def __call__(
        self,
        skill: Skill,
        component: dict | None,
        requested_fields: list[str],
        taxonomy: dict,
        current_values: dict | None = None,
    ) -> dict:
        """Ask the model to assign or review the given metadata fields.

        Modes:
        - Fill mode (``current_values is None``): every ``requested_fields``
          entry is treated as missing; the model picks a value for each.
        - Amend mode (``current_values`` provided): the model is shown the
          existing value for each requested field and asked to either keep it
          (return it verbatim) or change it to a better-fitting controlled
          value if the new skill content warrants it.
        Validation is identical in both modes — every requested key must be
        returned with a non-empty string from the controlled vocabulary, no
        extra keys, no ``UNRESOLVED`` placeholders.
        """
        if not self.api_key:
            raise EnrichmentError(
                "AI enrichment needed but INFERENCE_API_KEY is not set."
            )
        if not self.model:
            raise EnrichmentError(
                "AI enrichment needed but INFERENCE_MODEL is not set; configure "
                "the model name (no default is provided)."
            )

        amend_mode = current_values is not None
        system = textwrap.dedent(
            """
            You assign NVIDIA skill metadata. You must:
              - Return only a single JSON object, no prose.
              - Only use values from the provided controlled vocabularies. Do
                not invent values.
              - Return only the requested keys; no others.
              - For comma-separated fields, return a single string of allowed
                values joined by `,` with NO spaces.
              - If you cannot confidently pick a value, return the string
                "UNRESOLVED" for that key.
            """
        ).strip()

        controlled = {f: taxonomy[f]["values"] for f in requested_fields}
        kinds = {f: taxonomy[f]["kind"] for f in requested_fields}
        user: dict = {
            "skill_path": skill.path,
            "skill_name": skill.name,
            "skill_description": skill.description,
            "skill_frontmatter": skill.frontmatter,
            "component": component or {},
            "skill_card_excerpt": (skill.skill_card or "")[:4000],
            "requested_fields": requested_fields,
            "controlled_values_per_field": controlled,
            "field_kinds": kinds,
        }
        if amend_mode:
            user["existing_values"] = {
                f: current_values.get(f) for f in requested_fields
            }
            user["mode"] = "amend"
            user["instructions"] = (
                "The skill's name and/or description has materially changed "
                "since this metadata was assigned. Review each requested "
                "field's existing value against the current skill content. "
                "For each field, return either the existing value verbatim "
                "(if it still fits the new content) or a different "
                "controlled value (if the new content clearly warrants a "
                "different choice). Prefer keeping the existing value; only "
                "change a field when the new content makes a different "
                "controlled value clearly more appropriate. Return one "
                "string value per requested field. For multi-csv fields, "
                "use 1-5 values joined by commas with no spaces, ordered "
                "most-relevant first. Do not invent values, do not return "
                "arrays, do not return extra keys, do not include "
                "explanations."
            )
        else:
            user["mode"] = "fill"
            user["instructions"] = (
                "Choose values from controlled_values_per_field. For "
                "multi-csv fields, choose 1-5 most-applicable values, joined "
                "by commas with no spaces, ordered most-relevant first. Do "
                "not invent values, do not return arrays, do not return "
                "extra keys, do not include explanations."
            )

        # `temperature` is intentionally omitted: this is a strict
        # controlled-vocabulary classification task constrained by
        # response_format=json_object plus a small per-field enum, so the
        # model's default temperature is fine, and omitting the field keeps
        # us compatible with models (e.g. gpt-5.x) that reject any explicit
        # value.
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(user, ensure_ascii=False, default=str),
                },
            ],
            # Reasoning models spend completion budget on hidden reasoning
            # tokens BEFORE any visible content. 512 was routinely exhausted
            # mid-reasoning, returning an empty `content` with
            # finish_reason=length (seen deterministically on
            # nemo-relay-debug-runtime-integration, 2026-07-17). The task
            # output itself is tiny; the headroom is for reasoning.
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }

        # Lazy import so the deterministic --no-ai path needs no `requests`.
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise EnrichmentError(
                "AI enrichment needed but `requests` is not installed."
            ) from exc

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: str = ""
        for attempt in range(_RETRY_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    headers=headers,
                    json=body,
                    timeout=60,
                )
            except (requests.Timeout, requests.ConnectionError):
                last_error = "request timed out or connection failed"
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(_RETRY_BASE_SECONDS * (2 ** attempt))
                    continue
                raise EnrichmentError(
                    f"Inference API unreachable after {_RETRY_ATTEMPTS + 1} attempts."
                )
            except requests.RequestException as exc:
                raise EnrichmentError(f"Inference API request failed: {exc}")

            # Retry on rate-limit (429), transient server errors (5xx), and 403.
            # The gateway fronting the inference API returns 403 for transient
            # edge rejections as well as for genuine permission failures: on
            # 2026-08-03 an enrichment call failed with 403 three minutes after
            # the identical call succeeded. Retrying costs ~7s on a real
            # permission failure and saves an enrichment on a transient one.
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                if attempt < _RETRY_ATTEMPTS:
                    wait = _RETRY_BASE_SECONDS * (2 ** attempt)
                    print(
                        f"  [retry {attempt + 1}/{_RETRY_ATTEMPTS}] "
                        f"API returned {resp.status_code} for {skill.path}; "
                        f"retrying in {wait}s…",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                raise EnrichmentError(
                    f"Inference API returned HTTP {resp.status_code} after "
                    f"{_RETRY_ATTEMPTS + 1} attempts: {resp.text[:500]}"
                )
            if resp.status_code != 200:
                raise EnrichmentError(
                    f"Inference API returned HTTP {resp.status_code}: {resp.text[:500]}"
                )

            # Parse inside the retry loop: an empty or non-JSON `content`
            # (e.g. completion budget exhausted by reasoning tokens, or a
            # transiently garbled response) is retryable, not terminal.
            # Raising here used to silently drop the skill from the output.
            payload = None
            content = None
            try:
                payload = resp.json()
                content = payload["choices"][0]["message"]["content"]
                obj = json.loads(content)
            except (KeyError, ValueError, TypeError) as exc:
                finish = None
                if isinstance(payload, dict):
                    try:
                        finish = payload["choices"][0].get("finish_reason")
                    except (KeyError, IndexError, TypeError, AttributeError):
                        pass
                last_error = (
                    f"malformed JSON: {exc} "
                    f"(finish_reason={finish!r}, content={str(content)[:200]!r})"
                )
                if attempt < _RETRY_ATTEMPTS:
                    wait = _RETRY_BASE_SECONDS * (2 ** attempt)
                    print(
                        f"  [retry {attempt + 1}/{_RETRY_ATTEMPTS}] "
                        f"unusable response for {skill.path} ({last_error}); "
                        f"retrying in {wait}s…",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                raise EnrichmentError(
                    f"Inference API returned malformed JSON after "
                    f"{_RETRY_ATTEMPTS + 1} attempts: {last_error}"
                ) from exc
            break  # parsed successfully

        if not isinstance(obj, dict):
            raise EnrichmentError("Inference API JSON is not an object.")

        unexpected = set(obj.keys()) - set(requested_fields)
        if unexpected:
            raise EnrichmentError(
                f"Inference API returned unexpected keys: {sorted(unexpected)}"
            )
        for f in requested_fields:
            if f not in obj or not isinstance(obj[f], str) or not obj[f]:
                raise EnrichmentError(
                    f"Inference API did not return a usable string for `{f}`."
                )
            if obj[f] == "UNRESOLVED":
                raise EnrichmentError(
                    f"AI returned UNRESOLVED for `{f}` on skill "
                    f"`{skill.name}`; metadata cannot be auto-generated. "
                    f"Please set this field manually or update the skill."
                )
        return obj


# ---------------------------------------------------------------------------
# Build entries


class GeneratorError(Exception):
    pass


def validate_product_primary(value: Any, product_enum: list[str], location: str) -> None:
    if not isinstance(value, str) or value not in product_enum:
        raise GeneratorError(
            f"{location}: product_primary {value!r} must be a value in the "
            "`product.primary` enum in metadata.schema.json."
        )


def derive_product_from_component(component: dict | None, product_enum: list[str]) -> str | None:
    if not component:
        return None
    if "product_primary" in component:
        value = component["product_primary"]
        validate_product_primary(
            value, product_enum, component.get("components_file", "component"),
        )
        return value
    cname = component.get("component_name")
    if isinstance(cname, str) and cname in product_enum:
        return cname
    return None


def build_skill_entry(
    skill: Skill,
    components: dict[str, dict],
    baseline: dict | None,
    rename_map: dict[str, str],
    schema_validator: Draft202012Validator,
    taxonomy: dict,
    ai_client,
    skill_warnings: list[str],
    retained_notices: list[str] | None = None,
    is_materially_changed: bool = False,
) -> dict | None:
    component = components.get(skill.path)

    # Explicit declarations win, followed by an exact component-name match.
    # Both are authoritative for product.primary —
    # including over a carried baseline value. AI-enriched baselines can be
    # wrong (e.g. TAO skills labeled "Cosmos"/"Brev" from content inference)
    # and would otherwise persist forever, breaking product grouping on
    # downstream surfaces that read this file.
    product_enum = taxonomy["product.primary"]["values"]
    derived_product = derive_product_from_component(component, product_enum)
    authoritative = {}
    if derived_product:
        authoritative["product.primary"] = derived_product
    carried = existing_valid_metadata(
        skill, baseline, rename_map, schema_validator, authoritative=authoritative,
    ) or {}
    metadata: dict[str, str] = {k: carried[k] for k in MVP_FIELDS if k in carried}
    metadata.update(authoritative)

    missing = missing_required_fields(metadata)
    if missing:
        if ai_client is None:
            skill_warnings.append(
                f"{skill.path}: missing required fields {missing}; AI enrichment "
                f"is disabled (--no-ai)."
            )
            return None
        try:
            enrichment = ai_client(skill, component, missing, taxonomy)
        except EnrichmentError as exc:
            skill_warnings.append(f"{skill.path}: {exc}")
            return None
        for k in missing:
            metadata[k] = enrichment[k]
    elif is_materially_changed and ai_client is not None:
        # Skill name/description changed but all required fields already have
        # valid values. Ask the AI whether any of those values should be
        # amended to better fit the new content. If AI returns the same value
        # for a field, output stays byte-stable.
        # An amendment is optional: every required field already holds a valid
        # value, so a failed enrichment must leave the existing entry standing.
        # Returning None here would drop the skill from metadata.json and
        # skills.sh.json outright — delisting a published skill from the
        # marketplace on a transient AI failure, with the regenerated file
        # still internally consistent and so still passing --check.
        amended = None
        amendment_fields = [f for f in MVP_FIELDS if f not in authoritative]
        try:
            amended = ai_client(
                skill,
                component,
                amendment_fields,
                taxonomy,
                current_values=metadata,
            )
        except EnrichmentError as exc:
            # The entry is complete and is being emitted unchanged, so this is
            # not an omission. Recording it in skill_warnings would report the
            # skill as excluded from output and fail an otherwise correct run.
            if retained_notices is not None:
                retained_notices.append(
                    f"{skill.path}: {exc}; existing metadata values kept."
                )
        if amended:
            for k in amendment_fields:
                if amended.get(k):
                    metadata[k] = amended[k]

    # Re-order to schema field order.
    ordered_metadata = {k: metadata[k] for k in MVP_FIELDS if k in metadata}

    return {
        "path": skill.path,
        "name": skill.name,
        "description": skill.description,
        "metadata": ordered_metadata,
    }


# ---------------------------------------------------------------------------
# skills.sh.json


def build_skills_sh(metadata: dict, subdomains: dict) -> dict:
    sub_map = subdomains["subdomains"]
    groups: dict[str, list[str]] = {slug: [] for slug in sub_map}

    for entry in metadata["skills"]:
        slug = entry["metadata"]["catalog.subdomain"]
        if slug not in groups:
            raise GeneratorError(
                f"{entry['path']}: catalog.subdomain `{slug}` not found in "
                f"skills-subdomains.json. Add it there or fix the metadata."
            )
        groups[slug].append(entry["name"])

    # Stable sort within each group.
    for slug, skills in groups.items():
        groups[slug] = sorted(skills)

    # Emit groups in the order defined in skills-subdomains.json.
    out_groups = []
    for slug in sub_map.keys():
        if not groups[slug]:
            continue
        out_groups.append(
            {
                "title": sub_map[slug]["title"],
                "description": sub_map[slug]["description"],
                "skills": groups[slug],
            }
        )

    return {
        "$schema": "https://skills.sh/schemas/skills.sh.schema.json",
        "notGrouped": "bottom",
        "groupings": out_groups,
    }


# ---------------------------------------------------------------------------
# Plugins
#
# A plugin is a product-owned bundle declared by plugins.d/<name>.yml and built
# into plugins/<name>/ by .github/scripts/build-plugins.py. Its taxonomy is
# derived here rather than authored, because every skill it bundles already
# carries enriched metadata: `include_skills` lists canonical `skills/...`
# paths, which are exactly the keys of the entries built earlier in this run.
# The join is therefore deterministic and needs no inference.
#
# Where a plugin bundles no catalog skill -- an MCP-only plugin, or one that
# lands before its skills do -- there is nothing to roll up. That case is
# reported as an error naming the plugin, and the remedy is a `taxonomy:` block
# in plugins.d/<name>.yml. Inference is intentionally NOT wired in: it would
# put a nondeterministic path into a generator whose output is gated on being
# byte-stable. The hook for it is the empty-`resolved` branch below.


def _audience_values(entry: dict) -> list[str]:
    raw = entry["metadata"].get("audience", "")
    return [v for v in raw.split(",") if v]


PLUGIN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
# A plugin name becomes a directory name and, for the plugins.d precedence
# check, a filename with ".yml" appended. Without a bound, a long but
# otherwise legal name makes that check raise OSError (ENAMETOOLONG) straight
# out of the generator, which would stop regeneration of the whole catalog --
# the one outcome the plugin passes exist to prevent. 200 is well under the
# 255-byte limit every filesystem here enforces, and far above any real name.
PLUGIN_NAME_MAX = 200




def load_plugin_specs(warnings: list[str], declared: set[str] | None = None) -> list[dict]:
    """Read every plugins.d/<name>.yml that defines a plugin.

    Discovery matches build-plugins.py on everything that decides which files
    are plugins and what they contain: the same ``*.yml`` glob, the same
    leading-underscore rule for include files, the same ``_defaults.yml``
    shallow merge, the same lowercase-kebab-case name rule, and the same
    policy of skipping a bad file rather than aborting.

    One deliberate divergence: directory-form ``include_skills`` entries are
    refused by build_plugin_entries rather than expanded. See the note in
    plugins.d/README.md for why, and for what to implement if the catalog is
    ever regrouped by product.

    Any divergence shows up as a catalog that describes a different set of
    plugins from the one actually published -- an entry for a plugin that is
    never built, or a plugin on the marketplaces with no entry here.

    `declared`, when given, is filled with the identity of every plugin this
    directory declares, including ones skipped below. Recovery uses it to tell
    "failed to build" from "deliberately deleted". It is populated HERE, by
    the same parse that produces the specs, because resolving plugin identity
    in a second place is how discovery and recovery came to disagree twice:
    once over defaults, once over which names are legal. A name this function
    refuses is deliberately NOT declared -- build-plugins.py refuses it too,
    so the plugin is not published and the catalog should not keep describing
    it. A file too broken to parse falls back to its filename, matching that
    script leaving the previously built tree in place.
    """
    if declared is None:
        declared = set()
    if not PLUGINS_D_DIR.is_dir():
        return []

    # build-plugins.py merges _defaults.yml into every spec before reading any
    # field from it, so a `description` or `include_skills` declared there is
    # inherited by every plugin. Reading only the per-plugin file would make
    # this script see a different plugin than the one actually published.
    # The merge is shallow and the per-plugin file wins, matching that script.
    defaults: dict = {}
    defaults_path = PLUGINS_D_DIR / "_defaults.yml"
    if defaults_path.is_file():
        try:
            loaded = load_yaml(defaults_path)
        except Exception as exc:  # noqa: BLE001 - PyYAML raises builtins too
            warnings.append(f"plugins.d/_defaults.yml: could not be parsed: {exc}")
            loaded = None
        if isinstance(loaded, dict):
            defaults = loaded
        elif loaded is not None:
            warnings.append(
                "plugins.d/_defaults.yml: expected a YAML mapping; defaults ignored."
            )

    specs: list[dict] = []
    seen: dict[str, str] = {}
    for path in sorted(PLUGINS_D_DIR.glob("*.yml")):
        if path.name.startswith("_"):
            continue
        # Built from PLUGINS_D_DIR rather than relative_to(REPO_ROOT): the
        # result is identical in production and does not depend on where the
        # directory sits, so `source` stays a stable key that recovery can
        # match on and the function can be exercised in isolation.
        rel = f"{PLUGINS_D_DIR.name}/{path.name}"

        # User-authored YAML must never raise out of this generator: its
        # failures are reported into a GitHub issue body, where a traceback
        # tells the maintainer nothing.
        # Exception, not yaml.YAMLError. PyYAML's own constructors raise
        # ordinary builtins: `release_date: 2026-02-30` matches the timestamp
        # resolver and then raises ValueError out of datetime.date, and the
        # field need not be one this generator reads. A YAMLError-only handler
        # lets those through and stops the whole catalog.
        try:
            data = load_yaml(path)
        except Exception as exc:  # noqa: BLE001 - see above
            # The name is unknowable here, and it need not equal the filename.
            # Declaring the stem would miss a published entry built under a
            # different `name` and silently delist it, so the SOURCE PATH is
            # declared instead and recovery matches entries on it. Still
            # declared at all because build-plugins.py also fails to read this
            # file and leaves the previously built tree in place.
            declared.add(rel)
            warnings.append(f"{rel}: could not be parsed, plugin skipped: {exc}")
            continue

        if not isinstance(data, dict):
            declared.add(rel)
            warnings.append(
                f"{rel}: expected a YAML mapping at top level, got "
                f"{type(data).__name__}; plugin skipped."
            )
            continue

        data = {**defaults, **data}
        name = data.get("name")
        if not name or not isinstance(name, str):
            warnings.append(f"{rel}: missing or non-string `name`; plugin skipped.")
            continue
        if not PLUGIN_NAME_RE.fullmatch(name) or len(name) > PLUGIN_NAME_MAX:
            warnings.append(
                f"{rel}: plugin name `{name[:PLUGIN_NAME_MAX]}` is not "
                f"lowercase kebab-case of at most {PLUGIN_NAME_MAX} "
                f"characters; plugin skipped. build-plugins.py refuses to "
                f"build it, so an entry here would describe a plugin that is "
                f"never published."
            )
            continue
        if name in seen:
            warnings.append(
                f"{rel}: duplicate plugin name `{name}`, already declared in "
                f"{seen[name]}; plugin skipped."
            )
            continue
        if name != path.stem:
            # build-plugins.py builds plugins/<name>/ from the `name` field,
            # so this still works; it just makes the tree hard to navigate.
            warnings.append(
                f"{rel}: `name: {name}` does not match the file basename "
                f"`{path.stem}`. The plugin is built as `{name}`."
            )

        seen[name] = rel
        declared.add(name)
        data["__source"] = rel
        specs.append(data)
    return specs


def derive_plugin_taxonomy(included: list[dict]) -> dict:
    """Roll a plugin's taxonomy up from the skills it bundles.

    Domain  -- the most common ``classification.category.primary``. A plugin
               shows under one domain, so a single value has to be chosen even
               when the bundle spans several. Ties break alphabetically, which
               keeps the output byte-stable rather than dependent on input
               order.
    Library -- every distinct ``product.primary``, sorted. Usually one value;
               more when a product's skills are registered under several names.
    Audience -- values held by at least half the bundled skills. A union over
               all bundled skills returns most of the vocabulary and separates
               nothing; an intersection is frequently empty. The halfway rule
               keeps the values that actually characterise the bundle. When it
               selects nothing, the most common value(s) are used, so the field
               is never empty for a plugin that bundles any skill at all.
    """
    # Every entry reaching here came from build_skill_entry, which emits
    # nothing unless all of MVP_FIELDS is present, so these are populated in
    # practice. Reading them defensively keeps a future change to MVP_FIELDS
    # from turning into a KeyError in the plugin pass.
    missing = [
        e.get("path", "<unknown>")
        for e in included
        if not e.get("metadata", {}).get("classification.category.primary")
        or not e.get("metadata", {}).get("product.primary")
    ]
    if missing:
        raise GeneratorError(
            "cannot roll up a plugin from skills with incomplete metadata: "
            + ", ".join(sorted(missing))
        )

    cats = collections.Counter(
        e["metadata"]["classification.category.primary"] for e in included
    )
    domain = sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    libraries = sorted({e["metadata"]["product.primary"] for e in included})

    # set() because the skill schema permits a repeated audience token, and
    # counting tokens would let one skill vote twice and carry a value over
    # the halfway line on its own.
    aud: collections.Counter = collections.Counter()
    for e in included:
        for value in set(_audience_values(e)):
            aud[value] += 1
    audience = sorted(v for v, n in aud.items() if n * 2 >= len(included))
    if not audience and aud:
        top = max(aud.values())
        audience = sorted(v for v, n in aud.items() if n == top)

    return {
        "classification.category.primary": domain,
        "product.primary": libraries,
        "audience": ",".join(audience),
    }


def plugin_taxonomy_help(taxonomy: dict, src: str) -> str:
    """A ready-to-paste `taxonomy:` block for the error that asks for one.

    The person reading this error is usually adding their first plugin. Naming
    the three fields is not enough: they are dotted, unusual, and easy to
    mistype, and the product and audience values live in enums far too long to
    scan, in a schema file they have no reason to have opened. So the message
    carries the shape, and inlines the one vocabulary short enough to print.
    """
    domains = taxonomy["classification.category.primary"]["values"]
    return (
        f"declare it by hand in {src}:\n"
        f"\n"
        f"    taxonomy:\n"
        f"      classification.category.primary: {domains[0]}\n"
        f"      product.primary:\n"
        f"        - <product name, exactly as metadata.schema.json spells it>\n"
        f"      audience: developer\n"
        f"\n"
        f"    classification.category.primary is one of: "
        + ", ".join(domains)
        + ".\n"
        f"    product.primary and audience values come from "
        f"{SCHEMA_PATH.relative_to(REPO_ROOT).as_posix()}.\n"
        f"    All three are required when the plugin bundles no skill. When it\n"
        f"    does bundle skills, any field may be declared on its own and the\n"
        f"    rest still roll up.\n"
        f"    See plugins.d/README.md."
    )


def _suggest(value: str, allowed: Iterable[str]) -> str:
    """`Did you mean ...` for a mistyped controlled value.

    Without this, a single-character typo sends the maintainer to search the
    whole enum by eye. `product.primary` alone runs to several hundred values
    and grows with every product onboarded.
    """
    close = difflib.get_close_matches(str(value), list(allowed), n=3, cutoff=0.6)
    return f" Did you mean: {', '.join(close)}?" if close else ""


def validate_plugin_taxonomy(
    metadata: dict, taxonomy: dict, location: str, errors: list[str]
) -> None:
    """Check rolled-up or hand-declared values against the schema vocabulary.

    The schema is the single source of truth for the controlled values, so an
    override in plugins.d is held to exactly the same vocabulary as an enriched
    skill. Without this a typo in a hand-written `taxonomy:` block would only
    surface later, as a schema failure against the whole document.
    """
    domain = metadata.get("classification.category.primary")
    allowed = taxonomy["classification.category.primary"]["values"]
    if domain not in allowed:
        errors.append(
            f"{location}: classification.category.primary `{domain}` is not a "
            f"value in metadata.schema.json. Valid values: "
            + ", ".join(allowed)
            + "."
        )

    allowed_products = taxonomy["product.primary"]["values"]
    allowed_product_set = set(allowed_products)
    for value in metadata.get("product.primary", []):
        if value not in allowed_product_set:
            errors.append(
                f"{location}: product.primary `{value}` is not a value in "
                f"metadata.schema.json." + _suggest(value, allowed_products)
            )

    allowed_audience = taxonomy["audience"]["values"]
    allowed_audience_set = set(allowed_audience)
    for value in str(metadata.get("audience", "")).split(","):
        if value and value not in allowed_audience_set:
            errors.append(
                f"{location}: audience `{value}` is not a value in "
                f"metadata.schema.json." + _suggest(value, allowed_audience)
            )


def apply_taxonomy_override(metadata: dict, override: dict, src: str) -> str | None:
    """Apply a hand-declared `taxonomy:` block in place.

    A declared value always wins over the rolled-up one, the same precedence
    a components.d `product_primary` has over enrichment. Returns a message
    describing the first type problem found, or None on success.
    """
    for field in PLUGIN_FIELDS:
        if field not in override:
            continue
        value = override[field]
        if field == "product.primary":
            value = value if isinstance(value, list) else [value]
            if not all(isinstance(v, str) for v in value):
                return (
                    f"{src}: `taxonomy.product.primary` must be a string or a "
                    f"list of strings; plugin skipped."
                )
            value = sorted(value)
        elif not isinstance(value, str):
            return (
                f"{src}: `taxonomy.{field}` must be a string, got "
                f"{type(value).__name__}; plugin skipped."
            )
        metadata[field] = value
    return None


def build_plugin_entries(
    skill_entries: list[dict],
    taxonomy: dict,
    warnings: list[str],
    validator: Draft202012Validator,
    declared: set[str] | None = None,
    built_names: set[str] | None = None,
) -> list[dict]:
    """Build one metadata entry per plugin declared in plugins.d/.

    `built_names`, when given, is filled with the name of every plugin this
    directory declares, whether or not it builds. The mirrored pass uses it
    for precedence, and it must include the failures: a plugins.d plugin that
    fails to build still owns its directory under plugins/, and letting the
    mirrored pass claim it would describe one plugin from two sources.

    Nothing here is fatal. A plugin problem degrades that plugin -- it is
    warned about and left out of the catalog -- and never stops the skills
    from being emitted.

    That policy is not a preference, it is required for the repository to
    function. build-plugins.py soft-fails curation drift on purpose, so that
    "the daily sync PR is not blocked by curation drift in plugins.d", and
    validate-plugins.yml is advisory for the same reason. Hard-failing here
    would invert it: a skill renamed by an upstream sync, or dropped through
    metadata-exclusions.yaml, would leave a stale path in some team's
    include_skills and from that moment block regeneration of the entire
    catalog and red-line every unrelated PR in the repository.
    """
    by_path = {e["path"]: e for e in skill_entries}
    entries: list[dict] = []

    for spec in load_plugin_specs(warnings, declared):
        name = spec["name"]
        src = spec["__source"]
        if built_names is not None:
            built_names.add(name)

        description = spec.get("description")
        if not description or not isinstance(description, str):
            warnings.append(
                f"{src}: missing or non-string `description`; plugin skipped."
            )
            continue
        try:
            # The schema accepts any string, but the output is written as
            # UTF-8, and a lone surrogate from a YAML escape such as "\uD800"
            # only fails at that point -- too late to degrade just this plugin.
            description.encode("utf-8")
        except UnicodeEncodeError:
            warnings.append(
                f"{src}: `description` contains characters that cannot be "
                f"encoded as UTF-8; plugin skipped."
            )
            continue

        raw_includes = spec.get("include_skills") or []
        if not isinstance(raw_includes, list):
            # A bare string would otherwise be iterated one character at a
            # time, producing nonsense paths.
            warnings.append(
                f"{src}: `include_skills` must be a list, got "
                f"{type(raw_includes).__name__}; plugin skipped."
            )
            continue

        # build-plugins.py calls .rstrip() on each entry, so a non-string
        # raises AttributeError there and the build fails. Emitting an entry
        # for it would describe a plugin that is never published.
        bad_types = [r for r in raw_includes if not isinstance(r, str)]
        if bad_types:
            warnings.append(
                f"{src}: include_skills entries must be strings; found "
                + ", ".join(f"{type(r).__name__}" for r in bad_types)
                + ". build-plugins.py fails on these, so the plugin would not "
                "be published. Plugin skipped."
            )
            continue

        resolved: list[dict] = []
        unresolved: list[str] = []
        grouping: list[str] = []
        escaping: list[str] = []
        for raw in raw_includes:
            # Normalisation follows build-plugins.py exactly: it does
            # `(REPO_ROOT / entry.rstrip("/")).resolve()`, which collapses
            # "./" and ".." but does NOT strip surrounding whitespace. Doing
            # either differently makes the two scripts resolve different sets,
            # so this script would describe a plugin the other does not ship.
            path = posixpath.normpath(raw.rstrip("/"))
            # metadata.json keys are repo-relative. An absolute path, or one
            # climbing out of the repo, matches no key and so slipped past the
            # directory check below while build-plugins.py resolved it against
            # the filesystem and expanded it.
            if path.startswith("/") or path == ".." or path.startswith("../"):
                escaping.append(path)
                continue
            entry = by_path.get(path)
            if entry is not None:
                resolved.append(entry)
                continue
            # build-plugins.py also accepts a grouping directory, expanding it
            # to the immediate children that hold a SKILL.md. That is NOT
            # reimplemented here. The catalog is flat today -- 472 skills, all
            # at skills/<name>, no grouping directories, and no plugins.d file
            # using the form -- so the feature has no users, and approximating
            # its semantics is how this code got them wrong twice. It is
            # refused loudly rather than guessed at, so the divergence is
            # visible instead of silent. See plugins.d/README.md.
            if any(p.startswith(path + "/") for p in by_path):
                grouping.append(path)
            else:
                unresolved.append(path)

        if escaping:
            warnings.append(
                f"{src}: include_skills paths must be repo-relative, under "
                f"skills/: "
                + ", ".join(sorted(escaping))
                + ". build-plugins.py resolves these against the filesystem, "
                "so the published plugin and this catalog would disagree. "
                "Plugin skipped."
            )
            continue

        if grouping:
            warnings.append(
                f"{src}: directory-form include_skills entries are not "
                f"supported by the metadata generator: "
                + ", ".join(sorted(grouping))
                + ". build-plugins.py WOULD expand these, so the published "
                "plugin and this catalog would disagree. List each skill "
                "path explicitly. Plugin skipped."
            )
            continue

        # Held, not emitted. The plugin can still be skipped further down, and
        # a warning claiming it was "BUILT WITHOUT" some skills would then be
        # describing a plugin that is not in the output at all.
        unresolved_notice = ""
        if unresolved:
            # Not an error, and deliberately not advice to delete the path.
            # The usual causes are a skill that is legitimately absent from
            # this run -- newly added and not yet enriched under --no-ai, or
            # removed through metadata-exclusions.yaml -- where the path is
            # correct and deleting it would be wrong.
            unresolved_notice = (
                f"{src}: BUILT WITHOUT {len(unresolved)} SKILL(S). These "
                f"include_skills path(s) have no entry in metadata.json and "
                f"were left out of the plugin's taxonomy: "
                + ", ".join(sorted(unresolved))
                + ". This is expected for a skill that is new, excluded, or "
                "renamed upstream; check the path only if it should be there."
            )

        entry = assemble_plugin_entry(
            name=name,
            description=description,
            src=src,
            resolved=resolved,
            override=spec.get("taxonomy") or {},
            unresolved_notice=unresolved_notice,
            taxonomy=taxonomy,
            warnings=warnings,
            validator=validator,
            no_skills_remedy=(
                "Either list catalog skills under `include_skills:`, or "
                + plugin_taxonomy_help(taxonomy, src)
            ),
        )
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: e["name"])
    return entries


def assemble_plugin_entry(
    name: str,
    description: str,
    src: str,
    resolved: list[dict],
    override: Any,
    unresolved_notice: str,
    taxonomy: dict,
    warnings: list[str],
    validator: Draft202012Validator,
    no_skills_remedy: str,
) -> dict | None:
    """Turn a plugin's resolved skills into a validated entry, or warn and skip.

    Shared by both discovery paths. A plugin declared in plugins.d/ and a
    plugin mirrored from a product repo differ only in how their bundled
    skills are found; everything after that -- the roll-up, the `taxonomy:`
    override, the vocabulary check, the schema check, and the rule that a
    problem degrades one plugin and never the catalog -- must be identical,
    or the two kinds of plugin would be held to different standards in the
    same output file.

    Returns None when the plugin is skipped. Warnings are appended in place.
    """
    if not isinstance(override, dict):
        warnings.append(
            f"{src}: `taxonomy:` must be a mapping, got "
            f"{type(override).__name__}; plugin skipped."
        )
        return None

    # key=str because YAML keys need not be strings: a mapping such as
    # `{null: x, typo: y}` otherwise reaches sorted() with None beside a
    # string and raises TypeError out of the generator.
    unknown = sorted((set(override) - set(PLUGIN_FIELDS)), key=str)
    if unknown:
        # Silently ignoring these is how a dotted-name typo turns into a
        # plugin that quietly keeps its rolled-up values.
        warnings.append(
            f"{src}: unknown `taxonomy:` field(s) "
            + ", ".join(f"`{k}`" for k in unknown)
            + ". Valid fields are "
            + ", ".join(PLUGIN_FIELDS)
            + "; plugin skipped."
        )
        return None

    # De-duplicate before deriving, not at assembly. A skill counted twice
    # would otherwise vote twice in the domain plurality and the audience
    # threshold while the emitted `skills` list stayed byte-identical, so the
    # wrong taxonomy would be invisible in the output and in review. Both
    # discovery paths can produce a repeat: plugins.d through a path listed
    # twice in include_skills, a mirrored plugin through two bundled
    # directories resolving to the same catalog skill.
    resolved = list({e["path"]: e for e in resolved}.values())

    if resolved:
        try:
            metadata = derive_plugin_taxonomy(resolved)
        except GeneratorError as exc:
            warnings.append(f"{src}: {exc}; plugin skipped.")
            return None
    else:
        # Tier 2 hook. See the section comment above. With nothing to roll
        # up there is no partial case: every field has to be declared, or
        # the entry would be emitted incomplete and fail the schema with
        # an error blaming a value the maintainer never wrote.
        undeclared = [f for f in PLUGIN_FIELDS if f not in override]
        if undeclared:
            warnings.append(
                f"{src}: plugin `{name}` bundles no catalog skill, so its "
                f"taxonomy cannot be derived and every field has to be "
                f"declared. Missing: "
                + ", ".join(undeclared)
                + ". "
                + no_skills_remedy
            )
            return None
        metadata = {}

    problem = apply_taxonomy_override(metadata, override, src)
    if problem:
        warnings.append(problem)
        return None

    vocab_errors: list[str] = []
    validate_plugin_taxonomy(metadata, taxonomy, src, vocab_errors)
    if vocab_errors:
        warnings.extend(f"{e} Plugin skipped." for e in vocab_errors)
        return None

    entry = {
        "name": name,
        "path": f"plugins/{name}",
        # Carried so recovery can identify a published plugin whose source
        # declaration has since become unreadable, when its `name` cannot be
        # read. For a mirrored plugin this is its components.d registry file.
        "source": src,
        "description": description,
        "skills": sorted({e["path"] for e in resolved}),
        "metadata": {f: metadata[f] for f in PLUGIN_FIELDS},
    }

    # The assembled entry is checked against the published schema here,
    # not left to the whole-document validation later in main().
    #
    # This is the difference between a bad plugin degrading itself and a
    # bad plugin stopping the catalog. validate_plugin_taxonomy checks
    # that values are in the vocabulary, but the schema enforces more than
    # that -- minItems on product.primary, uniqueItems on skills, and an
    # audience pattern that rejects "" and a trailing comma. Anything the
    # vocabulary check does not cover used to pass this gate and fail the
    # whole document instead, which writes no output at all.
    schema_errors: list[str] = []
    validate_against_schema(
        {"skills": [], "plugins": [entry]}, validator, src, schema_errors
    )
    if schema_errors:
        warnings.extend(f"{e} Plugin skipped." for e in schema_errors)
        return None

    # Only now is the plugin certain to be in the output.
    if unresolved_notice:
        warnings.append(unresolved_notice)
    return entry


def load_mirrored_plugin_specs(
    warnings: list[str],
    built_names: set[str],
    declared: set[str] | None = None,
) -> list[dict]:
    """Read every `plugins:` entry declared in components.d/<product>.yml.

    These are the plugins a product team owns in its own repo. sync-plugins.yml
    mirrors the declared directory byte-for-byte into plugins/<catalog_dir>/.
    There is no plugins.d file for them, which is why they were invisible to
    this generator until now.

    Precedence follows sync-plugins.yml exactly: a plugins/<name>/ that has a
    plugins.d/<name>.yml is the catalog's own built plugin. The sync never
    mirrors, prunes or lists it, so this function must not claim it either, or
    one plugin would be described twice from two different sources.

    Each returned spec carries `__source_skills`: the product's declared
    skills as a source-path -> catalog-path map. That is one of the two keys
    used to identify a bundled skill, and it is read here because it lives in
    the same registry file as the plugin declaration.

    `built_names` is the set of names plugins.d declares, used for precedence.

    `declared`, when given, is filled so recovery can tell "failed to build"
    from "no longer registered". What goes in depends on whether the
    declaration could be READ, and the distinction matters because one
    registry file declares many plugins, where one plugins.d file declares
    exactly one:

    - A readable, validly named entry declares its NAME. Removing that entry
      from the registry then correctly delists the plugin, which is how a team
      retires one.
    - A declaration that cannot be read -- an unparseable file, a `plugins:`
      value that is not a list, an entry that is not a mapping, an entry with
      no `catalog_dir` -- declares the FILE instead. We cannot tell which
      published plugin it was meant to be, so every plugin published from that
      registry is kept rather than delisted on the strength of a syntax error.

    Declaring the file unconditionally, as an earlier version did, broke the
    first case: a registry that still declared any plugin kept every plugin it
    had ever published, so a retired plugin never left the catalog.

    A plugin registered but not yet mirrored is still declared by name: the
    sync may simply not have run yet, and delisting it on that basis would
    drop a live plugin from the catalog.
    """
    if declared is None:
        declared = set()
    if not COMPONENTS_DIR.is_dir():
        return []

    specs: list[dict] = []
    seen: dict[str, str] = {}
    for ymlf in sorted(COMPONENTS_DIR.glob("*.yml")):
        src = f"{COMPONENTS_DIR.name}/{ymlf.name}"
        try:
            data = load_yaml(ymlf)
        except Exception as exc:  # noqa: BLE001 - PyYAML raises builtins too
            # The skills pass reads this file too and reports its own errors.
            # Say nothing more than that no plugin could be read from it.
            warnings.append(
                f"{src}: could not be parsed, so any `plugins:` entry in it "
                f"was not read: {exc}"
            )
            # Declare the file, not any name: nothing in it can be read, so a
            # plugin already published from it must be kept rather than
            # delisted on the strength of a parse failure. See the note on
            # unreadable declarations below.
            declared.add(src)
            continue
        if not isinstance(data, dict):
            declared.add(src)
            continue

        raw_plugins = data.get("plugins")
        if raw_plugins is None:
            continue
        if not isinstance(raw_plugins, list):
            warnings.append(
                f"{src}: `plugins:` must be a list, got "
                f"{type(raw_plugins).__name__}; no plugin read from this file."
            )
            declared.add(src)
            continue

        # Source path -> catalog path for this product's standalone skills.
        # `catalog_dir` is the rename the catalog applies on the way in, so
        # the two sides genuinely differ: BioIR ships skills/bench-perf-oss/
        # and the catalog publishes it as skills/bioir-bench-perf-oss.
        source_skills: dict[str, str] = {}
        for entry in data.get("skills") or []:
            if not isinstance(entry, dict):
                continue
            spath = entry.get("path")
            cdir = entry.get("catalog_dir")
            if not isinstance(spath, str) or not spath.strip():
                continue
            if not isinstance(cdir, str) or not cdir.strip():
                continue
            source_skills[spath.strip().rstrip("/")] = (
                f"skills/{cdir.strip().rstrip('/')}"
            )

        for raw in raw_plugins:
            if not isinstance(raw, dict):
                warnings.append(
                    f"{src}: each `plugins:` entry must be a mapping, got "
                    f"{type(raw).__name__}; entry skipped."
                )
                # An unreadable entry names a plugin we cannot identify, so we
                # cannot tell whether a published plugin from this registry is
                # the one it meant. Declare the file so recovery keeps them.
                declared.add(src)
                continue
            name = raw.get("catalog_dir")
            if not isinstance(name, str) or not name.strip():
                warnings.append(
                    f"{src}: a `plugins:` entry has no `catalog_dir`; it names "
                    f"the directory under plugins/ and the install name, so "
                    f"the entry cannot be placed. Entry skipped."
                )
                declared.add(src)
                continue
            name = name.strip().rstrip("/")

            # sync-plugins.yml refuses a catalog_dir that is not lowercase
            # kebab-case, so one is never mirrored and must not be described.
            # This also refuses a path separator, which would otherwise let an
            # entry name ../ or another plugin's directory. The length bound
            # keeps the precedence check below from raising ENAMETOOLONG.
            #
            # Deliberately NOT declared: the name is readable and the sync
            # refuses it, so no such plugin is published and keeping an entry
            # for one would describe something that does not exist. This
            # matches how plugins.d treats a name it refuses.
            if not PLUGIN_NAME_RE.fullmatch(name) or len(name) > PLUGIN_NAME_MAX:
                warnings.append(
                    f"{src}: `catalog_dir` `{name[:PLUGIN_NAME_MAX]}` is not "
                    f"lowercase kebab-case of at most {PLUGIN_NAME_MAX} "
                    f"characters. The plugin sync refuses it, so it is never "
                    f"mirrored. Entry skipped."
                )
                continue

            if name in seen:
                # Same file or another one. Either way two declarations claim
                # one directory under plugins/, and emitting both would put
                # two entries with the same name and path in metadata.json.
                warnings.append(
                    f"{src}: plugin `{name}` is already declared by "
                    f"{seen[name]}. One plugin directory cannot have two "
                    f"declarations. Entry skipped."
                )
                continue
            seen[name] = src

            # Precedence: the catalog's own built plugins stay with plugins.d.
            # Keyed on the names plugins.d actually DECLARES, not on whether a
            # file called <name>.yml exists. build-plugins.py builds from the
            # `name` field inside the file, so plugins.d/bundle.yml declaring
            # `name: nvidia-skills` owns that directory while no
            # plugins.d/nvidia-skills.yml exists. A filename check misses it
            # and both passes would emit the same plugin.
            if name in built_names:
                continue

            declared.add(name)
            specs.append(
                {
                    "name": name,
                    "__source": src,
                    "__source_skills": source_skills,
                }
            )

    return specs


def resolve_bundled_skills(
    plugin_dir: Path,
    by_name: dict[str, dict],
    by_path: dict[str, dict],
    source_skills: dict[str, str],
) -> tuple[list[dict], list[str], list[str]]:
    """Identify which catalog skills a mirrored plugin bundles.

    A mirrored plugin carries copies of the skills, not references to them, so
    there is no declared list to read: `plugin.json`'s `skills` field is the
    container path "./skills/", not an inventory. The copies have to be
    identified from what they contain.

    Two keys are used, in order, and neither is the directory name:

    1. The `name` in the bundled SKILL.md frontmatter. Catalog `name` values
       are unique and are copied verbatim from that same frontmatter, so this
       is an identity match.
    2. The bundled directory's path within the plugin, resolved through the
       product's own `skills:` declarations to the catalog path. The product
       declares a full source path, which may nest (TAO declares
       `skills/applications/<skill>/`), so the bundled directory is matched on
       the last path segment of each declaration. A segment claimed by more
       than one declaration is ambiguous and is not used as a key.

    Matching on the directory name instead would be wrong for 43 of the 488
    skills in the catalog today, because `catalog_dir` renames a skill on the
    way in. BioIR publishes skills/bench-perf-oss/ as skills/bioir-bench-perf-oss,
    so the bundled directory and the catalog directory do not share a name.
    Both keys above survive that rename; the directory name does not.

    Returns (resolved entries, unidentified directory names, conflicts) where
    a conflict is a directory whose two keys resolve to different skills.
    """
    skills_dir = plugin_dir / "skills"
    if not skills_dir.is_dir():
        return [], [], []

    # Last path segment of each declared source path -> catalog path. A
    # product may nest its sources (TAO declares skills/applications/<skill>/),
    # in which case the bundled directory name matches no declared path
    # verbatim and the exact lookup below never fires. A segment claimed by
    # two declarations identifies nothing, so it is dropped rather than
    # resolved to whichever came first.
    by_segment: dict[str, str] = {}
    ambiguous: set[str] = set()
    for spath, cpath in source_skills.items():
        seg = spath.rsplit("/", 1)[-1]
        if not seg:
            continue
        if seg in by_segment and by_segment[seg] != cpath:
            ambiguous.add(seg)
        by_segment[seg] = cpath
    for seg in ambiguous:
        by_segment.pop(seg, None)

    resolved: list[dict] = []
    unresolved: list[str] = []
    conflicts: list[str] = []
    for child in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            # Not a skill directory. The sync mirrors the product's tree
            # byte-for-byte, so anything may sit here.
            continue

        try:
            front = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - unreadable or undecodable file
            front = {}

        by_header = None
        declared_name = front.get("name")
        if isinstance(declared_name, str) and declared_name.strip():
            by_header = by_name.get(declared_name.strip())

        by_registry = None
        catalog_path = source_skills.get(f"skills/{child.name}")
        if catalog_path is None:
            catalog_path = by_segment.get(child.name)
        if catalog_path:
            by_registry = by_path.get(catalog_path)

        # Both keys resolving to DIFFERENT skills means the bundled copy and
        # the registry disagree about what this directory holds. Taking either
        # one silently would give the plugin a taxonomy rolled up from a skill
        # it may not bundle. The usual cause is a stale SKILL.md header in the
        # mirrored copy after the standalone skill was renamed, so the header
        # is the side more likely to be wrong -- but it is also the stronger
        # identity when it is right, so it still wins, loudly.
        if (
            by_header is not None
            and by_registry is not None
            and by_header["path"] != by_registry["path"]
        ):
            conflicts.append(
                f"{child.name} (SKILL.md name -> {by_header['path']}, "
                f"registry path -> {by_registry['path']})"
            )

        entry = by_header if by_header is not None else by_registry
        if entry is None:
            unresolved.append(child.name)
        else:
            resolved.append(entry)

    return resolved, unresolved, conflicts


def build_mirrored_plugin_entries(
    skill_entries: list[dict],
    taxonomy: dict,
    warnings: list[str],
    validator: Draft202012Validator,
    built_names: set[str] | None = None,
    declared: set[str] | None = None,
) -> list[dict]:
    """Build one metadata entry per plugin mirrored from a product repo.

    Same contract as build_plugin_entries: nothing here is fatal, a plugin
    problem degrades that plugin and never the skills catalog.
    """
    by_path = {e["path"]: e for e in skill_entries}
    by_name = {e["name"]: e for e in skill_entries}
    entries: list[dict] = []

    for spec in load_mirrored_plugin_specs(warnings, built_names or set(), declared):
        name = spec["name"]
        src = spec["__source"]
        plugin_dir = PLUGINS_DIR / name

        if not plugin_dir.is_dir():
            # Registered but not mirrored yet, or reverted by the plugin sync
            # over signature integrity. A signature hold can persist, so the
            # warning does not promise it resolves on its own. The plugin
            # stays declared, so a previously published entry is recovered
            # rather than delisted.
            warnings.append(
                f"{src}: plugin `{name}` is registered but plugins/{name}/ is "
                f"not in the repository yet, so it has no entry this run. "
                f"Either the plugin sync has not mirrored it yet, or the sync "
                f"is holding it back over signature integrity. Check the "
                f"plugin sync's tracker issue if this persists."
            )
            continue

        description = None
        manifest = plugin_dir / "plugin.json"
        if manifest.is_file():
            try:
                loaded = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    description = loaded.get("description")
            except Exception:  # noqa: BLE001 - malformed or undecodable JSON
                description = None
        if not isinstance(description, str) or not description.strip():
            warnings.append(
                f"{src}: plugin `{name}` has no usable `description` in "
                f"plugins/{name}/plugin.json, which is the manifest the "
                f"marketplaces are built from. Plugin skipped."
            )
            continue
        try:
            # Matches build_plugin_entries: the schema accepts any string but
            # the output is written as UTF-8, and a lone surrogate only fails
            # at write time, too late to degrade just this plugin.
            description.encode("utf-8")
        except UnicodeEncodeError:
            warnings.append(
                f"{src}: plugin `{name}` has a `description` that cannot be "
                f"encoded as UTF-8; plugin skipped."
            )
            continue

        resolved, unidentified, conflicts = resolve_bundled_skills(
            plugin_dir, by_name, by_path, spec["__source_skills"]
        )

        if conflicts:
            warnings.append(
                f"{src}: plugin `{name}` has {len(conflicts)} bundled "
                f"directory(ies) whose SKILL.md name and registry path "
                f"identify DIFFERENT catalog skills: "
                + "; ".join(sorted(conflicts))
                + ". The SKILL.md name was used. Check whether the bundled "
                "copy is stale, because the plugin's taxonomy is rolled up "
                "from whichever skill wins."
            )

        unresolved_notice = ""
        if unidentified:
            unresolved_notice = (
                f"{src}: plugin `{name}` BUILT WITHOUT "
                f"{len(unidentified)} BUNDLED SKILL(S). These directories "
                f"under plugins/{name}/skills/ match no catalog skill by "
                f"SKILL.md name or by this registry's `skills:` paths, and "
                f"were left out of the plugin's taxonomy: "
                + ", ".join(sorted(unidentified))
                + ". This is expected for a skill that is new, excluded, or "
                "not published standalone; check it only if it should be there."
            )

        entry = assemble_plugin_entry(
            name=name,
            description=description.strip(),
            src=src,
            resolved=resolved,
            override={},
            unresolved_notice=unresolved_notice,
            taxonomy=taxonomy,
            warnings=warnings,
            validator=validator,
            # Single line, and no offer to declare taxonomy by hand. A
            # mirrored plugin has no `taxonomy:` key to declare it in -- this
            # path always passes override={} -- so the plugins.d remedy would
            # send a maintainer looking for a field that does not exist.
            no_skills_remedy=(
                f"No directory under plugins/{name}/skills/ matched a "
                f"published catalog skill by SKILL.md name or by this "
                f"registry's `skills:` paths. Check that the plugin bundles "
                f"skills that are published standalone in the catalog."
            ),
        )
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: e["name"])
    return entries


def recover_plugin_entries(
    entries: list[dict],
    baseline: dict | None,
    validator: Draft202012Validator,
    warnings: list[str],
    declared: set[str],
) -> list[dict]:
    """Keep a published plugin rather than delisting it when a rebuild fails.

    This mirrors the rule the skills pass already applies: an entry already in
    metadata.json must never regress, because a document missing it stays
    schema-valid and internally consistent, so nothing downstream notices the
    disappearance.

    Plugins need it more than skills do, not less. A plugin's taxonomy is
    derived from skills it does not own, so a skill being excluded for
    compliance, renamed upstream, or simply added unenriched under --no-ai can
    take a perfectly compliant plugin down with it. `nvidia-skills` bundles a
    single skill, so exactly one absent skill would otherwise remove the
    flagship plugin from build.nvidia.com.

    The deliberate case still works: a plugin whose plugins.d file is deleted
    is no longer declared, so it is not recovered and correctly disappears.

    `declared` is filled by the two discovery functions -- load_plugin_specs
    for plugins.d, load_mirrored_plugin_specs for components.d registries --
    and by nothing else. Recomputing identity here is what previously let
    recovery follow a name discovery had rejected, and miss one discovery had
    taken from _defaults.yml.

    It holds plugin NAMES, and also SOURCE paths for declarations that could
    not be read at all. A name means "this plugin is still declared". A source
    means "something in this file is unreadable, so keep whatever it
    published rather than delisting on a syntax error". The two registries
    differ in how often the second applies: a plugins.d file declares one
    plugin, so an unparseable one is ambiguous about exactly one entry, while
    a components.d file declares many.
    """
    built = {e["name"] for e in entries}

    # The baseline is a file on disk and may predate this code, be hand-edited,
    # or be truncated. It must never raise: a malformed `plugins` block there
    # would otherwise stop the independent skills path from being written.
    prior_plugins = (baseline or {}).get("plugins") or []
    if not isinstance(prior_plugins, list):
        prior_plugins = []
    prior_by_name = {
        e["name"]: e
        for e in prior_plugins
        if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"]
    }

    # `declared` holds plugin names, and -- for a file whose name could not be
    # read -- its source path. An entry matches on either, so a published
    # plugin whose file has become unparseable is still recognised as declared
    # even though its name is no longer readable.
    recovered: list[dict] = []
    for prior in sorted(prior_by_name.values(), key=lambda e: e["name"]):
        if prior["name"] in built:
            continue
        if prior["name"] not in declared and prior.get("source") not in declared:
            # Deliberately deleted, or its name is one discovery now rejects.
            continue
        try:
            # The validator covers the whole document, so a single entry is
            # checked by wrapping it the way existing_valid_metadata does.
            validator.validate({"skills": [], "plugins": [prior]})
        except Exception as exc:
            # Never drop it silently. This is the last path by which a
            # published plugin can still leave the catalog without a word, and
            # a baseline written by an older generator or edited by hand is
            # exactly what reaches it. Flatten the reason: these warnings are
            # rendered as single markdown bullets in the regeneration PR body.
            reason = " ".join(
                (getattr(exc, "message", None) or str(exc)).split()
            )
            warnings.append(
                f"plugin `{prior['name']}`: DROPPED. It could not be rebuilt "
                f"this run, and its entry in the previous metadata.json is not "
                f"valid, so it could not be carried forward either. The plugin "
                f"has left the catalog. Repair its plugins.d file to restore "
                f"it. Reason: {reason}"
            )
            continue
        try:
            # The schema accepts any string, but output is written as UTF-8.
            # A baseline entry predates this check, so it is re-checked here
            # rather than trusted; otherwise a surrogate already in the file
            # would be carried forward and fail at write time, stopping the
            # whole catalog.
            json.dumps(prior, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            warnings.append(
                f"plugin `{prior['name']}`: its previous entry cannot be "
                f"encoded as UTF-8 and was not carried forward."
            )
            continue
        recovered.append(prior)
        name = prior["name"]
        warnings.append(
            f"plugin `{name}`: KEPT. Could not be rebuilt this run, so the "
            f"entry from the previous metadata.json was retained and the "
            f"plugin stays in the catalog. Its taxonomy may now be stale."
        )

    if not recovered:
        return entries
    return sorted(entries + recovered, key=lambda e: e["name"])


# ---------------------------------------------------------------------------
# Validation helpers


def validate_against_schema(
    obj: Any, validator: Draft202012Validator, label: str, errors: list[str]
) -> None:
    for err in sorted(validator.iter_errors(obj), key=lambda e: e.path):
        loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{label}: {loc}: {err.message}")


def validate_skills_sh_uniqueness(
    obj: dict,
    errors: list[str],
    name_to_paths: dict[str, list[str]] | None = None,
) -> None:
    seen: dict[str, str] = {}
    for g in obj.get("groupings", []):
        for s in g.get("skills", []):
            if s in seen:
                group_a = seen[s]
                group_b = g.get("title")
                paths = (name_to_paths or {}).get(s, [])
                if group_a == group_b and len(paths) >= 2:
                    errors.append(
                        f"skills.sh.json: skill `{s}` appears at both "
                        f"`{paths[0]}` and `{paths[1]}` — both map to `{group_a}`."
                    )
                else:
                    errors.append(
                        f"skills.sh.json: skill `{s}` appears in both "
                        f"`{group_a}` and `{group_b}`."
                    )
            else:
                seen[s] = g.get("title")


def validate_inventory_round_trip(
    skills_now: list[Skill],
    metadata_obj: dict,
    skills_sh_obj: dict,
    errors: list[str],
    skipped_paths: set[str] | None = None,
) -> None:
    md_paths = {e["path"] for e in metadata_obj["skills"]}
    md_names = {e["name"] for e in metadata_obj["skills"]}
    found_paths = {s.path for s in skills_now}
    found_names = {s.name for s in skills_now}
    skipped = skipped_paths or set()

    for missing_path in sorted(found_paths - md_paths - skipped):
        errors.append(f"metadata.json: missing entry for {missing_path}.")
    for stale_path in sorted(md_paths - found_paths):
        errors.append(
            f"metadata.json: entry `{stale_path}` has no SKILL.md on disk."
        )

    sh_names: set[str] = set()
    for g in skills_sh_obj.get("groupings", []):
        sh_names.update(g.get("skills", []))
    for missing in sorted(md_names - sh_names):
        errors.append(f"skills.sh.json: missing skill `{missing}`.")
    for extra in sorted(sh_names - md_names):
        errors.append(
            f"skills.sh.json: skill `{extra}` is not present in metadata.json."
        )


# ---------------------------------------------------------------------------
# Output


def dumps_canonical(obj: Any) -> str:
    """Byte-stable JSON: 2-space indent, UTF-8 (non-ASCII preserved), trailing newline.

    Matches the canonical output style used by upstream NVIDIA/skills at
    .github/scripts/marketplace/metadata.json and skills.sh.json.
    """
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == content:
        return False
    # Encode before opening. `dumps_canonical` uses ensure_ascii=False, so a
    # lone surrogate reaching it survives into the string and only fails when
    # the file is written -- by which point write_text has already truncated
    # the file, destroying a good catalog to emit a bad one.
    encoded = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return True


def diff_text(label: str, expected: str, actual: str) -> str:
    if expected == actual:
        return ""
    return "".join(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
        )
    )


# ---------------------------------------------------------------------------
# Main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory; fail if checked-in files drift or are invalid.",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI enrichment; fail if any skill needs it.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print classification report and exit (no validation, no write).",
    )
    args = parser.parse_args(argv)

    schema = load_json(SCHEMA_PATH)
    skills_sh_schema = load_json(SKILLS_SH_SCHEMA_PATH)
    subdomains = load_json(SUBDOMAINS_PATH)
    taxonomy = taxonomy_from_schema(schema)

    metadata_validator = Draft202012Validator(schema)
    skills_sh_validator = Draft202012Validator(skills_sh_schema)

    exclusions = load_exclusions()
    try:
        components = load_components(taxonomy["product.primary"]["values"])
        skills_now, excluded_names = discover_skills(exclusions)
    except GeneratorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    baseline = load_json(METADATA_PATH) if METADATA_PATH.exists() else None
    cls = classify(skills_now, baseline, excluded_names)

    if args.report_only:
        _print_classification(cls)
        return 0

    ai_client = build_ai_client(allow_ai=not args.no_ai)

    errors: list[str] = []
    skill_warnings: list[str] = []
    retained_notices: list[str] = []
    entries: list[dict] = []
    materially_changed = set(cls.materially_changed)
    for skill in sorted(skills_now, key=lambda s: s.path):
        entry = build_skill_entry(
            skill,
            components,
            baseline,
            cls.renamed,
            metadata_validator,
            taxonomy,
            ai_client,
            skill_warnings,
            retained_notices,
            is_materially_changed=skill.path in materially_changed,
        )
        if entry is not None:
            entries.append(entry)

    # A skill already in metadata.json must never regress: a regenerated entry
    # is an improvement or byte-identical, and enrichment may add or amend but
    # never subtract. So when an entry fails to build for a skill that is still
    # on disk, fall back to its last-good entry rather than emitting a file
    # without it — the output would stay schema-valid and internally
    # consistent, so nothing downstream would notice the skill had been
    # delisted. Skills removed via metadata-exclusions.yaml are dropped in
    # discover_skills and never reach skills_now, so they cannot trip this.
    #
    # A genuinely new skill has no prior state to fall back on. That case is a
    # hard error whenever AI enrichment is available, since it should have been
    # enrichable; under --no-ai a new skill with no carried metadata is
    # expected to be skipped.
    baseline_entries = {
        e.get("path"): e for e in (baseline or {}).get("skills", []) if e.get("path")
    }
    emitted_paths = {e["path"] for e in entries}
    unbuilt = sorted(s.path for s in skills_now if s.path not in emitted_paths)

    recovered: list[str] = []
    unrecoverable: list[str] = []
    for path in unbuilt:
        prior = baseline_entries.get(path)
        # metadata_validator covers the whole document, so a single entry is
        # checked by wrapping it the same way existing_valid_metadata does.
        prior_is_valid = False
        if prior is not None:
            try:
                metadata_validator.validate({"skills": [prior]})
                prior_is_valid = True
            except Exception:
                prior_is_valid = False
        if prior_is_valid:
            entries.append(prior)
            recovered.append(path)
        elif prior is not None or ai_client is not None:
            unrecoverable.append(path)

    if recovered:
        entries.sort(key=lambda e: e["path"])
        for path in recovered:
            skill_warnings.append(
                f"{path}: no entry could be built; kept the entry from the "
                f"previous metadata.json."
            )

    if unrecoverable:
        errors.append(
            "no entry could be built for these skills, and no usable entry "
            "exists in the previous metadata.json to fall back on: "
            + ", ".join(unrecoverable)
            + ". Re-run, or add the skill name to metadata-exclusions.yaml to "
            "drop it deliberately."
        )

    # Plugins roll up from the entries built above, so this runs after the
    # recovery pass: a plugin that bundles a recovered skill still resolves.
    # Problems here are warnings, never errors -- see build_plugin_entries.
    plugin_warnings: list[str] = []
    declared_plugins: set[str] = set()
    # Two kinds of plugin, one output. plugins.d holds the ones the catalog
    # builds itself; components.d `plugins:` holds the ones product teams own
    # and the sync mirrors in. Both fill the same `declared` set, because
    # recovery has to recognise a published plugin of either kind -- without
    # that, a mirrored plugin whose skills fail to resolve for one run would
    # be delisted instead of kept.
    built_plugin_names: set[str] = set()
    plugin_entries = build_plugin_entries(
        entries,
        taxonomy,
        plugin_warnings,
        metadata_validator,
        declared_plugins,
        built_plugin_names,
    )
    plugin_entries += build_mirrored_plugin_entries(
        entries,
        taxonomy,
        plugin_warnings,
        metadata_validator,
        built_plugin_names,
        declared_plugins,
    )
    plugin_entries.sort(key=lambda e: e["name"])
    plugin_entries = recover_plugin_entries(
        plugin_entries, baseline, metadata_validator, plugin_warnings, declared_plugins
    )

    metadata_obj = {"skills": entries, "plugins": plugin_entries}

    # Validate metadata.json against the schema (all entries in one pass).
    if not errors:
        validate_against_schema(
            metadata_obj, metadata_validator, "metadata.json", errors
        )

    # Build and validate skills.sh.json.
    skills_sh_obj: dict = {}
    if not errors:
        try:
            skills_sh_obj = build_skills_sh(metadata_obj, subdomains)
        except GeneratorError as exc:
            errors.append(str(exc))

    if not errors:
        validate_against_schema(
            skills_sh_obj, skills_sh_validator, "skills.sh.json", errors
        )
        name_to_paths: dict[str, list[str]] = {}
        for e in metadata_obj["skills"]:
            name_to_paths.setdefault(e["name"], []).append(e["path"])
        validate_skills_sh_uniqueness(skills_sh_obj, errors, name_to_paths)
        skipped_paths = {s.path for s in skills_now if s.path not in {e["path"] for e in entries}}
        validate_inventory_round_trip(skills_now, metadata_obj, skills_sh_obj, errors, skipped_paths=skipped_paths)

    # Written before the error gate below. The per-skill reasons are needed
    # most on the runs that fail validation, but that path used to return
    # before reaching the reporting block, so the only thing reaching the log
    # and the tracking issue was the summary line naming which skills had no
    # entry -- never which field failed enrichment or why. The recovered case
    # (a published skill kept from the baseline) does not set `errors` and so
    # always reported correctly, which is why the gap stayed hidden.
    if skill_warnings or plugin_warnings:
        # Structured warnings file for CI to surface in issue bodies.
        warnings_path = Path("/tmp/skill-warnings.json")
        payload: dict[str, list[str]] = {}
        if skill_warnings:
            payload["stale_skills"] = skill_warnings
        if plugin_warnings:
            payload["plugins"] = plugin_warnings
        warnings_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        if skill_warnings:
            print("\nPer-skill reasons:", file=sys.stderr)
            for w in skill_warnings:
                print(f"  - {w}", file=sys.stderr)
        _print_classification(cls, stream=sys.stderr)
        return 1

    if retained_notices:
        # Not a failure: every one of these skills is present in the output
        # with its existing values intact.
        print(
            f"\n{len(retained_notices)} skill(s) kept existing metadata after an "
            f"enrichment failure:",
            file=sys.stderr,
        )
        for w in retained_notices:
            print(f"  - {w}", file=sys.stderr)

    if skill_warnings:
        print(
            f"\nPARTIAL SUCCESS: {len(skill_warnings)} skill(s) could not be "
            f"regenerated and were emitted from the previous metadata.json:",
            file=sys.stderr,
        )
        for w in skill_warnings:
            print(f"  - {w}", file=sys.stderr)
    if plugin_warnings:
        # Deliberately does not claim an outcome for the set as a whole. These
        # cover three different results -- a plugin skipped, a plugin kept from
        # the baseline, and a plugin built from a partial skill list -- and a
        # blanket "left out of metadata.json" header made a retained plugin
        # read as a lost one.
        print(
            f"\nPLUGIN WARNINGS: {len(plugin_warnings)} issue(s). Each line "
            f"states its own outcome. The skills catalog is unaffected in "
            f"every case:",
            file=sys.stderr,
        )
        for w in plugin_warnings:
            print(f"  - {w}", file=sys.stderr)

    # A skill that could not be built has to fail the run, in both --check and
    # write mode, so CI cannot read a partial success as a pass. Outputs are
    # still rendered first, so the catalog does not lose a skill either way.
    #
    # --no-ai is the exception. PR CI runs without an inference key on purpose,
    # so a genuinely new skill always lands there with its fields unenriched;
    # filling them is the regenerate job's work, not the PR's. Failing here
    # would red-line every open PR from the moment a sync adds a skill until
    # regeneration runs. The warnings are still printed either way.
    #
    # plugin_warnings deliberately does NOT feed this. A nonzero exit skips the
    # "Create regeneration PR" step in generate-skill-metadata.yml, so letting a
    # plugin problem set it would block regeneration of the whole catalog --
    # the exact failure this pass exists to prevent, reintroduced through the
    # exit code. Plugin problems surface in the log and in the warnings file.
    exit_code = 1 if skill_warnings and not args.no_ai else 0

    metadata_text = dumps_canonical(metadata_obj)
    skills_sh_text = dumps_canonical(skills_sh_obj)

    if args.check:
        diffs: list[str] = []
        for path, expected in (
            (METADATA_PATH, metadata_text),
            (SKILLS_SH_PATH, skills_sh_text),
        ):
            actual = path.read_text(encoding="utf-8") if path.exists() else ""
            d = diff_text(path.name, expected, actual)
            if d:
                diffs.append(d)
        if diffs:
            print(
                "DRIFT DETECTED. Run `python3 .github/scripts/"
                "generate-skill-metadata.py` and commit the regenerated files.",
                file=sys.stderr,
            )
            for d in diffs:
                print(d, file=sys.stderr)
            return 1
        _print_classification(cls)
        print("OK: generated outputs are byte-stable.")
        return exit_code

    md_changed = write_if_changed(METADATA_PATH, metadata_text)
    sh_changed = write_if_changed(SKILLS_SH_PATH, skills_sh_text)

    _print_classification(cls)
    print(
        f"metadata.json: {'updated' if md_changed else 'unchanged'} "
        f"({len(entries)} skills, {len(plugin_entries)} plugins)"
    )
    print(
        f"skills.sh.json: {'updated' if sh_changed else 'unchanged'} "
        f"({len(skills_sh_obj.get('groupings', []))} non-empty groups)"
    )

    return exit_code


def _print_classification(cls: Classification, stream=sys.stdout) -> None:
    def _line(label: str, items: Iterable[str]) -> None:
        items = list(items)
        print(f"  {label}: {len(items)}", file=stream)
        for it in items:
            print(f"    - {it}", file=stream)

    print("Skill change classification:", file=stream)
    _line("added", cls.added)
    _line("removed", cls.removed)
    _line(
        "renamed",
        (f"{old} -> {new}" for old, new in sorted(cls.renamed.items())),
    )
    _line("materially_changed", cls.materially_changed)
    _line("excluded", cls.excluded)
    print(f"  unchanged: {len(cls.unchanged)}", file=stream)


if __name__ == "__main__":
    sys.exit(main())
