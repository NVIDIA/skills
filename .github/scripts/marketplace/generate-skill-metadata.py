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
import dataclasses
import difflib
import json
import os
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


def load_components() -> dict[str, dict]:
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
) -> dict | None:
    """Return baseline metadata for ``skill`` if it is valid under the current schema.

    Honors renames (old_path -> new_path) by looking up the old path's entry.
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

            # Retry on rate-limit (429) and transient server errors (5xx).
            if resp.status_code == 429 or resp.status_code >= 500:
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


def derive_product_from_component(component: dict | None, product_enum: list[str]) -> str | None:
    if not component:
        return None
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
    is_materially_changed: bool = False,
) -> dict | None:
    component = components.get(skill.path)

    carried = existing_valid_metadata(skill, baseline, rename_map, schema_validator) or {}
    metadata: dict[str, str] = {k: carried[k] for k in MVP_FIELDS if k in carried}

    # Deterministic mapping wins: when the skill's registered component name
    # is a valid taxonomy value, it is authoritative for product.primary —
    # including over a carried baseline value. AI-enriched baselines can be
    # wrong (e.g. TAO skills labeled "Cosmos"/"Brev" from content inference)
    # and would otherwise persist forever, breaking product grouping on
    # downstream surfaces that read this file.
    product_enum = taxonomy["product.primary"]["values"]
    derived_product = derive_product_from_component(component, product_enum)
    if derived_product:
        metadata["product.primary"] = derived_product

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
        try:
            amended = ai_client(
                skill,
                component,
                list(MVP_FIELDS),
                taxonomy,
                current_values=metadata,
            )
        except EnrichmentError as exc:
            skill_warnings.append(f"{skill.path}: {exc}")
            return None
        for k in MVP_FIELDS:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
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
    components = load_components()
    try:
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
            is_materially_changed=skill.path in materially_changed,
        )
        if entry is not None:
            entries.append(entry)

    metadata_obj = {"skills": entries}

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

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        _print_classification(cls, stream=sys.stderr)
        return 1

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
        return 0

    md_changed = write_if_changed(METADATA_PATH, metadata_text)
    sh_changed = write_if_changed(SKILLS_SH_PATH, skills_sh_text)

    _print_classification(cls)
    print(
        f"metadata.json: {'updated' if md_changed else 'unchanged'} "
        f"({len(entries)} skills)"
    )
    print(
        f"skills.sh.json: {'updated' if sh_changed else 'unchanged'} "
        f"({len(skills_sh_obj.get('groupings', []))} non-empty groups)"
    )

    if skill_warnings:
        print(
            f"\nPARTIAL SUCCESS: {len(skill_warnings)} skill(s) could not be enriched "
            f"and were excluded from output:",
            file=sys.stderr,
        )
        for w in skill_warnings:
            print(f"  - {w}", file=sys.stderr)
        # Write a structured warnings file for CI to surface in issue bodies.
        warnings_path = Path("/tmp/skill-warnings.json")
        warnings_path.write_text(
            json.dumps({"omitted_skills": skill_warnings}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1

    return 0


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
