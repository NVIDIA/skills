# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
CATALOG_PATH = REPO_ROOT / "docs" / ".well-known" / "ai-catalog.json"
EXPECTED_IDENTIFIER_PREFIX = "urn:air:github.com:nvidia:skills:"
EXPECTED_URL_PREFIX = "https://github.com/NVIDIA/skills/blob/main/skills/"


def skill_names():
    return [
        d.name
        for d in SKILLS_DIR.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    ]


def catalog_entries():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8")).get("entries", [])


def catalog_display_names():
    return [entry["displayName"] for entry in catalog_entries()]


def test_catalog_exists():
    assert CATALOG_PATH.exists()


def test_catalog_is_valid_json():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert catalog["$schema"] == "https://agentdiscovery.org/schemas/catalog/v1.json"
    assert "entries" in catalog


def test_every_skill_has_catalog_entry():
    missing = [skill for skill in skill_names() if skill not in catalog_display_names()]
    assert not missing, "Skills missing from catalog:\n" + "\n".join(
        f"  - {skill}" for skill in missing
    )


def test_no_stale_catalog_entries():
    skills = set(skill_names())
    stale = [entry for entry in catalog_display_names() if entry not in skills]
    assert not stale, "Stale catalog entries:\n" + "\n".join(
        f"  - {entry}" for entry in stale
    )


def test_catalog_entries_have_ard_fields():
    for entry in catalog_entries():
        skill_name = entry["displayName"]
        assert entry["identifier"] == f"{EXPECTED_IDENTIFIER_PREFIX}{skill_name}"
        assert entry["type"] == "application/ai-skill"
        assert entry["url"] == f"{EXPECTED_URL_PREFIX}{skill_name}/SKILL.md"
        assert entry["description"]
        assert len(entry["representativeQueries"]) >= 3
