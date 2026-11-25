# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both skills run the same NIM container, so they must read the same NIM LLM docs and carry the
same environment variable checklist: the `docker (NIM LLM)` section of references/docs-sources.md
is identical in nim-deploy and nim-deploy-cli."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[2]
SECTION = "## docker (NIM LLM)"
NIM_LLM_PAGES = (
    "get-started/quickstart.html",
    "reference/architecture.html",
    "reference/environment-variables.html",
    "reference/advanced-configuration.html",
)
ENV_VARIABLES = (
    "NGC_API_KEY",
    "HF_TOKEN",
    "NIM_CACHE_PATH",
    "NIM_SERVER_PORT",
    "NIM_HEALTH_PORT",
    "NIM_MODEL_PROFILE",
    "NIM_MODEL_PATH",
    "NIM_SERVED_MODEL_NAME",
    "NIM_MAX_MODEL_LEN",
    "NIM_TENSOR_PARALLEL_SIZE",
    "NIM_PASSTHROUGH_ARGS",
)


def section(skill: str) -> str:
    text = (SKILLS / skill / "references" / "docs-sources.md").read_text()
    match = re.search(rf"^{re.escape(SECTION)}\n.*?(?=^## |\Z)", text, re.S | re.M)
    assert match, f"{skill}: no '{SECTION}' section in references/docs-sources.md"
    return match.group(0).strip()


def test_the_nim_llm_docs_section_is_the_same_in_both_skills() -> None:
    if not (SKILLS / "nim-deploy-cli").is_dir():
        pytest.skip("nim-deploy-cli is not installed next to nim-deploy")
    assert section("nim-deploy") == section("nim-deploy-cli")


@pytest.mark.parametrize("skill", ["nim-deploy", "nim-deploy-cli"])
def test_the_section_lists_the_pages_and_the_environment_variables(skill: str) -> None:
    if not (SKILLS / skill).is_dir():
        pytest.skip(f"{skill} is not installed")
    body = section(skill)
    assert [page for page in NIM_LLM_PAGES if page not in body] == []
    assert [name for name in ENV_VARIABLES if f"`{name}`" not in body] == []
