#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the portable plugin.json rendered by build-plugins.py.

plugins/<name>/plugin.json targets Agent Plugins 1.0.0
(https://agent-plugins.org/schemas/1.0.0/plugin.schema.json), which Agent
Plugins compatible clients such as Hermes load. The schema sets
additionalProperties: false and Hermes refuses the whole plugin when
`$schema` is missing or `author` carries anything beyond string
name/email/url, so the properties that matter are:

  schema pin      `$schema` is the exact v1 URL.
  closed fields   no key outside the v1 property list leaks in from the
                  Claude/Codex/Cursor renderers (skills, interface, logo...).
  author shape    author is projected down to string name/email/url.
  no extensions   v1 has no displayName, and `extensions` is for data a
                  client reads, so display_name is not emitted at all.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "build-plugins.py"
_spec = importlib.util.spec_from_file_location("build_plugins", SCRIPT)
bp = importlib.util.module_from_spec(_spec)
sys.modules["build_plugins"] = bp
_spec.loader.exec_module(bp)

AGENT_PLUGINS_V1_FIELDS = {
    "$schema", "name", "version", "description", "author", "homepage",
    "repository", "license", "keywords", "extensions",
}


def full_spec() -> dict:
    return {
        "name": "nvidia-skills",
        "version": "1.4.0",
        "description": "Find the right NVIDIA skill.",
        "display_name": "NVIDIA Skills",
        "short_description": "Find the right NVIDIA skill",
        "long_description": "Longer text.",
        "author": {"name": "NVIDIA", "url": "https://github.com/NVIDIA/skills"},
        "homepage": "https://build.nvidia.com/skills/",
        "repository": "https://github.com/NVIDIA/skills",
        "license": "Apache-2.0 AND CC-BY-4.0",
        "keywords": ["nvidia", "gpu"],
        "category": "Developer Tools",
        "capabilities": ["Interactive", "Write"],
        "logo": "./assets/nvidia.png",
        "composer_icon": "./assets/nvidia.png",
        "brand_color": "#76b900",
        "default_prompts": ["Which NVIDIA skills can help?"],
        "include_skills": ["skills/nvidia-skill-finder/"],
        "skill_files": "copy",
    }


class RenderAgentPluginJsonTest(unittest.TestCase):
    def test_full_spec_renders_only_v1_fields(self):
        out = bp.render_agent_plugin_json(full_spec())
        self.assertEqual(out["$schema"], "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json")
        self.assertLessEqual(set(out), AGENT_PLUGINS_V1_FIELDS)
        self.assertEqual(out["name"], "nvidia-skills")
        self.assertEqual(out["version"], "1.4.0")
        self.assertEqual(out["description"], "Find the right NVIDIA skill.")
        self.assertEqual(out["author"], {"name": "NVIDIA", "url": "https://github.com/NVIDIA/skills"})
        self.assertEqual(out["homepage"], "https://build.nvidia.com/skills/")
        self.assertEqual(out["repository"], "https://github.com/NVIDIA/skills")
        self.assertEqual(out["license"], "Apache-2.0 AND CC-BY-4.0")
        self.assertEqual(out["keywords"], ["nvidia", "gpu"])
        self.assertNotIn("extensions", out)

    def test_minimal_spec(self):
        out = bp.render_agent_plugin_json({"name": "demo", "description": "d"})
        self.assertEqual(
            out,
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
                "name": "demo",
                "version": "1.0.0",
                "description": "d",
            },
        )

    def test_author_projected_to_string_name_email_url(self):
        spec = full_spec()
        spec["author"] = {"name": "NVIDIA", "email": "x@nvidia.com", "url": "https://nvidia.com", "org": "NV", "id": 7}
        out = bp.render_agent_plugin_json(spec)
        self.assertEqual(out["author"], {"name": "NVIDIA", "email": "x@nvidia.com", "url": "https://nvidia.com"})

    def test_non_mapping_author_dropped(self):
        spec = full_spec()
        spec["author"] = "NVIDIA"
        self.assertNotIn("author", bp.render_agent_plugin_json(spec))


if __name__ == "__main__":
    unittest.main()
