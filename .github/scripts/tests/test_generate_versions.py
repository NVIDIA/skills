#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for generate_versions.py.

The properties that matter to a consumer, and what breaks if they fail:

  digest stability      A digest that moves without content moving makes every
                        sync look like an update to every mirror.
  order independence    The signer does not promise manifest ordering, so a
                        re-sign that shuffles resources must not change the
                        digest.
  separator safety      Concatenating name and digest without a separator lets
                        two different file lists collide.
  removal detection     A generated file stays schema-valid with one fewer
                        entry, which is how a skill silently vanished from
                        metadata.json on 2026-08-03.
"""

import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_versions as gv  # noqa: E402


def make_sig(resources: list[dict], subject: str = "demo-skill") -> str:
    """A minimal signature bundle carrying the given resource list."""
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": subject, "digest": {"sha256": "0" * 64}}],
        "predicateType": "https://model_signing/signature/v1.0",
        "predicate": {"resources": resources},
    }
    payload = base64.b64encode(json.dumps(statement).encode()).decode()
    return json.dumps({"dsseEnvelope": {"payload": payload, "payloadType": "x"}})


RESOURCES = [
    {"algorithm": "sha256", "name": "SKILL.md", "digest": "a" * 64},
    {"algorithm": "sha256", "name": "evals/evals.json", "digest": "b" * 64},
    {"algorithm": "sha256", "name": "skill-card.md", "digest": "c" * 64},
]


class TestSignedResources(unittest.TestCase):
    def test_decodes_dsse_payload(self):
        sig = Path(self.tmp) / "skill.oms.sig"
        sig.write_text(make_sig(RESOURCES))
        self.assertEqual(gv.signed_resources(sig), RESOURCES)

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class TestContentDigest(unittest.TestCase):
    def test_is_deterministic(self):
        self.assertEqual(gv.content_digest(RESOURCES), gv.content_digest(RESOURCES))

    def test_is_prefixed_sha256(self):
        digest = gv.content_digest(RESOURCES)
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), len("sha256:") + 64)

    def test_ignores_manifest_ordering(self):
        """A re-sign that shuffles resources must not look like a change."""
        shuffled = [RESOURCES[2], RESOURCES[0], RESOURCES[1]]
        self.assertEqual(gv.content_digest(RESOURCES), gv.content_digest(shuffled))

    def test_changes_when_a_file_digest_changes(self):
        edited = [dict(RESOURCES[0], digest="d" * 64), RESOURCES[1], RESOURCES[2]]
        self.assertNotEqual(gv.content_digest(RESOURCES), gv.content_digest(edited))

    def test_changes_when_a_file_is_added(self):
        extra = RESOURCES + [
            {"algorithm": "sha256", "name": "BENCHMARK.md", "digest": "e" * 64},
        ]
        self.assertNotEqual(gv.content_digest(RESOURCES), gv.content_digest(extra))

    def test_changes_when_a_file_is_removed(self):
        fewer = RESOURCES[:-1]
        self.assertNotEqual(gv.content_digest(RESOURCES), gv.content_digest(fewer))

    def test_changes_when_a_file_is_renamed(self):
        renamed = [dict(RESOURCES[0], name="SKILL.markdown")] + RESOURCES[1:]
        self.assertNotEqual(gv.content_digest(RESOURCES), gv.content_digest(renamed))

    def test_name_and_digest_cannot_collide(self):
        """Without a separator these two lists would serialize identically."""
        left = [{"name": "ab", "digest": "c"}]
        right = [{"name": "a", "digest": "bc"}]
        self.assertNotEqual(gv.content_digest(left), gv.content_digest(right))


class TestRemovedSkills(unittest.TestCase):
    def test_detects_a_dropped_skill(self):
        old = {"skills": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        new = {"skills": [{"name": "a"}, {"name": "c"}]}
        self.assertEqual(gv.removed_skills(new, old), ["b"])

    def test_additions_are_not_removals(self):
        old = {"skills": [{"name": "a"}]}
        new = {"skills": [{"name": "a"}, {"name": "b"}]}
        self.assertEqual(gv.removed_skills(new, old), [])

    def test_no_change_reports_nothing(self):
        doc = {"skills": [{"name": "a"}, {"name": "b"}]}
        self.assertEqual(gv.removed_skills(doc, doc), [])

    def test_reports_every_dropped_skill_sorted(self):
        old = {"skills": [{"name": "c"}, {"name": "a"}, {"name": "b"}]}
        new = {"skills": []}
        self.assertEqual(gv.removed_skills(new, old), ["a", "b", "c"])


class TestValidate(unittest.TestCase):
    def good(self) -> dict:
        return {
            "$schema": gv.SCHEMA_URL,
            "skills": [{
                "name": "demo-skill",
                "path": "skills/demo-skill",
                "content_digest": "sha256:" + "a" * 64,
                "last_commit": "b" * 40,
                "last_modified": "2026-08-28",
            }],
        }

    def test_accepts_a_well_formed_document(self):
        gv.validate(self.good())

    def test_rejects_an_unprefixed_digest(self):
        doc = self.good()
        doc["skills"][0]["content_digest"] = "a" * 64
        with self.assertRaises(Exception):
            gv.validate(doc)

    def test_rejects_a_nested_path(self):
        """Flat layout: a skill lives at skills/<name>, never deeper."""
        doc = self.good()
        doc["skills"][0]["path"] = "skills/product/demo-skill"
        with self.assertRaises(Exception):
            gv.validate(doc)

    def test_rejects_unknown_fields(self):
        """Closed schema — the contract cannot grow by accident."""
        doc = self.good()
        doc["skills"][0]["surprise"] = "x"
        with self.assertRaises(Exception):
            gv.validate(doc)

    def test_rejects_a_missing_required_field(self):
        doc = self.good()
        del doc["skills"][0]["last_modified"]
        with self.assertRaises(Exception):
            gv.validate(doc)

    def test_rejects_a_malformed_date(self):
        doc = self.good()
        doc["skills"][0]["last_modified"] = "28-08-2026"
        with self.assertRaises(Exception):
            gv.validate(doc)


class TestSerialize(unittest.TestCase):
    def test_ends_with_a_single_newline(self):
        out = gv.serialize({"$schema": "x", "skills": []})
        self.assertTrue(out.endswith("}\n"))
        self.assertFalse(out.endswith("\n\n"))

    def test_is_byte_stable(self):
        doc = {"$schema": "x", "skills": [{"name": "a"}]}
        self.assertEqual(gv.serialize(doc), gv.serialize(doc))

    def test_preserves_non_ascii(self):
        """ensure_ascii=False keeps descriptions readable rather than escaped."""
        out = gv.serialize({"$schema": "x", "skills": [{"name": "café"}]})
        self.assertIn("café", out)


if __name__ == "__main__":
    unittest.main()
