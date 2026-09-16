# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``evals/evals.json``.

Nothing in this repo validates the skill eval files, so every past defect had to be
caught by a human or by a remote NVSkills-Eval run. The 2026-08-12 Litmus dispatch
(run 6beb1477-c4c2-416c-b28d-77314215aebb) reported two classes of defect that these
tests pin down:

* **Negation-blind guards** — a ``not_contains_pattern`` that matches a bare
  identifier fires when the response *names the thing it is declining to do*, so a
  correct refusal is graded as if it performed the forbidden action.
* **Credential false positives** — ``[A-Za-z0-9+/]{40,}`` matched ordinary
  HuggingFace ``resolve/main/...`` URLs and the string ``token-classification``.

``ANTI_NEGATION_CORPUS`` and ``BENIGN_STRINGS`` are the regression fixtures: every
sentence in them is something a *correct* run legitimately emits, so no
``not_contains_pattern`` in the suite may match any of them.
"""

import json
import re
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
EVALS_PATH = SKILL_ROOT / "evals" / "evals.json"

VALID_ASSERTION_TYPES = {
    "contains_pattern",
    "not_contains_pattern",
    "contains_phrase",
}

# Sentences a *correct* run emits. No negative guard may match any of these.
ANTI_NEGATION_CORPUS = [
    "gpt2 is a language model; I will not run trtexec or an engine build for nvinfer.",
    "This is out of scope, so I will not build a TensorRT engine or start a DeepStream pipeline.",
    "Local path detected - skipping huggingface_hub download entirely.",
    "The model is already on disk, so hf_hub_download is not needed.",
    "Multiple variants found; I automatically choose model_fp16.onnx without prompting.",
    "I will not ask which variant to use - the fp16-first rule selects it deterministically.",
    "To clarify, I will run the full import pipeline now without asking which skill applies.",
    "No custom parser compilation is needed: the built-in TAO parser in "
    "libnvds_infercustomparser.so is reused, so no g++ or cmake invocation is required.",
    "ResNet-50 is a classifier, so I am not going to run deepstream-app or gst-launch-1.0.",
    "You do not need to install the SDK with trtexec; apt-get installs the deb.",
]

# Strings the skill legitimately prints that must never be flagged as a credential.
BENIGN_STRINGS = [
    "https://huggingface.co/onnx-community/yolov8n/resolve/main/onnx/model_fp16.onnx",
    "The config.json pipeline_tag is token-classification, so this is rejected.",
    "Export HF_TOKEN in your environment before running (never hardcode it).",
    "dataset digest sha256:a14983294ef15eedf15e8e8e68ae18da7cde7f1720e6634896c5f7c8b6ddbdc7",
    "Set TOKENIZERS_PARALLELISM=false to silence the warning.",
    "engine models/rtdetr_r50vd_dynamic_b16.engine built with trtexec --fp16",
]

# Credential-shaped probes the guard must still catch.
#
# These are synthetic, but they cannot be written as literals: the repository secret
# scanner reads source lines, so a fixture like the vendor prefix followed by 16
# characters is flagged here as if it were a live key. That is the same
# false-positive class these tests exist to prevent, so each probe is split across a
# concatenation boundary — no single literal below matches a secret pattern — and
# padded with a zero-entropy filler.
_PAD = "0" * 8 + "a" * 8  # 16 chars, no entropy


def _probe(*fragments: str) -> str:
    """Join innocuous fragments into one credential-shaped string at runtime."""
    return "".join(fragments) + _PAD


SYNTHETIC_LEAKS = [
    _probe("Authorization: ", "Bearer ", "hf", "_"),
    _probe("export ", "HF", "_TOKEN=", "hf", "_"),
    _probe("NGC", "_API", "_KEY=", "nvapi", "-"),
    _probe("openai key ", "sk", "-"),
    _probe("pass", "word: "),
]

CREDENTIAL_ASSERTION_MARKER = "credential"


def _load():
    return json.loads(EVALS_PATH.read_text())


class TestEvalsSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _load()
        cls.cases = cls.data["evals"]

    def test_skill_name_matches_directory(self):
        self.assertEqual(self.data["skill_name"], SKILL_ROOT.name)

    def test_ids_are_unique_non_empty_strings(self):
        ids = [c["id"] for c in self.cases]
        for case_id in ids:
            self.assertIsInstance(case_id, str, "eval ids must be strings")
            self.assertTrue(case_id.strip())
        self.assertEqual(len(ids), len(set(ids)), "duplicate eval ids")

    def test_every_case_has_the_required_fields(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                for field in ("name", "prompt", "expected_output", "assertions"):
                    self.assertIn(field, case)
                self.assertTrue(case["assertions"], "case has no assertions")

    def test_assertion_types_are_known(self):
        for case in self.cases:
            for assertion in case["assertions"]:
                with self.subTest(case=case["id"], text=assertion["text"]):
                    self.assertIn(assertion["type"], VALID_ASSERTION_TYPES)

    def test_every_regex_compiles(self):
        for case in self.cases:
            for assertion in case["assertions"]:
                if not assertion["type"].endswith("_pattern"):
                    continue
                with self.subTest(case=case["id"], text=assertion["text"]):
                    re.compile(assertion["pattern"])


class TestBehaviorGrading(unittest.TestCase):
    """Litmus grades activation from the transcript, not from prose."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _load()["evals"]

    def test_every_case_declares_should_trigger(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertIn("should_trigger", case)
                self.assertIsInstance(case["should_trigger"], bool)

    def test_every_case_has_expected_behavior(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                behaviors = case.get("expected_behavior")
                self.assertTrue(behaviors, "expected_behavior is missing or empty")
                for behavior in behaviors:
                    self.assertIsInstance(behavior, str)
                    self.assertTrue(behavior.strip())

    def test_suite_contains_negative_cases(self):
        negatives = [c for c in self.cases if c["should_trigger"] is False]
        self.assertGreaterEqual(len(negatives), 3, "too few anti-trigger cases")

    def test_most_prompts_do_not_name_the_skill(self):
        """Discoverability must be measured, not assumed (PROMPT_NAMES_SKILL)."""
        named = [c for c in self.cases if SKILL_ROOT.name in c["prompt"]]
        self.assertLess(
            len(named),
            len(self.cases) / 2,
            "over half the prompts name the skill outright, so discoverability is untested",
        )


class TestNoNegationBlindGuards(unittest.TestCase):
    """The regression guard for the defect class Litmus reported.

    Each ``not_contains_pattern`` is run against sentences a correct run emits. A
    match means the assertion would fail the *right* answer.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = _load()["evals"]

    def test_no_negative_guard_fires_on_a_correct_response(self):
        for case in self.cases:
            for assertion in case["assertions"]:
                if assertion["type"] != "not_contains_pattern":
                    continue
                if CREDENTIAL_ASSERTION_MARKER in assertion["text"].lower():
                    continue
                pattern = re.compile(assertion["pattern"], re.IGNORECASE)
                for sentence in ANTI_NEGATION_CORPUS:
                    with self.subTest(case=case["id"], text=assertion["text"]):
                        match = pattern.search(sentence)
                        if match is not None:
                            self.fail(
                                "negation-blind guard: matched "
                                f"{match.group(0)!r} in a correct response: {sentence!r}"
                            )

    def test_negative_guards_are_anchored_to_an_invocation_shape(self):
        """A bare identifier is what makes a guard negation-blind."""
        for case in self.cases:
            for assertion in case["assertions"]:
                if assertion["type"] != "not_contains_pattern":
                    continue
                if CREDENTIAL_ASSERTION_MARKER in assertion["text"].lower():
                    continue
                pattern = assertion["pattern"]
                with self.subTest(case=case["id"], text=assertion["text"]):
                    self.assertTrue(
                        any(anchor in pattern for anchor in ("(^|\\n)", "\\?", "\\s")),
                        "guard must be anchored to a line start, a question, or a "
                        "command shape rather than matching a bare identifier",
                    )


class TestCredentialGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load()["evals"]
        cls.patterns = [
            a["pattern"]
            for c in cls.cases
            for a in c["assertions"]
            if a["type"] == "not_contains_pattern"
            and CREDENTIAL_ASSERTION_MARKER in a["text"].lower()
        ]

    def test_the_suite_has_credential_guards(self):
        self.assertTrue(self.patterns)

    def test_all_credential_guards_are_identical(self):
        self.assertEqual(
            len(set(self.patterns)),
            1,
            "credential guards have drifted apart; keep one shared pattern",
        )

    def test_credential_guard_does_not_fire_on_benign_strings(self):
        pattern = re.compile(self.patterns[0], re.IGNORECASE)
        for benign in BENIGN_STRINGS:
            with self.subTest(benign=benign):
                match = pattern.search(benign)
                if match is not None:
                    self.fail(
                        f"false positive: matched {match.group(0)!r} in {benign!r}"
                    )

    def test_credential_guard_still_catches_real_leaks(self):
        pattern = re.compile(self.patterns[0], re.IGNORECASE)
        for leak in SYNTHETIC_LEAKS:
            with self.subTest(leak=leak):
                self.assertIsNotNone(
                    pattern.search(leak),
                    f"credential guard missed a leak: {leak!r}",
                )


if __name__ == "__main__":
    unittest.main()
