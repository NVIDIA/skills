#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Route ASR customization/orchestration prompts to the right bundled reference file.

This helper is intentionally small and deterministic. It does not run training,
evaluation, or invoke sub-skills; the orchestration skill drives those. The default
route is the orchestration workflow, since the skill's job is to scope the task and
sequence sub-skills before anything else.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterable


SKILL_NAME = "nemotron-asr-finetune"


@dataclass(frozen=True)
class Route:
    name: str
    reference: str
    reason: str
    patterns: tuple[str, ...]
    next_steps: tuple[str, ...]


DOMAIN_CUES = (
    r"\bfine[- ]?tun",
    r"\bfinetun",
    r"\bcustomiz",
    r"\badapt\b.*\b(asr|language|domain|model)\b",
    r"\bdomain adaptation\b",
    r"\bnew language\b",
    r"\bimprove\b.*\b(asr|accuracy|recognition|transcri)",
    r"\basr\b",
    r"\bspeech[- ]to[- ]text\b",
    r"\btranscri(?:be|bing|ption)\b",
    r"\brecogniz",
    r"\bmishear",
    r"\bword boost",
    r"\bn-?gram\b",
    r"\bkenlm\b",
    r"\blanguage model\b",
    r"\breduce (?:the )?wer\b",
    r"\bwer\b",
    r"\bnemo\b",
    r"\bparakeet\b",
    r"\bcanary\b",
    r"\bnemotron\b.*\b(asr|speech|streaming)\b",
)


ROUTES: tuple[Route, ...] = (
    Route(
        name="sub-skills",
        reference="references/sub-skills.md",
        reason="Which sub-skill to invoke and its handoff contract.",
        patterns=(
            r"\bsub-?skill",
            r"\bdelegate\b",
            r"\binvoke\b",
            r"\bwhich skill\b",
            r"\bdata[- ]?designer\b",
            r"\bsynthetic data\b",
            r"\bnemo-speech-asr-finetune\b",
            r"\bnemotron-speech\b",
            r"\bevaluator\b",
            r"\bdeploy",
            r"\bexport\b",
            r"\bserve\b|\bserving\b",
            r"\bnoise profil",
            r"\bvendor data\b",
        ),
        next_steps=(
            "Read references/sub-skills.md.",
            "Invoke the sub-skill by name; if it is a placeholder, use the interim guidance and tell the user.",
            "Pass the handoff inputs listed for that sub-skill and integrate what it returns.",
        ),
    ),
    Route(
        name="planning-answers",
        reference="references/planning-answers.md",
        reason="Cost/time/data planning questions (hours, synthetic vs real, GPU choice, cost).",
        patterns=(
            r"\bhow much data\b",
            r"\bhow many hours\b",
            r"\bhours\b",
            r"\bcost\b",
            r"\bexpensive\b|\bcheap",
            r"\bbudget\b",
            r"\bl40s\b",
            r"\bh100\b",
            r"\bgpu\b.*\b(choice|which|pick)\b",
            r"\bsynthetic vs real\b",
            r"\bhow long\b",
            r"\bhit\b.*\bwer\b",
            r"\bto (?:reach|hit) \d",
        ),
        next_steps=(
            "Read references/planning-answers.md.",
            "Give ranges with assumptions; prefer a measured pilot over a guessed hours-to-WER number.",
            "Lead with the cheapest sufficient path's cost, then the full fine-tuning plan's cost.",
        ),
    ),
    Route(
        name="path-selection",
        reference="references/path-selection.md",
        reason="Choosing the cheapest sufficient rung (boosting / vocab / n-gram LM / fine-tune).",
        patterns=(
            r"\bwhich\b.*\b(option|approach|way|method|path|rung)\b",
            r"\bcheapest\b",
            r"\bword boost",
            r"\bboost",
            r"\bn-?gram\b",
            r"\bkenlm\b",
            r"\blanguage model\b",
            r"\bcustom vocab",
            r"\bpronunciation\b",
            r"\bfull fine[- ]?tune\b",
            r"\bfrom scratch\b",
            r"\bnot recogniz|\bmis-?recogniz",
            r"\bjargon\b|\bvocabulary\b|\bnames?\b",
        ),
        next_steps=(
            "Read references/path-selection.md.",
            "Diagnose the failure mode and measure a baseline, then pick the lowest matching rung.",
            "Escalate the tuning method only if the target is missed; you may run a cheap-rung experiment now.",
        ),
    ),
    Route(
        name="workflow",
        reference="references/workflow.md",
        reason="The end-to-end orchestration loop: scope, choose path, data, train, evaluate, loop/ship, deploy.",
        patterns=(
            r"\bfine[- ]?tun",
            r"\bcustomiz",
            r"\bworkflow\b",
            r"\borchestrat",
            r"\bpipeline\b",
            r"\bscope\b|\bclarify\b",
            r"\bwhere do i start\b|\bhow do i start\b",
            r"\bnew language\b",
            r"\bdomain\b",
            r"\bimprove\b",
            r"\btrain\b",
            r"\bevaluat",
            r"\bloop\b|\bship\b",
        ),
        next_steps=(
            "Read references/workflow.md.",
            "Announce each step (Step N/8) and run the discovery questions before recommending a path.",
            "Delegate each stage to its sub-skill (see references/sub-skills.md) and answer planning questions along the way.",
        ),
    ),
)


def any_match(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def score_route(route: Route, text: str) -> int:
    return sum(1 for pattern in route.patterns if re.search(pattern, text, flags=re.IGNORECASE))


def classify(question: str) -> dict[str, object]:
    text = " ".join(question.strip().split())
    if not text:
        return {
            "expected_skill": None, "route": None, "reference": None, "confidence": "low",
            "reason": "Empty prompt.",
            "next_steps": ["Do not activate nemotron-asr-finetune without a concrete ASR customization task."],
        }

    if not any_match(DOMAIN_CUES, text):
        return {
            "expected_skill": None, "route": None, "reference": None, "confidence": "low",
            "reason": "No NeMo/Riva ASR customization or fine-tuning cue was found.",
            "next_steps": ["Keep the nemotron-asr-finetune skill silent and use a more relevant workflow."],
        }

    # Stay silent when another provider/framework is named without an in-domain cue.
    strong_domain = (r"\briva\b", r"\bnemo\b", r"\bparakeet\b", r"\bcanary\b", r"\bnemotron\b",
                     r"\bspeech_to_text")
    other_provider = (r"\bopenai\b", r"\bwhisper\b", r"\bgpt\b", r"\bllama\b", r"\bmistral\b",
                      r"\bgemini\b", r"\bcohere\b")
    text_llm = (r"\btext (?:llm|model)\b", r"\bchat log", r"\bllm\b")
    if any_match(other_provider, text) and not any_match(strong_domain, text):
        return {
            "expected_skill": None, "route": None, "reference": None, "confidence": "low",
            "reason": "Another provider/framework was requested; not a NeMo/Riva ASR customization task.",
            "next_steps": ["Keep the nemotron-asr-finetune skill silent and use the appropriate workflow."],
        }
    if any_match(text_llm, text) and not any_match((r"\basr\b", r"\bspeech", r"\btranscri", r"\bwer\b"), text):
        return {
            "expected_skill": None, "route": None, "reference": None, "confidence": "low",
            "reason": "Text-LLM task, not ASR customization.",
            "next_steps": ["Keep the nemotron-asr-finetune skill silent and use the LLM fine-tuning workflow."],
        }

    # Defer pure export/deployment execution to the nemotron-speech skill. These are concrete
    # commands that skill owns; the orchestrator only sequences the deploy stage, never runs them.
    defer_cues = (r"\bnemo2riva\b", r"\briva-build\b", r"\briva-deploy\b", r"\brmir\b", r"\.riva\b")
    if any_match(defer_cues, text):
        return {
            "expected_skill": None, "route": None, "reference": None, "confidence": "low",
            "reason": "Export/deployment execution is owned by the nemotron-speech skill.",
            "next_steps": ["Defer to the nemotron-speech skill for nemo2riva / riva-build / riva-deploy and serving."],
        }

    # ROUTES are declared so the orchestration workflow is last (highest index).
    # Break ties (and the all-zero case) toward it: scoping and sequencing come first.
    scored = [(score_route(route, text), index, route) for index, route in enumerate(ROUTES)]
    scored.sort(key=lambda item: (-item[0], -item[1]))
    score, _, route = scored[0]

    confidence = "high" if score >= 2 else "medium"
    return {
        "expected_skill": SKILL_NAME,
        "route": route.name,
        "reference": route.reference,
        "confidence": confidence,
        "reason": route.reason,
        "next_steps": list(route.next_steps),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="User prompt to classify.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    question = " ".join(args.question) if args.question else sys.stdin.read()
    result = classify(question)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
