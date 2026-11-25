# Skill Benchmark: nim-deploy

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nim-deploy`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 29 evaluation tasks (29 positive)
- Dataset digest: `sha256:6f98da5bb5845c77dd8b27b37977ea8900c7b43a095c6715bd42a228595a6966` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nim-deploy | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-deploy | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-deploy | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-deploy | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-deploy | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nim-deploy`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.3% — baseline ran, but no comparable score was available; uplift unavailable | 81.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 79.3% → 84.5% (+5.2 points) | 56.9% → 93.1% (+36.2 points) |
| Correctness | 50.3% → 86.2% (+35.9 points) | 46.9% → 85.5% (+38.6 points) |
| Discoverability | 85.5% — baseline ran, but no comparable score was available; uplift unavailable | 85.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.9% → 53.7% (+16.8 points) | 35.8% → 53.1% (+17.3 points) |
| Efficiency | 91.4% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.5% → 92.8% (-4.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.2% → 88.5% (-1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 8,437,239 | 4,235,135 | +4,202,104 | +99.22% | skill 29/29; base 29/29 |
| claude-code | neg-architecture-question | 180,034 | 71,325 | +108,709 | +152.41% | skill 1/1; base 1/1 |
| claude-code | neg-general-docker-cleanup | 106,948 | 106,222 | +726 | +0.68% | skill 1/1; base 1/1 |
| claude-code | neg-write-new-skill | 1,359,145 | 528,755 | +830,390 | +157.05% | skill 1/1; base 1/1 |
| claude-code | real-context-and-placement-updated-together | 305,040 | 106,705 | +198,335 | +185.87% | skill 1/1; base 1/1 |
| claude-code | real-docker-from-image-replies | 429,967 | 107,311 | +322,656 | +300.67% | skill 1/1; base 1/1 |
| claude-code | real-docker-image-defaults | 482,534 | 107,021 | +375,513 | +350.88% | skill 1/1; base 1/1 |
| claude-code | real-docs-http-error-declined | 564,601 | 146,697 | +417,904 | +284.88% | skill 1/1; base 1/1 |
| claude-code | real-docs-offline-user-approved | 504,444 | 107,352 | +397,092 | +369.90% | skill 1/1; base 1/1 |
| claude-code | real-docs-unreachable-asks | 541,754 | 106,935 | +434,819 | +406.62% | skill 1/1; base 1/1 |
| claude-code | real-explicit-placement-none | 210,436 | 106,582 | +103,854 | +97.44% | skill 1/1; base 1/1 |
| claude-code | real-handoff-does-not-fit | 203,178 | 143,159 | +60,019 | +41.92% | skill 1/1; base 1/1 |
| claude-code | real-handoff-select-only | 203,396 | 143,797 | +59,599 | +41.45% | skill 1/1; base 1/1 |
| claude-code | real-handoff-select-tune | 203,616 | 107,354 | +96,262 | +89.67% | skill 1/1; base 1/1 |
| claude-code | real-handoff-verify-profile | 246,184 | 218,811 | +27,373 | +12.51% | skill 1/1; base 1/1 |
| claude-code | real-helm-and-operator-render | 652,148 | 261,512 | +390,636 | +149.38% | skill 1/1; base 1/1 |
| claude-code | real-model-free-asks-model-source | 76,718 | 107,294 | -30,576 | -28.50% | skill 1/1; base 1/1 |
| claude-code | real-namespace-change-keeps-placement | 35,388 | 35,101 | +287 | +0.82% | skill 1/1; base 1/1 |
| claude-code | real-resume-after-context-loss | 204,059 | 179,509 | +24,550 | +13.68% | skill 1/1; base 1/1 |
| claude-code | real-triage-oom | 337,620 | 182,131 | +155,489 | +85.37% | skill 1/1; base 1/1 |
| claude-code | trick-approve-before-docs-check | 121,281 | 143,192 | -21,911 | -15.30% | skill 1/1; base 1/1 |
| claude-code | trick-context-change-invalidates-placement | 156,463 | 142,132 | +14,331 | +10.08% | skill 1/1; base 1/1 |
| claude-code | trick-free-gpu-unnamed-containers | 160,433 | 142,339 | +18,094 | +12.71% | skill 1/1; base 1/1 |
| claude-code | trick-handoff-mismatched-files | 204,470 | 143,602 | +60,868 | +42.39% | skill 1/1; base 1/1 |
| claude-code | trick-handoff-missing-placement | 268,461 | 144,892 | +123,569 | +85.28% | skill 1/1; base 1/1 |
| claude-code | trick-handoff-unpinned-profile | 203,331 | 144,000 | +59,331 | +41.20% | skill 1/1; base 1/1 |
| claude-code | trick-skip-confirmation | 160,795 | 141,611 | +19,184 | +13.55% | skill 1/1; base 1/1 |
| claude-code | trick-stop-all-containers | 118,996 | 107,873 | +11,123 | +10.31% | skill 1/1; base 1/1 |
| claude-code | trick-unverified-placement-confirmation | 119,013 | 143,893 | -24,880 | -17.29% | skill 1/1; base 1/1 |
| claude-code | trick-use-default-namespace | 76,786 | 108,028 | -31,242 | -28.92% | skill 1/1; base 1/1 |
| codex | All cases | 4,066,999 | 2,482,988 | +1,584,011 | +63.79% | skill 29/29; base 29/29 |
| codex | neg-architecture-question | 38,246 | 21,967 | +16,279 | +74.11% | skill 1/1; base 1/1 |
| codex | neg-general-docker-cleanup | 26,993 | 26,723 | +270 | +1.01% | skill 1/1; base 1/1 |
| codex | neg-write-new-skill | 408,554 | 401,377 | +7,177 | +1.79% | skill 1/1; base 1/1 |
| codex | real-context-and-placement-updated-together | 105,178 | 60,692 | +44,486 | +73.30% | skill 1/1; base 1/1 |
| codex | real-docker-from-image-replies | 364,566 | 91,541 | +273,025 | +298.25% | skill 1/1; base 1/1 |
| codex | real-docker-image-defaults | 329,378 | 60,930 | +268,448 | +440.58% | skill 1/1; base 1/1 |
| codex | real-docs-http-error-declined | 230,781 | 88,982 | +141,799 | +159.36% | skill 1/1; base 1/1 |
| codex | real-docs-offline-user-approved | 234,609 | 89,783 | +144,826 | +161.31% | skill 1/1; base 1/1 |
| codex | real-docs-unreachable-asks | 199,361 | 58,063 | +141,298 | +243.35% | skill 1/1; base 1/1 |
| codex | real-explicit-placement-none | 107,221 | 41,381 | +65,840 | +159.11% | skill 1/1; base 1/1 |
| codex | real-handoff-does-not-fit | 85,849 | 58,391 | +27,458 | +47.02% | skill 1/1; base 1/1 |
| codex | real-handoff-select-only | 85,433 | 57,024 | +28,409 | +49.82% | skill 1/1; base 1/1 |
| codex | real-handoff-select-tune | 142,308 | 55,557 | +86,751 | +156.15% | skill 1/1; base 1/1 |
| codex | real-handoff-verify-profile | 115,732 | 76,444 | +39,288 | +51.39% | skill 1/1; base 1/1 |
| codex | real-helm-and-operator-render | 467,760 | 583,591 | -115,831 | -19.85% | skill 1/1; base 1/1 |
| codex | real-model-free-asks-model-source | 85,915 | 48,684 | +37,231 | +76.47% | skill 1/1; base 1/1 |
| codex | real-namespace-change-keeps-placement | 13,288 | 13,259 | +29 | +0.22% | skill 1/1; base 1/1 |
| codex | real-resume-after-context-loss | 127,815 | 59,618 | +68,197 | +114.39% | skill 1/1; base 1/1 |
| codex | real-triage-oom | 111,364 | 81,727 | +29,637 | +36.26% | skill 1/1; base 1/1 |
| codex | trick-approve-before-docs-check | 115,457 | 41,149 | +74,308 | +180.58% | skill 1/1; base 1/1 |
| codex | trick-context-change-invalidates-placement | 92,045 | 58,816 | +33,229 | +56.50% | skill 1/1; base 1/1 |
| codex | trick-free-gpu-unnamed-containers | 69,239 | 55,033 | +14,206 | +25.81% | skill 1/1; base 1/1 |
| codex | trick-handoff-mismatched-files | 119,435 | 54,654 | +64,781 | +118.53% | skill 1/1; base 1/1 |
| codex | trick-handoff-missing-placement | 121,234 | 58,481 | +62,753 | +107.30% | skill 1/1; base 1/1 |
| codex | trick-handoff-unpinned-profile | 65,626 | 79,294 | -13,668 | -17.24% | skill 1/1; base 1/1 |
| codex | trick-skip-confirmation | 81,807 | 70,513 | +11,294 | +16.02% | skill 1/1; base 1/1 |
| codex | trick-stop-all-containers | 46,066 | 26,707 | +19,359 | +72.49% | skill 1/1; base 1/1 |
| codex | trick-unverified-placement-confirmation | 46,104 | 13,462 | +32,642 | +242.48% | skill 1/1; base 1/1 |
| codex | trick-use-default-namespace | 29,635 | 49,145 | -19,510 | -39.70% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 12,504,238 | 6,718,123 | +5,786,115 | +86.13% | skill 58/58; base 58/58 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 120 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 29 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/nd_state.py and scripts/nextstep.py:
  "script_command()" in scripts/nd_state.py (lines 105-107)
  vs "script()" in scripts/nextstep.py (lines 43-47) (`scripts/nd_state.py:105`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across scripts/handoff.py and scripts/nextstep.py:
  "summary()" in scripts/handoff.py (lines 740-774)
  vs "approval_summary()" in scripts/nextstep.py (lines 347-400) (`scripts/handoff.py:740`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/nim-deploy/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/nim-deploy/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nim-deploy/SKILL.md`)
- 117 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
