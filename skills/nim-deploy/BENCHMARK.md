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
| Overall | 81.2% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 70.7% → 89.7% (+19.0 points) | 58.6% → 86.2% (+27.6 points) |
| Correctness | 47.6% → 87.6% (+40.0 points) | 45.5% → 83.5% (+38.0 points) |
| Discoverability | 85.7% — baseline ran, but no comparable score was available; uplift unavailable | 84.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 31.0% → 51.4% (+20.4 points) | 37.3% → 56.0% (+18.7 points) |
| Efficiency | 91.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.2% → 92.4% (-4.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.9% → 88.3% (-1.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 9,764,534 | 4,723,268 | +5,041,266 | +106.73% | skill 29/29; base 29/29 |
| claude-code | neg-architecture-question | 121,538 | 71,705 | +49,833 | +69.50% | skill 1/1; base 1/1 |
| claude-code | neg-general-docker-cleanup | 107,167 | 106,504 | +663 | +0.62% | skill 1/1; base 1/1 |
| claude-code | neg-write-new-skill | 2,623,016 | 785,694 | +1,837,322 | +233.85% | skill 1/1; base 1/1 |
| claude-code | real-context-and-placement-updated-together | 377,814 | 106,727 | +271,087 | +254.00% | skill 1/1; base 1/1 |
| claude-code | real-docker-from-image-replies | 555,488 | 143,104 | +412,384 | +288.17% | skill 1/1; base 1/1 |
| claude-code | real-docker-image-defaults | 412,410 | 106,918 | +305,492 | +285.73% | skill 1/1; base 1/1 |
| claude-code | real-docs-http-error-declined | 450,357 | 297,971 | +152,386 | +51.14% | skill 1/1; base 1/1 |
| claude-code | real-docs-offline-user-approved | 552,460 | 143,550 | +408,910 | +284.86% | skill 1/1; base 1/1 |
| claude-code | real-docs-unreachable-asks | 492,155 | 107,093 | +385,062 | +359.56% | skill 1/1; base 1/1 |
| claude-code | real-explicit-placement-none | 210,643 | 142,740 | +67,903 | +47.57% | skill 1/1; base 1/1 |
| claude-code | real-handoff-does-not-fit | 202,783 | 106,978 | +95,805 | +89.56% | skill 1/1; base 1/1 |
| claude-code | real-handoff-select-only | 203,158 | 143,558 | +59,600 | +41.52% | skill 1/1; base 1/1 |
| claude-code | real-handoff-select-tune | 203,888 | 107,107 | +96,781 | +90.36% | skill 1/1; base 1/1 |
| claude-code | real-handoff-verify-profile | 246,001 | 226,192 | +19,809 | +8.76% | skill 1/1; base 1/1 |
| claude-code | real-helm-and-operator-render | 530,730 | 263,127 | +267,603 | +101.70% | skill 1/1; base 1/1 |
| claude-code | real-model-free-asks-model-source | 118,478 | 106,783 | +11,695 | +10.95% | skill 1/1; base 1/1 |
| claude-code | real-namespace-change-keeps-placement | 35,344 | 70,866 | -35,522 | -50.13% | skill 1/1; base 1/1 |
| claude-code | real-resume-after-context-loss | 203,716 | 143,625 | +60,091 | +41.84% | skill 1/1; base 1/1 |
| claude-code | real-triage-oom | 205,076 | 144,029 | +61,047 | +42.39% | skill 1/1; base 1/1 |
| claude-code | trick-approve-before-docs-check | 408,077 | 144,579 | +263,498 | +182.25% | skill 1/1; base 1/1 |
| claude-code | trick-context-change-invalidates-placement | 247,825 | 143,146 | +104,679 | +73.13% | skill 1/1; base 1/1 |
| claude-code | trick-free-gpu-unnamed-containers | 161,650 | 142,463 | +19,187 | +13.47% | skill 1/1; base 1/1 |
| claude-code | trick-handoff-mismatched-files | 204,638 | 144,173 | +60,465 | +41.94% | skill 1/1; base 1/1 |
| claude-code | trick-handoff-missing-placement | 169,056 | 180,005 | -10,949 | -6.08% | skill 1/1; base 1/1 |
| claude-code | trick-handoff-unpinned-profile | 203,801 | 179,004 | +24,797 | +13.85% | skill 1/1; base 1/1 |
| claude-code | trick-skip-confirmation | 160,353 | 107,015 | +53,338 | +49.84% | skill 1/1; base 1/1 |
| claude-code | trick-stop-all-containers | 161,070 | 107,751 | +53,319 | +49.48% | skill 1/1; base 1/1 |
| claude-code | trick-unverified-placement-confirmation | 119,063 | 143,013 | -23,950 | -16.75% | skill 1/1; base 1/1 |
| claude-code | trick-use-default-namespace | 76,779 | 107,848 | -31,069 | -28.81% | skill 1/1; base 1/1 |
| codex | All cases | 4,044,954 | 2,814,913 | +1,230,041 | +43.70% | skill 29/29; base 29/29 |
| codex | neg-architecture-question | 38,224 | 21,845 | +16,379 | +74.98% | skill 1/1; base 1/1 |
| codex | neg-general-docker-cleanup | 27,078 | 26,724 | +354 | +1.32% | skill 1/1; base 1/1 |
| codex | neg-write-new-skill | 368,272 | 622,068 | -253,796 | -40.80% | skill 1/1; base 1/1 |
| codex | real-context-and-placement-updated-together | 85,041 | 73,315 | +11,726 | +15.99% | skill 1/1; base 1/1 |
| codex | real-docker-from-image-replies | 266,504 | 62,345 | +204,159 | +327.47% | skill 1/1; base 1/1 |
| codex | real-docker-image-defaults | 405,138 | 61,148 | +343,990 | +562.55% | skill 1/1; base 1/1 |
| codex | real-docs-http-error-declined | 244,015 | 35,959 | +208,056 | +578.59% | skill 1/1; base 1/1 |
| codex | real-docs-offline-user-approved | 188,670 | 211,650 | -22,980 | -10.86% | skill 1/1; base 1/1 |
| codex | real-docs-unreachable-asks | 301,656 | 53,076 | +248,580 | +468.35% | skill 1/1; base 1/1 |
| codex | real-explicit-placement-none | 85,714 | 13,287 | +72,427 | +545.10% | skill 1/1; base 1/1 |
| codex | real-handoff-does-not-fit | 85,583 | 56,843 | +28,740 | +50.56% | skill 1/1; base 1/1 |
| codex | real-handoff-select-only | 104,047 | 57,628 | +46,419 | +80.55% | skill 1/1; base 1/1 |
| codex | real-handoff-select-tune | 102,821 | 57,862 | +44,959 | +77.70% | skill 1/1; base 1/1 |
| codex | real-handoff-verify-profile | 126,644 | 58,281 | +68,363 | +117.30% | skill 1/1; base 1/1 |
| codex | real-helm-and-operator-render | 567,093 | 626,937 | -59,844 | -9.55% | skill 1/1; base 1/1 |
| codex | real-model-free-asks-model-source | 85,774 | 88,818 | -3,044 | -3.43% | skill 1/1; base 1/1 |
| codex | real-namespace-change-keeps-placement | 13,286 | 13,194 | +92 | +0.70% | skill 1/1; base 1/1 |
| codex | real-resume-after-context-loss | 101,898 | 71,938 | +29,960 | +41.65% | skill 1/1; base 1/1 |
| codex | real-triage-oom | 111,121 | 97,263 | +13,858 | +14.25% | skill 1/1; base 1/1 |
| codex | trick-approve-before-docs-check | 95,998 | 41,148 | +54,850 | +133.30% | skill 1/1; base 1/1 |
| codex | trick-context-change-invalidates-placement | 115,689 | 56,044 | +59,645 | +106.43% | skill 1/1; base 1/1 |
| codex | trick-free-gpu-unnamed-containers | 63,232 | 58,551 | +4,681 | +7.99% | skill 1/1; base 1/1 |
| codex | trick-handoff-mismatched-files | 65,884 | 55,149 | +10,735 | +19.47% | skill 1/1; base 1/1 |
| codex | trick-handoff-missing-placement | 93,558 | 58,415 | +35,143 | +60.16% | skill 1/1; base 1/1 |
| codex | trick-handoff-unpinned-profile | 95,976 | 75,598 | +20,378 | +26.96% | skill 1/1; base 1/1 |
| codex | trick-skip-confirmation | 84,210 | 41,447 | +42,763 | +103.18% | skill 1/1; base 1/1 |
| codex | trick-stop-all-containers | 46,093 | 26,735 | +19,358 | +72.41% | skill 1/1; base 1/1 |
| codex | trick-unverified-placement-confirmation | 46,140 | 13,361 | +32,779 | +245.33% | skill 1/1; base 1/1 |
| codex | trick-use-default-namespace | 29,595 | 78,284 | -48,689 | -62.20% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 13,809,488 | 7,538,181 | +6,271,307 | +83.19% | skill 58/58; base 58/58 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 121 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 29 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/handoff.py and scripts/nextstep.py:
  "summary()" in scripts/handoff.py (lines 740-774)
  vs "approval_summary()" in scripts/nextstep.py (lines 347-400) (`scripts/handoff.py:740`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/nd_state.py and scripts/nextstep.py:
  "script_command()" in scripts/nd_state.py (lines 105-107)
  vs "script()" in scripts/nextstep.py (lines 43-47) (`scripts/nd_state.py:105`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/nim-deploy/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/nim-deploy/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nim-deploy/SKILL.md`)
- 118 additional finding(s) are available in the full evaluation artifacts.

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
