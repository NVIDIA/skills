# Skill Benchmark: nemo-gym-run-and-babysit-eval

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-gym-run-and-babysit-eval`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 21 evaluation tasks (19 positive, 2 negative)
- Dataset digest: `sha256:38877db796fe827d3ad2f80d9eb7d118184b7825826f9590140d2f514ef01b8c` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemo-gym-run-and-babysit-eval | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-run-and-babysit-eval | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-run-and-babysit-eval | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-run-and-babysit-eval | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-run-and-babysit-eval | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nemo-gym-run-and-babysit-eval`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 93.9% — baseline ran, but no comparable score was available; uplift unavailable | 95.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 83.8% → 98.1% (+14.3 points) | 75.2% → 99.1% (+23.9 points) |
| Discoverability | 93.7% — baseline ran, but no comparable score was available; uplift unavailable | 92.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 68.9% → 85.0% (+16.1 points) | 62.1% → 88.8% (+26.7 points) |
| Efficiency | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.8% → 89.3% (-6.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.7% → 90.0% (-2.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,659,608 | 2,472,180 | +1,187,428 | +48.03% | skill 21/21; base 21/21 |
| claude-code | gym-eval-001-existing-config | 93,526 | 146,895 | -53,369 | -36.33% | skill 1/1; base 1/1 |
| claude-code | gym-eval-002-new-config | 554,782 | 185,202 | +369,580 | +199.56% | skill 1/1; base 1/1 |
| claude-code | gym-eval-003-running | 91,105 | 71,389 | +19,716 | +27.62% | skill 1/1; base 1/1 |
| claude-code | gym-eval-004-walltime | 92,011 | 145,219 | -53,208 | -36.64% | skill 1/1; base 1/1 |
| claude-code | gym-eval-005-saved-answer | 92,309 | 72,295 | +20,014 | +27.68% | skill 1/1; base 1/1 |
| claude-code | gym-eval-006-semantic-change | 128,429 | 72,486 | +55,943 | +77.18% | skill 1/1; base 1/1 |
| claude-code | gym-eval-007-unverified | 204,038 | 72,045 | +131,993 | +183.21% | skill 1/1; base 1/1 |
| claude-code | gym-eval-008-success | 72,749 | 72,245 | +504 | +0.70% | skill 1/1; base 1/1 |
| claude-code | gym-eval-009-subset | 92,227 | 72,286 | +19,941 | +27.59% | skill 1/1; base 1/1 |
| claude-code | gym-eval-010-fault-boundary | 128,431 | 72,260 | +56,171 | +77.73% | skill 1/1; base 1/1 |
| claude-code | gym-eval-011-unpinned | 128,048 | 144,827 | -16,779 | -11.59% | skill 1/1; base 1/1 |
| claude-code | gym-eval-012-negative-benchmark | 237,033 | 144,893 | +92,140 | +63.59% | skill 1/1; base 1/1 |
| claude-code | gym-eval-013-negative-harness | 228,431 | 144,997 | +83,434 | +57.54% | skill 1/1; base 1/1 |
| claude-code | gym-eval-014-chat-endpoint | 213,416 | 145,928 | +67,488 | +46.25% | skill 1/1; base 1/1 |
| claude-code | gym-eval-015-responses-endpoint | 276,298 | 145,136 | +131,162 | +90.37% | skill 1/1; base 1/1 |
| claude-code | gym-eval-016-local-health | 150,133 | 72,139 | +77,994 | +108.12% | skill 1/1; base 1/1 |
| claude-code | gym-eval-017-task-selection | 220,089 | 145,570 | +74,519 | +51.19% | skill 1/1; base 1/1 |
| claude-code | gym-eval-018-local-resume | 207,832 | 108,490 | +99,342 | +91.57% | skill 1/1; base 1/1 |
| claude-code | gym-eval-019-supported-agent | 224,279 | 255,689 | -31,410 | -12.28% | skill 1/1; base 1/1 |
| claude-code | gym-eval-020-changed-task-resume | 151,204 | 72,970 | +78,234 | +107.21% | skill 1/1; base 1/1 |
| claude-code | gym-eval-021-benchmark-metric | 73,238 | 109,219 | -35,981 | -32.94% | skill 1/1; base 1/1 |
| codex | All cases | 1,860,522 | 1,107,184 | +753,338 | +68.04% | skill 21/21; base 21/21 |
| codex | gym-eval-001-existing-config | 137,476 | 56,771 | +80,705 | +142.16% | skill 1/1; base 1/1 |
| codex | gym-eval-002-new-config | 157,637 | 73,024 | +84,613 | +115.87% | skill 1/1; base 1/1 |
| codex | gym-eval-003-running | 65,676 | 26,837 | +38,839 | +144.72% | skill 1/1; base 1/1 |
| codex | gym-eval-004-walltime | 89,630 | 55,874 | +33,756 | +60.41% | skill 1/1; base 1/1 |
| codex | gym-eval-005-saved-answer | 86,494 | 71,191 | +15,303 | +21.50% | skill 1/1; base 1/1 |
| codex | gym-eval-006-semantic-change | 85,384 | 27,302 | +58,082 | +212.74% | skill 1/1; base 1/1 |
| codex | gym-eval-007-unverified | 103,577 | 27,379 | +76,198 | +278.31% | skill 1/1; base 1/1 |
| codex | gym-eval-008-success | 65,914 | 27,246 | +38,668 | +141.92% | skill 1/1; base 1/1 |
| codex | gym-eval-009-subset | 49,246 | 27,581 | +21,665 | +78.55% | skill 1/1; base 1/1 |
| codex | gym-eval-010-fault-boundary | 89,465 | 41,073 | +48,392 | +117.82% | skill 1/1; base 1/1 |
| codex | gym-eval-011-unpinned | 87,429 | 27,612 | +59,817 | +216.63% | skill 1/1; base 1/1 |
| codex | gym-eval-012-negative-benchmark | 70,946 | 41,636 | +29,310 | +70.40% | skill 1/1; base 1/1 |
| codex | gym-eval-013-negative-harness | 109,585 | 55,643 | +53,942 | +96.94% | skill 1/1; base 1/1 |
| codex | gym-eval-014-chat-endpoint | 72,762 | 72,681 | +81 | +0.11% | skill 1/1; base 1/1 |
| codex | gym-eval-015-responses-endpoint | 73,123 | 74,243 | -1,120 | -1.51% | skill 1/1; base 1/1 |
| codex | gym-eval-016-local-health | 48,770 | 27,188 | +21,582 | +79.38% | skill 1/1; base 1/1 |
| codex | gym-eval-017-task-selection | 98,920 | 142,352 | -43,432 | -30.51% | skill 1/1; base 1/1 |
| codex | gym-eval-018-local-resume | 91,062 | 72,960 | +18,102 | +24.81% | skill 1/1; base 1/1 |
| codex | gym-eval-019-supported-agent | 98,656 | 74,558 | +24,098 | +32.32% | skill 1/1; base 1/1 |
| codex | gym-eval-020-changed-task-resume | 114,108 | 56,452 | +57,656 | +102.13% | skill 1/1; base 1/1 |
| codex | gym-eval-021-benchmark-metric | 64,662 | 27,581 | +37,081 | +134.44% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,520,130 | 3,579,364 | +1,940,766 | +54.22% | skill 42/42; base 42/42 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 21 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nemo-gym-run-and-babysit-eval/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-gym-run-and-babysit-eval/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (12541 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nemo-gym-run-and-babysit-eval/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in local-and-api.md (`skills/nemo-gym-run-and-babysit-eval/SKILL.md`)
- **MEDIUM** SCHEMA/line_count: SKILL.md has 521 lines (limit: 500) (`skills/nemo-gym-run-and-babysit-eval/SKILL.md`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
