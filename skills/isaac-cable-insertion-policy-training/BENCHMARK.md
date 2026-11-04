# Skill Benchmark: isaac-cable-insertion-policy-training

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `isaac-cable-insertion-policy-training`
- Validation status: `passed`
- Evaluation date: 2026-11-04
- Evaluator version: `1.6.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 11 evaluation tasks (11 positive)
- Dataset digest: `sha256:6b39d6615f27b196d7e789fac463b897743bdb4790de0b766a0bd345819b45a5` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| isaac-cable-insertion-policy-training | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| isaac-cable-insertion-policy-training | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| isaac-cable-insertion-policy-training | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| isaac-cable-insertion-policy-training | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| isaac-cable-insertion-policy-training | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |

- Controls for `isaac-cable-insertion-policy-training`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 86.6% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 100.0% → 100.0% (±0.0 points) | Not available |
| Correctness | 56.4% → 85.5% (+29.1 points) | Not available |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 36.5% → 56.1% (+19.6 points) | Not available |
| Efficiency | 91.2% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.5% → 89.7% (-7.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,364,809 | 1,386,560 | +978,249 | +70.55% | skill 11/11; base 11/11 |
| claude-code | isaac-cable-insertion-policy-training-plan-default | 230,291 | 148,278 | +82,013 | +55.31% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-plan-missing-exit-marker | 280,869 | 109,352 | +171,517 | +156.85% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-plan-physx-retry | 322,357 | 109,628 | +212,729 | +194.05% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-plan-short-run | 168,659 | 108,499 | +60,160 | +55.45% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-setup-blocked | 119,513 | 108,456 | +11,057 | +10.19% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-setup-fresh | 228,402 | 147,465 | +80,937 | +54.89% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-setup-no-path | 178,325 | 109,023 | +69,302 | +63.57% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-setup-path-collision | 243,601 | 144,241 | +99,360 | +68.88% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-setup-public-access | 156,747 | 109,439 | +47,308 | +43.23% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-setup-repair | 224,275 | 182,246 | +42,029 | +23.06% | skill 1/1; base 1/1 |
| claude-code | isaac-cable-insertion-policy-training-standalone-dependency-conflicts | 211,770 | 109,933 | +101,837 | +92.64% | skill 1/1; base 1/1 |
| codex | All cases | 1,309,289 | 605,879 | N/A | N/A | skill 9/11; base 9/11; With-skill token coverage is incomplete (9/11 attempts). Some token counters are missing. Baseline token coverage is incomplete (9/11 attempts). Some token counters are missing. |
| codex | isaac-cable-insertion-policy-training-plan-default | 168,495 | 82,102 | +86,393 | +105.23% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-plan-missing-exit-marker | 110,824 | 71,821 | +39,003 | +54.31% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-plan-physx-retry | 170,544 | 79,963 | +90,581 | +113.28% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-plan-short-run | 88,087 | 57,692 | +30,395 | +52.68% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-setup-blocked | 87,168 | 57,109 | +30,059 | +52.63% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-setup-fresh | 203,809 | 87,610 | +116,199 | +132.63% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-setup-no-path | 199,354 | 70,814 | +128,540 | +181.52% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-setup-path-collision | N/A | 41,862 | N/A | N/A | skill 0/1; base 1/1; With-skill token coverage is incomplete (0/1 attempts). Some token counters are missing. |
| codex | isaac-cable-insertion-policy-training-setup-public-access | N/A | N/A | N/A | N/A | skill 0/1; base 0/1; With-skill token coverage is incomplete (0/1 attempts). Some token counters are missing. Baseline token coverage is incomplete (0/1 attempts). Some token counters are missing. |
| codex | isaac-cable-insertion-policy-training-setup-repair | 166,129 | 56,906 | +109,223 | +191.94% | skill 1/1; base 1/1 |
| codex | isaac-cable-insertion-policy-training-standalone-dependency-conflicts | 114,879 | N/A | N/A | N/A | skill 1/1; base 0/1; Baseline token coverage is incomplete (0/1 attempts). Some token counters are missing. |
| ALL AGENTS | Dataset aggregate | 3,674,098 | 1,992,439 | N/A | N/A | skill 20/22; base 20/22; With-skill token coverage is incomplete (20/22 attempts). Some token counters are missing. Baseline token coverage is incomplete (20/22 attempts). Some token counters are missing. |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 66 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** PII/ip_addresses: Public IP address (`references/setup.md:67`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/setup.md:94`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_dependencies.py:47`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_dependencies.py:51`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_dependencies.py:59`)
- 61 additional finding(s) are available in the full evaluation artifacts.

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
