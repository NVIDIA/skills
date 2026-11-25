# Skill Benchmark: nemo-gym-analyzing-results

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-gym-analyzing-results`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 23 evaluation tasks (13 positive, 10 negative)
- Dataset digest: `sha256:25cbc6115a6f8728688b469531f1d6218c33aa5090c954a932cb37562f6cfb48` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemo-gym-analyzing-results | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nemo-gym-analyzing-results`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 94.1% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 97.8% → 97.8% (±0.0 points) |
| Correctness | 83.5% → 95.7% (+12.2 points) | 96.5% → 99.1% (+2.6 points) |
| Discoverability | 96.2% — baseline ran, but no comparable score was available; uplift unavailable | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 79.5% → 85.6% (+6.1 points) | 83.0% → 92.4% (+9.4 points) |
| Efficiency | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 83.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.5% → 93.0% (-2.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.7% → 83.8% (+0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,633,155 | 3,801,460 | +831,695 | +21.88% | skill 23/23; base 23/23 |
| claude-code | baseline-counted-failures-as-zero | 325,048 | 173,874 | +151,174 | +86.94% | skill 1/1; base 1/1 |
| claude-code | different-task-sets | 326,791 | 292,145 | +34,646 | +11.86% | skill 1/1; base 1/1 |
| claude-code | failed-rollouts-inflate-candidate | 451,503 | 252,324 | +199,179 | +78.94% | skill 1/1; base 1/1 |
| claude-code | flipped-task-is-noise | 276,404 | 286,904 | -10,500 | -3.66% | skill 1/1; base 1/1 |
| claude-code | gap-to-published-number | 238,869 | 256,900 | -18,031 | -7.02% | skill 1/1; base 1/1 |
| claude-code | healthy-but-unchecked | 236,743 | 246,000 | -9,257 | -3.76% | skill 1/1; base 1/1 |
| claude-code | is-candidate-better-interval-contains-zero | 235,531 | 124,314 | +111,217 | +89.46% | skill 1/1; base 1/1 |
| claude-code | judge-failures-in-score | 124,748 | 143,101 | -18,353 | -12.83% | skill 1/1; base 1/1 |
| claude-code | managed-run-selected-attempts | 231,747 | 161,458 | +70,289 | +43.53% | skill 1/1; base 1/1 |
| claude-code | negative-add-public-benchmark | 155,504 | 72,537 | +82,967 | +114.38% | skill 1/1; base 1/1 |
| claude-code | negative-choose-training-tasks | 153,384 | 107,502 | +45,882 | +42.68% | skill 1/1; base 1/1 |
| claude-code | negative-custom-metric | 110,762 | 217,807 | -107,045 | -49.15% | skill 1/1; base 1/1 |
| claude-code | negative-plan-repeats-in-advance | 36,875 | 36,950 | -75 | -0.20% | skill 1/1; base 1/1 |
| claude-code | negative-resume-killed-run | 148,784 | 107,514 | +41,270 | +38.39% | skill 1/1; base 1/1 |
| claude-code | negative-run-a-benchmark | 189,640 | 143,786 | +45,854 | +31.89% | skill 1/1; base 1/1 |
| claude-code | negative-run-cannot-connect | 71,408 | 144,812 | -73,404 | -50.69% | skill 1/1; base 1/1 |
| claude-code | negative-slow-evaluation | 36,770 | 36,346 | +424 | +1.17% | skill 1/1; base 1/1 |
| claude-code | negative-training-reward-flat | 36,580 | 36,760 | -180 | -0.49% | skill 1/1; base 1/1 |
| claude-code | negative-welch-function | 35,993 | 35,869 | +124 | +0.35% | skill 1/1; base 1/1 |
| claude-code | pass-at-k-drops-tasks | 277,262 | 243,487 | +33,775 | +13.87% | skill 1/1; base 1/1 |
| claude-code | single-repeat-no-interval | 220,850 | 191,618 | +29,232 | +15.26% | skill 1/1; base 1/1 |
| claude-code | tasks-that-never-pass | 355,299 | 247,320 | +107,979 | +43.66% | skill 1/1; base 1/1 |
| claude-code | tasks-that-sometimes-pass | 356,660 | 242,132 | +114,528 | +47.30% | skill 1/1; base 1/1 |
| codex | All cases | 3,242,282 | 3,119,306 | +122,976 | +3.94% | skill 23/23; base 23/23 |
| codex | baseline-counted-failures-as-zero | 252,946 | 285,902 | -32,956 | -11.53% | skill 1/1; base 1/1 |
| codex | different-task-sets | 197,241 | 253,879 | -56,638 | -22.31% | skill 1/1; base 1/1 |
| codex | failed-rollouts-inflate-candidate | 224,002 | 269,975 | -45,973 | -17.03% | skill 1/1; base 1/1 |
| codex | flipped-task-is-noise | 154,459 | 98,563 | +55,896 | +56.71% | skill 1/1; base 1/1 |
| codex | gap-to-published-number | 299,231 | 243,379 | +55,852 | +22.95% | skill 1/1; base 1/1 |
| codex | healthy-but-unchecked | 196,704 | 210,346 | -13,642 | -6.49% | skill 1/1; base 1/1 |
| codex | is-candidate-better-interval-contains-zero | 151,321 | 178,779 | -27,458 | -15.36% | skill 1/1; base 1/1 |
| codex | judge-failures-in-score | 79,596 | 82,320 | -2,724 | -3.31% | skill 1/1; base 1/1 |
| codex | managed-run-selected-attempts | 133,489 | 79,160 | +54,329 | +68.63% | skill 1/1; base 1/1 |
| codex | negative-add-public-benchmark | 80,614 | 91,379 | -10,765 | -11.78% | skill 1/1; base 1/1 |
| codex | negative-choose-training-tasks | 85,280 | 57,343 | +27,937 | +48.72% | skill 1/1; base 1/1 |
| codex | negative-custom-metric | 81,137 | 84,047 | -2,910 | -3.46% | skill 1/1; base 1/1 |
| codex | negative-plan-repeats-in-advance | 18,744 | 73,733 | -54,989 | -74.58% | skill 1/1; base 1/1 |
| codex | negative-resume-killed-run | 39,535 | 57,021 | -17,486 | -30.67% | skill 1/1; base 1/1 |
| codex | negative-run-a-benchmark | 77,880 | 99,024 | -21,144 | -21.35% | skill 1/1; base 1/1 |
| codex | negative-run-cannot-connect | 61,717 | 91,304 | -29,587 | -32.40% | skill 1/1; base 1/1 |
| codex | negative-slow-evaluation | 91,612 | 73,645 | +17,967 | +24.40% | skill 1/1; base 1/1 |
| codex | negative-training-reward-flat | 47,645 | 47,412 | +233 | +0.49% | skill 1/1; base 1/1 |
| codex | negative-welch-function | 13,699 | 13,602 | +97 | +0.71% | skill 1/1; base 1/1 |
| codex | pass-at-k-drops-tasks | 183,359 | 148,262 | +35,097 | +23.67% | skill 1/1; base 1/1 |
| codex | single-repeat-no-interval | 100,243 | 129,290 | -29,047 | -22.47% | skill 1/1; base 1/1 |
| codex | tasks-that-never-pass | 366,560 | 233,342 | +133,218 | +57.09% | skill 1/1; base 1/1 |
| codex | tasks-that-sometimes-pass | 305,268 | 217,599 | +87,669 | +40.29% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,875,437 | 6,920,766 | +954,671 | +13.79% | skill 46/46; base 46/46 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 23 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (5977 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nemo-gym-analyzing-results/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (474 chars, recommend 50-150) (`skills/nemo-gym-analyzing-results/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nemo-gym-analyzing-results/SKILL.md`)
- **LOW** SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/nemo-gym-analyzing-results/SKILL.md`)

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
