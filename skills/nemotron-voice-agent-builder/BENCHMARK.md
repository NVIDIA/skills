# Skill Benchmark: nemotron-voice-agent-builder

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemotron-voice-agent-builder`
- Evaluation date: 2026-10-19
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:820305d412563ab26a2312566e2cd07ced079c651a147528990951cee89bcebc` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

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
| Overall | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | 85.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 70.0% → 80.0% (+10.0 points) | 40.0% → 80.0% (+40.0 points) |
| Correctness | 48.0% → 88.0% (+40.0 points) | 68.0% → 100.0% (+32.0 points) |
| Discoverability | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.5% → 86.5% (+39.0 points) | 48.5% → 86.5% (+38.0 points) |
| Efficiency | 82.2% — baseline ran, but no comparable score was available; uplift unavailable | 69.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 11,598,700 | 5,274,254 | +6,324,446 | +119.91% | skill 5/5; base 5/5 |
| claude-code | nemotron-voice-agent-builder-contextual-high-concurrency-workstation | 709,644 | 97,839 | +611,805 | +625.32% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-explicit-dgx-spark | 1,195,656 | 186,462 | +1,009,194 | +541.23% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-implicit-low-concurrency-workstation | 626,820 | 696,920 | -70,100 | -10.06% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-livekit-omni-conflict | 130,318 | 1,967,971 | -1,837,653 | -93.38% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-negative-text-rag | 8,936,262 | 2,325,062 | +6,611,200 | +284.35% | skill 1/1; base 1/1 |
| codex | All cases | 1,067,133 | 3,887,505 | -2,820,372 | -72.55% | skill 5/5; base 5/5 |
| codex | nemotron-voice-agent-builder-contextual-high-concurrency-workstation | 102,035 | 25,501 | +76,534 | +300.12% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-explicit-dgx-spark | 182,167 | 47,921 | +134,246 | +280.14% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-implicit-low-concurrency-workstation | 239,160 | 1,926,704 | -1,687,544 | -87.59% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-livekit-omni-conflict | 129,335 | 1,253,480 | -1,124,145 | -89.68% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-negative-text-rag | 414,436 | 633,899 | -219,463 | -34.62% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 12,665,833 | 9,161,759 | +3,504,074 | +38.25% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 23 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nemotron-voice-agent-builder/SKILL.md (`BENCHMARK.md:86`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nemotron-voice-agent-builder/SKILL.md (`BENCHMARK.md:87`)
- 18 additional finding(s) are available in the full evaluation artifacts.

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
