# Skill Benchmark: foundationpose-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `foundationpose-setup`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:630465a8eabab5dc1555b505a71728ed5e9bf504f29da89fde0808baaa447d61` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
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
| Overall | 95.2% — baseline ran, but no comparable score was available; uplift unavailable | 92.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 85.0% → 100.0% (+15.0 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 74.4% → 99.4% (+25.0 points) | 62.1% → 86.3% (+24.2 points) |
| Efficiency | 76.5% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 597,942 | 379,728 | +218,214 | +57.47% | skill 4/4; base 4/4 |
| claude-code | setup-context-engine-shape | 190,170 | 145,461 | +44,709 | +30.74% | skill 1/1; base 1/1 |
| claude-code | setup-explicit-os-floor | 97,439 | 97,282 | +157 | +0.16% | skill 1/1; base 1/1 |
| claude-code | setup-implicit-repair | 279,618 | 106,036 | +173,582 | +163.70% | skill 1/1; base 1/1 |
| claude-code | setup-negative-dataset-results | 30,715 | 30,949 | -234 | -0.76% | skill 1/1; base 1/1 |
| codex | All cases | 276,000 | 261,531 | +14,469 | +5.53% | skill 4/4; base 4/4 |
| codex | setup-context-engine-shape | 81,772 | 88,279 | -6,507 | -7.37% | skill 1/1; base 1/1 |
| codex | setup-explicit-os-floor | 76,845 | 101,292 | -24,447 | -24.14% | skill 1/1; base 1/1 |
| codex | setup-implicit-repair | 103,490 | 58,226 | +45,264 | +77.74% | skill 1/1; base 1/1 |
| codex | setup-negative-dataset-results | 13,893 | 13,734 | +159 | +1.16% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 873,942 | 641,259 | +232,683 | +36.29% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/foundationpose-setup/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: The command `docker run --rm --gpus=all ubuntu:24.04 nvidia-smi` uses `ubuntu:24.04` without a digest (SHA256 pinned ref (`references/installation.md:17`)
- **LOW** QUALITY/quality_discoverability: Description very long (214 chars, recommend 50-150) (`skills/foundationpose-setup/SKILL.md`)

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
