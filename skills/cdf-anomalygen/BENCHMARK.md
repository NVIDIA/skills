# Skill Benchmark: cdf-anomalygen

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-anomalygen`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (5 positive)
- Dataset digest: `sha256:70cd2d0187ec4a98b6dabaca1b4fab7c7a4aea3c53115acdcd5a15d75872cc18` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 71.4% — baseline ran, but no comparable score was available; uplift unavailable | 71.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 56.0% (+56.0 points) | 0.0% → 48.0% (+48.0 points) |
| Discoverability | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 6.5% → 17.8% (+11.3 points) | 4.5% → 20.7% (+16.2 points) |
| Efficiency | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 95.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,081,766 | 1,772,534 | +309,232 | +17.45% | skill 5/5; base 5/5 |
| claude-code | 1 | 310,839 | 292,125 | +18,714 | +6.41% | skill 1/1; base 1/1 |
| claude-code | 2 | 406,507 | 379,371 | +27,136 | +7.15% | skill 1/1; base 1/1 |
| claude-code | 3 | 358,612 | 213,331 | +145,281 | +68.10% | skill 1/1; base 1/1 |
| claude-code | 4 | 646,836 | 703,491 | -56,655 | -8.05% | skill 1/1; base 1/1 |
| claude-code | 5 | 358,972 | 184,216 | +174,756 | +94.86% | skill 1/1; base 1/1 |
| codex | All cases | 852,353 | 513,367 | +338,986 | +66.03% | skill 5/5; base 5/5 |
| codex | 1 | 134,242 | 113,129 | +21,113 | +18.66% | skill 1/1; base 1/1 |
| codex | 2 | 279,857 | 115,590 | +164,267 | +142.11% | skill 1/1; base 1/1 |
| codex | 3 | 141,050 | 113,490 | +27,560 | +24.28% | skill 1/1; base 1/1 |
| codex | 4 | 227,295 | 114,234 | +113,061 | +98.97% | skill 1/1; base 1/1 |
| codex | 5 | 69,909 | 56,924 | +12,985 | +22.81% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,934,119 | 2,285,901 | +648,218 | +28.36% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 38 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (9845 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-anomalygen/SKILL.md`)
- **MEDIUM** SCHEMA/line_count: SKILL.md has 639 lines (limit: 500) (`skills/cdf-anomalygen/SKILL.md`)
- **MEDIUM** SECURITY/Session Persistence (RA2): Rogue Agent: mkdir -p ~/.anomalygen && cp scripts/agentic_mask_generation_serving/endpoints.example.yaml ~/.anomalygen (`SKILL.md:98`)
- **MEDIUM** SECURITY/Whitespace Padding (P9): Prompt Injection: U+0020 x117 (`SKILL.md:581`)
- **MEDIUM** SECURITY/Whitespace Padding (P9): Prompt Injection: U+0020 x117 (`SKILL.md:638`)
- 33 additional finding(s) are available in the full evaluation artifacts.

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
