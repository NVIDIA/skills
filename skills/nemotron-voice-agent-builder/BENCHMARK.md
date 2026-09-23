# Skill Benchmark: nemotron-voice-agent-builder

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemotron-voice-agent-builder`
- Evaluation date: 2026-09-23
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:820305d412563ab26a2312566e2cd07ced079c651a147528990951cee89bcebc` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 87.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 36.4% → 80.0% (+43.6 points) |
| Correctness | Not available | 54.6% → 100.0% (+45.4 points) |
| Discoverability | Not available | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 34.3% → 88.5% (+54.2 points) |
| Efficiency | Not available | 76.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,141,398 | 20,667,844 | N/A | N/A | skill 5/5; base 10/15 |
| claude-code | nemotron-voice-agent-builder-contextual-high-concurrency-workstation | 853,172 | 77,026 | +776,146 | +1007.64% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-explicit-dgx-spark | 691,580 | 437,502 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nemotron-voice-agent-builder-implicit-low-concurrency-workstation | 456,973 | 14,983,825 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemotron-voice-agent-builder-livekit-omni-conflict | 132,199 | 3,188,790 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemotron-voice-agent-builder-negative-text-rag | 1,007,474 | 1,980,701 | -973,227 | -49.14% | skill 1/1; base 1/1 |
| codex | All cases | 1,388,532 | 7,397,001 | N/A | N/A | skill 5/5; base 11/11 |
| codex | nemotron-voice-agent-builder-contextual-high-concurrency-workstation | 229,734 | 27,692 | +202,042 | +729.60% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-explicit-dgx-spark | 170,703 | 79,503 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-voice-agent-builder-implicit-low-concurrency-workstation | 335,169 | 5,002,150 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-voice-agent-builder-livekit-omni-conflict | 83,428 | 1,762,066 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-voice-agent-builder-negative-text-rag | 569,498 | 525,590 | +43,908 | +8.35% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,529,930 | 28,064,845 | N/A | N/A | skill 10/10; base 21/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 17 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without checking (`references/domain/speech-customization.md:110`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without checking (`references/domain/speech-customization.md:189`)
- 12 additional finding(s) are available in the full evaluation artifacts.

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
