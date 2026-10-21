# Skill Benchmark: holoscan-install-container

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `holoscan-install-container`
- Evaluation date: 2026-10-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:ed38125dd6f2c6f949645eb559436a68b8acd45f40b70699f3c00835ad4f6f29` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 93.3% — baseline ran, but no comparable score was available; uplift unavailable | 94.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 75.0% (±0.0 points) | 75.0% → 87.5% (+12.5 points) |
| Correctness | 75.0% → 100.0% (+25.0 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 73.1% → 91.3% (+18.2 points) | 71.5% → 90.4% (+18.9 points) |
| Efficiency | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 100.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 756,691 | 983,319 | N/A | N/A | skill 1/4; base 1/4 |
| claude-code | holoscan-install-container-5x-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | holoscan-install-container-5x-002 | 756,691 | 983,319 | -226,628 | -23.05% | skill 1/1; base 1/1 |
| claude-code | holoscan-install-container-5x-003 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | holoscan-install-container-5x-004 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | All cases | 750,625 | 616,870 | N/A | N/A | skill 1/4; base 1/4 |
| codex | holoscan-install-container-5x-001 | N/A | 616,870 | N/A | N/A | skill 0/1; base 1/1 |
| codex | holoscan-install-container-5x-002 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | holoscan-install-container-5x-003 | 750,625 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | holoscan-install-container-5x-004 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| ALL AGENTS | Dataset aggregate | 1,507,316 | 1,600,189 | N/A | N/A | skill 2/8; base 2/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`team-skills/holoscan/holoscan-sdk/holoscan-install-container/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`team-skills/holoscan/holoscan-sdk/holoscan-install-container/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: The `docker pull "$IMAGE"` command uses a shell variable `$IMAGE` constructed from `SDK_VERSION` and a hardcoded suffix  (`SKILL.md:54`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: The `docker run --rm --gpus all "$IMAGE" nvidia-smi` command runs a container from an image referenced by a mutable tag  (`SKILL.md:56`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: The verification `docker run` command at line 73 also uses `$IMAGE` (a mutable tag) and executes `bash -eu -c '...'` ins (`SKILL.md:73`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
