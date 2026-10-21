# Skill Benchmark: cdf-orchestration-write-dag

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-orchestration-write-dag`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 9 evaluation tasks (7 positive, 2 negative)
- Dataset digest: `sha256:2bc7705f144b721a29bf402452bef3e40301d230497b3d784778b0746b748162` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 87.2% — baseline ran, but no comparable score was available; uplift unavailable | 76.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 100.0% (+16.7 points) | 66.7% → 72.2% (+5.5 points) |
| Correctness | 33.3% → 95.6% (+62.3 points) | 31.1% → 73.3% (+42.2 points) |
| Discoverability | 95.7% — baseline ran, but no comparable score was available; uplift unavailable | 82.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 24.5% → 66.6% (+42.1 points) | 15.8% → 67.3% (+51.5 points) |
| Efficiency | 78.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,924,810 | 10,665,207 | -4,740,397 | -44.45% | skill 9/9; base 9/9 |
| claude-code | negative-endpoint-change-evg | 907,635 | 440,483 | +467,152 | +106.05% | skill 1/1; base 1/1 |
| claude-code | negative-payload-only-iaa | 401,629 | 125,570 | +276,059 | +219.84% | skill 1/1; base 1/1 |
| claude-code | write-dag-augmentation-only-positive | 943,196 | 290,367 | +652,829 | +224.83% | skill 1/1; base 1/1 |
| claude-code | write-dag-deploy-confirmation-positive | 460,226 | 6,003,918 | -5,543,692 | -92.33% | skill 1/1; base 1/1 |
| claude-code | write-dag-existing-dag-negative | 539,076 | 868,053 | -328,977 | -37.90% | skill 1/1; base 1/1 |
| claude-code | write-dag-gpu-footprint-positive | 860,675 | 2,282,602 | -1,421,927 | -62.29% | skill 1/1; base 1/1 |
| claude-code | write-dag-labeling-only-positive | 947,565 | 286,645 | +660,920 | +230.57% | skill 1/1; base 1/1 |
| claude-code | write-dag-nvcf-scope-negative | 315,746 | 187,495 | +128,251 | +68.40% | skill 1/1; base 1/1 |
| claude-code | write-dag-parity-check-positive | 549,062 | 180,074 | +368,988 | +204.91% | skill 1/1; base 1/1 |
| codex | All cases | 4,547,573 | 1,928,192 | +2,619,381 | +135.85% | skill 9/9; base 9/9 |
| codex | negative-endpoint-change-evg | 198,369 | 129,149 | +69,220 | +53.60% | skill 1/1; base 1/1 |
| codex | negative-payload-only-iaa | 270,749 | 99,770 | +170,979 | +171.37% | skill 1/1; base 1/1 |
| codex | write-dag-augmentation-only-positive | 1,965,233 | 236,776 | +1,728,457 | +730.00% | skill 1/1; base 1/1 |
| codex | write-dag-deploy-confirmation-positive | 212,001 | 339,095 | -127,094 | -37.48% | skill 1/1; base 1/1 |
| codex | write-dag-existing-dag-negative | 204,772 | 174,524 | +30,248 | +17.33% | skill 1/1; base 1/1 |
| codex | write-dag-gpu-footprint-positive | 214,577 | 208,904 | +5,673 | +2.72% | skill 1/1; base 1/1 |
| codex | write-dag-labeling-only-positive | 274,516 | 254,541 | +19,975 | +7.85% | skill 1/1; base 1/1 |
| codex | write-dag-nvcf-scope-negative | 183,656 | 338,295 | -154,639 | -45.71% | skill 1/1; base 1/1 |
| codex | write-dag-parity-check-positive | 1,023,700 | 147,138 | +876,562 | +595.74% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 10,472,383 | 12,593,399 | -2,121,016 | -16.84% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (10429 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-orchestration-write-dag/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cdf-orchestration-write-dag/SKILL.md`)
- **MEDIUM** SCHEMA/line_count: SKILL.md has 581 lines (limit: 500) (`skills/cdf-orchestration-write-dag/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cdf-orchestration-write-dag/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: no confirmation (`SKILL.md:485`)
- 9 additional finding(s) are available in the full evaluation artifacts.

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
