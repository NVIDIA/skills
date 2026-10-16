# Skill Benchmark: tao-run-dinov3-ssl-deft

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-dinov3-ssl-deft`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:4af41a3d00a986b758547fb383084946dab341ed73b17b47c2a29eadf2a70683` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 96.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 10.0% → 100.0% (+90.0 points) | 45.0% → 90.0% (+45.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 62.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 23.9% → 88.1% (+64.2 points) | 41.3% → 64.4% (+23.1 points) |
| Efficiency | 92.3% — baseline ran, but no comparable score was available; uplift unavailable | 96.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 512,059 | 1,462,799 | -950,740 | -64.99% | skill 4/4; base 4/4 |
| claude-code | tao-run-dinov3-ssl-deft-approval-monitoring | 180,094 | 294,691 | -114,597 | -38.89% | skill 1/1; base 1/1 |
| claude-code | tao-run-dinov3-ssl-deft-benchmark-isolation | 109,774 | 122,671 | -12,897 | -10.51% | skill 1/1; base 1/1 |
| claude-code | tao-run-dinov3-ssl-deft-layered-interface | 111,317 | 378,781 | -267,464 | -70.61% | skill 1/1; base 1/1 |
| claude-code | tao-run-dinov3-ssl-deft-traceability | 110,874 | 666,656 | -555,782 | -83.37% | skill 1/1; base 1/1 |
| codex | All cases | 426,687 | 131,839 | +294,848 | +223.64% | skill 4/4; base 4/4 |
| codex | tao-run-dinov3-ssl-deft-approval-monitoring | 14,829 | 13,935 | +894 | +6.42% | skill 1/1; base 1/1 |
| codex | tao-run-dinov3-ssl-deft-benchmark-isolation | 84,191 | 14,167 | +70,024 | +494.28% | skill 1/1; base 1/1 |
| codex | tao-run-dinov3-ssl-deft-layered-interface | 175,446 | 90,181 | +85,265 | +94.55% | skill 1/1; base 1/1 |
| codex | tao-run-dinov3-ssl-deft-traceability | 152,221 | 13,556 | +138,665 | +1022.90% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 938,746 | 1,594,638 | -655,892 | -41.13% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 24 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/adapter-contracts.md and references/architecture.md:
  "## Task Scoring" in references/adapter-contracts.md (lines 150-158)
  vs "## Run Trace" in references/architecture.md (lines 78-86) (`references/adapter-contracts.md:150`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/applications/tao-run-dinov3-ssl-deft/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in architecture.md (`skills/applications/tao-run-dinov3-ssl-deft/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/applications/tao-run-dinov3-ssl-deft/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/applications/tao-run-dinov3-ssl-deft`)
- 20 additional finding(s) are available in the full evaluation artifacts.

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
