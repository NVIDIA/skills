# Skill Benchmark: nemotron-asr-finetune

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemotron-asr-finetune`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 17 evaluation tasks (14 positive, 3 negative)
- Dataset digest: `sha256:1265bdc5a4c5f3bcc2526b4d79cee536998a77dcc82765150ae3cbc7bcb05c2d` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.7% — baseline ran, but no comparable score was available; uplift unavailable | 80.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 94.1% (-5.9 points) | 85.2% → 80.0% (-5.2 points) |
| Correctness | 61.0% → 94.1% (+33.1 points) | 58.5% → 83.0% (+24.5 points) |
| Discoverability | 98.2% — baseline ran, but no comparable score was available; uplift unavailable | 91.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.1% → 83.3% (+46.2 points) | 38.1% → 66.7% (+28.6 points) |
| Efficiency | 83.7% — baseline ran, but no comparable score was available; uplift unavailable | 80.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,696,703 | 2,131,523 | N/A | N/A | skill 17/17; base 21/21 |
| claude-code | nemotron-asr-orchestrate-envcheck-001 | 195,423 | 412,637 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nemotron-asr-orchestrate-envcheck-insist-001 | 198,794 | 29,519 | +169,275 | +573.44% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-escalate-001 | 174,424 | 32,270 | +142,154 | +440.51% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-eval-001 | 157,832 | 92,250 | +65,582 | +71.09% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-lowdata-001 | 103,542 | 31,654 | +71,888 | +227.11% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-negative-deploy-001 | 356,276 | 126,930 | +229,346 | +180.69% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-negative-llm-001 | 836,221 | 157,238 | +678,983 | +431.82% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-negative-openai-001 | 62,476 | 62,600 | -124 | -0.20% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-ngram-rnnt-deploy-001 | 106,297 | 64,965 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nemotron-asr-orchestrate-path-001 | 169,680 | 30,781 | +138,899 | +451.25% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-planning-001 | 174,715 | 97,090 | +77,625 | +79.95% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-preflight-001 | 308,210 | 214,595 | +93,615 | +43.62% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-preflight-8khz-001 | 156,654 | 31,400 | +125,254 | +398.90% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-scope-001 | 170,261 | 94,315 | +75,946 | +80.52% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-subskill-reachability-001 | 248,758 | 219,466 | +29,292 | +13.35% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-subskills-001 | 112,186 | 366,568 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemotron-asr-orchestrate-wordboost-nemo-pilot-001 | 164,954 | 67,245 | +97,709 | +145.30% | skill 1/1; base 1/1 |
| codex | All cases | 2,563,746 | 3,277,730 | N/A | N/A | skill 20/20; base 27/27 |
| codex | nemotron-asr-orchestrate-envcheck-001 | 65,836 | 229,187 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-asr-orchestrate-envcheck-insist-001 | 47,508 | 13,301 | +34,207 | +257.18% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-escalate-001 | 30,616 | 13,774 | +16,842 | +122.27% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-eval-001 | 50,060 | 55,432 | -5,372 | -9.69% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-lowdata-001 | 30,379 | 20,243 | +10,136 | +50.07% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-negative-deploy-001 | 439,456 | 709,131 | -269,675 | -38.03% | skill 2/2; base 2/2 |
| codex | nemotron-asr-orchestrate-negative-llm-001 | 1,093,941 | 1,264,495 | -170,554 | -13.49% | skill 3/3; base 3/3 |
| codex | nemotron-asr-orchestrate-negative-openai-001 | 39,625 | 110,770 | -71,145 | -64.23% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-ngram-rnnt-deploy-001 | 74,378 | 50,802 | +23,576 | +46.41% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-path-001 | 54,704 | 18,269 | +36,435 | +199.44% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-planning-001 | 73,787 | 18,913 | +54,874 | +290.14% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-preflight-001 | 186,530 | 85,429 | +101,101 | +118.35% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-preflight-8khz-001 | 35,326 | 236,698 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-asr-orchestrate-scope-001 | 34,659 | 25,257 | +9,402 | +37.23% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-subskill-reachability-001 | 169,468 | 198,340 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nemotron-asr-orchestrate-subskills-001 | 49,903 | 194,982 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-asr-orchestrate-wordboost-nemo-pilot-001 | 87,570 | 32,707 | +54,863 | +167.74% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,260,449 | 5,409,253 | N/A | N/A | skill 37/37; base 48/48 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 17 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/nemotron-asr-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/nemotron-asr-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/nemotron-asr-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in path-selection.md (`skills/nemotron-asr-finetune/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/nemotron-asr-finetune/SKILL.md`)
- 21 additional finding(s) are available in the full evaluation artifacts.

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
