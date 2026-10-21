# Skill Benchmark: cdf-defect-image-generation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-defect-image-generation`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 18 evaluation tasks (12 positive, 6 negative)
- Dataset digest: `sha256:93d8e87a1a272081e57030da99fbc687536d36b31369ccc1b3286ee5f9a91abe` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 93.3% — baseline ran, but no comparable score was available; uplift unavailable | 92.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 94.4% → 100.0% (+5.6 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 43.3% → 86.7% (+43.4 points) | 61.1% → 87.8% (+26.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.1% → 85.8% (+43.7 points) | 48.8% → 88.3% (+39.5 points) |
| Efficiency | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | 97.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,090,482 | 3,997,944 | -907,462 | -22.70% | skill 18/18; base 18/18 |
| claude-code | dig-anomalygen-finetune-negative | 262,722 | 154,852 | +107,870 | +69.66% | skill 1/1; base 1/1 |
| claude-code | dig-api-timeout | 109,053 | 94,060 | +14,993 | +15.94% | skill 1/1; base 1/1 |
| claude-code | dig-capacity | 112,873 | 403,268 | -290,395 | -72.01% | skill 1/1; base 1/1 |
| claude-code | dig-clothing-augmentation-negative | 274,039 | 731,837 | -457,798 | -62.55% | skill 1/1; base 1/1 |
| claude-code | dig-controller-setup-negative | 30,145 | 29,749 | +396 | +1.33% | skill 1/1; base 1/1 |
| claude-code | dig-count-merge | 114,139 | 156,379 | -42,240 | -27.01% | skill 1/1; base 1/1 |
| claude-code | dig-dag-authoring-negative | 30,026 | 29,963 | +63 | +0.21% | skill 1/1; base 1/1 |
| claude-code | dig-day1-attached-image | 116,317 | 189,031 | -72,714 | -38.47% | skill 1/1; base 1/1 |
| claude-code | dig-event-video-negative | 163,573 | 29,827 | +133,746 | +448.41% | skill 1/1; base 1/1 |
| claude-code | dig-full-payload | 405,704 | 358,622 | +47,082 | +13.13% | skill 1/1; base 1/1 |
| claude-code | dig-golden-prompt-override | 111,858 | 280,221 | -168,363 | -60.08% | skill 1/1; base 1/1 |
| claude-code | dig-missing-inputs | 221,296 | 217,068 | +4,228 | +1.95% | skill 1/1; base 1/1 |
| claude-code | dig-mode-selection | 113,025 | 183,965 | -70,940 | -38.56% | skill 1/1; base 1/1 |
| claude-code | dig-new-board | 161,024 | 257,953 | -96,929 | -37.58% | skill 1/1; base 1/1 |
| claude-code | dig-pose-defect-coverage | 274,682 | 154,687 | +119,995 | +77.57% | skill 1/1; base 1/1 |
| claude-code | dig-result-provenance | 106,070 | 225,434 | -119,364 | -52.95% | skill 1/1; base 1/1 |
| claude-code | dig-texture-full-coverage | 297,206 | 317,527 | -20,321 | -6.40% | skill 1/1; base 1/1 |
| claude-code | dig-video-style-negative | 186,730 | 183,501 | +3,229 | +1.76% | skill 1/1; base 1/1 |
| codex | All cases | 1,761,117 | 1,471,917 | +289,200 | +19.65% | skill 18/18; base 18/18 |
| codex | dig-anomalygen-finetune-negative | 101,088 | 443,801 | -342,713 | -77.22% | skill 1/1; base 1/1 |
| codex | dig-api-timeout | 50,674 | 87,642 | -36,968 | -42.18% | skill 1/1; base 1/1 |
| codex | dig-capacity | 66,866 | 42,838 | +24,028 | +56.09% | skill 1/1; base 1/1 |
| codex | dig-clothing-augmentation-negative | 346,731 | 63,420 | +283,311 | +446.72% | skill 1/1; base 1/1 |
| codex | dig-controller-setup-negative | 30,533 | 87,881 | -57,348 | -65.26% | skill 1/1; base 1/1 |
| codex | dig-count-merge | 82,675 | 13,973 | +68,702 | +491.68% | skill 1/1; base 1/1 |
| codex | dig-dag-authoring-negative | 13,707 | 13,419 | +288 | +2.15% | skill 1/1; base 1/1 |
| codex | dig-day1-attached-image | 71,513 | 70,578 | +935 | +1.32% | skill 1/1; base 1/1 |
| codex | dig-event-video-negative | 30,133 | 29,643 | +490 | +1.65% | skill 1/1; base 1/1 |
| codex | dig-full-payload | 142,736 | 85,826 | +56,910 | +66.31% | skill 1/1; base 1/1 |
| codex | dig-golden-prompt-override | 149,720 | 90,897 | +58,823 | +64.71% | skill 1/1; base 1/1 |
| codex | dig-missing-inputs | 130,494 | 133,611 | -3,117 | -2.33% | skill 1/1; base 1/1 |
| codex | dig-mode-selection | 67,612 | 41,412 | +26,200 | +63.27% | skill 1/1; base 1/1 |
| codex | dig-new-board | 158,525 | 14,175 | +144,350 | +1018.34% | skill 1/1; base 1/1 |
| codex | dig-pose-defect-coverage | 91,364 | 84,149 | +7,215 | +8.57% | skill 1/1; base 1/1 |
| codex | dig-result-provenance | 48,209 | 70,798 | -22,589 | -31.91% | skill 1/1; base 1/1 |
| codex | dig-texture-full-coverage | 94,065 | 13,887 | +80,178 | +577.36% | skill 1/1; base 1/1 |
| codex | dig-video-style-negative | 84,472 | 83,967 | +505 | +0.60% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,851,599 | 5,469,861 | -618,262 | -11.30% | skill 36/36; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/cdf-defect-image-generation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in troubleshooting.md (`skills/cdf-defect-image-generation/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cdf-defect-image-generation/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: Without declared permissions the skill's intent is opaque and cannot be validated. (`SKILL.md:1`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -s -X POST "$AIRFLOW_URL/auth/token" -H "Content-Type: application/json" -d @"$AUTH_BODY")
rm -f "$AUTH_BODY"
RESPONSE="$RESPONSE" python3 -c 'import json, os; print("Authorization: Bearer " + js (`references/airflow-direct-api.md:28`)
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
