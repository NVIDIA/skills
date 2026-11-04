# Skill Benchmark: cdf-video-data-augmentation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-video-data-augmentation`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 21 evaluation tasks (16 positive, 5 negative)
- Dataset digest: `sha256:3e630c8191a34b59d215c6cad8c34972de2d6012dc3124eb268311b9e98d4d26` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 92.1% — baseline ran, but no comparable score was available; uplift unavailable | 86.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 95.2% → 100.0% (+4.8 points) | 83.3% → 90.5% (+7.2 points) |
| Correctness | 36.2% → 90.5% (+54.3 points) | 49.5% → 82.9% (+33.4 points) |
| Discoverability | 89.4% — baseline ran, but no comparable score was available; uplift unavailable | 88.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 35.6% → 87.6% (+52.0 points) | 39.96% → 79.04% (+39.08 points) |
| Efficiency | 93.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,585,292 | 6,568,628 | -2,983,336 | -45.42% | skill 21/21; base 21/21 |
| claude-code | vda-augmentation-component-routing | 30,023 | 154,119 | -124,096 | -80.52% | skill 1/1; base 1/1 |
| claude-code | vda-cleanup-does-not-mask-failure | 103,796 | 180,995 | -77,199 | -42.65% | skill 1/1; base 1/1 |
| claude-code | vda-component-reference-inaccessible | 69,023 | 152,805 | -83,782 | -54.83% | skill 1/1; base 1/1 |
| claude-code | vda-custom-dag-routing | 30,220 | 181,710 | -151,490 | -83.37% | skill 1/1; base 1/1 |
| claude-code | vda-event-generation-routing-negative | 517,560 | 1,750,906 | -1,233,346 | -70.44% | skill 1/1; base 1/1 |
| claude-code | vda-full-workflow-component-routing | 69,023 | 316,661 | -247,638 | -78.20% | skill 1/1; base 1/1 |
| claude-code | vda-generation-only-unsupported | 350,534 | 222,162 | +128,372 | +57.78% | skill 1/1; base 1/1 |
| claude-code | vda-ignored-fields-negative | 105,974 | 153,496 | -47,522 | -30.96% | skill 1/1; base 1/1 |
| claude-code | vda-labeling-component-routing | 29,737 | 29,453 | +284 | +0.96% | skill 1/1; base 1/1 |
| claude-code | vda-minimum-compute-all-checkers-positive | 66,612 | 151,974 | -85,362 | -56.17% | skill 1/1; base 1/1 |
| claude-code | vda-minimum-compute-no-plausibility-positive | 66,004 | 91,324 | -25,320 | -27.73% | skill 1/1; base 1/1 |
| claude-code | vda-missing-dag-no-deploy | 315,769 | 254,332 | +61,437 | +24.16% | skill 1/1; base 1/1 |
| claude-code | vda-no-input-negative | 184,590 | 184,430 | +160 | +0.09% | skill 1/1; base 1/1 |
| claude-code | vda-no-secret-leak-security | 29,577 | 29,261 | +316 | +1.08% | skill 1/1; base 1/1 |
| claude-code | vda-orchestration-setup-routing | 194,392 | 192,270 | +2,122 | +1.10% | skill 1/1; base 1/1 |
| claude-code | vda-osmo-internal-services-negative | 146,152 | 469,263 | -323,111 | -68.85% | skill 1/1; base 1/1 |
| claude-code | vda-payload-layer-routing | 98,290 | 30,080 | +68,210 | +226.76% | skill 1/1; base 1/1 |
| claude-code | vda-render-external-k8s-positive | 303,345 | 579,643 | -276,298 | -47.67% | skill 1/1; base 1/1 |
| claude-code | vda-render-internal-k8s-positive | 256,853 | 444,919 | -188,066 | -42.27% | skill 1/1; base 1/1 |
| claude-code | vda-render-osmo-positive | 302,558 | 442,965 | -140,407 | -31.70% | skill 1/1; base 1/1 |
| claude-code | vda-retrieval-routing-negative | 315,260 | 555,860 | -240,600 | -43.28% | skill 1/1; base 1/1 |
| codex | All cases | 3,825,155 | 3,313,506 | +511,649 | +15.44% | skill 21/21; base 21/21 |
| codex | vda-augmentation-component-routing | 27,399 | 18,504 | +8,895 | +48.07% | skill 1/1; base 1/1 |
| codex | vda-cleanup-does-not-mask-failure | 47,572 | 83,900 | -36,328 | -43.30% | skill 1/1; base 1/1 |
| codex | vda-component-reference-inaccessible | 69,341 | 53,053 | +16,288 | +30.70% | skill 1/1; base 1/1 |
| codex | vda-custom-dag-routing | 30,074 | 42,144 | -12,070 | -28.64% | skill 1/1; base 1/1 |
| codex | vda-event-generation-routing-negative | 27,970 | 13,861 | +14,109 | +101.79% | skill 1/1; base 1/1 |
| codex | vda-full-workflow-component-routing | 396,912 | 14,470 | +382,442 | +2643.00% | skill 1/1; base 1/1 |
| codex | vda-generation-only-unsupported | 1,654,496 | 112,659 | +1,541,837 | +1368.59% | skill 1/1; base 1/1 |
| codex | vda-ignored-fields-negative | 47,923 | 55,263 | -7,340 | -13.28% | skill 1/1; base 1/1 |
| codex | vda-labeling-component-routing | 13,649 | 13,499 | +150 | +1.11% | skill 1/1; base 1/1 |
| codex | vda-minimum-compute-all-checkers-positive | 108,135 | 745,037 | -636,902 | -85.49% | skill 1/1; base 1/1 |
| codex | vda-minimum-compute-no-plausibility-positive | 65,149 | 94,853 | -29,704 | -31.32% | skill 1/1; base 1/1 |
| codex | vda-missing-dag-no-deploy | 544,626 | 542,343 | +2,283 | +0.42% | skill 1/1; base 1/1 |
| codex | vda-no-input-negative | 49,393 | 69,276 | -19,883 | -28.70% | skill 1/1; base 1/1 |
| codex | vda-no-secret-leak-security | 29,788 | 13,463 | +16,325 | +121.26% | skill 1/1; base 1/1 |
| codex | vda-orchestration-setup-routing | 46,603 | 52,990 | -6,387 | -12.05% | skill 1/1; base 1/1 |
| codex | vda-osmo-internal-services-negative | 164,978 | 423,481 | -258,503 | -61.04% | skill 1/1; base 1/1 |
| codex | vda-payload-layer-routing | 29,850 | 84,998 | -55,148 | -64.88% | skill 1/1; base 1/1 |
| codex | vda-render-external-k8s-positive | 106,435 | 101,217 | +5,218 | +5.16% | skill 1/1; base 1/1 |
| codex | vda-render-internal-k8s-positive | 92,537 | 146,475 | -53,938 | -36.82% | skill 1/1; base 1/1 |
| codex | vda-render-osmo-positive | 86,374 | 514,064 | -427,690 | -83.20% | skill 1/1; base 1/1 |
| codex | vda-retrieval-routing-negative | 185,951 | 117,956 | +67,995 | +57.64% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,410,447 | 9,882,134 | -2,471,687 | -25.01% | skill 42/42; base 42/42 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 27 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 21 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/cdf-video-data-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/cdf-video-data-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/cdf-video-data-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cdf-video-data-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in airflow-direct-api.md (`skills/cdf-video-data-augmentation/SKILL.md`)
- 22 additional finding(s) are available in the full evaluation artifacts.

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
