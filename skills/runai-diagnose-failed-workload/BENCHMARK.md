# Skill Benchmark: runai-diagnose-failed-workload

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `runai-diagnose-failed-workload`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 22 evaluation tasks (19 positive, 3 negative)
- Dataset digest: `sha256:11671384f2bddc73199555ab2ec1df1138baead68ab48fc4efd782b89946e0e0` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 98.1% — baseline ran, but no comparable score was available; uplift unavailable | 83.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 81.8% → 98.2% (+16.4 points) | 78.2% → 84.6% (+6.4 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 70.4% → 94.4% (+24.0 points) | 65.1% → 79.5% (+14.4 points) |
| Efficiency | 97.7% — baseline ran, but no comparable score was available; uplift unavailable | 56.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,211,466 | 2,055,783 | +1,155,683 | +56.22% | skill 22/22; base 22/22 |
| claude-code | v1-missing-cluster | 66,596 | 94,061 | -27,465 | -29.20% | skill 1/1; base 1/1 |
| claude-code | v1-negative-cluster-health | 99,488 | 134,541 | -35,053 | -26.05% | skill 1/1; base 1/1 |
| claude-code | v1-negative-inference-latency | 63,231 | 62,792 | +439 | +0.70% | skill 1/1; base 1/1 |
| claude-code | v1-negative-workload-health-summary | 135,738 | 135,747 | -9 | -0.01% | skill 1/1; base 1/1 |
| claude-code | v1-trigger-explicit | 66,565 | 170,913 | -104,348 | -61.05% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-cluster-mismatch | 103,535 | 63,688 | +39,847 | +62.57% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-init-failure | 143,323 | 96,216 | +47,107 | +48.96% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-mount-failure | 181,597 | 98,876 | +82,721 | +83.66% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-name-project-resolution | 223,406 | 131,649 | +91,757 | +69.70% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-name-project-running-crashloopbackoff | 219,185 | 95,585 | +123,600 | +129.31% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-node-failure | 180,463 | 64,367 | +116,096 | +180.37% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-pending-boundary | 104,271 | 97,582 | +6,689 | +6.85% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-policy-termination | 141,591 | 64,647 | +76,944 | +119.02% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-preemption-boundary | 104,255 | 130,242 | -25,987 | -19.95% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-recurring-oom | 182,624 | 65,867 | +116,757 | +177.26% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-running-crashloopbackoff | 143,736 | 97,112 | +46,624 | +48.01% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-unexplained-sigkill | 181,168 | 95,934 | +85,234 | +88.85% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-unknown | 180,459 | 95,905 | +84,554 | +88.16% | skill 1/1; base 1/1 |
| claude-code | v2-fixture-untrusted-event | 180,357 | 64,704 | +115,653 | +178.74% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-application-crash | 144,064 | 65,176 | +78,888 | +121.04% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-image-pull-failure | 182,366 | 65,243 | +117,123 | +179.52% | skill 1/1; base 1/1 |
| claude-code | v2-recorded-oom | 183,448 | 64,936 | +118,512 | +182.51% | skill 1/1; base 1/1 |
| codex | All cases | 3,378,759 | 4,273,802 | -895,043 | -20.94% | skill 22/22; base 22/22 |
| codex | v1-missing-cluster | 103,600 | 72,973 | +30,627 | +41.97% | skill 1/1; base 1/1 |
| codex | v1-negative-cluster-health | 31,641 | 288,760 | -257,119 | -89.04% | skill 1/1; base 1/1 |
| codex | v1-negative-inference-latency | 226,506 | 294,444 | -67,938 | -23.07% | skill 1/1; base 1/1 |
| codex | v1-negative-workload-health-summary | 183,943 | 201,298 | -17,355 | -8.62% | skill 1/1; base 1/1 |
| codex | v1-trigger-explicit | 30,981 | 291,031 | -260,050 | -89.35% | skill 1/1; base 1/1 |
| codex | v2-fixture-cluster-mismatch | 196,029 | 165,089 | +30,940 | +18.74% | skill 1/1; base 1/1 |
| codex | v2-fixture-init-failure | 102,754 | 168,836 | -66,082 | -39.14% | skill 1/1; base 1/1 |
| codex | v2-fixture-mount-failure | 175,528 | 166,462 | +9,066 | +5.45% | skill 1/1; base 1/1 |
| codex | v2-fixture-name-project-resolution | 237,484 | 219,175 | +18,309 | +8.35% | skill 1/1; base 1/1 |
| codex | v2-fixture-name-project-running-crashloopbackoff | 213,976 | 198,305 | +15,671 | +7.90% | skill 1/1; base 1/1 |
| codex | v2-fixture-node-failure | 194,057 | 235,100 | -41,043 | -17.46% | skill 1/1; base 1/1 |
| codex | v2-fixture-pending-boundary | 156,876 | 206,431 | -49,555 | -24.01% | skill 1/1; base 1/1 |
| codex | v2-fixture-policy-termination | 193,513 | 202,099 | -8,586 | -4.25% | skill 1/1; base 1/1 |
| codex | v2-fixture-preemption-boundary | 66,414 | 254,145 | -187,731 | -73.87% | skill 1/1; base 1/1 |
| codex | v2-fixture-recurring-oom | 215,189 | 209,197 | +5,992 | +2.86% | skill 1/1; base 1/1 |
| codex | v2-fixture-running-crashloopbackoff | 194,227 | 182,856 | +11,371 | +6.22% | skill 1/1; base 1/1 |
| codex | v2-fixture-unexplained-sigkill | 120,842 | 104,074 | +16,768 | +16.11% | skill 1/1; base 1/1 |
| codex | v2-fixture-unknown | 192,781 | 168,126 | +24,655 | +14.66% | skill 1/1; base 1/1 |
| codex | v2-fixture-untrusted-event | 175,010 | 169,042 | +5,968 | +3.53% | skill 1/1; base 1/1 |
| codex | v2-recorded-application-crash | 103,603 | 212,663 | -109,060 | -51.28% | skill 1/1; base 1/1 |
| codex | v2-recorded-image-pull-failure | 140,873 | 173,259 | -32,386 | -18.69% | skill 1/1; base 1/1 |
| codex | v2-recorded-oom | 122,932 | 90,437 | +32,495 | +35.93% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,590,225 | 6,329,585 | +260,640 | +4.12% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 9 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/runai-diagnose-failed-workload/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/runai-diagnose-failed-workload/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (321 chars, recommend 50-150) (`skills/runai-diagnose-failed-workload/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/runai-diagnose-failed-workload/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/runai-diagnose-failed-workload/SKILL.md`)
- 4 additional finding(s) are available in the full evaluation artifacts.

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
