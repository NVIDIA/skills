# Skill Benchmark: vss-build-vision-ai

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `vss-build-vision-ai`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 11 evaluation tasks (10 positive, 1 negative)
- Dataset digest: `sha256:503e7e991283a099b167cb9f095067f9045f52afd152fef5ee094b8dbc4eff29` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 67.9% — baseline ran, but no comparable score was available; uplift unavailable | 60.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 72.7% → 59.1% (-13.6 points) | 27.3% → 63.6% (+36.3 points) |
| Correctness | 27.3% → 67.3% (+40.0 points) | 47.3% → 50.9% (+3.6 points) |
| Discoverability | 99.3% — baseline ran, but no comparable score was available; uplift unavailable | 77.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 3.8% → 30.2% (+26.4 points) | 15.9% → 28.7% (+12.8 points) |
| Efficiency | 83.6% — baseline ran, but no comparable score was available; uplift unavailable | 81.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 8,321,750 | 12,636,150 | -4,314,400 | -34.14% | skill 11/11; base 11/11 |
| claude-code | build-alerts-gb300 | 826,168 | 187,896 | +638,272 | +339.69% | skill 1/1; base 1/1 |
| claude-code | build-base-default-tag | 634,188 | 248,403 | +385,785 | +155.31% | skill 1/1; base 1/1 |
| claude-code | build-base-gb300 | 986,685 | 1,204,477 | -217,792 | -18.08% | skill 1/1; base 1/1 |
| claude-code | build-base-gb300-release-tag | 800,127 | 294,671 | +505,456 | +171.53% | skill 1/1; base 1/1 |
| claude-code | build-base-harness-default-ref | 476,062 | 150,874 | +325,188 | +215.54% | skill 1/1; base 1/1 |
| claude-code | build-base-harness-ref | 644,420 | 150,311 | +494,109 | +328.72% | skill 1/1; base 1/1 |
| claude-code | build-base-release-tag | 521,452 | 249,897 | +271,555 | +108.67% | skill 1/1; base 1/1 |
| claude-code | build-lvs-gb300 | 927,521 | 1,873,278 | -945,757 | -50.49% | skill 1/1; base 1/1 |
| claude-code | build-search-gb300 | 1,452,301 | 151,341 | +1,300,960 | +859.62% | skill 1/1; base 1/1 |
| claude-code | build-search-profile | 782,065 | 7,878,376 | -7,096,311 | -90.07% | skill 1/1; base 1/1 |
| claude-code | search-running-archive | 270,761 | 246,626 | +24,135 | +9.79% | skill 1/1; base 1/1 |
| codex | All cases | 6,679,537 | 8,588,008 | -1,908,471 | -22.22% | skill 11/11; base 11/11 |
| codex | build-alerts-gb300 | 544,823 | 576,166 | -31,343 | -5.44% | skill 1/1; base 1/1 |
| codex | build-base-default-tag | 167,910 | 511,025 | -343,115 | -67.14% | skill 1/1; base 1/1 |
| codex | build-base-gb300 | 956,678 | 1,007,214 | -50,536 | -5.02% | skill 1/1; base 1/1 |
| codex | build-base-gb300-release-tag | 1,465,179 | 2,150,885 | -685,706 | -31.88% | skill 1/1; base 1/1 |
| codex | build-base-harness-default-ref | 310,855 | 420,137 | -109,282 | -26.01% | skill 1/1; base 1/1 |
| codex | build-base-harness-ref | 585,652 | 129,582 | +456,070 | +351.95% | skill 1/1; base 1/1 |
| codex | build-base-release-tag | 235,224 | 2,120,677 | -1,885,453 | -88.91% | skill 1/1; base 1/1 |
| codex | build-lvs-gb300 | 280,793 | 908,544 | -627,751 | -69.09% | skill 1/1; base 1/1 |
| codex | build-search-gb300 | 298,373 | 98,419 | +199,954 | +203.17% | skill 1/1; base 1/1 |
| codex | build-search-profile | 1,604,123 | 609,976 | +994,147 | +162.98% | skill 1/1; base 1/1 |
| codex | search-running-archive | 229,927 | 55,383 | +174,544 | +315.16% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 15,001,287 | 21,224,158 | -6,222,871 | -29.32% | skill 22/22; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 289 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/chunk_count_limit: Tier 2 produced more than 512 content chunks. (`references/services/sop/build-ds-sop.md`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/troubleshooting.md:130`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/prerequisites.md:213`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/prerequisites.md:220`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/prerequisites.md:221`)
- 285 additional finding(s) are available in the full evaluation artifacts.

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
