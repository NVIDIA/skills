# Skill Benchmark: nvflare-autofl

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvflare-autofl`
- Evaluation date: 2026-09-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:47bb5f4f9de5f1a56ecb009c45deee42349c7c12a196a16f05072e36e82b36ec` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 76.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 100.0% → 100.0% (±0.0 points) |
| Correctness | Not available | 46.2% → 77.5% (+31.3 points) |
| Discoverability | Not available | 63.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 30.7% → 50.0% (+19.3 points) |
| Efficiency | Not available | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,450,847 | 7,933,373 | N/A | N/A | skill 7/7; base 12/21 |
| claude-code | autofl-diversify-after-family-repeat | 339,356 | 228,515 | +110,841 | +48.50% | skill 1/1; base 1/1 |
| claude-code | autofl-global-negative-web-app | 1,093,673 | 1,201,596 | -107,923 | -8.98% | skill 1/1; base 1/1 |
| claude-code | autofl-literature-batch-before-tuning | 316,181 | 468,304 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | autofl-natural-phrasing-low-accuracy | 208,443 | 575,259 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | autofl-negative-diagnose-job | 123,230 | 283,949 | -160,719 | -56.60% | skill 1/1; base 1/1 |
| claude-code | autofl-negative-pytorch-conversion | 121,450 | 212,766 | -91,316 | -42.92% | skill 1/1; base 1/1 |
| claude-code | autofl-optimize-existing-job | 248,514 | 4,962,984 | N/A | N/A | skill 1/1; base 2/2 |
| codex | All cases | 996,161 | 1,643,074 | N/A | N/A | skill 8/8; base 13/13 |
| codex | autofl-diversify-after-family-repeat | 153,489 | 82,457 | +71,032 | +86.14% | skill 1/1; base 1/1 |
| codex | autofl-global-negative-web-app | 314,206 | 347,084 | -32,878 | -9.47% | skill 1/1; base 1/1 |
| codex | autofl-literature-batch-before-tuning | 105,477 | 59,693 | +45,784 | +76.70% | skill 1/1; base 1/1 |
| codex | autofl-natural-phrasing-low-accuracy | 80,962 | 327,004 | N/A | N/A | skill 1/1; base 3/3 |
| codex | autofl-negative-diagnose-job | 208,327 | 329,168 | N/A | N/A | skill 2/2; base 3/3 |
| codex | autofl-negative-pytorch-conversion | 70,115 | 83,605 | -13,490 | -16.14% | skill 1/1; base 1/1 |
| codex | autofl-optimize-existing-job | 63,585 | 414,063 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 3,447,008 | 9,576,447 | N/A | N/A | skill 15/15; base 25/34 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCRIPT_LINT/deep_nesting: campaign_guard.py has deeply nested code (depth 7, max 6) (`skills/nvflare-autofl/scripts/campaign_guard.py`)
- **MEDIUM** SCRIPT_LINT/deep_nesting: run_job_campaign.py has deeply nested code (depth 9, max 6) (`skills/nvflare-autofl/scripts/run_job_campaign.py`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill declares no explicit tool scope ('permissions' or 'allowed-tools') in its metadata, yet the skill content clea (`SKILL.md:1`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0, (`scripts/run_job_campaign.py:781`)
- **MEDIUM** SECURITY/Unbounded Resource Access (EA4): Excessive Agency: timeout=0 (`scripts/run_job_campaign.py:810`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
