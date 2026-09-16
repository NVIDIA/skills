# Skill Benchmark: cuopt-server-api-python

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuopt-server-api-python`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (8 positive)
- Dataset digest: `sha256:89ad304517df359a8c33a1f67da3ad9df8aeae5ec234d2adbed1593bf49c4ebc` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 94.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 90.9% → 100.0% (+9.1 points) |
| Correctness | 68.0% → 100.0% (+32.0 points) | 67.3% → 100.0% (+32.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.8% → 95.2% (+47.4 points) | 48.5% → 91.2% (+42.7 points) |
| Efficiency | 96.1% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 990,844 | 2,338,962 | N/A | N/A | skill 8/8; base 10/10 |
| claude-code | srv-py-eval-001-rest-routing-workflow | 61,737 | 31,761 | +29,976 | +94.38% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-002-rest-lp-payload-fields | 159,833 | 224,737 | -64,904 | -28.88% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-003-rest-milp-vs-lp | 258,072 | 30,448 | +227,624 | +747.58% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-004-qp-not-supported-over-rest | 61,254 | 366,123 | -304,869 | -83.27% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-005-debug-422-field-names | 61,453 | 30,853 | +30,600 | +99.18% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-006-docker-deployment | 61,610 | 155,592 | -93,982 | -60.40% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-007-client-hardening | 197,723 | 30,615 | +167,108 | +545.84% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-008-runnable-assets | 129,162 | 1,468,833 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 378,610 | 1,697,844 | N/A | N/A | skill 8/8; base 11/11 |
| codex | srv-py-eval-001-rest-routing-workflow | 45,986 | 193,794 | -147,808 | -76.27% | skill 1/1; base 1/1 |
| codex | srv-py-eval-002-rest-lp-payload-fields | 45,220 | 47,739 | -2,519 | -5.28% | skill 1/1; base 1/1 |
| codex | srv-py-eval-003-rest-milp-vs-lp | 46,076 | 42,547 | +3,529 | +8.29% | skill 1/1; base 1/1 |
| codex | srv-py-eval-004-qp-not-supported-over-rest | 78,889 | 576,772 | N/A | N/A | skill 1/1; base 2/2 |
| codex | srv-py-eval-005-debug-422-field-names | 28,843 | 118,005 | -89,162 | -75.56% | skill 1/1; base 1/1 |
| codex | srv-py-eval-006-docker-deployment | 41,121 | 25,518 | +15,603 | +61.15% | skill 1/1; base 1/1 |
| codex | srv-py-eval-007-client-hardening | 29,126 | 25,289 | +3,837 | +15.17% | skill 1/1; base 1/1 |
| codex | srv-py-eval-008-runnable-assets | 63,349 | 668,180 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 1,369,454 | 4,036,806 | N/A | N/A | skill 16/16; base 21/21 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cuopt-server-api-python/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: Without declared permissions the skill's intent is opaque and cannot be validated. (`SKILL.md:1`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: The Docker run command references `nvidia/cuopt:latest-cu13` (a mutable floating tag) without pinning to an immutable di (`SKILL.md:55`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: requests.post(f"{SERVER}/cuopt/request", json= (`SKILL.md:88`)
- **MEDIUM** SECURITY/Tainted flow: 'req_id' from requests.post (line 81, network input) → requests.get (network output) (TT2): Data Flow:         response = requests.get(
            f"{server}/cuopt/solution/{req_id}",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        ) (`assets/lp_basic/client.py:88`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
