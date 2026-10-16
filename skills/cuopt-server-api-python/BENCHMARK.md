# Skill Benchmark: cuopt-server-api-python

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuopt-server-api-python`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (8 positive)
- Dataset digest: `sha256:c54a53a22db535b2e2101bfcd2c43a8cdfef342f5c166ba343b8fdba517c741d` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 77.5% → 100.0% (+22.5 points) | 82.5% → 100.0% (+17.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 58.1% → 94.3% (+36.2 points) | 64.1% → 90.5% (+26.4 points) |
| Efficiency | 97.1% — baseline ran, but no comparable score was available; uplift unavailable | 95.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 885,077 | 597,044 | +288,033 | +48.24% | skill 8/8; base 8/8 |
| claude-code | srv-py-eval-001-rest-routing-workflow | 61,314 | 29,004 | +32,310 | +111.40% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-002-rest-lp-payload-fields | 158,831 | 29,008 | +129,823 | +447.54% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-003-rest-milp-vs-lp | 159,973 | 28,976 | +130,997 | +452.09% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-004-qp-not-supported-over-rest | 60,627 | 59,196 | +1,431 | +2.42% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-005-debug-422-field-names | 60,837 | 29,007 | +31,830 | +109.73% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-006-docker-deployment | 60,693 | 28,988 | +31,705 | +109.37% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-007-client-hardening | 162,999 | 28,980 | +134,019 | +462.45% | skill 1/1; base 1/1 |
| claude-code | srv-py-eval-008-runnable-assets | 159,803 | 363,885 | -204,082 | -56.08% | skill 1/1; base 1/1 |
| codex | All cases | 388,941 | 481,460 | -92,519 | -19.22% | skill 8/8; base 8/8 |
| codex | srv-py-eval-001-rest-routing-workflow | 45,413 | 92,911 | -47,498 | -51.12% | skill 1/1; base 1/1 |
| codex | srv-py-eval-002-rest-lp-payload-fields | 45,088 | 46,097 | -1,009 | -2.19% | skill 1/1; base 1/1 |
| codex | srv-py-eval-003-rest-milp-vs-lp | 45,998 | 30,983 | +15,015 | +48.46% | skill 1/1; base 1/1 |
| codex | srv-py-eval-004-qp-not-supported-over-rest | 28,544 | 46,936 | -18,392 | -39.19% | skill 1/1; base 1/1 |
| codex | srv-py-eval-005-debug-422-field-names | 28,579 | 61,511 | -32,932 | -53.54% | skill 1/1; base 1/1 |
| codex | srv-py-eval-006-docker-deployment | 35,800 | 32,176 | +3,624 | +11.26% | skill 1/1; base 1/1 |
| codex | srv-py-eval-007-client-hardening | 79,361 | 27,750 | +51,611 | +185.99% | skill 1/1; base 1/1 |
| codex | srv-py-eval-008-runnable-assets | 80,158 | 143,096 | -62,938 | -43.98% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,274,018 | 1,078,504 | +195,514 | +18.13% | skill 16/16; base 16/16 |

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
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill demonstrates network-capable code (HTTP requests via `requests`, Docker networking) but declares no tool scope (`SKILL.md:1`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: The Docker run command uses `nvidia/cuopt:latest-cu13` (a floating/mutable tag). The skill itself acknowledges this risk (`SKILL.md:55`)
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
