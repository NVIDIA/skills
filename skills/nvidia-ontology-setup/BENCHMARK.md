# Skill Benchmark: nvidia-ontology-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-setup`
- Evaluation date: 2026-10-08
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 10 evaluation tasks (10 positive)
- Dataset digest: `sha256:dcbcf0848aefb19cd5b010e9a0d69ed8920e61dc3fdba560ca0f88cd48a8da96` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.6% — baseline ran, but no comparable score was available; uplift unavailable | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 90.0% → 100.0% (+10.0 points) | 60.0% → 85.0% (+25.0 points) |
| Correctness | 36.0% → 100.0% (+64.0 points) | 68.0% → 100.0% (+32.0 points) |
| Discoverability | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.8% → 95.5% (+52.7 points) | 54.5% → 88.5% (+34.0 points) |
| Efficiency | 88.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,266,901 | 2,038,854 | -771,953 | -37.86% | skill 10/10; base 10/10 |
| claude-code | setup-eval-001-underspecified | 99,720 | 29,740 | +69,980 | +235.31% | skill 1/1; base 1/1 |
| claude-code | setup-eval-002-use-existing-scripts | 203,860 | 119,696 | +84,164 | +70.31% | skill 1/1; base 1/1 |
| claude-code | setup-eval-003-auth-secret-cookies | 99,782 | 472,431 | -372,649 | -78.88% | skill 1/1; base 1/1 |
| claude-code | setup-eval-004-postgres-port-host-only | 63,562 | 120,299 | -56,737 | -47.16% | skill 1/1; base 1/1 |
| claude-code | setup-eval-005-mcp-already-up | 137,280 | 387,881 | -250,601 | -64.61% | skill 1/1; base 1/1 |
| claude-code | setup-eval-006-vault-partial | 93,388 | 91,617 | +1,771 | +1.93% | skill 1/1; base 1/1 |
| claude-code | setup-eval-007-embed-triplet-mismatch | 100,692 | 149,507 | -48,815 | -32.65% | skill 1/1; base 1/1 |
| claude-code | setup-eval-008-connect-source-workflow | 144,807 | 322,080 | -177,273 | -55.04% | skill 1/1; base 1/1 |
| claude-code | setup-eval-009-env-vs-ui-connections | 64,877 | 192,173 | -127,296 | -66.24% | skill 1/1; base 1/1 |
| claude-code | setup-eval-010-helm-install | 258,933 | 153,430 | +105,503 | +68.76% | skill 1/1; base 1/1 |
| codex | All cases | 1,610,140 | 2,515,404 | -905,264 | -35.99% | skill 10/10; base 10/10 |
| codex | setup-eval-001-underspecified | 45,064 | 355,321 | -310,257 | -87.32% | skill 1/1; base 1/1 |
| codex | setup-eval-002-use-existing-scripts | 47,468 | 65,756 | -18,288 | -27.81% | skill 1/1; base 1/1 |
| codex | setup-eval-003-auth-secret-cookies | 66,752 | 156,942 | -90,190 | -57.47% | skill 1/1; base 1/1 |
| codex | setup-eval-004-postgres-port-host-only | 63,392 | 83,348 | -19,956 | -23.94% | skill 1/1; base 1/1 |
| codex | setup-eval-005-mcp-already-up | 206,529 | 13,377 | +193,152 | +1443.91% | skill 1/1; base 1/1 |
| codex | setup-eval-006-vault-partial | 125,282 | 68,955 | +56,327 | +81.69% | skill 1/1; base 1/1 |
| codex | setup-eval-007-embed-triplet-mismatch | 185,401 | 127,201 | +58,200 | +45.75% | skill 1/1; base 1/1 |
| codex | setup-eval-008-connect-source-workflow | 557,521 | 1,028,990 | -471,469 | -45.82% | skill 1/1; base 1/1 |
| codex | setup-eval-009-env-vs-ui-connections | 125,220 | 213,094 | -87,874 | -41.24% | skill 1/1; base 1/1 |
| codex | setup-eval-010-helm-install | 187,511 | 402,420 | -214,909 | -53.40% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,877,041 | 4,554,258 | -1,677,217 | -36.83% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/nvidia-ontology-setup/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: Never ask the user (`SKILL.md:62`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -s -o /dev/null -w '%{http_code}\n' "$AUTO_ONTOLOGY_API_URL/.well-known/oauth-authorization-server"
```

`200` is required. Anything else: run the frontend from the checkout
(`pnpm dev`) and poin (`SKILL.md:200`)
- **LOW** QUALITY/quality_discoverability: Skill uses exclusivity language that conflicts with composability (`skills/nvidia-ontology-setup/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/nvidia-ontology-setup/SKILL.md`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
