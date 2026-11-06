# Skill Benchmark: isaac-humanoid-cloud-control

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `isaac-humanoid-cloud-control`
- Validation status: `passed`
- Evaluation date: 2026-11-05
- Evaluator version: `1.6.8`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 15 evaluation tasks (15 positive)
- Dataset digest: `sha256:bbb620a371ba91241218df5a85d2791daf339409c06a2e021e1fd2f5ff2f1c09` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| isaac-humanoid-cloud-control | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| isaac-humanoid-cloud-control | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| isaac-humanoid-cloud-control | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| isaac-humanoid-cloud-control | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| isaac-humanoid-cloud-control | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `isaac-humanoid-cloud-control`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 84.2% — baseline ran, but no comparable score was available; uplift unavailable | 75.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 93.3% (-6.7 points) | 66.7% → 76.7% (+10.0 points) |
| Correctness | 29.3% → 88.0% (+58.7 points) | 22.7% → 81.3% (+58.6 points) |
| Discoverability | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 26.2% → 53.0% (+26.8 points) | 28.3% → 43.0% (+14.7 points) |
| Efficiency | 88.1% — baseline ran, but no comparable score was available; uplift unavailable | 85.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.2% → 91.5% (-5.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.2% → 87.4% (-3.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,518,724 | 2,057,321 | +1,461,403 | +71.03% | skill 15/15; base 15/15 |
| claude-code | isaac-humanoid-cloud-control-001 | 282,355 | 144,064 | +138,291 | +95.99% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-002 | 251,801 | 70,851 | +180,950 | +255.40% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-003 | 231,955 | 179,148 | +52,807 | +29.48% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-004 | 190,403 | 108,244 | +82,159 | +75.90% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-005 | 247,746 | 107,761 | +139,985 | +129.90% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-006 | 305,425 | 183,493 | +121,932 | +66.45% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-007 | 168,259 | 107,534 | +60,725 | +56.47% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-008 | 276,298 | 144,235 | +132,063 | +91.56% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-009 | 267,752 | 147,257 | +120,495 | +81.83% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-010 | 170,894 | 145,593 | +25,301 | +17.38% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-011 | 359,293 | 144,591 | +214,702 | +148.49% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-012 | 308,950 | 180,940 | +128,010 | +70.75% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-013 | 166,367 | 144,218 | +22,149 | +15.36% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-release-retry | 125,181 | 106,638 | +18,543 | +17.39% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-security-boundary | 166,045 | 142,754 | +23,291 | +16.32% | skill 1/1; base 1/1 |
| codex | All cases | 2,496,789 | 1,336,075 | +1,160,714 | +86.87% | skill 15/15; base 15/15 |
| codex | isaac-humanoid-cloud-control-001 | 202,259 | 266,129 | -63,870 | -24.00% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-002 | 121,580 | 13,206 | +108,374 | +820.64% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-003 | 204,733 | 73,697 | +131,036 | +177.80% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-004 | 51,531 | 186,627 | -135,096 | -72.39% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-005 | 149,057 | 54,434 | +94,623 | +173.83% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-006 | 239,101 | 133,594 | +105,507 | +78.98% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-007 | 174,510 | 75,290 | +99,220 | +131.78% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-008 | 164,756 | 49,750 | +115,006 | +231.17% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-009 | 149,390 | 42,304 | +107,086 | +253.13% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-010 | 235,933 | 61,672 | +174,261 | +282.56% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-011 | 308,196 | 110,258 | +197,938 | +179.52% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-012 | 169,981 | 56,117 | +113,864 | +202.90% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-013 | 154,961 | 62,722 | +92,239 | +147.06% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-release-retry | 48,054 | 55,394 | -7,340 | -13.25% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-security-boundary | 122,747 | 94,881 | +27,866 | +29.37% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,015,513 | 3,393,396 | +2,622,117 | +77.27% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 64 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/isaac-orchestration/isaac-humanoid-cloud-control`)
- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/isaac-orchestration/isaac-humanoid-cloud-control`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in dependencies.md (`skills/isaac-orchestration/isaac-humanoid-cloud-control/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/isaac-orchestration/isaac-humanoid-cloud-control`)
- **MEDIUM** SCRIPT_LINT/flat_script: simulation_identity.py has no function definitions (flat script) (`skills/isaac-orchestration/isaac-humanoid-cloud-control/scripts/simulation_identity.py`)
- 61 additional finding(s) are available in the full evaluation artifacts.

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
