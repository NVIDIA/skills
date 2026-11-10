# Skill Benchmark: cuopt-numerical-optimization-api

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuopt-numerical-optimization-api`
- Validation status: `passed`
- Evaluation date: 2026-11-10
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (9 positive)
- Dataset digest: `sha256:f385c69ce235035e8e5cbfd91a8d3245b75473f0bf0d694451e988e439f6e9e1` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cuopt-numerical-optimization-api | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `cuopt-numerical-optimization-api`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 95.2% — baseline ran, but no comparable score was available; uplift unavailable | 86.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 86.7% → 100.0% (+13.3 points) | 91.1% → 95.6% (+4.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 92.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 65.1% → 91.4% (+26.3 points) | 56.2% → 74.1% (+17.9 points) |
| Efficiency | 84.6% — baseline ran, but no comparable score was available; uplift unavailable | 70.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.7% → 97.0% (-1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.3% → 88.3% (±0.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,121,154 | 501,132 | +620,022 | +123.72% | skill 9/9; base 9/9 |
| claude-code | numopt-c-eval-001-milp-api-call-sequence | 112,671 | 71,924 | +40,747 | +56.65% | skill 1/1; base 1/1 |
| claude-code | numopt-c-eval-002-parameter-function-wrong-name | 149,126 | 71,816 | +77,310 | +107.65% | skill 1/1; base 1/1 |
| claude-code | numopt-c-eval-003-csr-constraint-matrix | 150,215 | 35,779 | +114,436 | +319.84% | skill 1/1; base 1/1 |
| claude-code | numopt-c-eval-004-qp-restrictions | 111,370 | 71,217 | +40,153 | +56.38% | skill 1/1; base 1/1 |
| claude-code | numopt-cli-eval-001-mps-sections-and-cli-command | 112,014 | 71,445 | +40,569 | +56.78% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-001-lp-api-call-sequence | 111,558 | 35,759 | +75,799 | +211.97% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-002-status-case-sensitivity | 111,038 | 35,707 | +75,331 | +210.97% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-003-integer-vs-continuous-workers | 110,549 | 71,226 | +39,323 | +55.21% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-004-qp-maximize-workaround | 152,613 | 36,259 | +116,354 | +320.90% | skill 1/1; base 1/1 |
| codex | All cases | 535,086 | 242,605 | +292,481 | +120.56% | skill 9/9; base 9/9 |
| codex | numopt-c-eval-001-milp-api-call-sequence | 52,596 | 25,850 | +26,746 | +103.47% | skill 1/1; base 1/1 |
| codex | numopt-c-eval-002-parameter-function-wrong-name | 52,572 | 21,721 | +30,851 | +142.03% | skill 1/1; base 1/1 |
| codex | numopt-c-eval-003-csr-constraint-matrix | 48,334 | 17,912 | +30,422 | +169.84% | skill 1/1; base 1/1 |
| codex | numopt-c-eval-004-qp-restrictions | 73,110 | 47,107 | +26,003 | +55.20% | skill 1/1; base 1/1 |
| codex | numopt-cli-eval-001-mps-sections-and-cli-command | 52,456 | 25,905 | +26,551 | +102.49% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-001-lp-api-call-sequence | 48,187 | 26,163 | +22,024 | +84.18% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-002-status-case-sensitivity | 66,940 | 21,741 | +45,199 | +207.90% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-003-integer-vs-continuous-workers | 62,236 | 17,540 | +44,696 | +254.82% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-004-qp-maximize-workaround | 78,655 | 38,666 | +39,989 | +103.42% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,656,240 | 743,737 | +912,503 | +122.69% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across assets/python/mps_solver/README.md and references/cli_api.md:
  "## MPS File Format" in assets/python/mps_solver/README.md (lines 15-38)
  vs "## MPS Format (required sections, in order)" in references/cli_api.md (lines 44-54) (`assets/python/mps_solver/README.md:15`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in c_api.md (`skills/cuopt-numerical-optimization-api/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cuopt-numerical-optimization-api/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cuopt-numerical-optimization-api/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cuopt-numerical-optimization-api/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
