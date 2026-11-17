# Skill Benchmark: compileiq-bootstrap

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compileiq-bootstrap`
- Validation status: `passed`
- Evaluation date: 2026-11-14
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:d47065e1d904ad38bab2c34451ead408ce01931003282433cbb592165dbd715c` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| compileiq-bootstrap | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-bootstrap | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-bootstrap | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-bootstrap | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-bootstrap | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `compileiq-bootstrap`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 86.9% — baseline ran, but no comparable score was available; uplift unavailable | 86.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 93.8% → 93.8% (±0.0 points) |
| Correctness | 72.5% → 100.0% (+27.5 points) | 95.0% → 100.0% (+5.0 points) |
| Discoverability | 70.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 58.8% → 83.4% (+24.6 points) | 76.3% → 83.1% (+6.8 points) |
| Efficiency | 80.4% — baseline ran, but no comparable score was available; uplift unavailable | 63.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.5% → 96.5% (-1.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.2% → 87.5% (+0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,129,566 | 1,218,486 | -88,920 | -7.30% | skill 8/8; base 8/8 |
| claude-code | compileiq-bootstrap-airgap-mirror-manifest | 193,976 | 146,141 | +47,835 | +32.73% | skill 1/1; base 1/1 |
| claude-code | compileiq-bootstrap-check-fresh-container | 161,236 | 311,347 | -150,111 | -48.21% | skill 1/1; base 1/1 |
| claude-code | compileiq-bootstrap-check-log-no-network | 145,716 | 71,296 | +74,420 | +104.38% | skill 1/1; base 1/1 |
| claude-code | compileiq-bootstrap-first-eval-hang-no-root | 151,631 | 146,581 | +5,050 | +3.45% | skill 1/1; base 1/1 |
| claude-code | compileiq-bootstrap-negative-csv-dedupe | 179,383 | 178,728 | +655 | +0.37% | skill 1/1; base 1/1 |
| claude-code | compileiq-bootstrap-negative-no-kernel-image | 36,031 | 36,114 | -83 | -0.23% | skill 1/1; base 1/1 |
| claude-code | compileiq-bootstrap-ptxas-version-mismatch | 73,410 | 108,705 | -35,295 | -32.47% | skill 1/1; base 1/1 |
| claude-code | compileiq-bootstrap-search-space-source-fork | 188,183 | 219,574 | -31,391 | -14.30% | skill 1/1; base 1/1 |
| codex | All cases | 551,940 | 437,030 | +114,910 | +26.29% | skill 8/8; base 8/8 |
| codex | compileiq-bootstrap-airgap-mirror-manifest | 115,166 | 45,733 | +69,433 | +151.82% | skill 1/1; base 1/1 |
| codex | compileiq-bootstrap-check-fresh-container | 172,232 | 129,323 | +42,909 | +33.18% | skill 1/1; base 1/1 |
| codex | compileiq-bootstrap-check-log-no-network | 33,490 | 36,157 | -2,667 | -7.38% | skill 1/1; base 1/1 |
| codex | compileiq-bootstrap-first-eval-hang-no-root | 43,592 | 35,309 | +8,283 | +23.46% | skill 1/1; base 1/1 |
| codex | compileiq-bootstrap-negative-csv-dedupe | 87,365 | 56,261 | +31,104 | +55.29% | skill 1/1; base 1/1 |
| codex | compileiq-bootstrap-negative-no-kernel-image | 25,970 | 26,488 | -518 | -1.96% | skill 1/1; base 1/1 |
| codex | compileiq-bootstrap-ptxas-version-mismatch | 32,618 | 38,086 | -5,468 | -14.36% | skill 1/1; base 1/1 |
| codex | compileiq-bootstrap-search-space-source-fork | 41,507 | 69,673 | -28,166 | -40.43% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,681,506 | 1,655,516 | +25,990 | +1.57% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/compileiq-bootstrap/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/compileiq-bootstrap/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/compileiq-bootstrap/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/compileiq-bootstrap/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (515 chars, recommend 50-150) (`skills/compileiq-bootstrap/SKILL.md`)
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
