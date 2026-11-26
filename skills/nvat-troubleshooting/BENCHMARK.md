# Skill Benchmark: nvat-troubleshooting

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvat-troubleshooting`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 15 evaluation tasks (12 positive, 3 negative)
- Dataset digest: `sha256:1b3402000c3036b9e92b4adb7263edcb219d4f76029082126af37622fbfa2d22` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvat-troubleshooting | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-troubleshooting | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-troubleshooting | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-troubleshooting | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-troubleshooting | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nvat-troubleshooting`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 97.3% — baseline ran, but no comparable score was available; uplift unavailable | 88.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 93.3% → 96.7% (+3.4 points) |
| Correctness | 73.3% → 100.0% (+26.7 points) | 94.7% → 100.0% (+5.3 points) |
| Discoverability | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 62.7% → 100.0% (+37.3 points) | 74.2% → 83.4% (+9.2 points) |
| Efficiency | 92.0% — baseline ran, but no comparable score was available; uplift unavailable | 71.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 96.5% (-1.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.0% → 87.5% (+4.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,824,096 | 1,306,027 | +518,069 | +39.67% | skill 15/15; base 15/15 |
| claude-code | nvat-troubleshoot-bug-report-bundle | 149,900 | 148,532 | +1,368 | +0.92% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-digest-hex | 196,749 | 36,271 | +160,478 | +442.44% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-ear-only | 161,405 | 74,723 | +86,682 | +116.00% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-hopper-format-mismatch | 110,619 | 107,677 | +2,942 | +2.73% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-loader | 110,653 | 71,098 | +39,555 | +55.63% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-policy-versus-error | 112,797 | 107,250 | +5,547 | +5.17% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-reference-stages | 152,460 | 144,478 | +7,982 | +5.52% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-tcg-mismatch-records | 115,226 | 36,994 | +78,232 | +211.47% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-trace-replay | 117,246 | 73,037 | +44,209 | +60.53% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshoot-unknown-oid | 113,508 | 36,013 | +77,495 | +215.19% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshooting-fsp-hint-versus-mismatch | 154,288 | 109,984 | +44,304 | +40.28% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshooting-fsp-required-coev | 112,038 | 143,471 | -31,433 | -21.91% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshooting-negative-cuda-kernel | 108,880 | 108,236 | +644 | +0.59% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshooting-negative-iso-digest | 36,300 | 36,194 | +106 | +0.29% | skill 1/1; base 1/1 |
| claude-code | nvat-troubleshooting-negative-web-tls | 72,027 | 72,069 | -42 | -0.06% | skill 1/1; base 1/1 |
| codex | All cases | 1,037,937 | 1,118,982 | -81,045 | -7.24% | skill 15/15; base 15/15 |
| codex | nvat-troubleshoot-bug-report-bundle | 150,146 | 58,807 | +91,339 | +155.32% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-digest-hex | 86,250 | 22,037 | +64,213 | +291.39% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-ear-only | 70,422 | 54,642 | +15,780 | +28.88% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-hopper-format-mismatch | 49,138 | 65,201 | -16,063 | -24.64% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-loader | 78,427 | 50,917 | +27,510 | +54.03% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-policy-versus-error | 52,694 | 70,623 | -17,929 | -25.39% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-reference-stages | 77,302 | 85,144 | -7,842 | -9.21% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-tcg-mismatch-records | 66,133 | 104,258 | -38,125 | -36.57% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-trace-replay | 53,464 | 57,848 | -4,384 | -7.58% | skill 1/1; base 1/1 |
| codex | nvat-troubleshoot-unknown-oid | 48,222 | 21,777 | +26,445 | +121.44% | skill 1/1; base 1/1 |
| codex | nvat-troubleshooting-fsp-hint-versus-mismatch | 117,360 | 345,650 | -228,290 | -66.05% | skill 1/1; base 1/1 |
| codex | nvat-troubleshooting-fsp-required-coev | 67,976 | 79,207 | -11,231 | -14.18% | skill 1/1; base 1/1 |
| codex | nvat-troubleshooting-negative-cuda-kernel | 58,092 | 54,084 | +4,008 | +7.41% | skill 1/1; base 1/1 |
| codex | nvat-troubleshooting-negative-iso-digest | 31,049 | 26,561 | +4,488 | +16.90% | skill 1/1; base 1/1 |
| codex | nvat-troubleshooting-negative-web-tls | 31,262 | 22,226 | +9,036 | +40.66% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,862,033 | 2,425,009 | +437,024 | +18.02% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nvat-troubleshooting/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nvat-troubleshooting/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in bug-reports.md (`skills/nvat-troubleshooting/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvat-troubleshooting/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvat-troubleshooting/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

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
