# Skill Benchmark: cudnn-compile-and-serialize

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cudnn-compile-and-serialize`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (8 positive, 2 negative)
- Dataset digest: `sha256:857a292a1ec4966f7b9f33223a18dc874378a31a1824a17ec975e90792335902` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cudnn-compile-and-serialize | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-compile-and-serialize | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-compile-and-serialize | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-compile-and-serialize | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-compile-and-serialize | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `cudnn-compile-and-serialize`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | 82.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 75.0% (-25.0 points) |
| Correctness | 88.0% → 92.0% (+4.0 points) | 100.0% → 98.0% (-2.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 87.1% → 86.7% (-0.4 points) | 77.9% → 88.3% (+10.4 points) |
| Efficiency | 90.5% — baseline ran, but no comparable score was available; uplift unavailable | 65.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.0% → 93.5% (-1.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.5% → 84.8% (+0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 980,592 | 522,506 | +458,086 | +87.67% | skill 10/10; base 10/10 |
| claude-code | cudnn-compile-and-serialize-binary-round-trip | 117,559 | 74,636 | +42,923 | +57.51% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-cache-failure-policy | 120,385 | 76,573 | +43,812 | +57.22% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-cross-process-bindings | 159,074 | 39,598 | +119,476 | +301.72% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-deviceless-deployment | 77,065 | 73,925 | +3,140 | +4.25% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-heuristics-versus-tuning | 117,787 | 38,138 | +79,649 | +208.84% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-negative-dataloader-workers | 37,005 | 37,481 | -476 | -1.27% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-negative-python-install | 35,870 | 35,887 | -17 | -0.05% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-parallel-build-race | 161,752 | 72,671 | +89,081 | +122.58% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-status-preserving-fragment | 77,094 | 36,903 | +40,191 | +108.91% | skill 1/1; base 1/1 |
| claude-code | cudnn-compile-and-serialize-version-and-skip-evidence | 77,001 | 36,694 | +40,307 | +109.85% | skill 1/1; base 1/1 |
| codex | All cases | 897,872 | 410,930 | +486,942 | +118.50% | skill 10/10; base 10/10 |
| codex | cudnn-compile-and-serialize-binary-round-trip | 279,562 | 39,387 | +240,175 | +609.78% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-cache-failure-policy | 88,237 | 51,762 | +36,475 | +70.47% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-cross-process-bindings | 141,740 | 47,577 | +94,163 | +197.92% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-deviceless-deployment | 36,101 | 55,580 | -19,479 | -35.05% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-heuristics-versus-tuning | 118,195 | 47,299 | +70,896 | +149.89% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-negative-dataloader-workers | 31,200 | 30,970 | +230 | +0.74% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-negative-python-install | 42,156 | 34,391 | +7,765 | +22.58% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-parallel-build-race | 51,304 | 57,847 | -6,543 | -11.31% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-status-preserving-fragment | 30,233 | 18,273 | +11,960 | +65.45% | skill 1/1; base 1/1 |
| codex | cudnn-compile-and-serialize-version-and-skip-evidence | 79,144 | 27,844 | +51,300 | +184.24% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,878,464 | 933,436 | +945,028 | +101.24% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/cudnn-compile-and-serialize/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cudnn-compile-and-serialize/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cudnn-compile-and-serialize/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cudnn-compile-and-serialize/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/cudnn-compile-and-serialize/SKILL.md`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
