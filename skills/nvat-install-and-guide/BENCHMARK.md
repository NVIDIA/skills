# Skill Benchmark: nvat-install-and-guide

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvat-install-and-guide`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 12 evaluation tasks (9 positive, 3 negative)
- Dataset digest: `sha256:dfc8e7ed487842d39a52f196f73afa1c0c86a819f644e534a2f98b05273c3e88` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvat-install-and-guide | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-install-and-guide | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-install-and-guide | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-install-and-guide | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-install-and-guide | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nvat-install-and-guide`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | Not available | 84.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 83.3% → 87.5% (+4.2 points) |
| Correctness | Not available | 98.3% → 98.3% (±0.0 points) |
| Discoverability | Not available | 90.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 69.8% → 79.5% (+9.7 points) |
| Efficiency | Not available | 68.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.2% → 87.9% (+5.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,653,891 | 1,118,958 | +534,933 | +47.81% | skill 12/12; base 12/12 |
| claude-code | nvat-guide-blackwell-responder | 114,376 | 108,376 | +6,000 | +5.54% | skill 1/1; base 1/1 |
| claude-code | nvat-guide-gpu-nvml-checks | 36,858 | 0 | +36,858 | N/A | skill 1/1; base 1/1; Baseline total tokens are zero; percentage change is undefined. |
| claude-code | nvat-guide-hopper-legacy-evidence | 157,722 | 146,848 | +10,874 | +7.40% | skill 1/1; base 1/1 |
| claude-code | nvat-guide-rust-native-dependency | 242,960 | 107,509 | +135,451 | +125.99% | skill 1/1; base 1/1 |
| claude-code | nvat-guide-unavailable-docs | 114,170 | 107,022 | +7,148 | +6.68% | skill 1/1; base 1/1 |
| claude-code | nvat-guide-version-drift | 114,380 | 142,161 | -27,781 | -19.54% | skill 1/1; base 1/1 |
| claude-code | nvat-install-and-guide-negative-cublas-api | 36,660 | 36,988 | -328 | -0.89% | skill 1/1; base 1/1 |
| claude-code | nvat-install-and-guide-negative-cuda-toolkit | 456,307 | 144,927 | +311,380 | +214.85% | skill 1/1; base 1/1 |
| claude-code | nvat-install-and-guide-negative-nvml-monitoring | 35,916 | 35,767 | +149 | +0.42% | skill 1/1; base 1/1 |
| claude-code | nvat-install-container-arm64 | 115,925 | 74,151 | +41,774 | +56.34% | skill 1/1; base 1/1 |
| claude-code | nvat-install-ubuntu-network-repository | 113,689 | 143,969 | -30,280 | -21.03% | skill 1/1; base 1/1 |
| claude-code | nvat-install-unsupported-host | 114,928 | 71,240 | +43,688 | +61.33% | skill 1/1; base 1/1 |
| codex | All cases | 763,109 | 1,295,965 | -532,856 | -41.12% | skill 12/12; base 12/12 |
| codex | nvat-guide-blackwell-responder | 86,934 | 626,744 | -539,810 | -86.13% | skill 1/1; base 1/1 |
| codex | nvat-guide-gpu-nvml-checks | 66,024 | 18,436 | +47,588 | +258.13% | skill 1/1; base 1/1 |
| codex | nvat-guide-hopper-legacy-evidence | 55,680 | 66,469 | -10,789 | -16.23% | skill 1/1; base 1/1 |
| codex | nvat-guide-rust-native-dependency | 49,781 | 27,756 | +22,025 | +79.35% | skill 1/1; base 1/1 |
| codex | nvat-guide-unavailable-docs | 47,184 | 17,614 | +29,570 | +167.88% | skill 1/1; base 1/1 |
| codex | nvat-guide-version-drift | 48,684 | 48,818 | -134 | -0.27% | skill 1/1; base 1/1 |
| codex | nvat-install-and-guide-negative-cublas-api | 18,935 | 22,180 | -3,245 | -14.63% | skill 1/1; base 1/1 |
| codex | nvat-install-and-guide-negative-cuda-toolkit | 36,275 | 97,113 | -60,838 | -62.65% | skill 1/1; base 1/1 |
| codex | nvat-install-and-guide-negative-nvml-monitoring | 17,892 | 25,743 | -7,851 | -30.50% | skill 1/1; base 1/1 |
| codex | nvat-install-container-arm64 | 178,843 | 215,049 | -36,206 | -16.84% | skill 1/1; base 1/1 |
| codex | nvat-install-ubuntu-network-repository | 109,463 | 94,313 | +15,150 | +16.06% | skill 1/1; base 1/1 |
| codex | nvat-install-unsupported-host | 47,414 | 35,730 | +11,684 | +32.70% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,417,000 | 2,414,923 | +2,077 | +0.09% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nvat-install-and-guide/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nvat-install-and-guide/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvat-install-and-guide/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvat-install-and-guide/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/nvat-install-and-guide/SKILL.md`)
- 9 additional finding(s) are available in the full evaluation artifacts.

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
