# Skill Benchmark: isaac-humanoid-cloud-control

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `isaac-humanoid-cloud-control`
- Validation status: `passed`
- Evaluation date: 2026-11-06
- Evaluator version: `1.6.9`
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
| Overall | 83.8% — baseline ran, but no comparable score was available; uplift unavailable | 78.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.3% → 93.3% (±0.0 points) | 76.7% → 90.0% (+13.3 points) |
| Correctness | 33.3% → 84.0% (+50.7 points) | 16.0% → 81.3% (+65.3 points) |
| Discoverability | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 30.6% → 56.3% (+25.7 points) | 25.3% → 45.5% (+20.2 points) |
| Efficiency | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.0% → 91.3% (-6.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.6% → 87.9% (-3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,722,186 | 2,208,053 | +1,514,133 | +68.57% | skill 15/15; base 15/15 |
| claude-code | isaac-humanoid-cloud-control-001 | 353,360 | 107,422 | +245,938 | +228.95% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-002 | 212,553 | 106,588 | +105,965 | +99.42% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-003 | 235,718 | 142,871 | +92,847 | +64.99% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-004 | 157,865 | 108,308 | +49,557 | +45.76% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-005 | 291,494 | 144,428 | +147,066 | +101.83% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-006 | 425,967 | 187,303 | +238,664 | +127.42% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-007 | 169,679 | 180,278 | -10,599 | -5.88% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-008 | 368,128 | 218,066 | +150,062 | +68.81% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-009 | 214,072 | 146,360 | +67,712 | +46.26% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-010 | 169,984 | 182,245 | -12,261 | -6.73% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-011 | 311,354 | 107,216 | +204,138 | +190.40% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-012 | 269,284 | 143,504 | +125,780 | +87.65% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-013 | 165,602 | 145,100 | +20,502 | +14.13% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-release-retry | 124,902 | 106,775 | +18,127 | +16.98% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-security-boundary | 252,224 | 181,589 | +70,635 | +38.90% | skill 1/1; base 1/1 |
| codex | All cases | 2,302,040 | 986,989 | +1,315,051 | +133.24% | skill 15/15; base 15/15 |
| codex | isaac-humanoid-cloud-control-001 | 202,423 | 96,532 | +105,891 | +109.70% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-002 | 118,059 | 13,174 | +104,885 | +796.15% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-003 | 197,354 | 73,814 | +123,540 | +167.37% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-004 | 33,813 | 108,453 | -74,640 | -68.82% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-005 | 186,527 | 41,478 | +145,049 | +349.70% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-006 | 235,112 | 74,025 | +161,087 | +217.61% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-007 | 202,990 | 79,144 | +123,846 | +156.48% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-008 | 138,441 | 49,682 | +88,759 | +178.65% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-009 | 118,233 | 42,200 | +76,033 | +180.17% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-010 | 143,719 | 61,667 | +82,052 | +133.06% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-011 | 226,707 | 61,314 | +165,393 | +269.75% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-012 | 203,427 | 55,706 | +147,721 | +265.18% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-013 | 153,183 | 54,916 | +98,267 | +178.94% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-release-retry | 48,009 | 55,805 | -7,796 | -13.97% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-security-boundary | 94,043 | 119,079 | -25,036 | -21.02% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,024,226 | 3,195,042 | +2,829,184 | +88.55% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 63 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/ordered-startup.md and references/release-4.6.md:
  "# Ordered startup and mission workflow" in references/ordered-startup.md (lines 86-107)
  vs "## Isaac Sim navigation" in references/release-4.6.md (lines 47-63) (`references/ordered-startup.md:86`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in dependencies.md (`skills/isaac-humanoid-cloud-control/SKILL.md`)
- **MEDIUM** SCRIPT_LINT/flat_script: simulation_identity.py has no function definitions (flat script) (`skills/isaac-humanoid-cloud-control/scripts/simulation_identity.py`)
- **MEDIUM** SECURITY/MCP Least Privilege (LP3): MCP Least Privilege: Without declared permissions the skill's intent is opaque and cannot be validated. (`SKILL.md:1`)
- **MEDIUM** SECURITY/Security finding (SQP-2): Step 1 (Isaac Sim) and step 4 (web-viewer clone) both require explicit approval. Step 2 doesn't: it tells the agent to i (`references/bootstrap.md:65`)
- 59 additional finding(s) are available in the full evaluation artifacts.

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
