# Skill Benchmark: nv-reason-pathwsi

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nv-reason-pathwsi`
- Validation status: `passed`
- Evaluation date: 2026-11-26
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:03c9fb52f3e06656d761b9b4cdaf6c393a178b8147beb68e409cb91e72a764da` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nv-reason-pathwsi | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-pathwsi | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-pathwsi | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-pathwsi | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-pathwsi | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nv-reason-pathwsi`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 82.7% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 30.0% → 100.0% (+70.0 points) | 25.0% → 100.0% (+75.0 points) |
| Discoverability | 47.5% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.8% → 77.5% (+43.7 points) | 33.8% → 77.5% (+43.7 points) |
| Efficiency | 88.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.2% → 96.6% (-1.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.5% → 91.3% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 610,072 | 611,779 | -1,707 | -0.28% | skill 4/4; base 4/4 |
| claude-code | clinical-boundary | 35,419 | 143,215 | -107,796 | -75.27% | skill 1/1; base 1/1 |
| claude-code | offline-synthetic-mock | 244,040 | 217,682 | +26,358 | +12.11% | skill 1/1; base 1/1 |
| claude-code | raw-slide-boundary | 35,451 | 106,936 | -71,485 | -66.85% | skill 1/1; base 1/1 |
| claude-code | upstream-sample-live | 295,162 | 143,946 | +151,216 | +105.05% | skill 1/1; base 1/1 |
| codex | All cases | 364,647 | 257,156 | +107,491 | +41.80% | skill 4/4; base 4/4 |
| codex | clinical-boundary | 29,141 | 38,380 | -9,239 | -24.07% | skill 1/1; base 1/1 |
| codex | offline-synthetic-mock | 164,222 | 105,611 | +58,611 | +55.50% | skill 1/1; base 1/1 |
| codex | raw-slide-boundary | 46,798 | 41,123 | +5,675 | +13.80% | skill 1/1; base 1/1 |
| codex | upstream-sample-live | 124,486 | 72,042 | +52,444 | +72.80% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 974,719 | 868,935 | +105,784 | +12.17% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 19 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nv-reason-pathwsi/SKILL.md`)
- **MEDIUM** SECURITY/skillspector_execution_failure: skillspector scan did not execute reliably (agent-control analysis limitation): scripts/run_nv_reason_pathwsi.py [static; obfuscated_instruction_text; analyzers=static_patterns_agent_snooping,static_patterns_anti_refusal,static_patterns_deserialization,static_patterns_excessive_agency,static_patterns_harmful_content,static_patterns_memory_poisoning,static_patterns_privilege_escalation,static_patterns_prompt_injection,static_patterns_rogue_agent,static_patterns_ssrf,static_patterns_supply_chain,static_patterns_system_prompt_leakage,static_patterns_tool_misuse]: Obfuscated instruction text could not be fully evaluated by the deterministic layer.; tests/test_nv_reason_pathwsi.py [static; static_parse_limit; analyzers=static_patterns_tool_misuse]: A security-relevant expression could not be fully resolved by the bounded static parser. SkillSpector could not determine this command or code fragment's boundaries. Check: command argument boundary; location: reported source range; unchecked scope: command arguments. Relevant parser input not fully checked. Parser measurements: characters: observed at least 8192, limit 8192. (`.`)
- **MEDIUM** SECURITY/Incomplete referenced artifact analysis (AE1): analysis-evasion: scripts/run_nv_reason_pathwsi.py (partial) (`SKILL.md:29`)
- **MEDIUM** SECURITY/Incomplete referenced artifact analysis (AE1): analysis-evasion: scripts/run_nv_reason_pathwsi.py (partial) (`SKILL.md:31`)
- **MEDIUM** SECURITY/Incomplete referenced artifact analysis (AE1): analysis-evasion: scripts/run_nv_reason_pathwsi.py (partial) (`SKILL.md:74`)
- 14 additional finding(s) are available in the full evaluation artifacts.

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
