# Skill Benchmark: nvat-corim-attestation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvat-corim-attestation`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 16 evaluation tasks (13 positive, 3 negative)
- Dataset digest: `sha256:839cf32c19d82f932ba2e5d6be7de923270121fada773b2d20db88ce2700bde5` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvat-corim-attestation | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-corim-attestation | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-corim-attestation | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-corim-attestation | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nvat-corim-attestation | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nvat-corim-attestation`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | Not available | 88.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 90.6% → 96.9% (+6.3 points) |
| Correctness | Not available | 85.0% → 100.0% (+15.0 points) |
| Discoverability | Not available | 93.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 70.6% → 83.0% (+12.4 points) |
| Efficiency | Not available | 66.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 80.8% → 87.4% (+6.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,771,351 | 1,333,511 | +437,840 | +32.83% | skill 16/16; base 16/16 |
| claude-code | nvat-corim-attestation-fsp-backup-inputs | 151,008 | 143,910 | +7,098 | +4.93% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-attestation-fsp-malformed-hint | 112,403 | 142,832 | -30,429 | -21.30% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-attestation-fsp-missing-hint | 111,706 | 107,769 | +3,937 | +3.65% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-attestation-negative-corim-authoring | 38,755 | 39,539 | -784 | -1.98% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-attestation-negative-oauth-jwt | 72,878 | 72,323 | +555 | +0.77% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-attestation-negative-tls-nonce | 37,463 | 37,216 | +247 | +0.66% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-blackwell-routing | 72,403 | 71,293 | +1,110 | +1.56% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-default-trust | 72,869 | 107,276 | -34,407 | -32.07% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-file-appraisal | 115,756 | 38,401 | +77,355 | +201.44% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-file-nonce | 110,881 | 71,170 | +39,711 | +55.80% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-file-not-offline | 197,321 | 142,855 | +54,466 | +38.13% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-legacy-input-routing | 148,586 | 107,704 | +40,882 | +37.96% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-live-challenge | 156,640 | 107,737 | +48,903 | +45.39% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-negative-ear | 150,249 | 71,764 | +78,485 | +109.37% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-policy-selection | 110,386 | 71,722 | +38,664 | +53.91% | skill 1/1; base 1/1 |
| claude-code | nvat-corim-transport-freshness | 112,047 | 0 | +112,047 | N/A | skill 1/1; base 1/1; Baseline total tokens are zero; percentage change is undefined. |
| codex | All cases | 943,735 | 1,241,728 | -297,993 | -24.00% | skill 16/16; base 16/16 |
| codex | nvat-corim-attestation-fsp-backup-inputs | 92,676 | 78,973 | +13,703 | +17.35% | skill 1/1; base 1/1 |
| codex | nvat-corim-attestation-fsp-malformed-hint | 52,994 | 128,308 | -75,314 | -58.70% | skill 1/1; base 1/1 |
| codex | nvat-corim-attestation-fsp-missing-hint | 53,253 | 68,366 | -15,113 | -22.11% | skill 1/1; base 1/1 |
| codex | nvat-corim-attestation-negative-corim-authoring | 57,206 | 48,203 | +9,003 | +18.68% | skill 1/1; base 1/1 |
| codex | nvat-corim-attestation-negative-oauth-jwt | 22,427 | 30,851 | -8,424 | -27.31% | skill 1/1; base 1/1 |
| codex | nvat-corim-attestation-negative-tls-nonce | 31,036 | 30,919 | +117 | +0.38% | skill 1/1; base 1/1 |
| codex | nvat-corim-blackwell-routing | 36,296 | 63,659 | -27,363 | -42.98% | skill 1/1; base 1/1 |
| codex | nvat-corim-default-trust | 32,015 | 30,037 | +1,978 | +6.59% | skill 1/1; base 1/1 |
| codex | nvat-corim-file-appraisal | 55,614 | 47,473 | +8,141 | +17.15% | skill 1/1; base 1/1 |
| codex | nvat-corim-file-nonce | 57,112 | 64,920 | -7,808 | -12.03% | skill 1/1; base 1/1 |
| codex | nvat-corim-file-not-offline | 82,945 | 40,932 | +42,013 | +102.64% | skill 1/1; base 1/1 |
| codex | nvat-corim-legacy-input-routing | 32,151 | 63,034 | -30,883 | -48.99% | skill 1/1; base 1/1 |
| codex | nvat-corim-live-challenge | 46,806 | 21,900 | +24,906 | +113.73% | skill 1/1; base 1/1 |
| codex | nvat-corim-negative-ear | 61,708 | 224,111 | -162,403 | -72.47% | skill 1/1; base 1/1 |
| codex | nvat-corim-policy-selection | 170,450 | 256,182 | -85,732 | -33.47% | skill 1/1; base 1/1 |
| codex | nvat-corim-transport-freshness | 59,046 | 43,860 | +15,186 | +34.62% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,715,086 | 2,575,239 | +139,847 | +5.43% | skill 32/32; base 32/32 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 16 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nvat-corim-attestation/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nvat-corim-attestation/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvat-corim-attestation/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvat-corim-attestation/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/nvat-corim-attestation/SKILL.md`)
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
