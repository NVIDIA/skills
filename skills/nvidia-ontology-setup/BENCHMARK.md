# Skill Benchmark: nvidia-ontology-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-setup`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (10 positive)
- Dataset digest: `sha256:dcbcf0848aefb19cd5b010e9a0d69ed8920e61dc3fdba560ca0f88cd48a8da96` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-ontology-setup | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `nvidia-ontology-setup`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 95.2% — baseline ran, but no comparable score was available; uplift unavailable | 92.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 80.0% → 100.0% (+20.0 points) |
| Correctness | 42.0% → 100.0% (+58.0 points) | 56.0% → 98.0% (+42.0 points) |
| Discoverability | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.3% → 95.0% (+57.7 points) | 56.2% → 90.7% (+34.5 points) |
| Efficiency | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | 81.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 96.8% (-1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.1% → 90.5% (+0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,147,298 | 1,472,943 | -325,645 | -22.11% | skill 10/10; base 10/10 |
| claude-code | setup-eval-001-underspecified | 116,219 | 106,949 | +9,270 | +8.67% | skill 1/1; base 1/1 |
| claude-code | setup-eval-002-use-existing-scripts | 75,561 | 106,903 | -31,342 | -29.32% | skill 1/1; base 1/1 |
| claude-code | setup-eval-003-auth-secret-cookies | 117,292 | 143,920 | -26,628 | -18.50% | skill 1/1; base 1/1 |
| claude-code | setup-eval-004-postgres-port-host-only | 115,756 | 142,007 | -26,251 | -18.49% | skill 1/1; base 1/1 |
| claude-code | setup-eval-005-mcp-already-up | 115,175 | 182,127 | -66,952 | -36.76% | skill 1/1; base 1/1 |
| claude-code | setup-eval-006-vault-partial | 107,248 | 143,085 | -35,837 | -25.05% | skill 1/1; base 1/1 |
| claude-code | setup-eval-007-embed-triplet-mismatch | 155,543 | 142,502 | +13,041 | +9.15% | skill 1/1; base 1/1 |
| claude-code | setup-eval-008-connect-source-workflow | 117,488 | 217,998 | -100,510 | -46.11% | skill 1/1; base 1/1 |
| claude-code | setup-eval-009-env-vs-ui-connections | 110,730 | 143,292 | -32,562 | -22.72% | skill 1/1; base 1/1 |
| claude-code | setup-eval-010-helm-install | 116,286 | 144,160 | -27,874 | -19.34% | skill 1/1; base 1/1 |
| codex | All cases | 495,379 | 543,819 | -48,440 | -8.91% | skill 10/10; base 10/10 |
| codex | setup-eval-001-underspecified | 29,652 | 78,780 | -49,128 | -62.36% | skill 1/1; base 1/1 |
| codex | setup-eval-002-use-existing-scripts | 50,231 | 49,048 | +1,183 | +2.41% | skill 1/1; base 1/1 |
| codex | setup-eval-003-auth-secret-cookies | 46,935 | 41,105 | +5,830 | +14.18% | skill 1/1; base 1/1 |
| codex | setup-eval-004-postgres-port-host-only | 33,975 | 48,623 | -14,648 | -30.13% | skill 1/1; base 1/1 |
| codex | setup-eval-005-mcp-already-up | 51,029 | 36,697 | +14,332 | +39.05% | skill 1/1; base 1/1 |
| codex | setup-eval-006-vault-partial | 79,118 | 58,064 | +21,054 | +36.26% | skill 1/1; base 1/1 |
| codex | setup-eval-007-embed-triplet-mismatch | 50,989 | 44,874 | +6,115 | +13.63% | skill 1/1; base 1/1 |
| codex | setup-eval-008-connect-source-workflow | 46,497 | 58,004 | -11,507 | -19.84% | skill 1/1; base 1/1 |
| codex | setup-eval-009-env-vs-ui-connections | 72,843 | 70,447 | +2,396 | +3.40% | skill 1/1; base 1/1 |
| codex | setup-eval-010-helm-install | 34,110 | 58,177 | -24,067 | -41.37% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,642,677 | 2,016,762 | -374,085 | -18.55% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

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
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: Never ask the user (`SKILL.md:65`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -s -o /dev/null -w '%{http_code}\n' "$AUTO_ONTOLOGY_API_URL/.well-known/oauth-authorization-server"
```

`200` is required. Anything else: run the frontend from the checkout
(`pnpm dev`) and poin (`SKILL.md:203`)
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
