# Skill Benchmark: ising-calibration-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `ising-calibration-setup`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (6 positive, 3 negative)
- Dataset digest: `sha256:7bcd71995dd5970325a3c55dca88a58c524bf50d9aef5648358643d00f01b048` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| ising-calibration-setup | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ising-calibration-setup | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ising-calibration-setup | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ising-calibration-setup | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| ising-calibration-setup | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `ising-calibration-setup`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 97.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 88.9% → 100.0% (+11.1 points) |
| Correctness | 62.2% → 97.8% (+35.6 points) | 80.0% → 91.1% (+11.1 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 49.4% → 89.0% (+39.6 points) | 63.2% → 68.5% (+5.3 points) |
| Efficiency | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 72.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 96.1% (-2.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.5% → 88.7% (+1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 860,359 | 752,363 | +107,996 | +14.35% | skill 9/9; base 9/9 |
| claude-code | connectivity-checklist-auth-failure | 117,288 | 107,179 | +10,109 | +9.43% | skill 1/1; base 1/1 |
| claude-code | connectivity-checklist-not-auth | 122,483 | 107,687 | +14,796 | +13.74% | skill 1/1; base 1/1 |
| claude-code | hosted-api-connect | 117,115 | 106,729 | +10,386 | +9.73% | skill 1/1; base 1/1 |
| claude-code | hosted-api-hello-world | 118,644 | 107,081 | +11,563 | +10.80% | skill 1/1; base 1/1 |
| claude-code | negative-generic-docker-gpu | 35,981 | 35,873 | +108 | +0.30% | skill 1/1; base 1/1 |
| claude-code | negative-model-usage | 71,137 | 70,926 | +211 | +0.30% | skill 1/1; base 1/1 |
| claude-code | negative-other-nim-model | 35,944 | 35,748 | +196 | +0.55% | skill 1/1; base 1/1 |
| claude-code | self-hosted-hello-world | 122,007 | 71,037 | +50,970 | +71.75% | skill 1/1; base 1/1 |
| claude-code | self-hosted-nim-setup | 119,760 | 110,103 | +9,657 | +8.77% | skill 1/1; base 1/1 |
| codex | All cases | 406,447 | 271,490 | +134,957 | +49.71% | skill 9/9; base 9/9 |
| codex | connectivity-checklist-auth-failure | 51,436 | 43,788 | +7,648 | +17.47% | skill 1/1; base 1/1 |
| codex | connectivity-checklist-not-auth | 58,070 | 30,636 | +27,434 | +89.55% | skill 1/1; base 1/1 |
| codex | hosted-api-connect | 51,819 | 23,583 | +28,236 | +119.73% | skill 1/1; base 1/1 |
| codex | hosted-api-hello-world | 56,485 | 30,800 | +25,685 | +83.39% | skill 1/1; base 1/1 |
| codex | negative-generic-docker-gpu | 29,419 | 29,099 | +320 | +1.10% | skill 1/1; base 1/1 |
| codex | negative-model-usage | 13,517 | 13,378 | +139 | +1.04% | skill 1/1; base 1/1 |
| codex | negative-other-nim-model | 31,476 | 33,830 | -2,354 | -6.96% | skill 1/1; base 1/1 |
| codex | self-hosted-hello-world | 53,528 | 29,955 | +23,573 | +78.69% | skill 1/1; base 1/1 |
| codex | self-hosted-nim-setup | 60,697 | 36,421 | +24,276 | +66.65% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,266,806 | 1,023,853 | +242,953 | +23.73% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/hosted-api-quickstart.md and references/self-hosted-nim-quickstart.md:
  "## Hello world: describe a real calibration plot" in references/hosted-api-quickstart.md (lines 95-108)
  vs "# 1. Grab a real calibration-plot image URL from the public QCalEval dataset" in references/hosted-api-quickstart.md (lines 109-112)
  vs "# 2. Ask the model to describe it" in references/hosted-api-quickstart.md (lines 113-141)
  vs "### Hello world: describe a real calibration plot" in references/self-hosted-nim-quickstart.md (lines 210-216)
  vs "# 1. Grab a real calibration-plot image URL from the public QCalEval dataset" in references/self-hosted-nim-quickstart.md (lines 217-220)
  vs "# 2. Ask your container to describe it" in references/self-hosted-nim-quickstart.md (lines 221-251) (`references/hosted-api-quickstart.md:95`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/ising-calibration-setup/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/ising-calibration-setup/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: Never ask the
user (`SKILL.md:47`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`SKILL.md:96`)
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
