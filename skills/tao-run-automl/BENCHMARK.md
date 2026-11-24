# Skill Benchmark: tao-run-automl

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-automl`
- Validation status: `passed`
- Evaluation date: 2026-11-24
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 2 evaluation tasks (2 positive)
- Dataset digest: `sha256:6e6a8b1880c4283f37cfa5a864e7adf4d95f63ad1728111063fdc09ab2ac0604` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| tao-run-automl | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-automl | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-automl | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-automl | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-run-automl | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `tao-run-automl`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 99.5% — baseline ran, but no comparable score was available; uplift unavailable | 80.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 40.0% → 100.0% (+60.0 points) | 50.0% → 90.0% (+40.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 47.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 58.3% → 100.0% (+41.7 points) | 58.3% → 66.7% (+8.4 points) |
| Efficiency | 97.4% — baseline ran, but no comparable score was available; uplift unavailable | 96.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.7% → 94.8% (-3.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 93.7% → 92.9% (-0.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 159,838 | 107,627 | +52,211 | +48.51% | skill 2/2; base 2/2 |
| claude-code | tao-run-automl-basic | 80,228 | 35,500 | +44,728 | +125.99% | skill 1/1; base 1/1 |
| claude-code | tao-run-automl-container-default | 79,610 | 72,127 | +7,483 | +10.37% | skill 1/1; base 1/1 |
| codex | All cases | 44,158 | 26,997 | +17,161 | +63.57% | skill 2/2; base 2/2 |
| codex | tao-run-automl-basic | 13,836 | 13,391 | +445 | +3.32% | skill 1/1; base 1/1 |
| codex | tao-run-automl-container-default | 30,322 | 13,606 | +16,716 | +122.86% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 203,996 | 134,624 | +69,372 | +51.53% | skill 4/4; base 4/4 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 35 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 5 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 2 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/automl-compression-literature.md and references/automl-intent-algorithms.md:
  "## Action Policy" in references/automl-compression-literature.md (lines 20-39)
  vs "### LLM/Agentic Algorithms (NEW)" in references/automl-intent-algorithms.md (lines 306-319) (`references/automl-compression-literature.md:20`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/automl-intent-algorithms.md:
  "## Pre-Launch Review Gate" in SKILL.md (lines 115-143)
  vs "## Automatic Baseline Eval Job" in SKILL.md (lines 144-168)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 95-103)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 104-109) (`SKILL.md:115`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/automl-advanced-monitoring.md and references/automl-compression-literature.md and references/automl-examples.md and references/automl-intent-algorithms.md and references/automl-preflight-concepts.md and references/automl-runner-configuration.md:
  "# TAO AutoML" in SKILL.md (lines 1-14)
  vs "# TAO AutoML Advanced Monitoring" in references/automl-advanced-monitoring.md (lines 1-6)
  vs "## Contents" in references/automl-advanced-monitoring.md (lines 7-43)
  vs "# AutoML For Distill, Prune, And Quantize" in references/automl-compression-literature.md (lines 1-4)
  vs "# TAO AutoML Example Conversations" in references/automl-examples.md (lines 1-6)
  vs "## Contents" in references/automl-examples.md (lines 7-16)
  vs "### User: "Use the LLM to figure out the best hyperparameters"" in references/automl-examples.md (lines 46-68)
  vs "### User: "Run a fully autonomous search — let the AI figure everything out"" in references/automl-examples.md (lines 69-98)
  vs "# TAO AutoML Intent And Algorithms" in references/automl-intent-algorithms.md (lines 1-6)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 18-18)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 28-47)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 49-56)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 58-60)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 61-68)
  vs "## Step 1: Parse User Intent" in references/automl-intent-algorithms.md (lines 112-120)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 197-197)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 198-198)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 199-201)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 204-204)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 205-209)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 210-210)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 212-212)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 222-222)
  vs "# metadata at skill_dir, which was passed to AutoMLRunner(...) at construction." in references/automl-intent-algorithms.md (lines 236-236)
  vs "### LLM/Agentic Algorithms (NEW)" in references/automl-intent-algorithms.md (lines 297-297)
  vs "### LLM/Agentic Algorithms (NEW)" in references/automl-intent-algorithms.md (lines 304-304)
  vs "## Contents" in references/automl-preflight-concepts.md (lines 7-15)
  vs "# TAO AutoML Skill" in references/automl-preflight-concepts.md (lines 16-34)
  vs "## Concepts: What is TAO AutoML?" in references/automl-preflight-concepts.md (lines 146-166)
  vs "### LLM-Powered Algorithm Example" in references/automl-runner-configuration.md (lines 203-212)
  vs "### `automl_settings` keys" in references/automl-runner-configuration.md (lines 244-262) (`SKILL.md:1`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/detailed-guide.md:
  "## Reference Map" in SKILL.md (lines 29-49)
  vs "## Reference Map" in references/detailed-guide.md (lines 7-14) (`SKILL.md:29`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found within cards/driver.sh:
  "(comment)" in cards/driver.sh (lines 94-96)
  vs "finish_if_terminal()" in cards/driver.sh (lines 93-101) (`cards/driver.sh:94`)
- 35 additional finding(s) are available in the full evaluation artifacts.

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
