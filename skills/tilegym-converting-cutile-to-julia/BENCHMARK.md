# Skill Benchmark: tilegym-converting-cutile-to-julia

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tilegym-converting-cutile-to-julia`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.3`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (1 positive, 4 negative)
- Dataset digest: `sha256:6c40b3ade1f94e92fef9c18ae5ede5d84019b586662db8dd15b37ef7e88c5ef4` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| tilegym-converting-cutile-to-julia | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| tilegym-converting-cutile-to-julia | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| tilegym-converting-cutile-to-julia | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| tilegym-converting-cutile-to-julia | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| tilegym-converting-cutile-to-julia | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `tilegym-converting-cutile-to-julia`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 95.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 100.0% (+20.0 points) | 76.0% → 100.0% (+24.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 82.5% → 100.0% (+17.5 points) | 72.5% → 87.5% (+15.0 points) |
| Efficiency | 99.1% — baseline ran, but no comparable score was available; uplift unavailable | 93.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.6% → 98.1% (-0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.9% → 87.1% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 222,963 | 288,592 | -65,629 | -22.74% | skill 5/5; base 5/5 |
| claude-code | 01-overview-cutile-to-julia | 75,165 | 107,359 | -32,194 | -29.99% | skill 1/1; base 1/1 |
| claude-code | 02-terraform-state-negative | 36,736 | 71,910 | -35,174 | -48.91% | skill 1/1; base 1/1 |
| claude-code | 03-graphql-federation-negative | 36,815 | 36,748 | +67 | +0.18% | skill 1/1; base 1/1 |
| claude-code | 04-redis-cluster-negative | 37,365 | 36,396 | +969 | +2.66% | skill 1/1; base 1/1 |
| claude-code | 05-elasticsearch-negative | 36,882 | 36,179 | +703 | +1.94% | skill 1/1; base 1/1 |
| codex | All cases | 177,350 | 192,630 | -15,280 | -7.93% | skill 5/5; base 5/5 |
| codex | 01-overview-cutile-to-julia | 48,222 | 55,963 | -7,741 | -13.83% | skill 1/1; base 1/1 |
| codex | 02-terraform-state-negative | 26,726 | 26,725 | +1 | +0.00% | skill 1/1; base 1/1 |
| codex | 03-graphql-federation-negative | 31,386 | 39,457 | -8,071 | -20.46% | skill 1/1; base 1/1 |
| codex | 04-redis-cluster-negative | 31,103 | 30,985 | +118 | +0.38% | skill 1/1; base 1/1 |
| codex | 05-elasticsearch-negative | 39,913 | 39,500 | +413 | +1.05% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 400,313 | 481,222 | -80,909 | -16.81% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 7 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/tilegym-converting-cutile-to-julia`)
- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/tilegym-converting-cutile-to-julia`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/testing.md and translations/workflow.md:
  "### Step 2: Register in `julia/test/runtests.jl`" in references/testing.md (lines 67-79)
  vs "### Step 2: Register in `julia/test/runtests.jl`" in translations/workflow.md (lines 366-374) (`references/testing.md:67`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/debugging.md and references/testing.md and translations/workflow.md:
  "# Dump emitted Tile IR bytecode (disassemble with `tileirdisasm <file>.cutile`)" in references/debugging.md (lines 52-55)
  vs "# Dump Tile IR bytecode for compilation issues" in references/testing.md (lines 35-40)
  vs "# Dump emitted Tile IR bytecode (compilation issues); disassemble with tileirdisasm" in translations/workflow.md (lines 109-114) (`references/debugging.md:52`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/api-mapping.md and references/critical-rules.md and translations/workflow.md:
  "## Memory Layout Considerations" in references/api-mapping.md (lines 260-284)
  vs "# Critical Rules for cuTile Python → Julia Conversion" in references/critical-rules.md (lines 8-8)
  vs "### Step 4: Memory Layout Considerations" in translations/workflow.md (lines 295-312) (`references/api-mapping.md:260`)
- 20 additional finding(s) are available in the full evaluation artifacts.

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
