# Skill Benchmark: doca-devemu

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-devemu`
- Evaluation date: 2026-10-12
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:bddc3102319aeedfab57309e910e03e8c1185f748c84845eadf3d5b4a3799de2` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

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
| Overall | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 94.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 75.0% → 100.0% (+25.0 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 99.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 52.5% → 93.1% (+40.6 points) | 56.3% → 98.1% (+41.8 points) |
| Efficiency | 81.8% — baseline ran, but no comparable score was available; uplift unavailable | 79.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 905,585 | 495,799 | +409,786 | +82.65% | skill 4/4; base 4/4 |
| claude-code | doca-devemu.explicit-positive.v1 | 284,966 | 183,669 | +101,297 | +55.15% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.implicit-positive.v1 | 285,000 | 34,345 | +250,655 | +729.82% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.implicit-positive.v2 | 304,250 | 123,543 | +180,707 | +146.27% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.negative.v1 | 31,369 | 154,242 | -122,873 | -79.66% | skill 1/1; base 1/1 |
| codex | All cases | 475,973 | 164,568 | +311,405 | +189.23% | skill 4/4; base 4/4 |
| codex | doca-devemu.explicit-positive.v1 | 141,609 | 39,797 | +101,812 | +255.83% | skill 1/1; base 1/1 |
| codex | doca-devemu.implicit-positive.v1 | 123,214 | 36,124 | +87,090 | +241.09% | skill 1/1; base 1/1 |
| codex | doca-devemu.implicit-positive.v2 | 136,979 | 19,209 | +117,770 | +613.10% | skill 1/1; base 1/1 |
| codex | doca-devemu.negative.v1 | 74,171 | 69,438 | +4,733 | +6.82% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,381,558 | 660,367 | +721,191 | +109.21% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 17 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md and references/details.md:
  "# DOCA Device Emulation capabilities, version overlay, errors, observability, safety" in CAPABILITIES.md (lines 1-23)
  vs "## Pattern overview" in CAPABILITIES.md (lines 30-37)
  vs "## Pattern overview" in CAPABILITIES.md (lines 39-58)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 73-76)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 77-85)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 93-100)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 141-153)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 224-230)
  vs "## Safety policy" in CAPABILITIES.md (lines 301-312)
  vs "## Safety policy" in CAPABILITIES.md (lines 315-321)
  vs "## Safety policy" in CAPABILITIES.md (lines 322-332)
  vs "## Safety policy" in CAPABILITIES.md (lines 347-355)
  vs "## Deferred topic boundaries" in CAPABILITIES.md (lines 368-426)
  vs "## Examples" in SKILL.md (lines 3-11)
  vs "## Instructions" in SKILL.md (lines 12-47)
  vs "## Audience" in SKILL.md (lines 48-80)
  vs "## When to load this skill" in SKILL.md (lines 81-123)
  vs "## What this skill provides" in SKILL.md (lines 124-162)
  vs "## Related skills" in SKILL.md (lines 196-247)
  vs "## configure" in TASKS.md (lines 47-55)
  vs "## configure" in TASKS.md (lines 57-170)
  vs "## run" in TASKS.md (lines 250-253)
  vs "## run" in TASKS.md (lines 255-300)
  vs "## test" in TASKS.md (lines 310-316)
  vs "## test" in TASKS.md (lines 329-372)
  vs "## test" in TASKS.md (lines 374-381)
  vs "## debug" in TASKS.md (lines 417-441)
  vs "## debug" in TASKS.md (lines 466-490)
  vs "## Example questions this skill answers well" in references/details.md (lines 10-75)
  vs "## What this skill deliberately does not ship" in references/details.md (lines 79-123)
  vs "## Related skills" in references/details.md (lines 125-176)
  vs "## Related skills" in references/details.md (lines 177-184) (`CAPABILITIES.md:1`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-devemu/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in details.md (`skills/libs/doca-devemu/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/libs/doca-devemu`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`BENCHMARK.md:120`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
