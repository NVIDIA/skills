# Skill Benchmark: nv-generate-mr

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nv-generate-mr`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 2 evaluation tasks (1 positive, 1 negative)
- Dataset digest: `sha256:007333dbd58286ca619f7a25b5e556d18fa250add2d728c04f99375bd018a8dd` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nv-generate-mr | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-generate-mr | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-generate-mr | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-generate-mr | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-generate-mr | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nv-generate-mr`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | Not available | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 25.0% → 100.0% (+75.0 points) |
| Correctness | Not available | 0.0% → 100.0% (+100.0 points) |
| Discoverability | Not available | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 16.7% → 66.7% (+50.0 points) |
| Efficiency | Not available | 94.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.4% → 89.6% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 274,392 | 144,863 | N/A | N/A | skill 2/2; base 1/2; Baseline token coverage is incomplete (1/2 attempts). Some token counters are missing. |
| claude-code | brain-request-redirect | 113,822 | N/A | N/A | N/A | skill 1/1; base 0/1; Baseline token coverage is incomplete (0/1 attempts). Some token counters are missing. |
| claude-code | generate-body-mr-t1 | 160,570 | 144,863 | +15,707 | +10.84% | skill 1/1; base 1/1 |
| codex | All cases | 165,102 | 260,270 | -95,168 | -36.57% | skill 2/2; base 2/2 |
| codex | brain-request-redirect | 50,540 | 107,533 | -56,993 | -53.00% | skill 1/1; base 1/1 |
| codex | generate-body-mr-t1 | 114,562 | 152,737 | -38,175 | -24.99% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 439,494 | 405,133 | N/A | N/A | skill 4/4; base 3/4; Baseline token coverage is incomplete (3/4 attempts). Some token counters are missing. |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 2 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and fixtures/README.md and scripts/run_mr.py:
  "## Purpose" in SKILL.md (lines 3-8)
  vs "## Limitations" in SKILL.md (lines 27-33)
  vs "## Exact Runnable Surface" in SKILL.md (lines 49-61)
  vs "## Usage" in SKILL.md (lines 101-137)
  vs "# Curated Fixture Catalog - `nv_generate_mr`" in fixtures/README.md (lines 1-24)
  vs "(module docstring)" in scripts/run_mr.py (lines 1-10) (`SKILL.md:3`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nv-generate-mr/SKILL.md`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=int("10"),
    (`scripts/run_mr.py:109`)
- **MEDIUM** SECURITY/Tainted flow: 'root' from os.environ.get (line 662, credential/environment) → subprocess.run (code execution) (TT2): Data Flow: subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=int("10"),
    (`scripts/run_mr.py:109`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(
            cmd,
            cwd=str(upstream_root),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            c (`scripts/run_mr.py:611`)
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
