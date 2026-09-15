# Skill Benchmark: nv-generate-mr-brain

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nv-generate-mr-brain`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:cd7314cae4d2269a61962e6b3e8e7cc821983453218701162a4526ceecd5ee28` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
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
| Overall | 97.3% — baseline ran, but no comparable score was available; uplift unavailable | 97.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 20.0% → 100.0% (+80.0 points) | 36.0% → 100.0% (+64.0 points) |
| Discoverability | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 18.4% → 96.7% (+78.3 points) | 38.0% → 96.7% (+58.7 points) |
| Efficiency | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 98.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 430,102 | 1,920,822 | N/A | N/A | skill 3/3; base 8/8 |
| claude-code | generate-brain-mra-v1 | 205,531 | 658,031 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | generate-brain-mri-t1 | 99,119 | 869,013 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | skull-stripped-modality-supported | 125,452 | 393,778 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 124,659 | 244,322 | N/A | N/A | skill 3/3; base 5/5 |
| codex | generate-brain-mra-v1 | 48,632 | 102,435 | -53,803 | -52.52% | skill 1/1; base 1/1 |
| codex | generate-brain-mri-t1 | 46,409 | 52,860 | -6,451 | -12.20% | skill 1/1; base 1/1 |
| codex | skull-stripped-modality-supported | 29,618 | 89,027 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 554,761 | 2,165,144 | N/A | N/A | skill 6/6; base 13/13 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 15 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nv-generate-mr-brain/SKILL.md (`BENCHMARK.md:75`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            env=_child_process_env(),
            check=False,
            capture_output=True,
     (`scripts/run_mr_brain.py:160`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root),
            env=_child_process_env(),
            check=False,
          (`scripts/run_mr_brain.py:178`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         proc = subprocess.run(
            cmd,
            cwd=str(upstream_root),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds (`scripts/run_mr_brain.py:734`)
- **MEDIUM** SECURITY/Tainted flow: 'cmd' from os.environ.get (line 718, credential/environment) → subprocess.run (code execution) (TT2): Data Flow:         proc = subprocess.run(
            cmd,
            cwd=str(upstream_root),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds (`scripts/run_mr_brain.py:734`)
- 10 additional finding(s) are available in the full evaluation artifacts.

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
