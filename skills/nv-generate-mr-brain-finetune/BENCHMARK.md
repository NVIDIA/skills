# Skill Benchmark: nv-generate-mr-brain-finetune

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nv-generate-mr-brain-finetune`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 2 evaluation tasks (2 positive)
- Dataset digest: `sha256:b454dd144a46b515c2d9de6de25fc7fb92c1ed1c66479f52678fa03cb000bc64` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 96.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 40.0% → 100.0% (+60.0 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 20.0% → 90.0% (+70.0 points) | 60.0% → 90.0% (+30.0 points) |
| Efficiency | 97.6% — baseline ran, but no comparable score was available; uplift unavailable | 97.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 135,326 | 1,362,815 | N/A | N/A | skill 2/2; base 4/4 |
| claude-code | finetune-mr-brain-from-datalist | 67,793 | 1,202,352 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | preflight-before-gpu-run | 67,533 | 160,463 | -92,930 | -57.91% | skill 1/1; base 1/1 |
| codex | All cases | 80,382 | 145,101 | -64,719 | -44.60% | skill 2/2; base 2/2 |
| codex | finetune-mr-brain-from-datalist | 49,758 | 53,505 | -3,747 | -7.00% | skill 1/1; base 1/1 |
| codex | preflight-before-gpu-run | 30,624 | 91,596 | -60,972 | -66.57% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 215,708 | 1,507,916 | N/A | N/A | skill 4/4; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 2 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nv-generate-mr-brain-finetune/SKILL.md (`BENCHMARK.md:75`)
- **MEDIUM** SECURITY/Session Persistence (RA2): Rogue Agent: write model caches under the upstream checkout and `~/.cache (`SKILL.md:79`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            env=_child_process_env(),
            check=False,
            capture_output=True,
     (`scripts/run_mr_brain_finetune.py:250`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root),
            env=_child_process_env(),
            check=False,
          (`scripts/run_mr_brain_finetune.py:266`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:     return subprocess.run(
        command,
        cwd=str(upstream_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    ) (`scripts/run_mr_brain_finetune.py:517`)
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
