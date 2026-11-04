# Skill Benchmark: ambient-healthcare-agent-with-nemotron-voice-agent

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `ambient-healthcare-agent-with-nemotron-voice-agent`
- Evaluation date: 2026-09-24
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:d8ae4021cef430b548b0c29cac10d8517b49a7c5910f714f06a474939295ff30` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.3% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 14.3% → 85.0% (+70.7 points) | 33.3% → 95.0% (+61.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 61.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 82.9% (+32.9 points) | 34.3% → 88.3% (+54.0 points) |
| Efficiency | 83.5% — baseline ran, but no comparable score was available; uplift unavailable | 96.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 416,038 | 491,598 | N/A | N/A | skill 4/4; base 7/7 |
| claude-code | nva-ambient-explicit-welcome | 63,708 | 58,594 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nva-ambient-negative-ordinary-deploy | 106,016 | 93,771 | +12,245 | +13.06% | skill 1/1; base 1/1 |
| claude-code | nva-ambient-public-endpoint-env-disclosure | 143,933 | 248,890 | -104,957 | -42.17% | skill 1/1; base 1/1 |
| claude-code | nva-ambient-scenario-disclosure | 102,381 | 90,343 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 152,390 | 272,051 | N/A | N/A | skill 4/4; base 9/9 |
| codex | nva-ambient-explicit-welcome | 16,493 | 39,885 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nva-ambient-negative-ordinary-deploy | 58,304 | 36,126 | +22,178 | +61.39% | skill 1/1; base 1/1 |
| codex | nva-ambient-public-endpoint-env-disclosure | 47,903 | 168,827 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nva-ambient-scenario-disclosure | 29,690 | 27,213 | N/A | N/A | skill 1/1; base 2/2 |
| ALL AGENTS | Dataset aggregate | 568,428 | 763,649 | N/A | N/A | skill 8/8; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 43 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/ambient-healthcare-agent-with-nemotron-voice-agent/SKILL.md (`BENCHMARK.md:82`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/ambient-healthcare-agent-with-nemotron-voice-agent/SKILL.md (`BENCHMARK.md:83`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         completed = subprocess.run(
            ["git", "-C", str(nva_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            tex (`scripts/nva_generic_defaults.py:466`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        ) (`scripts/verify_docker_compose_access.py:67`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution:         subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans", "--volumes"],
            capture_output=True,
            text=True,
            timeout (`scripts/verify_docker_compose_access.py:103`)
- 38 additional finding(s) are available in the full evaluation artifacts.

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
