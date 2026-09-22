# Skill Benchmark: tao-run-on-slurm

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-on-slurm`
- Evaluation date: 2026-09-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 1 evaluation tasks (1 positive)
- Dataset digest: `sha256:dfd65a7abfa0842fb6b147b51eb76b88ccc9ef8a74bc920d184c85aac501756b` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 99.3% — baseline ran, but no comparable score was available; uplift unavailable | 62.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 20.0% → 60.0% (+40.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 16.7% → 100.0% (+83.3 points) | 43.3% → 53.3% (+10.0 points) |
| Efficiency | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 99.8% → 98.4% (-1.4 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 73,480 | 90,721 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-on-slurm-basic | 73,480 | 90,721 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 14,040 | 13,432 | +608 | +4.53% | skill 1/1; base 1/1 |
| codex | tao-run-on-slurm-basic | 14,040 | 13,432 | +608 | +4.53% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 87,520 | 104,153 | N/A | N/A | skill 2/2; base 4/4 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 19 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 10 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 1 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/slurm-preflight-storage.md and references/slurm-ssh-credentials.md:
  "# 1. SSH to the login node works without a password prompt" in references/slurm-preflight-storage.md (lines 27-39)
  vs "# 1. SSH to the login node works without a password prompt" in references/slurm-ssh-credentials.md (lines 8-20) (`references/slurm-preflight-storage.md:27`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/slurm-preflight-storage.md and references/slurm-ssh-credentials.md:
  "## SSH Failure Remediation Prompt" in references/slurm-preflight-storage.md (lines 220-251)
  vs "## SSH Failure Remediation Prompt" in references/slurm-ssh-credentials.md (lines 190-221) (`references/slurm-preflight-storage.md:220`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/slurm-preflight-storage.md and references/slurm-ssh-credentials.md:
  "## Backend Details" in references/slurm-preflight-storage.md (lines 172-188)
  vs "## Backend Details" in references/slurm-ssh-credentials.md (lines 142-158) (`references/slurm-preflight-storage.md:172`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/slurm-preflight-storage.md and references/slurm-ssh-credentials.md:
  "## Required inputs" in SKILL.md (lines 281-300)
  vs "## Prerequisites" in references/slurm-preflight-storage.md (lines 83-146)
  vs "## Credentials" in references/slurm-preflight-storage.md (lines 147-171)
  vs "## Prerequisites" in references/slurm-ssh-credentials.md (lines 53-116)
  vs "## Credentials" in references/slurm-ssh-credentials.md (lines 117-141) (`SKILL.md:281`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/slurm-preflight-storage.md and references/slurm-ssh-credentials.md:
  "## Storage" in SKILL.md (lines 164-179)
  vs "## Storage" in references/slurm-preflight-storage.md (lines 189-219)
  vs "## Storage" in references/slurm-ssh-credentials.md (lines 159-189) (`SKILL.md:164`)
- 24 additional finding(s) are available in the full evaluation artifacts.

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
