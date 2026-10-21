# Skill Benchmark: cdf-orchestration-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-orchestration-setup`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 11 evaluation tasks (9 positive, 2 negative)
- Dataset digest: `sha256:4387b9f4a4d9ac65bee1d14597318ac2da37624b65c0d788a1f0689f00881a44` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 85.8% — baseline ran, but no comparable score was available; uplift unavailable | 70.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 90.9% → 95.5% (+4.6 points) | 54.6% → 77.3% (+22.7 points) |
| Correctness | 16.4% → 85.5% (+69.1 points) | 16.4% → 74.6% (+58.2 points) |
| Discoverability | 91.4% — baseline ran, but no comparable score was available; uplift unavailable | 72.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 24.9% → 62.7% (+37.8 points) | 19.9% → 45.8% (+25.9 points) |
| Efficiency | 93.9% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,228,358 | 2,528,789 | -300,431 | -11.88% | skill 11/11; base 11/11 |
| claude-code | blocked-preflight-stops-setup | 66,351 | 183,917 | -117,566 | -63.92% | skill 1/1; base 1/1 |
| claude-code | dag-update-rebuilds-image | 130,124 | 89,567 | +40,557 | +45.28% | skill 1/1; base 1/1 |
| claude-code | first-install-builds-controller | 301,238 | 547,889 | -246,651 | -45.02% | skill 1/1; base 1/1 |
| claude-code | k3s-is-not-supported | 130,509 | 185,728 | -55,219 | -29.73% | skill 1/1; base 1/1 |
| claude-code | negative-prerequisite-audit | 152,522 | 121,844 | +30,678 | +25.18% | skill 1/1; base 1/1 |
| claude-code | negative-run-event-video | 158,859 | 61,370 | +97,489 | +158.85% | skill 1/1; base 1/1 |
| claude-code | one-gpu-service-mode | 176,294 | 563,827 | -387,533 | -68.73% | skill 1/1; base 1/1 |
| claude-code | osmo-external-only | 294,777 | 222,505 | +72,272 | +32.48% | skill 1/1; base 1/1 |
| claude-code | post-install-verification-retained | 184,709 | 152,217 | +32,492 | +21.35% | skill 1/1; base 1/1 |
| claude-code | preflight-required-before-install | 298,502 | 185,039 | +113,463 | +61.32% | skill 1/1; base 1/1 |
| claude-code | values-only-redeploy | 334,473 | 214,886 | +119,587 | +55.65% | skill 1/1; base 1/1 |
| codex | All cases | 1,942,229 | 3,798,041 | -1,855,812 | -48.86% | skill 11/11; base 11/11 |
| codex | blocked-preflight-stops-setup | 227,851 | 466,172 | -238,321 | -51.12% | skill 1/1; base 1/1 |
| codex | dag-update-rebuilds-image | 18,091 | 13,676 | +4,415 | +32.28% | skill 1/1; base 1/1 |
| codex | first-install-builds-controller | 195,825 | 1,187,951 | -992,126 | -83.52% | skill 1/1; base 1/1 |
| codex | k3s-is-not-supported | 180,257 | 353,728 | -173,471 | -49.04% | skill 1/1; base 1/1 |
| codex | negative-prerequisite-audit | 41,396 | 57,585 | -16,189 | -28.11% | skill 1/1; base 1/1 |
| codex | negative-run-event-video | 103,013 | 222,210 | -119,197 | -53.64% | skill 1/1; base 1/1 |
| codex | one-gpu-service-mode | 174,954 | 108,128 | +66,826 | +61.80% | skill 1/1; base 1/1 |
| codex | osmo-external-only | 232,334 | 769,077 | -536,743 | -69.79% | skill 1/1; base 1/1 |
| codex | post-install-verification-retained | 382,787 | 131,925 | +250,862 | +190.16% | skill 1/1; base 1/1 |
| codex | preflight-required-before-install | 165,850 | 146,858 | +18,992 | +12.93% | skill 1/1; base 1/1 |
| codex | values-only-redeploy | 219,871 | 340,731 | -120,860 | -35.47% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,170,587 | 6,326,830 | -2,156,243 | -34.08% | skill 22/22; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 15 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/deploy-controller.md:
  "## Deploy a controller on the current cluster" in SKILL.md (lines 129-132)
  vs "# Deploy the controller" in references/deploy-controller.md (lines 32-35) (`SKILL.md:129`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/deploy-controller.md:
  "## Deploy a controller on the current cluster" in SKILL.md (lines 116-121)
  vs "# Deploy the controller" in references/deploy-controller.md (lines 26-31) (`SKILL.md:116`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/deploy-controller.md:
  "## Deploy a controller on the current cluster" in SKILL.md (lines 107-108)
  vs "# Deploy the controller" in references/deploy-controller.md (lines 19-20) (`SKILL.md:107`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/deploy-controller.md:
  "## Deploy a controller on the current cluster" in SKILL.md (lines 72-85)
  vs "## Deploy a controller on the current cluster" in SKILL.md (lines 87-99)
  vs "# Deploy the controller" in references/deploy-controller.md (lines 7-13) (`SKILL.md:72`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in controller-connection.md (`skills/cdf-orchestration-setup/SKILL.md`)
- 14 additional finding(s) are available in the full evaluation artifacts.

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
