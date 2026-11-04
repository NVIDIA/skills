# Skill Benchmark: nvidia-broadcast

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-broadcast`
- Evaluation date: 2026-10-02
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 22 evaluation tasks (19 positive, 3 negative)
- Dataset digest: `sha256:e64ad6bef7a96002e979cd95f36177197892f9a1fc780874e2ca376e1faecd9b` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 77.9% — baseline ran, but no comparable score was available; uplift unavailable | 74.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 95.5% → 100.0% (+4.5 points) |
| Correctness | 29.1% → 53.6% (+24.5 points) | 30.0% → 49.1% (+19.1 points) |
| Discoverability | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 86.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 39.2% → 49.1% (+9.9 points) | 38.9% → 45.6% (+6.7 points) |
| Efficiency | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,108,867 | 1,301,936 | +1,806,931 | +138.79% | skill 22/22; base 22/22 |
| claude-code | nvidia-broadcast-batch-017 | 163,643 | 90,496 | +73,147 | +80.83% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-cancel-011 | 119,337 | 58,599 | +60,738 | +103.65% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-conflict-016 | 165,720 | 122,497 | +43,223 | +35.28% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-consent-010 | 120,882 | 152,952 | -32,070 | -20.97% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-contextual-003 | 208,564 | 30,051 | +178,513 | +594.03% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-describe-014 | 120,607 | 29,885 | +90,722 | +303.57% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-device-013 | 163,313 | 29,433 | +133,880 | +454.86% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-edge-004 | 163,065 | 89,766 | +73,299 | +81.66% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-explicit-001 | 210,331 | 29,716 | +180,615 | +607.80% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-file-005 | 165,633 | 125,940 | +39,693 | +31.52% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-implicit-002 | 208,385 | 30,138 | +178,247 | +591.44% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-install-021 | 119,186 | 30,038 | +89,148 | +296.78% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-install-safety-023 | 30,525 | 30,169 | +356 | +1.18% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-negative-007 | 30,298 | 30,201 | +97 | +0.32% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-negative-008 | 64,397 | 62,773 | +1,624 | +2.59% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-negative-009 | 89,592 | 29,777 | +59,815 | +200.88% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-profile-015 | 207,627 | 29,803 | +177,824 | +596.66% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-restore-012 | 163,609 | 180,620 | -17,011 | -9.42% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-scope-019 | 75,349 | 30,276 | +45,073 | +148.87% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-state-006 | 119,159 | 29,675 | +89,484 | +301.55% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-update-022 | 120,155 | 29,454 | +90,701 | +307.94% | skill 1/1; base 1/1 |
| claude-code | nvidia-broadcast-wsl-020 | 279,490 | 29,677 | +249,813 | +841.77% | skill 1/1; base 1/1 |
| codex | All cases | 1,813,207 | 942,846 | +870,361 | +92.31% | skill 22/22; base 22/22 |
| codex | nvidia-broadcast-batch-017 | 90,737 | 104,509 | -13,772 | -13.18% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-cancel-011 | 41,740 | 26,985 | +14,755 | +54.68% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-conflict-016 | 107,893 | 62,845 | +45,048 | +71.68% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-consent-010 | 69,405 | 79,595 | -10,190 | -12.80% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-contextual-003 | 69,067 | 41,258 | +27,809 | +67.40% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-describe-014 | 31,444 | 24,638 | +6,806 | +27.62% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-device-013 | 95,173 | 13,426 | +81,747 | +608.87% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-edge-004 | 69,436 | 40,721 | +28,715 | +70.52% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-explicit-001 | 110,012 | 13,603 | +96,409 | +708.73% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-file-005 | 68,757 | 69,785 | -1,028 | -1.47% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-implicit-002 | 121,409 | 13,660 | +107,749 | +788.79% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-install-021 | 95,207 | 36,791 | +58,416 | +158.78% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-install-safety-023 | 66,620 | 13,517 | +53,103 | +392.86% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-negative-007 | 13,812 | 18,227 | -4,415 | -24.22% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-negative-008 | 98,824 | 110,531 | -11,707 | -10.59% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-negative-009 | 13,705 | 13,635 | +70 | +0.51% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-profile-015 | 115,395 | 24,681 | +90,714 | +367.55% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-restore-012 | 90,019 | 84,416 | +5,603 | +6.64% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-scope-019 | 169,449 | 70,494 | +98,955 | +140.37% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-state-006 | 88,907 | 43,577 | +45,330 | +104.02% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-update-022 | 95,730 | 18,103 | +77,627 | +428.81% | skill 1/1; base 1/1 |
| codex | nvidia-broadcast-wsl-020 | 90,466 | 17,849 | +72,617 | +406.84% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,922,074 | 2,244,782 | +2,677,292 | +119.27% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Windows-style paths detected (`skills/nvidia-broadcast/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (10432 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nvidia-broadcast/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nvidia-broadcast/SKILL.md (`BENCHMARK.md:120`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nvidia-broadcast/SKILL.md (`BENCHMARK.md:121`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nvidia-broadcast/SKILL.md (`BENCHMARK.md:122`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
