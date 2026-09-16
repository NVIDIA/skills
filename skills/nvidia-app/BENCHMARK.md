# Skill Benchmark: nvidia-app

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-app`
- Evaluation date: 2026-10-01
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 32 evaluation tasks (29 positive, 3 negative)
- Dataset digest: `sha256:24a80b6a8c68db3347169c24fd3020997f52e4a2b1d6de22376feab66226e32f` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 77.9% — baseline ran, but no comparable score was available; uplift unavailable | 70.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.9% → 68.8% (-28.1 points) | 85.9% → 76.6% (-9.3 points) |
| Correctness | 15.6% → 90.0% (+74.4 points) | 20.6% → 61.3% (+40.7 points) |
| Discoverability | 97.9% — baseline ran, but no comparable score was available; uplift unavailable | 83.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 29.7% → 37.0% (+7.3 points) | 30.2% → 35.7% (+5.5 points) |
| Efficiency | 95.9% — baseline ran, but no comparable score was available; uplift unavailable | 93.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,143,668 | 3,842,556 | +2,301,112 | +59.88% | skill 32/32; base 32/32 |
| claude-code | nvidia-app-applications-list-024 | 197,146 | 120,873 | +76,273 | +63.10% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-compound-010 | 229,111 | 121,839 | +107,272 | +88.04% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-connection-default-012 | 104,070 | 121,988 | -17,918 | -14.69% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-current-game-008 | 184,171 | 29,417 | +154,754 | +526.07% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-driver-discovery-018 | 195,717 | 119,396 | +76,321 | +63.92% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-driver-release-notes-026 | 197,794 | 118,858 | +78,936 | +66.41% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-game-optimization-027 | 243,597 | 283,683 | -40,086 | -14.13% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-game-optimization-battery-031 | 200,184 | 91,163 | +109,021 | +119.59% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-game-optimization-configuration-prerequisite-033 | 154,641 | 154,251 | +390 | +0.25% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-game-optimization-default-ac-030 | 242,112 | 60,167 | +181,945 | +302.40% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-game-optimization-discovery-019 | 103,302 | 29,921 | +73,381 | +245.25% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-game-optimization-resolution-prerequisite-032 | 240,417 | 1,173,633 | -933,216 | -79.52% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-highlights-disable-020 | 232,943 | 91,124 | +141,819 | +155.63% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-highlights-enable-014 | 187,868 | 90,550 | +97,318 | +107.47% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-instant-replay-disable-023 | 195,196 | 29,600 | +165,596 | +559.45% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-instant-replay-enable-003 | 228,748 | 29,887 | +198,861 | +665.38% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-instant-replay-save-005 | 188,237 | 122,716 | +65,521 | +53.39% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-instant-replay-toggle-004 | 226,994 | 119,643 | +107,351 | +89.73% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-laptop-global-read-028 | 195,840 | 60,262 | +135,578 | +224.98% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-laptop-global-set-029 | 196,972 | 59,332 | +137,640 | +231.98% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-launch-025 | 198,512 | 29,175 | +169,337 | +580.42% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-negative-broadcast-016 | 65,746 | 29,998 | +35,748 | +119.17% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-negative-desktop-capture-015 | 65,109 | 91,498 | -26,389 | -28.84% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-negative-obs-017 | 126,722 | 60,707 | +66,015 | +108.74% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-record-start-001 | 187,088 | 29,273 | +157,815 | +539.11% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-record-stop-002 | 311,439 | 29,834 | +281,605 | +943.91% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-rtx-dvc-enable-021 | 188,582 | 91,405 | +97,177 | +106.31% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-rtx-dvc-toggle-022 | 236,133 | 121,691 | +114,442 | +94.04% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-screenshot-006 | 230,896 | 59,294 | +171,602 | +289.41% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-stats-007 | 271,498 | 90,813 | +180,685 | +198.96% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-status-009 | 227,449 | 120,517 | +106,932 | +88.73% | skill 1/1; base 1/1 |
| claude-code | nvidia-app-unsupported-qualifier-011 | 89,434 | 60,048 | +29,386 | +48.94% | skill 1/1; base 1/1 |
| codex | All cases | 4,310,327 | 2,717,877 | +1,592,450 | +58.59% | skill 32/32; base 32/32 |
| codex | nvidia-app-applications-list-024 | 234,283 | 277,300 | -43,017 | -15.51% | skill 1/1; base 1/1 |
| codex | nvidia-app-compound-010 | 161,893 | 44,183 | +117,710 | +266.41% | skill 1/1; base 1/1 |
| codex | nvidia-app-connection-default-012 | 47,528 | 65,550 | -18,022 | -27.49% | skill 1/1; base 1/1 |
| codex | nvidia-app-current-game-008 | 126,140 | 105,185 | +20,955 | +19.92% | skill 1/1; base 1/1 |
| codex | nvidia-app-driver-discovery-018 | 148,681 | 66,941 | +81,740 | +122.11% | skill 1/1; base 1/1 |
| codex | nvidia-app-driver-release-notes-026 | 187,033 | 56,038 | +130,995 | +233.76% | skill 1/1; base 1/1 |
| codex | nvidia-app-game-optimization-027 | 158,412 | 593,761 | -435,349 | -73.32% | skill 1/1; base 1/1 |
| codex | nvidia-app-game-optimization-battery-031 | 107,351 | 56,609 | +50,742 | +89.64% | skill 1/1; base 1/1 |
| codex | nvidia-app-game-optimization-configuration-prerequisite-033 | 148,030 | 134,129 | +13,901 | +10.36% | skill 1/1; base 1/1 |
| codex | nvidia-app-game-optimization-default-ac-030 | 148,806 | 86,381 | +62,425 | +72.27% | skill 1/1; base 1/1 |
| codex | nvidia-app-game-optimization-discovery-019 | 48,505 | 18,048 | +30,457 | +168.76% | skill 1/1; base 1/1 |
| codex | nvidia-app-game-optimization-resolution-prerequisite-032 | 127,469 | 71,837 | +55,632 | +77.44% | skill 1/1; base 1/1 |
| codex | nvidia-app-highlights-disable-020 | 125,806 | 102,778 | +23,028 | +22.41% | skill 1/1; base 1/1 |
| codex | nvidia-app-highlights-enable-014 | 86,673 | 83,678 | +2,995 | +3.58% | skill 1/1; base 1/1 |
| codex | nvidia-app-instant-replay-disable-023 | 171,484 | 27,256 | +144,228 | +529.16% | skill 1/1; base 1/1 |
| codex | nvidia-app-instant-replay-enable-003 | 106,106 | 13,464 | +92,642 | +688.07% | skill 1/1; base 1/1 |
| codex | nvidia-app-instant-replay-save-005 | 108,027 | 57,709 | +50,318 | +87.19% | skill 1/1; base 1/1 |
| codex | nvidia-app-instant-replay-toggle-004 | 231,301 | 27,055 | +204,246 | +754.93% | skill 1/1; base 1/1 |
| codex | nvidia-app-laptop-global-read-028 | 133,105 | 57,485 | +75,620 | +131.55% | skill 1/1; base 1/1 |
| codex | nvidia-app-laptop-global-set-029 | 170,462 | 78,709 | +91,753 | +116.57% | skill 1/1; base 1/1 |
| codex | nvidia-app-launch-025 | 134,752 | 27,233 | +107,519 | +394.81% | skill 1/1; base 1/1 |
| codex | nvidia-app-negative-broadcast-016 | 34,397 | 17,900 | +16,497 | +92.16% | skill 1/1; base 1/1 |
| codex | nvidia-app-negative-desktop-capture-015 | 50,388 | 25,008 | +25,380 | +101.49% | skill 1/1; base 1/1 |
| codex | nvidia-app-negative-obs-017 | 157,356 | 191,849 | -34,493 | -17.98% | skill 1/1; base 1/1 |
| codex | nvidia-app-record-start-001 | 106,569 | 41,524 | +65,045 | +156.64% | skill 1/1; base 1/1 |
| codex | nvidia-app-record-stop-002 | 164,882 | 27,255 | +137,627 | +504.96% | skill 1/1; base 1/1 |
| codex | nvidia-app-rtx-dvc-enable-021 | 175,658 | 41,180 | +134,478 | +326.56% | skill 1/1; base 1/1 |
| codex | nvidia-app-rtx-dvc-toggle-022 | 264,116 | 31,467 | +232,649 | +739.34% | skill 1/1; base 1/1 |
| codex | nvidia-app-screenshot-006 | 27,615 | 100,075 | -72,460 | -72.41% | skill 1/1; base 1/1 |
| codex | nvidia-app-stats-007 | 105,425 | 17,984 | +87,441 | +486.22% | skill 1/1; base 1/1 |
| codex | nvidia-app-status-009 | 255,302 | 114,462 | +140,840 | +123.05% | skill 1/1; base 1/1 |
| codex | nvidia-app-unsupported-qualifier-011 | 56,772 | 57,844 | -1,072 | -1.85% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 10,453,995 | 6,560,433 | +3,893,562 | +59.35% | skill 64/64; base 64/64 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 32 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in overlay-state.md (`skills/nvidia-app/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description doesn't mention WHEN to use this skill (`skills/nvidia-app/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/nvidia-app/SKILL.md`)

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
