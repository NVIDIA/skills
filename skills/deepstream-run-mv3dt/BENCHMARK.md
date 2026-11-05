# Skill Benchmark: deepstream-run-mv3dt

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `deepstream-run-mv3dt`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:2944e4346697d995493b73c4f6b3fedcac029b04b99b55b4dc94838a20457537` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 78.8% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 30.0% → 85.0% (+55.0 points) | 45.0% → 70.0% (+25.0 points) |
| Discoverability | 71.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 38.4% → 56.3% (+17.9 points) | 43.4% → 59.9% (+16.5 points) |
| Efficiency | 81.4% — baseline ran, but no comparable score was available; uplift unavailable | 87.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 862,438 | 620,382 | +242,056 | +39.02% | skill 4/4; base 4/4 |
| claude-code | deepstream-run-mv3dt-001 | 286,635 | 184,264 | +102,371 | +55.56% | skill 1/1; base 1/1 |
| claude-code | deepstream-run-mv3dt-002 | 293,643 | 183,670 | +109,973 | +59.88% | skill 1/1; base 1/1 |
| claude-code | deepstream-run-mv3dt-003 | 252,131 | 222,567 | +29,564 | +13.28% | skill 1/1; base 1/1 |
| claude-code | deepstream-run-mv3dt-neg-001 | 30,029 | 29,881 | +148 | +0.50% | skill 1/1; base 1/1 |
| codex | All cases | 294,124 | 452,044 | -157,920 | -34.93% | skill 4/4; base 4/4 |
| codex | deepstream-run-mv3dt-001 | 100,395 | 32,360 | +68,035 | +210.24% | skill 1/1; base 1/1 |
| codex | deepstream-run-mv3dt-002 | 72,487 | 303,221 | -230,734 | -76.09% | skill 1/1; base 1/1 |
| codex | deepstream-run-mv3dt-003 | 72,857 | 98,690 | -25,833 | -26.18% | skill 1/1; base 1/1 |
| codex | deepstream-run-mv3dt-neg-001 | 48,385 | 17,773 | +30,612 | +172.24% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,156,562 | 1,072,426 | +84,136 | +7.85% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 24 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/custom-dataset.md and references/sample-run.md:
  "## Docker Access" in references/custom-dataset.md (lines 118-121)
  vs "## Launch DeepStream" in references/custom-dataset.md (lines 122-144)
  vs "## Docker Access" in references/sample-run.md (lines 93-119) (`references/custom-dataset.md:118`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/custom-dataset.md and references/sample-run.md and references/visualization-metadata.md:
  "## BEV Visualizer And Capture" in references/custom-dataset.md (lines 100-117)
  vs "# Run after the broker endpoint checks above. Use DOCKER_CMD="sudo docker" only if Docker needs sudo." in references/custom-dataset.md (lines 149-203)
  vs "# run the display workflow" in references/sample-run.md (lines 51-57)
  vs "# Run after the broker endpoint checks above. Use DOCKER_CMD="sudo docker" only if Docker needs sudo." in references/sample-run.md (lines 124-141)
  vs "# Run after the broker endpoint checks above. Use DOCKER_CMD="sudo docker" only if Docker needs sudo." in references/sample-run.md (lines 142-157)
  vs "# Run after the broker endpoint checks above. Use DOCKER_CMD="sudo docker" only if Docker needs sudo." in references/sample-run.md (lines 158-158)
  vs "## BEV Visualizer" in references/visualization-metadata.md (lines 47-79)
  vs "# After the DeepStream run command returns:" in references/visualization-metadata.md (lines 80-97)
  vs "## Offline BEV Video Capture" in references/visualization-metadata.md (lines 99-99)
  vs "## Offline BEV Video Capture" in references/visualization-metadata.md (lines 131-131)
  vs "## Offline BEV Video Capture" in references/visualization-metadata.md (lines 133-133) (`references/custom-dataset.md:100`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in amc-calibration-handoff.md (`skills/deepstream-run-mv3dt/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill declares capabilities involving file_read, file_write, and shell execution but does not define any explicit to (`SKILL.md:1`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without approval (`SKILL.md:103`)
- 21 additional finding(s) are available in the full evaluation artifacts.

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
