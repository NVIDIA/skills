# Skill Benchmark: nvidia-skill-finder

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-skill-finder`
- Evaluation date: 2026-10-01
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 18 evaluation tasks (13 positive, 5 negative)
- Dataset digest: `sha256:df412e229dfed3687e1112c8f5bc439a6f3695291f3d357f049002621e90845f` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 92.0% — baseline ran, but no comparable score was available; uplift unavailable | 89.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 98.2% → 97.2% (-1.0 points) | 88.5% → 94.4% (+5.9 points) |
| Correctness | 69.6% → 96.7% (+27.1 points) | 70.8% → 100.0% (+29.2 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.1% → 87.9% (+51.8 points) | 40.9% → 89.7% (+48.8 points) |
| Efficiency | 78.4% — baseline ran, but no comparable score was available; uplift unavailable | 66.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,931,632 | 4,494,280 | N/A | N/A | skill 18/18; base 27/27 |
| claude-code | nvidia-skill-finder-diff-bare-invocation | 175,571 | 151,559 | +24,012 | +15.84% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-browse-no-keywords | 136,133 | 120,437 | +15,696 | +13.03% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-deep-research | 170,078 | 363,184 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvidia-skill-finder-neg-express-route | 356,710 | 483,643 | -126,933 | -26.25% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-generic-kubernetes | 154,997 | 185,511 | -30,514 | -16.45% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-python-refactor | 120,042 | 119,199 | +843 | +0.71% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-react-optimize | 153,960 | 152,401 | +1,559 | +1.02% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-video-editing | 159,058 | 154,957 | +4,101 | +2.65% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-cad-to-simready | 129,524 | 162,375 | -32,851 | -20.23% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dicom | 177,291 | 157,064 | +20,227 | +12.88% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dynamo-kv | 129,903 | 376,086 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvidia-skill-finder-pos-gpu-pandas | 130,132 | 124,405 | +5,727 | +4.60% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-isaac-lab | 211,676 | 60,437 | +151,239 | +250.24% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-driver-install | 167,216 | 59,583 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nvidia-skill-finder-pos-jetson-setup | 130,190 | 335,895 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvidia-skill-finder-pos-multigpu-training | 169,065 | 1,276,278 | -1,107,213 | -86.75% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-openusd-optimization | 129,543 | 89,238 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvidia-skill-finder-pos-vehicle-routing | 130,543 | 122,028 | +8,515 | +6.98% | skill 1/1; base 1/1 |
| codex | All cases | 1,642,585 | 2,128,791 | N/A | N/A | skill 18/18; base 26/26 |
| codex | nvidia-skill-finder-diff-bare-invocation | 73,710 | 685,152 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvidia-skill-finder-diff-browse-no-keywords | 128,756 | 240,494 | -111,738 | -46.46% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-deep-research | 86,599 | 40,331 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvidia-skill-finder-neg-express-route | 177,679 | 69,297 | +108,382 | +156.40% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-generic-kubernetes | 201,213 | 213,286 | -12,073 | -5.66% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-python-refactor | 41,853 | 69,141 | -27,288 | -39.47% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-react-optimize | 98,985 | 127,238 | -28,253 | -22.20% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-video-editing | 69,541 | 55,477 | +14,064 | +25.35% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-cad-to-simready | 74,083 | 25,338 | +48,745 | +192.38% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dicom | 74,117 | 31,845 | +42,272 | +132.74% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dynamo-kv | 94,291 | 307,770 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvidia-skill-finder-pos-gpu-pandas | 71,906 | 18,127 | +53,779 | +296.68% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-isaac-lab | 85,260 | 74,373 | +10,887 | +14.64% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-driver-install | 71,998 | 18,014 | +53,984 | +299.68% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-setup | 71,193 | 66,628 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvidia-skill-finder-pos-multigpu-training | 74,250 | 59,276 | +14,974 | +25.26% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-openusd-optimization | 79,947 | 13,501 | +66,446 | +492.16% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-vehicle-routing | 67,204 | 13,503 | +53,701 | +397.70% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,574,217 | 6,623,071 | N/A | N/A | skill 36/36; base 53/53 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 19 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvidia-skill-finder/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (SQP-1): L032 instructs activation on any "how do I do X" request where X "might be a common task that an NVIDIA skill can help w (`SKILL.md:32`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: The `npx skills add nvidia/skills --list` command is invoked without a pinned package version. When npx resolves an unpi (`SKILL.md:74`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: Same unpinned `npx skills` invocation as above, used as the primary catalog-check mechanism. Without a version pin, any  (`SKILL.md:103`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: Another instance of the same unpinned `npx skills` pattern. The cumulative effect of multiple unpinned invocations acros (`SKILL.md:113`)
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
