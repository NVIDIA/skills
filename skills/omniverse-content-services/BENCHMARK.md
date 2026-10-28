# Skill Benchmark: omniverse-content-services

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `omniverse-content-services`
- Evaluation date: 2026-10-24
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (8 positive, 2 negative)
- Dataset digest: `sha256:26ba432fafb9b59ca9f07418d47b8d52deb447fa024755733b0bc0958056faa1` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 88.2% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 85.0% (-15.0 points) | 100.0% → 85.0% (-15.0 points) |
| Correctness | 64.0% → 100.0% (+36.0 points) | 74.0% → 92.0% (+18.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 83.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 45.3% → 86.3% (+41.0 points) | 50.7% → 69.4% (+18.7 points) |
| Efficiency | 74.6% — baseline ran, but no comparable score was available; uplift unavailable | 70.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 10,464,332 | 3,148,814 | +7,315,518 | +232.33% | skill 10/10; base 10/10 |
| claude-code | clarify-local-caching | 427,742 | 64,136 | +363,606 | +566.93% | skill 1/1; base 1/1 |
| claude-code | custom-backend-contract | 1,824,221 | 1,901,998 | -77,777 | -4.09% | skill 1/1; base 1/1 |
| claude-code | existing-systems-to-simulation | 269,967 | 94,200 | +175,767 | +186.59% | skill 1/1; base 1/1 |
| claude-code | local-versioned-storage-handoff | 751,580 | 333,593 | +417,987 | +125.30% | skill 1/1; base 1/1 |
| claude-code | mesh-modeling-negative | 30,666 | 30,902 | -236 | -0.76% | skill 1/1; base 1/1 |
| claude-code | prebuilds-and-cache-routing | 229,340 | 156,269 | +73,071 | +46.76% | skill 1/1; base 1/1 |
| claude-code | reproducible-package-inputs | 6,691,988 | 125,688 | +6,566,300 | +5224.29% | skill 1/1; base 1/1 |
| claude-code | simulation-algorithm-negative | 33,265 | 62,133 | -28,868 | -46.46% | skill 1/1; base 1/1 |
| claude-code | team-and-agent-collaboration | 104,514 | 132,460 | -27,946 | -21.10% | skill 1/1; base 1/1 |
| claude-code | version-and-source-identities | 101,049 | 247,435 | -146,386 | -59.16% | skill 1/1; base 1/1 |
| codex | All cases | 1,182,435 | 454,263 | +728,172 | +160.30% | skill 10/10; base 10/10 |
| codex | clarify-local-caching | 65,611 | 30,866 | +34,745 | +112.57% | skill 1/1; base 1/1 |
| codex | custom-backend-contract | 560,062 | 63,834 | +496,228 | +777.37% | skill 1/1; base 1/1 |
| codex | existing-systems-to-simulation | 68,189 | 30,618 | +37,571 | +122.71% | skill 1/1; base 1/1 |
| codex | local-versioned-storage-handoff | 171,438 | 53,572 | +117,866 | +220.01% | skill 1/1; base 1/1 |
| codex | mesh-modeling-negative | 13,652 | 13,506 | +146 | +1.08% | skill 1/1; base 1/1 |
| codex | prebuilds-and-cache-routing | 86,091 | 71,553 | +14,538 | +20.32% | skill 1/1; base 1/1 |
| codex | reproducible-package-inputs | 120,259 | 127,091 | -6,832 | -5.38% | skill 1/1; base 1/1 |
| codex | simulation-algorithm-negative | 22,617 | 22,750 | -133 | -0.58% | skill 1/1; base 1/1 |
| codex | team-and-agent-collaboration | 60,615 | 26,904 | +33,711 | +125.30% | skill 1/1; base 1/1 |
| codex | version-and-source-identities | 13,901 | 13,569 | +332 | +2.45% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 11,646,767 | 3,603,077 | +8,043,690 | +223.25% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in prebuilt-artifacts.md (`skills/omniverse-content-services/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/omniverse-content-services/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/omniverse-content-services/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/omniverse-content-services/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/ovstorage-expert/SKILL.md (`references/repositories.md:17`)
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
