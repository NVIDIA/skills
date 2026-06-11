# Evaluation Report

Evaluation of the `vss-manage-alerts` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `vss-manage-alerts`
- Evaluation date: 2026-06-11
- NVSkills-Eval profile: `external`
- Environment: `astra-sandbox`
- Dataset: 14 evaluation tasks
- Attempts per task: 1
- Pass threshold: 50%
- Overall verdict: PASS

## Agents Used

- `claude-code`
- `codex`

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- `security` (Security): checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency` (Efficiency): checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy` (Accuracy): grades final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): checks whether the overall user task completed successfully.
- `behavior_check` (Behavior Check): verifies expected behavior steps, including safety expectations.
- `token_efficiency` (Token Efficiency): compares token usage with and without the skill.

## Test Tasks

The benchmark included 14 recorded Tier 3 trials, but the source evaluation dataset was not available in this report payload.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 7 | 100% (+0%) | 79% (-7%) |
| Correctness | 7 | 92% (+60%) | 95% (+50%) |
| Discoverability | 7 | 96% (+57%) | 82% (+21%) |
| Effectiveness | 7 | 66% (+49%) | 68% (+45%) |
| Efficiency | 7 | 88% (+52%) | 72% (+16%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed. NVSkills-Eval ran 9 checks and found 0 total findings.

Notable observations:

- SECURITY: no findings reported.
- SCHEMA: Found skill manifest: SKILL.md
- VERSION: Valid semantic version: 3.2.0
- PII: Scanning 10 files for PII
- LICENSE: no findings reported.

## Tier 2: Deduplication Summary

Tier 2 validation passed. NVSkills-Eval ran 2 checks and found 0 total findings.

Notable observations:

- Context Deduplication: Collected 9 file(s)
- Inter-Skill Deduplication: Parsed skill 'vss-manage-alerts': 166 char description

## Publication Recommendation

The skill is suitable to proceed toward NVSkills-Eval publication based on this benchmark. Skill owners should keep this file with the skill and refresh it when the evaluation dataset, skill behavior, or target agents materially change.
