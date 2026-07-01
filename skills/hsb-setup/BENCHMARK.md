# Evaluation Report

Evaluation of the `hsb-setup` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `hsb-setup`
- Evaluation date: 2026-06-30
- NVSkills-Eval profile: `external`
- Environment: `astra-sandbox`
- Dataset: 1 evaluation tasks
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

The benchmark dataset contained 1 evaluation tasks:

- Positive tasks: 1 tasks where the skill was expected to activate.
- Negative tasks: 0 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 1 | 100% (+0%) | 100% (+0%) |
| Correctness | 1 | 0% (-50%) | 97% (+81%) |
| Discoverability | 1 | 0% (-100%) | 89% (+48%) |
| Effectiveness | 1 | 0% (+0%) | 78% (+78%) |
| Efficiency | 1 | 27% (-68%) | 81% (+26%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 1 checks and found 4 total findings.

Top findings:

- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`team-skills/holoscan/holoscan-sensor-bridge/hsb-setup/SKILL.md`)
- LOW SCHEMA/unexpected_file: Unexpected 'linux' in skill root (`team-skills/holoscan/holoscan-sensor-bridge/hsb-setup/linux`)
- LOW SCHEMA/unexpected_file: Unexpected 'windows' in skill root (`team-skills/holoscan/holoscan-sensor-bridge/hsb-setup/windows`)
- LOW SCHEMA/unexpected_file: Unexpected 'docs' in skill root (`team-skills/holoscan/holoscan-sensor-bridge/hsb-setup/docs`)

## Tier 2: Deduplication Summary

This tier was not run or did not produce findings in this report.

## Publication Recommendation

The skill is suitable to proceed toward NVSkills-Eval publication based on this benchmark. Skill owners should keep this file with the skill and refresh it when the evaluation dataset, skill behavior, or target agents materially change.
