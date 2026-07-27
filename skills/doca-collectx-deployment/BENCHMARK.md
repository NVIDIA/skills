# Evaluation Report

Evaluation of the `doca-collectx-deployment` skill before publication through Skill Evaluator.

This benchmark summarizes 3-Tier Evaluation from Skill Evaluator results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `doca-collectx-deployment`
- Evaluation date: 2026-07-25
- Environment: `k8s-sandbox`
- Dataset: 2 evaluation tasks
- Attempts per task: 1
- Pass threshold: 50%
- Overall verdict: PASS

## Agents Used

- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`)
- Codex (`openai/openai/gpt-5.5`)

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

## Test Tasks

The benchmark dataset contained 2 evaluation tasks:

- Positive tasks: 1 tasks where the skill was expected to activate.
- Negative tasks: 1 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) | Codex (`openai/openai/gpt-5.5`) |
|---|---:|---:|---:|
| Security | 2 | 100% (+0%) | 100% (+0%) |
| Correctness | 2 | 100% (+80%) | 100% (+50%) |
| Discoverability | 2 | 100% (+25%) | 97% (+47%) |
| Effectiveness | 2 | 100% (+29%) | 84% (+38%) |
| Efficiency | 2 | 94% (+19%) | 69% (+19%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. Skill Evaluator ran 1 checks and found 5 total findings.

Top findings:

- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/doca-collectx-deployment/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/doca-collectx-deployment/SKILL.md`)
- MEDIUM SCHEMA/author_missing: Author not specified in metadata (`skills/doca-collectx-deployment/SKILL.md`)
- LOW SCHEMA/unexpected_file: Unexpected 'CAPABILITIES.md' in skill root (`skills/doca-collectx-deployment/CAPABILITIES.md`)
- LOW SCHEMA/unexpected_file: Unexpected 'TASKS.md' in skill root (`skills/doca-collectx-deployment/TASKS.md`)

## Tier 2: Deduplication Summary

This tier was not run or did not produce findings in this report.

## Publication Recommendation

The skill is suitable to proceed toward Skill Evaluator publication based on this benchmark. Skill owners should keep this file with the skill and refresh it when the evaluation dataset, skill behavior, or target agents materially change.
