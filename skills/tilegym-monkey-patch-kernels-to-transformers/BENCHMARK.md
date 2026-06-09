# Evaluation Report

Evaluation of the `tilegym-monkey-patch-kernels-to-transformers` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `tilegym-monkey-patch-kernels-to-transformers`
- Evaluation date: 2026-06-09
- NVSkills-Eval profile: `external`
- Environment: `astra-sandbox`
- Dataset: 5 evaluation tasks
- Attempts per task: 1
- Pass threshold: 50%
- Overall verdict: FAIL
The skill should be reviewed before NVSkills-Eval publication. **Skill owners should address the applicable findings below and rerun NVSkills-Eval to refresh this benchmark.**

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

The benchmark dataset contained 5 evaluation tasks:

- Positive tasks: 1 tasks where the skill was expected to activate.
- Negative tasks: 4 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 5 | 100% (+0%) | 100% (+0%) |
| Correctness | 5 | 97% (+0%) | 99% (+0%) |
| Discoverability | 5 | 87% (+0%) | 94% (+0%) |
| Effectiveness | 5 | 96% (-0%) | 100% (+1%) |
| Efficiency | 5 | 81% (-0%) | 89% (+1%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 9 total findings.

Top findings:

- MEDIUM QUALITY/quality_efficiency: Instructions lack clear action verbs (`skills/tilegym-monkey-patch-kernels-to-transformers/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/tilegym-monkey-patch-kernels-to-transformers/SKILL.md`)
- LOW QUALITY/quality_discoverability: Description very long (325 chars, recommend 50-150) (`skills/tilegym-monkey-patch-kernels-to-transformers/SKILL.md`)
- LOW QUALITY/quality_discoverability: No '## Purpose' section (`skills/tilegym-monkey-patch-kernels-to-transformers/SKILL.md`)
- LOW QUALITY/quality_reliability: No mention of error handling or validation (`skills/tilegym-monkey-patch-kernels-to-transformers/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 1 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found within references/kernel-integration.md:
  "# Integrate TileGym kernels to Transformers" in references/kernel-integration.md (lines 27-31)
  vs "# Integrate TileGym kernels to Transformers" in references/kernel-integration.md (lines 32-35)
  vs "# Integrate TileGym kernels to Transformers" in references/kernel-integration.md (lines 36-42) (`references/kernel-integration.md:27`)
