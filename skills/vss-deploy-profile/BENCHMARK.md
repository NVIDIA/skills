# Evaluation Report

Evaluation of the `vss-deploy-profile` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `vss-deploy-profile`
- Evaluation date: 2026-06-10
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

- Positive tasks: 5 tasks where the skill was expected to activate.
- Negative tasks: 0 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 5 | 100% (+0%) | 100% (+10%) |
| Correctness | 5 | 94% (+66%) | 88% (+52%) |
| Discoverability | 5 | 94% (+56%) | 85% (+28%) |
| Effectiveness | 5 | 69% (+63%) | 64% (+58%) |
| Efficiency | 5 | 78% (+41%) | 81% (+31%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 13 total findings.

Top findings:

- MEDIUM PII/ip_addresses: Non-RFC1918 IP address (`SKILL.md:152`)
- MEDIUM PII/ip_addresses: Non-RFC1918 IP address (`references/prerequisites.md:141`)
- MEDIUM PII/ip_addresses: Non-RFC1918 IP address (`references/prerequisites.md:148`)
- MEDIUM PII/ip_addresses: Non-RFC1918 IP address (`references/prerequisites.md:149`)
- MEDIUM PII/ip_addresses: Non-RFC1918 IP address (`references/prerequisites.md:169`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 1 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across references/prerequisites.md and references/warehouse.md:
  "### 2. Docker" in references/prerequisites.md (lines 264-283)
  vs "#### 2.2 Docker" in references/warehouse.md (lines 478-492) (`references/prerequisites.md:264`)
