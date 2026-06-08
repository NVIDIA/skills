# Evaluation Report

Evaluation of the `vss-deploy-profile` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `vss-deploy-profile`
- Evaluation date: 2026-06-08
- NVSkills-Eval profile: `external`
- Environment: `astra-sandbox`
- Dataset: 5 evaluation tasks
- Attempts per task: 2
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
| Security | 8 | 100% (+0%) | 95% (+10%) |
| Correctness | 8 | 96% (+77%) | 91% (+54%) |
| Discoverability | 8 | 93% (+72%) | 85% (+28%) |
| Effectiveness | 8 | 64% (+59%) | 61% (+54%) |
| Efficiency | 8 | 77% (+51%) | 76% (+22%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 5 total findings.

Top findings:

- MEDIUM QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/vss-deploy-profile/SKILL.md`)
- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/vss-deploy-profile/SKILL.md`)
- MEDIUM QUALITY/quality_efficiency: Deeply nested references in search.md (`skills/vss-deploy-profile/SKILL.md`)
- MEDIUM SCHEMA/author_missing: Author not specified in metadata (`skills/vss-deploy-profile/SKILL.md`)
- LOW QUALITY/quality_efficiency: Non-descriptive filename: ngc.md (`skills/vss-deploy-profile/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 1 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/alerts.md and references/base.md and references/lvs-profile.md and references/search.md:
  "# 1. cp dev-profile-<profile>/.env dev-profile-<profile>/generated.env  (clean copy)" in SKILL.md (lines 41-41)
  vs "# 5. docker compose --env-file generated.env -f resolved.yml up -d" in SKILL.md (lines 45-49)
  vs "### Step 1c — Initialize `generated.env`" in SKILL.md (lines 165-178)
  vs "### Step 3 — Apply overrides + dry-run" in SKILL.md (lines 199-205)
  vs "## Env file location" in references/alerts.md (lines 279-285)
  vs "## Env File Location" in references/base.md (lines 453-459)
  vs "## Env file location" in references/lvs-profile.md (lines 205-211)
  vs "## Env file location" in references/search.md (lines 278-284) (`SKILL.md:41`)
