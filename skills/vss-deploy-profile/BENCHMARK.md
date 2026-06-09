# Evaluation Report

Evaluation of the `vss-deploy-profile` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `vss-deploy-profile`
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

- Positive tasks: 5 tasks where the skill was expected to activate.
- Negative tasks: 0 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 5 | 90% (+10%) | 80% (+0%) |
| Correctness | 5 | 92% (+11%) | 82% (+32%) |
| Discoverability | 5 | 92% (+15%) | 75% (+24%) |
| Effectiveness | 5 | 64% (+10%) | 56% (+35%) |
| Efficiency | 5 | 67% (+15%) | 59% (+20%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 6 total findings.

Top findings:

- MEDIUM QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/vss-deploy-profile/SKILL.md`)
- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/vss-deploy-profile/SKILL.md`)
- MEDIUM QUALITY/quality_efficiency: Deeply nested references in search.md (`skills/vss-deploy-profile/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/vss-deploy-profile/SKILL.md`)
- MEDIUM SCHEMA/author_missing: Author not specified in metadata (`skills/vss-deploy-profile/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 2 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across references/alerts.md and references/base.md and references/lvs-profile.md and references/search.md and references/warehouse-debug.md and references/warehouse.md:
  "## Endpoints (after deploy)" in references/alerts.md (lines 204-222)
  vs "## Endpoints (after deploy)" in references/base.md (lines 431-458)
  vs "## Endpoints (after deploy)" in references/lvs-profile.md (lines 189-204)
  vs "## Endpoints (after deploy)" in references/search.md (lines 259-277)
  vs "## Service Access Points" in references/warehouse-debug.md (lines 237-284)
  vs "### Agent + UI + ingress (`bp_wh` only)" in references/warehouse.md (lines 84-93)
  vs "### Via HAProxy ingress (`http://<EXTERNAL_IP>:<HAPROXY_PORT>` — default `<EXTERNAL_IP>:7777`)" in references/warehouse.md (lines 140-156)
  vs "### Direct ports (no HAProxy route — diagnostics only)" in references/warehouse.md (lines 157-172)
  vs "##### Internal service-to-service URLs (no Brev override needed)" in references/warehouse.md (lines 874-894)
  vs "## After deploy" in references/warehouse.md (lines 983-1029) (`references/alerts.md:204`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/prerequisites.md and references/warehouse.md:
  "### 2. Docker" in references/prerequisites.md (lines 167-186)
  vs "#### 2.2 Docker" in references/warehouse.md (lines 478-492) (`references/prerequisites.md:167`)
