# Evaluation Report

Evaluation of the `nemo-mbridge-perf-cpu-offloading` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `nemo-mbridge-perf-cpu-offloading`
- Evaluation date: 2026-05-28
- NVSkills-Eval profile: `external`
- Overall verdict: FAIL
- Tier 3 live agent evaluation: not available in this report

## Agents Used

- Tier 3 agent details were not available in this report.

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- No Tier 3 evaluation signal details were available in this report.

## Test Tasks

Tier 3 evaluation task details were not available in this report.

## Results

Tier 3 dimension rollup was not available in this report.

## Tier 1: Static Validation Summary

Tier 1 validation reported findings. NVSkills-Eval ran 9 checks and found 14 total findings.

Top findings:

- MEDIUM PII/gps_coordinates: GPS coordinates (location information) (`SKILL.md:101`)
- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nemo-mbridge-perf-cpu-offloading/SKILL.md`)
- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-mbridge-perf-cpu-offloading/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-mbridge-perf-cpu-offloading/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-mbridge-perf-cpu-offloading/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 1 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found within SKILL.md:
  "### Activation CPU offloading (small/medium models only)" in SKILL.md (lines 46-58)
  vs "### Activation offloading" in SKILL.md (lines 70-79)
  vs "### Activation offloading" in SKILL.md (lines 82-89)
  vs "### Activation offload (small model, PP=1)" in SKILL.md (lines 119-129)
  vs "### Weight offload only (small model, PP=1)" in SKILL.md (lines 130-140)
  vs "### Both activations and weights (small model, PP=1)" in SKILL.md (lines 141-156)
  vs "### MCore activation offload constraints" in SKILL.md (lines 188-206)
  vs "### MCore fine-grained offloading mutual exclusion" in SKILL.md (lines 214-222)
  vs "### MCore model_parallel_config fields" in SKILL.md (lines 257-267) (`SKILL.md:46`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
