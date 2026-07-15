# Evaluation Report

Evaluation of the `earth2studio-create-prognostic` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `earth2studio-create-prognostic`
- Evaluation date: 2026-06-03
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

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 6 total findings.

Top findings:

- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/earth2studio-create-prognostic/SKILL.md`)
- LOW QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/earth2studio-create-prognostic/SKILL.md`)
- LOW QUALITY/quality_reliability: No limitations documented (`skills/earth2studio-create-prognostic/SKILL.md`)
- LOW QUALITY/quality_efficiency: Non-doc file in references/: testing-guide.py (`skills/earth2studio-create-prognostic/SKILL.md`)
- LOW UNICODE/isolated_invisible_char: Isolated invisible character(s) (1): VARIATION SELECTOR-16 (`SKILL.md:25`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 5 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found within references/validation-guide.md:
  "# ============================================================" in references/validation-guide.md (lines 93-97)
  vs "# Load model" in references/validation-guide.md (lines 353-356) (`references/validation-guide.md:93`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/method-templates.py and references/skeleton-template.py:
  "default_generator_template()" in references/method-templates.py (lines 215-255)
  vs "create_iterator_template()" in references/method-templates.py (lines 258-277)
  vs "_default_generator()" in references/skeleton-template.py (lines 357-392)
  vs "create_iterator()" in references/skeleton-template.py (lines 397-416) (`references/method-templates.py:215`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/method-templates.py and references/skeleton-template.py:
  "load_default_package_template()" in references/method-templates.py (lines 286-321)
  vs "load_default_package()" in references/skeleton-template.py (lines 202-226) (`references/method-templates.py:286`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/method-templates.py and references/skeleton-template.py:
  "call_template()" in references/method-templates.py (lines 157-211)
  vs "__call__()" in references/skeleton-template.py (lines 303-351) (`references/method-templates.py:157`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/method-templates.py and references/skeleton-template.py:
  "load_model_template()" in references/method-templates.py (lines 326-365)
  vs "load_model()" in references/skeleton-template.py (lines 233-262) (`references/method-templates.py:326`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
