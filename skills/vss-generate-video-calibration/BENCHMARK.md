# Evaluation Report

Evaluation of the `vss-generate-video-calibration` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `vss-generate-video-calibration`
- Evaluation date: 2026-05-29
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

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 2 total findings.

Top findings:

- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/vss-generate-video-calibration/SKILL.md`)
- MEDIUM SCHEMA/author_missing: Author not specified in metadata (`skills/vss-generate-video-calibration/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 3 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across references/sample-dataset.md and references/videos.md:
  "# iterating over this script's `videos` (the bundled cam_*.mp4)." in references/sample-dataset.md (lines 222-231)
  vs "# Step 2 — Upload videos (sorted)" in references/videos.md (lines 167-176) (`references/sample-dataset.md:222`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/calibration-tail.md and references/common-steps.md and references/rtsp.md and references/sample-dataset.md and references/videos.md:
  "# Step A — Verify project" in references/calibration-tail.md (lines 13-15)
  vs "## Create project" in references/common-steps.md (lines 7-25)
  vs "## Step 3 — Create Project" in references/rtsp.md (lines 87-91)
  vs "# Step 3 — Create project" in references/rtsp.md (lines 218-223)
  vs "# Step 1 — Create project" in references/sample-dataset.md (lines 213-219)
  vs "### Step 1 — Create Project" in references/videos.md (lines 40-43)
  vs "# Step 1 — Create project" in references/videos.md (lines 161-166) (`references/calibration-tail.md:13`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/rtsp.md and references/videos.md:
  "## Complete Python Script" in references/rtsp.md (lines 166-174)
  vs "## Complete Python Script" in references/videos.md (lines 104-112) (`references/rtsp.md:166`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
