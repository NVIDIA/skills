# Evaluation Report

Evaluation of the `vss-generate-video-calibration` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `vss-generate-video-calibration`
- Evaluation date: 2026-06-01
- NVSkills-Eval profile: `external`
- Environment: `local`
- Dataset: 6 evaluation tasks
- Attempts per task: 2
- Pass threshold: 50%
- Overall verdict: FAIL

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

The benchmark dataset contained 6 evaluation tasks:

- Positive tasks: 6 tasks where the skill was expected to activate.
- Negative tasks: 0 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 96% (+12%) | 79% (+12%) |
| Correctness | 8 | 87% (+1%) | 82% (+26%) |
| Discoverability | 8 | 89% (+9%) | 69% (+7%) |
| Effectiveness | 8 | 57% (-3%) | 55% (+24%) |
| Efficiency | 8 | 71% (+14%) | 53% (+6%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 4 total findings.

Top findings:

- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/vss-generate-video-calibration/SKILL.md`)
- MEDIUM SCHEMA/author_missing: Author not specified in metadata (`skills/vss-generate-video-calibration/SKILL.md`)
- MEDIUM SECURITY/Unknown (SDI-2): The script uses a curl-pipe-sh pattern to download and execute the `uv` installer from astral.sh without any integrity v (`references/sample-dataset.md:132`)
- MEDIUM SECURITY/Unknown (SQP-2): SSL verification is explicitly disabled (`ssl_verify: false`) in the RTSP capture request, and the Python script also im (`references/rtsp.md:106`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 4 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across references/sample-dataset.md and references/videos.md:
  "# iterating over this script's `videos` (the bundled cam_*.mp4)." in references/sample-dataset.md (lines 222-231)
  vs "# Step 2 — Upload videos (sorted)" in references/videos.md (lines 168-177) (`references/sample-dataset.md:222`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/rtsp.md and references/videos.md:
  "# (verify_project → calibrate → poll get_project_info → fetch evaluation_statistics)" in references/rtsp.md (lines 300-302)
  vs "# (verify_project → calibrate → poll get_project_info → fetch evaluation_statistics)" in references/videos.md (lines 243-244) (`references/rtsp.md:300`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/calibration-tail.md and references/common-steps.md and references/rtsp.md and references/sample-dataset.md and references/videos.md:
  "# Step A — Verify project" in references/calibration-tail.md (lines 15-17)
  vs "## Create project" in references/common-steps.md (lines 7-25)
  vs "## Step 3 — Create Project" in references/rtsp.md (lines 88-92)
  vs "# Step 3 — Create project" in references/rtsp.md (lines 219-224)
  vs "# Step 1 — Create project" in references/sample-dataset.md (lines 213-219)
  vs "### Step 1 — Create Project" in references/videos.md (lines 41-44)
  vs "# Step 1 — Create project" in references/videos.md (lines 162-167) (`references/calibration-tail.md:15`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/rtsp.md and references/videos.md:
  "## Complete Python Script" in references/rtsp.md (lines 167-175)
  vs "## Complete Python Script" in references/videos.md (lines 105-113) (`references/rtsp.md:167`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
