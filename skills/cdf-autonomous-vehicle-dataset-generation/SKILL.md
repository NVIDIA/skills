---
name: cdf-autonomous-vehicle-dataset-generation
description: Prepare payloads, run and monitor the Autonomous Vehicle Dataset Generation Airflow workflow, and inspect results for driving-clip retrieval and Cosmos3-Nano variants. Existing clips with no retrieval use cdf-video-data-augmentation; seed-image event or anomaly generation uses cdf-event-video-generation.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  owner: NVIDIA
  service: cosmos-data-factory
  version: "1.1.0"
  tags:
    - physical-ai
    - cdf-orchestration
    - video-curation
    - cosmos
---

# Autonomous Vehicle Dataset Generation

Find driving clips similar to an example video, or generate rain, snow and
daylight variations. Prepare a payload (a JSON file of run settings), run
`autonomous_vehicle_dataset_generation_dag_k8s`, and inspect the results.
The workflow uses Cosmos-Curator, Cosmos Dataset Search (CDS) and Cosmos3-Nano.
It starts and stops its own services.

For augmenting existing clips without retrieval, use
[cdf-video-data-augmentation](../cdf-video-data-augmentation/SKILL.md).
For event or anomaly videos from a seed image, use
[cdf-event-video-generation](../cdf-event-video-generation/SKILL.md).
For clip-augmentation requests where the user says no search or retrieval is
needed, stop and reroute to video data augmentation rather than building an AV
search payload or working around the search requirement.

Use a checkout matching the deployed workflow. Keep the repository documentation
and payloads available when this skill is copied into an agent sandbox.

## Component details

This workflow skill and its references are the primary guide for configuring and running the
workflow. Consult a component skill only when the requested configuration or diagnosis requires
component-specific details not covered here, and do not invent keys or values.

| Component | Covers | Component skill |
|---|---|---|
| CDF Curation and Retrieval | Cosmos-Curator clip splitting, embedding and curation; Cosmos Dataset Search collection ingest and similarity search | [`cdf-curation-and-retrieval` skill][curation-skill] |
| CDF Augmentation | Cosmos3-Nano variant generation with RGB or WSM controls, and augmentation errors | [`cdf-augmentation` skill][augmentation-skill] |

[curation-skill]: https://github.com/NVIDIA/cdf-curation-and-retrieval/tree/b4d553d00cc00c1293ad997fa8693af809489c71/skills/cdf-curation-and-retrieval
[augmentation-skill]: https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation

If the repository is inaccessible, continue tasks supported by this workflow's guidance. Identify
the unavailable reference and the unresolved component-specific detail only when that missing
guidance prevents completing the requested work; do not invent component settings or treat
component access as a prerequisite for ordinary workflow use.

## Examples

Choose a starter from the current [Payload Guide](../../docs/autonomous-vehicle-dataset-generation/payload-guide.md):

| Request | Starter payload |
| --- | --- |
| Find one similar driving clip | [Retrieval](../../payloads/dev/autonomous_vehicle_dataset_generation_dag/cdf-avdg-real-retrieval.json) |
| Generate one rainy variation | [Rain](../../payloads/dev/autonomous_vehicle_dataset_generation_dag/cdf-avdg-rain-generation.json) |
| Generate one snowy variation | [Snow](../../payloads/dev/autonomous_vehicle_dataset_generation_dag/cdf-avdg-snow-generation.json) |
| Generate one daylight variation | [Daylight](../../payloads/dev/autonomous_vehicle_dataset_generation_dag/cdf-avdg-day-generation.json) |

All starters process one source scene and request one match. Generation starters
produce one 93-frame variation at 1280x720. Rain and daylight use RGB controls,
24 frames per second and 35 inference steps. Snow uses WSM controls, 10 frames
per second and 50 inference steps.

Example request:

> Prepare the retrieval starter using storage profiles `inputs` and `outputs`.
> Keep the example paths and use collection `demo_av_retrieval_001`. Return the
> JSON and Airflow steps; do not submit.

Return the edited payload with generation disabled and the linked submission
steps. Explain that a successful run writes `search_result.json` and retrieved
clips under `results/` in its output folder.

For more examples, use:

- [Workflow requests](evals/evals.json): the `question` fields cover preparing
  retrieval, rain, snow and daylight payloads and explaining their results.
- [Generation prompt example](../../docs/autonomous-vehicle-dataset-generation/payload-guide.md#generate-a-variation):
  a snow edit for `cosmos.prompt`. The generation starters above contain the
  complete rain, snow and daylight prompts.
- [Retrieval walkthrough](../../docs/autonomous-vehicle-dataset-generation/getting-started.md#run-the-workflow)
  and [RGB-generation walkthrough](../../docs/autonomous-vehicle-dataset-generation/getting-started.md#next-step-generate-a-variation):
  input layout, run settings, Airflow submission and expected outputs.

## Instructions

### Prepare the Payload

Copy the selected starter. For S3-backed AV runs, use explicit
`s3://<bucket>/<path>` URIs for every storage path, including `input_video_path`,
`query_video` and `output_directory`. Replace all `msc://` placeholders in the
starter payload; use the actual backing bucket name, which may differ from the
MSC profile name. The current AV runtime does not support `msc://` paths throughout
the workflow.

For Azure requests, follow [Azure Storage with AV Dataset Generation](../../docs/autonomous-vehicle-dataset-generation/azure-support.md).
That guide requires deployment and code changes before running; changing URI
prefixes alone does not enable Azure support.

Set the user's paths and choose a unique collection name. Keep other starter
settings unless the user requests a documented change. Ask for missing inputs
or clarify whether the user wants retrieval or generation when needed.

- Source videos use `<scene-id>/camera_front_wide_120fov.mp4` and must be at
  least 9.3 seconds long. Keep outputs separate from inputs.
- The starters use `search_mode: "embedding"`. The query is processed separately
  from `limit`; all clips from its source are excluded. No remaining matches
  means the run fails.
- Retrieval keeps `cosmos.enabled: false`. Generation sets it to `true` and
  uses `cosmos.prompt` for the requested edit. `num_clips` selects matches;
  `num_augmentation` produces variations with the same prompt.
- For documented adjustments, read the [Payload Guide](../../docs/autonomous-vehicle-dataset-generation/payload-guide.md).
  It also describes indexing with `search_mode: "none"`. External service URLs,
  model selection and custom generation templates are not payload options.

For a payload-only request, return the edited JSON and unresolved prerequisites
without accessing the cluster or submitting a run.

For generation payload responses, explain the selected control mechanism:

- RGB: retrieved clips are resampled to `cosmos.fps` without changing playback
  speed, then used to derive edge controls. No WSM manifest is needed.
- WSM: each source scene's `wsm/wsm-manifest.json` identifies prepared controls
  matching the camera, frame rate, size and selected clip span. The workflow
  extracts the matching control-video section; it does not render controls.

### Check Prerequisites

Read [Prerequisites](../../docs/autonomous-vehicle-dataset-generation/prerequisites.md)
for GPU capacity, storage and the shared FFmpeg installation at `/opt/ffmpeg`.
Reuse the existing controller. Use shared [orchestration preflight](../cdf-orchestration-preflight/SKILL.md)
when readiness needs checking and [orchestration setup](../cdf-orchestration-setup/SKILL.md)
when installation or connection is requested.

Generation defaults to `cosmos.control_mode: "rgb"`. The workflow resamples
retrieved clips to `cosmos.fps` without changing playback speed and derives
edge controls from them. No WSM manifest is needed.

Only `cosmos.control_mode: "wsm"` requires prepared World Scenario Model (WSM)
controls. Each eligible source scene needs `wsm/wsm-manifest.json` linking its
video to a control that matches the camera, frame rate and size and covers the
selected span. The workflow extracts the matching section; it does not render
controls. Follow [WSM Controls](../../docs/autonomous-vehicle-dataset-generation/payload-guide.md#wsm-controls)
for preparation. Report missing or misaligned controls before WSM submission.

When verifying readiness, check storage access for the configured paths.
Accepting the payload format does not confirm that the workflow can read them.
Keep credentials in protected configuration, outside payloads.

### Run and Monitor

An explicit run request authorizes submission. Verify the deployed Airflow
workflow (DAG) supports
the payload. From the repository root, run
`python3 skills/cdf-orchestration-preflight/scripts/storage_check.py --payload <payload.json>`
and submit only if it prints `Storage mapping: ready`. An `UNMAPPED` line means
the run would fail at runtime; report the unmapped field and configured
prefixes. Using that bucket requires changing the controller's storage
configuration and redeploying it; with the user's authorization, route that
change to [orchestration setup](../cdf-orchestration-setup/SKILL.md), then rerun
the check. Then follow [Getting Started](../../docs/autonomous-vehicle-dataset-generation/getting-started.md#run-the-workflow)
for the Airflow UI or [Direct Airflow API](references/airflow-direct-api.md)
for API submission. If execution access is unavailable, return the payload and
UI steps, including how to follow task state in Airflow's Grid or Graph view,
without claiming a run started.

Record the run ID and continue observing it through terminal success or failure
within the user's requested observation window. A background poller alone does
not complete a request to monitor the run; wait for its result before ending
the turn. If observation must end early because of an observation limit or
loss of access, report the run ID, last observed state and how to resume checking
it. Do not promise a later update unless a supported continuation mechanism is
active.

If a submission response is uncertain, look up the same ID and compare its payload before
retrying. Do not create a duplicate run. For status-only requests, inspect the
existing run.

On failure, read the first failed task's log and report the cause supported by
that evidence. Successful cleanup alone does not establish workflow success.
For cancellation, mark the running task failed as documented so cleanup can
run; do not delete the DagRun or DAG.

### Read Results

Use the run folder under `output_directory`, with unsupported run-ID filename
characters replaced by `-`:

| Output | Meaning |
| --- | --- |
| `search_result.json` | Matching clips and source locations, or an indexing summary. |
| `results/` | Retrieved clips when `collect_result_clips` is true. |
| `cosmos3-nano/videos/` | Accepted videos that pass physical-plausibility and media checks; may be empty. |
| `cosmos3-nano/result_manifest.json` | Accepted `outputs` with `published_video` paths, `rejected` and `failed` jobs, and validation results when a manifest is published. |

Report the run state, output locations and available validation results.
Generation requires video validation and physical-plausibility checks. Candidates
must score strictly above `cosmos.plausibility_threshold` (default `0.5`) to be
accepted. Rejected candidates are filtered out; a run can succeed with no accepted
videos. Missing or invalid evaluation results, evaluator errors and media-validation
failures still fail the run. Watch accessible generated media to
confirm the requested edit and preservation of the scene; physical plausibility
alone does not establish either. State when visual review was not performed.
