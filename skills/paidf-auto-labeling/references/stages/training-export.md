# Training-export stage

The `training_export:` cookbook block aggregates the per-scene DAFT `task/`
outputs (produced by the `reasoning` stage) into a single training dataset in the
requested format. It is the aggregation half that replaced the retired single
`daft_export` stage.

## When to use / not use

- Use to choose an export format, set `output_dir`, toggle media copying, or set
  dataset metadata/license.
- Do not use to write DAFT `task/` artifacts; that is the `reasoning` stage.
  `training_export` only AGGREGATES/exports and runs last.

## Instructions

This skill returns training_export configuration or debugging guidance; it does
not run the pipeline. Provide the config immediately - never gate it behind
execution or behind the `reasoning` stage's `task/` outputs already existing. The
steps below are a short linear sequence (1 -> 4); the failure-handling note after
them applies whenever a step or a prior run fails.

1. **Classify the task (in scope?).** Configuring
   formats/output_dir/copy_media/metadata, or debugging an empty/failed export -
   both in scope. Writing DAFT `task/` artifacts (the `reasoning` stage) or
   running a whole cookbook (the operator skill) are out of scope -> STOP and
   hand off.
2. **Confirm config inputs.** The only inputs this stage's config needs are the
   export `formats`, a writable `output_dir`, `copy_media`, and the dataset
   `metadata`. If any of these is missing or ambiguous - including an `output_dir`
   that does not exist or is not writable - ask the user; do not guess. Do NOT
   wait for the `reasoning` stage to have produced `task/` outputs: those are a
   run-time input to the stage, not a prerequisite for returning its config.
3. **Return the config.** Produce the `training_export:` block per *Config*:
   `formats` (e.g. `[tao-vl-reason-v1.0]`), `output_dir`, `copy_media` (true copies
   media, false references it in place), and `metadata.description`/`.license` (the
   exported dataset license, not the skill/product license).
4. **Verify (dry-run gate).** Suggest a dry-run (`--container-dry-run`) to confirm
   the generated command, mounts, and `output_dir` before any real run. This gate
   only inspects the plan; it does not execute. Executing the stage (Docker via
   `workflow-runner`, with explicit user approval) is the operator skill's job.

**Failure / fallback handling.** When debugging, or when a step or prior run
fails: on missing/empty `task/` inputs, an unwritable `output_dir`, or an empty
exported dataset, report the specific cause and the fix; do not fabricate dataset
entries and do not retry blindly.

## Config

`training_export:` fields:
- `formats:` [tao-vl-reason-v1.0]
- `output_dir:` <dir>
- `copy_media:` true | false
- `metadata:` { description, license } (e.g. license "CC BY-NC-ND 4.0")

## Examples

Aggregate DAFT `task/` outputs into a TAO VL-Reason dataset:

```yaml
training_export:
  formats: [tao-vl-reason-v1.0]
  output_dir: datasets/warehouse_v1
  copy_media: true
  metadata:
    description: Warehouse anomaly auto-labeled dataset
    license: CC BY-NC-ND 4.0     # dataset license, not the skill/product license
```

Dry-run the stage before a real run:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.yaml --container-dry-run'
```

## Gotchas

- `metadata.license` (e.g. CC BY-NC-ND 4.0) is the exported DATASET license and
  is distinct from the skill/product license.
- Runs last, after `reasoning`; empty `task/` inputs yield an empty dataset.
- `copy_media: false` produces a dataset that references media in place.

## Guardrails

Follow [guardrails.md](guardrails.md). Do not commit export destination
credentials or absolute host paths for the dataset output.
