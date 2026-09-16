# PAIDF Workflow Runner Debugging

Use this skill when the user asks for an end-to-end agentic auto-labeling run, a
container-by-container experiment, generated container command review, or a
data-flow debugging pass, especially when mixed-resolution inputs may need
conditional super-resolution.

The primary executor is the `workflow-runner` service from
`services/workflow_runner`, not ad hoc agent-authored `docker run` commands.
Follow the numbered Workflow below for the exact procedure and its gates. OSMO or
Airflow should own scheduling, distributed execution, retry policy, queues, and
resource placement; this skill is only repo-specific runner usage and debugging
guidance.

## Workflow

The flow is a linear spine Step 1 -> 2 -> 3 -> 4 -> 5 with two explicitly
bounded loops and four terminal states: review-only report (Step 4),
invalid-plan stop (Step 3), validated success (Step 5), and escalation after a
failed rerun (Step 5). Steps 1-3 only inspect and plan (no side effects); Step 4
is the only step that executes, builds images, or mutates outputs, and runs only
with explicit user approval.

1. Identify the cookbook or explicit stage sequence.

2. Run a dry plan:
   `make run SCRIPT=workflow-runner:main ARGS='--cookbook-file <config> --container-dry-run'`.

3. **Is the plan valid?** (single decision)
   - Trigger: the dry-run plan from Step 2 is available.
   - The plan is valid only if all hold: selected stages appear in the fixed
     canonical relative order (see Guardrails) with no retired `daft_export`
     stage; images, build targets, mounts, endpoints, and GPU settings are
     correct; when SR is selected, the SeedVR runtime is baked into the image and
     checkpoints exist under the model cache; and, for mixed-resolution inputs,
     SR `--resolution-policy auto` is set and the caller `media_path` /
     active-media handoff is preserved.
   - Invalid -> fix the cookbook, stage list, build flags, mounts, endpoints, or
     model availability and return to Step 2. Bounded loop: at most three
     correction passes; if still invalid, STOP and report the blocker (terminal).
   - Valid -> go to Step 4.

4. **Is execution requested and approved?** (execution gate)
   - Trigger: a valid plan from Step 3.
   - Dry-run / command-review only, or the user says not to execute -> report the
     plan and STOP (terminal); do not execute, build images, or mutate outputs.
   - Execution requested and explicitly approved -> run the whole selected
     pipeline once through `workflow-runner:main`. The runner executes the
     selected stages internally in canonical order; the agent does not re-invoke
     per stage. Choose the build mode as a value, not a branch:
     `--container-ensure-images build-if-missing` builds only missing images;
     `--container-build-images` rebuilds all images. Continue to Step 5.

5. **Validate outputs.** After the single run completes, confirm each selected
   stage produced its expected outputs, in canonical order. This is a finite
   check over the selected stages, not a re-execution loop.
   - All expected outputs present -> report success (terminal).
   - A stage failed or is missing outputs -> follow *Failure Handling* in
     [runner-debugging-patterns.md](workflow-runner-debugging/runner-debugging-patterns.md):
     stop unless the stage is optional; after fixing the root cause, rerun from
     the failed stage. Bounded loop: at most one rerun per fixed root cause - if
     the same stage fails again with no new cause, STOP and escalate with the
     saved log and exact command (terminal).

## References

- Read [data-handoff.md](workflow-runner-debugging/data-handoff.md) before changing or
  debugging active-media behavior.
- Read [runner-debugging-patterns.md](workflow-runner-debugging/runner-debugging-patterns.md)
  before writing experiment scripts, inspecting generated container commands, or
  debugging cookbook stages.

## Examples

Inspect the plan, execute while building only missing images, then validate:

```bash
# Inspect the generated container plan (no execution)
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'

# Execute, building only images that are missing
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-ensure-images build-if-missing'

# Validate stage outputs before reporting success
ls -lhR output/auto_labeling/<scenario>/
```

## Guardrails

- Do not bypass the runner for production experiments unless the user is
  explicitly debugging a single container command.
- Do not modify the original `media_path` in caller-authored input JSONL;
  runtime components such as `prepare_input` may rewrite
  `data_entry.media_path` to `sidecars/active.*`, and downstream code such as
  `SceneContext.from_input(data_entry.media_path)` is expected to operate on
  those runtime rewrites.
- For mixed-resolution video, prefer SR `--resolution-policy auto` over
  unconditional SR so high-resolution clips keep the existing active media.
- Do not mount broad host paths when runner-managed media/data/config/cache
  mounts are sufficient.
- Prefer existing `make build IMAGE=...` registrations and runner image flags
  over ad hoc image names.
- Plain `stages` dry-runs follow the canonical relative order. Cookbooks using
  `workflow.nodes` follow their validated stable topological order and may
  repeat a stage. Confirm every plan/result/log entry preserves its node ID and
  repeated nodes use distinct sidecar namespaces. Shared `stage_args` apply to
  every occurrence; node `args` are appended last.
- The single `daft_export` stage is retired: DAFT `task/` writing belongs to
  `reasoning`, and dataset aggregation belongs to the new `training_export`
  stage. Do not plan, run, or debug a `daft_export` stage.
- The `person_attribute_search` stage (Visual Attribute Search product)
  builds the `event-and-person-attribute-search-service` image; the stage key stays
  `person_attribute_search`.
