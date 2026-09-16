# Workflow Runner Debugging Patterns

Use the PAIDF `workflow-runner` service as the local execution layer. The runner
owns Docker or Podman command construction, path normalization, identity bind
mounts, stage ordering, image build dispatch, dry-run mode, and policy handling.
It is not a scheduler; OSMO or Airflow should own production scheduling,
distributed execution, retries, queues, and resource placement.

Raw `docker run` commands are useful as diagnostics, but they should not be the
primary production path when the runner can express the workflow.

## Runner Commands

Dry-run a cookbook:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
```

Run the same cookbook and build only the missing images first
(`--container-ensure-images build-if-missing`); use `--container-build-images`
instead only when you intend to rebuild all images:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <cookbook-config> --container-ensure-images build-if-missing'
```

Run explicit stages (selected stages keep the canonical relative order):

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--input-file payloads/simple.jsonl --stages captioning visual_qa reasoning training_export'
```

## Build Pattern

Prefer repo registrations:

```bash
make build IMAGE=super-resolution-service:build
make build IMAGE=detection-and-tracking-service:sam3
make build IMAGE=captioning-service:main
make build IMAGE=workflow-runner:build
```

Cookbooks should reference registered service images and any required build
target through runner config and stage flags.

## Runner Runtime Pattern

- DataEntry local media parents are mounted read-only.
- DataEntry local scene directories are mounted read-write.
- Runner-generated input manifests are mounted read-only.
- Cookbook prompt/question-bank/config paths are mounted read-only.
- Model cache paths are mounted read-write when configured.
- Remote URI-like paths are passed through and not mounted by the local runner.

## SeedVR2 SR Pattern

The SR service image bakes the pinned SeedVR runtime into `/opt/seedvr`; do not
mount or symlink SeedVR source for normal auto-labeling runs. The large
checkpoints still come from the configured model cache unless the service is
explicitly run with checkpoint download enabled.

Expected model-cache layout:

```text
<model-cache>/seedvr2/ema_vae.pth
<model-cache>/seedvr2/seedvr2_ema_3b.pth
```

For mixed-resolution media, pass SR gate args through the cookbook or
`--stage-arg`:

```yaml
stage_args:
  super_resolution:
    - --resolution-policy
    - auto
    - --min-input-short-side
    - "720"
    - --min-input-long-side
    - "1280"
```

The gate probes each staged active media file. Inputs below either threshold run
SeedVR2; inputs meeting both thresholds are recorded as skipped and keep the
existing active media. Validate real SR on representative hardware before a
large run; the 3B 720p path can exceed A40 memory depending on resolution,
window size, and active GPU load.

## Failure Handling

- Stop after a failed stage unless the cookbook marks the stage as optional.
- If a failed stage is marked optional, continue to the next stage, but still
  validate that any expected outputs from the optional stage exist before
  proceeding; a missing optional output can still break downstream inputs.
- Keep the failed stage log and the exact command.
- Do not repair downstream outputs manually; after fixing mounts, endpoints, or
  model availability, rerun from the failed stage. Rerun at most once per fixed
  root cause: if the same stage fails again with no new cause identified, STOP
  and escalate with the saved log and command instead of looping.
