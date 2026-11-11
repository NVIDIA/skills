---
name: tao-run-dinov3-ssl-deft
description: Run iterative DINOv3 SSL DEFT data selection and retraining from an unlabeled pool, using GRIT or multi-task weakness and fixed-encoder retrieval. Use for active-learning-style DINOv3 mining, not plain pretraining/fine-tuning (tao-train-dinov3) or AOI/PAS DEFT.
license: Apache-2.0
compatibility: Requires the selected platform's native launcher and a TAO Data Services image containing the DINOv3 SSL DEFT modules; no host TAO or Python data-stack installation.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- application
- workflow
- deft
- dinov3
- ssl
---


# DINOv3 SSL DEFT

The DS-packaged Python controller owns the loop and durable state. This skill
prepares and approves its inputs; it does not implement orchestration.

## Instructions

1. Resolve the DINOv3 model using the model resolver. Choose `grit-score`
   for unlabeled target ranking; its scorer is built in. Offer
   `multi-task-round-robin` only when the user already has a score adapter
   that meets the [task scoring contract](references/adapter-contracts.md#task-scoring):
   DS ships no multi-task scorer, and `validate` rejects the recipe's
   placeholder. Read [architecture.md](references/architecture.md) for the
   scientific meaning, task weighting, and stopping policy.
2. Read [tao-container-integration.md](references/tao-container-integration.md).
   Check release readiness before offering a launch: the resolved
   `images.tao_toolkit.data_services` image must be a release build whose
   installed packages contain the reviewed DEFT modules and pass packaged
   validation. An older DS build is not sufficient. The placeholder rejection
   (step 1), the cohort checks and the canceled-run exit code 143 need a DS
   build that contains tao-data-services#59: the first DS 7.3.0 RC after
   rc-90. DS rc-90 and older accept the placeholder at `validate`, fail in
   round 1 with `Missing configured inputs: [... 'score_implementation_0']`,
   and exit 0 after a cancel. The `release/7.3.0` cherry-pick of this skill
   must bump `images.tao_toolkit.data_services` in `versions.yaml` to that RC.
   The executable profile is
   one allocated DS container; no external runner ships with this application.
   For release QA, the repository script `scripts/deft_container_handoff.py`
   runs each repository's regression tests (`--suite source`) and the shipped
   package checks (`--suite packaged`) inside the approved images; the
   packaged suite also rejects a DS image that predates #59. Run it from
   clean checkouts of the release SHAs; the default `--base-ref origin/main`
   diffs each release from its merge base. It is a QA tool, not a launcher.
3. Follow the selected platform's launch gate. For Docker, use the concrete
   allocation, preflight, and `docker exec` lifecycle in the integration
   reference. Inside that runtime, follow the preparation sequence in
   [adapter-contracts.md](references/adapter-contracts.md#prepare-fixed-mining-embeddings):
   produce source/target embeddings with the same fixed encoder, register the
   source store with its payload contract, and write the target contract.
   No additional host/model package installation is needed. Generate a packaged recipe with `init`,
   fill paths/resources, then run `validate <config>`, `preflight <config> --gpu`
   in the launch runtime, and `plan <config>`.
   Read [adapter-contracts.md](references/adapter-contracts.md) when declaring a
   held-out benchmark ([isolation](references/adapter-contracts.md#held-out-benchmark-isolation)),
   or configuring custom scoring/evaluation, indexed retrieval, or resource policies.
4. Inspect the plan and present the approval contract below. Read
   `references/skill_info.yaml:path_contract` as the input-mount checklist:
   every applicable path must be visible in the allocated container, and
   `output.run_dir` must match the platform's durable results directory.
   Skill Bank resolves image keys on the host; DS accepts explicit image
   references only and never reads this repository at runtime.
5. After approval, execute `run <config>` inside the allocated container.
   Monitor to a terminal result using the commands below. Do not modify
   controller state or locks manually.

## Approval contract

Populate these six items from preflight and the plan, not from assumptions:

- Loop: chosen strategy, maximum rounds, stops, evaluation scope and patience.
  For `multi-task-round-robin`, the user's score adapter command and its
  `implementation_files`.
- Data: immutable target/source versions, verified row/shard counts, parent
  history, storage capacity and mount locations. For any held-out benchmark,
  the identity sidecar and its declared columns, even with evaluation disabled,
  and preflight's `contracts.benchmark_isolation` (at least 1; otherwise stop)
  ([held-out benchmark isolation](references/adapter-contracts.md#held-out-benchmark-isolation)).
- Sampling: fixed target budget, resolved integer task quotas, neighbors,
  radius/floor and duplicate thresholds; index, audit, probes/depth for ANN.
- Training: passes, resolution, resources/update floor and immutable original
  checkpoint. Every candidate starts from that checkpoint, never its predecessor.
  Check the per-round update budget against LR warm-up and last-layer freeze;
  follow [short-round schedule sizing](references/adapter-contracts.md#short-round-schedule-sizing).
- Cache/output: read-only images/embeddings, any platform cache, node-local
  `TAO_LOCAL_SCRATCH`, cumulative
  locator manifests, checkpoint retention, and durable results directory.
- Monitoring: cadence, stage/round updates, retry/failure handling and final reason.

Do not request launch approval while any of these items is unknown.

## Lifecycle and monitoring

All commands below follow `python -m nvidia_tao_ds.mining.dinov3.workflow`
inside the same allocated DS runtime:

- `preflight [<config>] [--gpu]` (with the config, also checks the allocation)
- `init --recipe grit-score --output run.yaml` (or `multi-task-round-robin`,
  which needs the user's score adapter; see step 1)
- `validate <config>`, `plan <config>`, `run <config>`, `resume <config>`
- `status <run_dir> [--config <config>]`
- `logs <run_dir> <client_job_id> [--cursor <cursor>] [--config <config>]`
- `cancel <run_dir> [--config <config>]`
- `report <run_dir> [--config <config>]`
- `adopt-training <continuation_config>` only for an explicitly approved
  continuation into a new run directory.

Read the job ID from status before requesting logs. Resume takes the config
path, not the run directory. Cancellation is terminal for that run: `run` and
`resume` then exit with code 143, not 0. Use an explicit new-run continuation
when appropriate.

Keep monitoring attached until the workflow is terminal. Use the user's
cadence, otherwise about five minutes for short stages and 20–30 minutes for
long search/training stages. Report stage/round transitions, mining yield,
metrics, retries, failures, and the final stop reason. Submission is not success.
If the container has exited, use the integration reference's read-only
inspection recipe rather than attempting `docker exec` against it.

## Related Skills

- `tao-train-dinov3`: ordinary DINOv3 training without a mining loop.
- `tao-launch-workflow`, `tao-run-on-docker`, `tao-data-io`: launch and storage.
- `tao-run-deft-aoi`, `tao-run-deft-pas`: different supervised DEFT workflows.
