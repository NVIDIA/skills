# Troubleshooting

Use the [skill routes](../SKILL.md#component-details) for container-internal configuration or
runtime failures: augmentation YAML/model/evaluator errors belong to `cdf-augmentation`;
auto-labeling stage/prompt errors belong to `cdf-auto-labeling`. Include the failing task,
sanitized error, generated config, and deployed image version when consulting those references.
Environment preparation belongs to `cdf-orchestration-setup`; a requested custom stage graph
belongs to `cdf-orchestration-write-dag`. The checks below diagnose the existing VDA integration.

## Payload rejected or behaves unexpectedly

- Use a positive `max_videos` for a bounded run. `null`, zero, and negative integers all select
  every matching input.
- Match every nested `output_directory` and `external_services` value to the top level.
- External mode needs endpoints in `cosmos`, `captioning`, `visual_qa`, and `reasoning`; the
  runtime schema does not currently reject a missing Cosmos Transfer URL, but the task needs it.
- Include a non-empty video distribution whenever `cosmos` is present. Omitting it can select the
  shared model's clothing-attribute default.
- `cosmos_transfer_model` does not override the bundled `nvidia/Cosmos3-Nano` model.
- `visual_qa.question_bank_file` is replaced by the DAG's bundled question bank.
- Unknown fields may be ignored. Validate spelling and inspect the normalized payload.

If the run processes defaults rather than the confirmed payload, inspect the DagRun `conf`. The
trigger body must use `{"conf":{"payload":{...}}}`; payload fields directly under `conf` are
not the same contract.

## OSMO run fails before processing

- OSMO requires `external_services: true` and disabled lifecycle entries.
- Supply every VLM, LLM, and Cosmos Transfer endpoint used by all stages.
- Supply an external Cosmos Evaluator URL when any check is enabled, or disable every check.
- Confirm the endpoint network is reachable from OSMO tasks and `osmo_rtx6000_1` has capacity.

## DAG not found

- Confirm the requested ID is `video_data_augmentation_dag_k8s` or
  `video_data_augmentation_dag_osmo`.
- Check Airflow import errors and the corresponding manifest under
  `airflow/dags/workflows/video_data_augmentation_dag/configs/`.
- If an updated DAG is not present in the existing installation, report that mismatch. Do not
  reinstall Airflow without explicit deployment authorization.

## No input videos

- Confirm the storage prefix contains `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm` objects.
- For local input, use `scripts/upload_videos.py --validate-only` before uploading.
- Confirm storage credentials can list the input and that the selected positive limit includes at
  least one file.

## Tasks remain queued or services never become ready

- Check `k8s_rtx6000_1` plus the internal/external VDA augmentation pool.
- Inspect actual free GPUs and existing workloads, not only node allocatable totals.
- Internal Kubernetes mode needs one shared language-model GPU, two Cosmos Transfer GPUs, batch
  task GPUs, and one additional GPU for each enabled managed checker. Hallucination, attribute
  verification, and physical plausibility are all GPU-backed.
- A lifecycle entry can wait forever when the pool has slots but Kubernetes cannot place the pod;
  inspect pod events, image pulls, PVCs, model access, and node resources.
- Service pods restarting with `Orchestrator did not become ready within <N>s` mean model
  initialization exceeded its `--init-timeout`. Follow
  [model startup timeouts](../../cdf-orchestration-setup/references/model-startup-timeouts.md).
  Propose a timeout increase to the user.

## Cosmos candidates are missing, incomplete, or rejected

- Inspect `cosmos_augmentation` task logs and each candidate's `config.yaml`, `caption.txt`, and
  `metadata.json`.
- A missing `output.mp4` indicates generation did not finish. Missing sidecars make the candidate
  incomplete.
- Evaluator `operational_success=false` is an evaluator failure; `quality_passed=false` is a
  quality rejection. Report them separately.
- The output branch proceeds with complete passing candidates and fails when none remain.

## Run failed but cleanup succeeded

Use the DagRun terminal state and the first failing processing task. Shutdown tasks use cleanup
semantics and can succeed after a processing failure; successful cleanup does not mean the dataset
is ready.

## Stopping a run

Mark the running task failed in Airflow so shutdown tasks can execute. Do not delete the DagRun or
DAG, which can bypass cleanup and leave deployments, services, or GPU pods.
