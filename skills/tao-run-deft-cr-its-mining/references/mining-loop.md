# DEFT CR ITS Mining Loop Reference

Use this reference when running the iteration loop after the DEFT workspace and `workflow.yaml` have been validated.

Resolve `DEFT_SKILL_ROOT` to the absolute directory containing the installed `tao-run-deft-cr-its-mining/SKILL.md`; never ask the user for it. Export `WORKSPACE_DIR="$WORKSPACE"`, select `DEFT_PYTHON` with `scripts/deft_python.sh`, and invoke every bundled helper with that interpreter. The user's working directory does not need to be the plugin or repository root.

## Stage Skills

Read each listed underlying skill before invoking its stage. Cosmos Reason is the exception: this workflow owns its specs and commands, and the selected platform owns execution.

| Stage | Owner | Owns |
| --- | --- | --- |
| Cosmos Reason train/evaluate | This workflow + selected platform | DEFT image key, generated TOMLs, and exact commands; platform submission, logs, status, and terminal-state detection. |
| Cosmos Embed inference | `tao-finetune-cosmos-embed` | Inference command, image, credentials, mounts, and submitted job. |
| VLM BCQ gap analysis | `tao-analyze-gaps-vlm-bcq` | Gap spec generation, data-services action, image, and outputs. |
| Nearest-neighbor mining | `tao-mine-nearest-neighbors` | Default template, spec validation, data-services action, image, and outputs. |
| Job execution | Selected `tao-run-on-*` platform skill | Resource allocation, submission, status, logs, and terminal-state detection. |

## Initialization

If the user does not provide custom Cosmos Reason or Cosmos Embed templates, copy them from `$DEFT_SKILL_ROOT/assets/` into `<deft_workspace>/specs/`. Use `tao-mine-nearest-neighbors` to copy its bundled default template when the user does not provide a mining template. Point `workflow.yaml` at the workspace copies.

Initialize a run once:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/initialize_workflow.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

Use the printed `run_dir` as `RUN_DIR` for every later command. If `run.name` is `null`, never let another preparation script derive a new run directory; pass `--run-dir "$RUN_DIR"`.
An orchestrator that resolves the run directory before initialization may also pass that absolute path to this command with `--run-dir "$RUN_DIR"`.

`--force` rewrites `deft_state.json` and the workflow snapshot but does not remove the existing loop log or stage artifacts. Use it only to repair the snapshot for the same run. Start a clean run with a new `run.name` instead.

Append one stage event after each completed, skipped, or failed stage:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/log_stage.py" \
  --log-path "$RUN_DIR/loop_log.jsonl" \
  --iter-label "iter_${ITER}" \
  --stage mine_nearest_neighbors \
  --status ok \
  --summary "mined nearest train neighbors" \
  --duration-sec "$DURATION_SEC" \
  --artifact "$RUN_DIR/iter_${ITER}/mining/mined_neighbors.parquet"
```

Each append rebuilds `deft_state.json` atomically from valid log events. A truncated or otherwise malformed log line is ignored without blocking later appends. Before resuming, refresh state and ask the helper for the next unfinished stage:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/resume_position.py" \
  --run-dir "$RUN_DIR"
```

Use the reported stage; do not infer resume position from directory names or rerun iteration 1 by default.

## Cosmos Reason Platform Runtime

Resolve `images.tao_toolkit.deft_cosmos_reason` from the installed skill bank's `versions.yaml` and keep the resulting literal image URI as `DEFT_COSMOS_REASON_IMAGE`. Do not substitute `images.tao_toolkit.cosmos_rl` or any image selected by another skill.

Generate Cosmos Reason TOMLs only with this workflow's bundled helpers. `prepare_cosmos_reason_train.py` loads `cosmos_reason.base_train_toml`, and `prepare_cosmos_reason_evaluate.py` loads `cosmos_reason.base_evaluate_toml`; when the user does not provide custom base TOMLs, initialization copies the bundled files from `assets/` and points `workflow.yaml` at them. Do not load templates from `tao-finetune-cosmos-reason` or invoke its planner or action bundle.

The pinned runtime commands are:

```text
cosmos-rl --config <train.toml> /opt/cosmos_rl/tao_sft_example.py
cosmos-rl-evaluate --config <evaluate.toml>
```

Before the first Cosmos Reason launch, inspect `DEFT_COSMOS_REASON_IMAGE` through the selected platform and require `cosmos-rl`, `cosmos-rl-evaluate`, and `/opt/cosmos_rl/tao_sft_example.py` to exist. Stop with a runtime-contract error if any is missing. Do not invoke `cosmos_workflow.py`, inspect `skill_info.yaml` for a replacement command, derive the hook from `cosmos_rl.__file__`, or submit a model-skill action bundle.

Run every Cosmos Reason train/evaluate action through the selected platform skill with the pinned commands above and the TOML prepared by this workflow. Confirm the submitted job records `DEFT_COSMOS_REASON_IMAGE` before launch. For local or remote single-node Docker, select `tao-run-on-docker`; do not construct a competing unmanaged Docker launch. Require its submitted container command to include `--shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864`. Kubernetes or Slurm must provide equivalent shared-memory and memlock resources. Keep the platform job id and use the platform's status and log operations until the job reaches a terminal state.

Before every Cosmos Reason launch, set `STAGE_DIR` and `TAO_JOB_ID` from this table:

| Launch | `STAGE_DIR` | `TAO_JOB_ID` |
| --- | --- | --- |
| Baseline evaluate | `$RUN_DIR/baseline/evaluate` | `baseline-evaluate` |
| Iteration train | `$RUN_DIR/iter_${ITER}/train` | `iter-${ITER}-train` |
| Iteration evaluate | `$RUN_DIR/iter_${ITER}/evaluate` | `iter-${ITER}-evaluate` |

Set `CONTAINER_STAGE_DIR` to the exact writable container destination where the selected platform mounts `STAGE_DIR`. The submitted job must record the read-write mapping `STAGE_DIR:CONTAINER_STAGE_DIR`; a one-to-one mapping is valid. Do not infer `CONTAINER_STAGE_DIR` from the image working directory. If the platform submission does not expose this mapping, stop before launch.

Create `$STAGE_DIR/.tao-status/$TAO_JOB_ID/` on the host and verify it is writable. Because it is below the stage mount, its container path is `$CONTAINER_STAGE_DIR/.tao-status/$TAO_JOB_ID/`. Set:

```text
TAO_API_JOB_ID=$TAO_JOB_ID
TAO_API_RESULTS_DIR=$CONTAINER_STAGE_DIR/.tao-status
TAO_STATUS_FILE=$CONTAINER_STAGE_DIR/.tao-status/$TAO_JOB_ID/status.json
```

Confirm the submitted job contains the mount and all three environment variables before launch; otherwise the TAO status decorator may fall back to writing `./results` under the image's unwritable `/workspace` directory.

## Baseline Evaluation

Generate the baseline evaluate TOML:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_cosmos_reason_evaluate.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR"
```

Run the pinned `cosmos-rl-evaluate --config` command with `$RUN_DIR/baseline/evaluate/specs/evaluate.toml`. After the job exits successfully, restore host write access before result discovery:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/restore_docker_mount_permissions.py" \
  --path "$RUN_DIR/baseline/evaluate" \
  --docker-image "$DEFT_COSMOS_REASON_IMAGE"
```

The helper exits without changing anything when the directory is already writable. Do not continue unless it succeeds. Then locate the result and compute the authoritative binary metrics:

```bash
BASELINE_RESULTS_JSON="$("$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/find_cosmos_reason_results.py" \
  --evaluate-dir "$RUN_DIR/baseline/evaluate")"

"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/compute_bcq_accuracy_metrics.py" \
  --results-json "$BASELINE_RESULTS_JSON" \
  --output-json "$RUN_DIR/baseline/evaluate/bcq_accuracy_metrics.json"
```

Ignore metrics emitted by Cosmos-RL evaluate; `compute_bcq_accuracy_metrics.py` and its `bcq_accuracy_metrics.json` output are the source of truth for this workflow. The metrics script reads `response`/`gt` and also accepts `answer`/`ground_truth`. It extracts `yes` and `no` defensively from capitalization, punctuation, and short free-form answers. An unparseable prediction is reported and counted as incorrect; an unparseable ground truth is a hard error. Report the printed accuracy, balanced accuracy, false-positive count, false-negative count, and unparseable count in the baseline stage update. Log both `results.json` and `bcq_accuracy_metrics.json` as `baseline_evaluate` artifacts. The stage is not complete until the metrics file exists.

## Mining Embeddings

Prepare fixed KPI/train Cosmos Embed inputs once:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_cosmos_embed_inference.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR"
```

Run `tao-finetune-cosmos-embed` inference for every generated spec under:

```text
$RUN_DIR/cosmos_embed_output/kpi/specs/
$RUN_DIR/cosmos_embed_output/train/specs/
```

Only run specs that exist. KPI specs follow `mining.embeddings_modality`; train specs always cover text and video. If `mining.embedding_parquets.kpi` or `.train` supplied a complete combined dataset Parquet, there are no specs for that dataset and its mining-ready output is already staged under `$RUN_DIR/embedding_parquets/<dataset>/embeddings.parquet`. Staging remaps text identifiers to the current run's lookup by reading the source question files, so those files must remain readable; video identifiers and embedding vectors are unchanged.

Read `inference.num_gpus` from each generated spec and request exactly that many GPUs from the selected platform. The bundled template defaults to 8 GPUs. The preparation script prints the count for every generated modality; stop before launch if the platform cannot satisfy it.

Use the registered inference action exactly as `cosmos-embed1 inference -e <generated-spec>`. Do not append a `results_dir=...` CLI override: OmegaConf would replace the generated spec's modality-specific output path. Before submission, inspect the rendered command and require the selected platform to make the spec's absolute workspace paths visible and writable at those same paths inside the container. Do not infer output location from the container working directory.

Keep the exact container image selected by `tao-finetune-cosmos-embed` as `COSMOS_EMBED_IMAGE`. Handle every generated spec independently. Set `INFERENCE_SPEC` to that spec's absolute path and `DATASET` to its `kpi` or `train` parent dataset. Run the inference action through `tao-finetune-cosmos-embed` and the selected platform. Monitor it to a terminal state and record its exact numeric exit code as `COSMOS_EMBED_EXIT_CODE`, including when the platform classifies the job as failed. After the container exits, restore host write access if needed with the same image and the dataset directory containing that spec:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/restore_docker_mount_permissions.py" \
  --path "$RUN_DIR/cosmos_embed_output/$DATASET" \
  --docker-image "$COSMOS_EMBED_IMAGE"
```

Then validate the terminal result and existing outputs:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/validate_cosmos_embed_output.py" \
  --inference-spec "$INFERENCE_SPEC" \
  --exit-code "$COSMOS_EMBED_EXIT_CODE"
```

The validator reads `results_dir` from the generated spec and requires metadata and a finite two-dimensional NPY array there, exact input/output counts and identifiers, and one unique `npy_row` for every embedding. Exit `0` passes only when those checks pass. Exit `130` is the one permitted nonzero code because the affected runtime can return it during `torchrun` teardown after successful output; the validation artifact records `ok_with_teardown_warning`. Every other nonzero exit fails. Existing outputs remain reusable when they still match the unchanged inference spec. If the platform cannot report an exact numeric code, do not infer `130` from logs or files; stop and ask how the user wants to proceed.

After every generated spec passes `check` or `complete`, log one `cosmos_embed` `ok` event with every `completion_validation.json` as an artifact. If there are no generated specs because both combined Parquets were supplied, log the stage as skipped. Do not log this stage complete from platform status alone.

Convert completed outputs only for datasets that generated inference specs:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/cosmos_embed_outputs_to_parquet.py" \
  --output-dir "$RUN_DIR/cosmos_embed_output/kpi" \
  --parquet-dir "$RUN_DIR/embedding_parquets/kpi" \
  --embedding-modality "$EMBEDDING_MODALITY"

"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/cosmos_embed_outputs_to_parquet.py" \
  --output-dir "$RUN_DIR/cosmos_embed_output/train" \
  --parquet-dir "$RUN_DIR/embedding_parquets/train" \
  --embedding-modality both
```

These commands write `$RUN_DIR/embedding_parquets/kpi/embeddings.parquet` and `$RUN_DIR/embedding_parquets/train/embeddings.parquet`. The converter reads each modality's `results_dir` from its generated spec; it never reconstructs the producer path or searches for misplaced files. Skip the corresponding command when preparation already staged a supplied combined Parquet at that path. The KPI file contains the selected modality or modalities. The train file always contains both, and each row records its `modality` alongside `filepath` and `embedding`.

For text outputs, conversion matches each Cosmos Embed metadata `text` value to the corresponding lookup question. Repeated questions consume their lookup rows in occurrence order. `npy_row` selects only the embedding vector; it is not treated as a lookup-row index. Conversion fails if Cosmos Embed returns missing, extra, or unmatched text occurrences.

## Iteration Loop

Run iterations `1..run.max_iterations`. Before each stage, run `resume_position.py`; `loop_log.jsonl` is the event source and `deft_state.json` is its refreshed snapshot.

1. **Gap analysis**: for iteration 1, source predictions come from the baseline `results.json`; for later iterations, they come from the previous iteration's `results.json`. Cosmos Reason may write a LLaVA annotation id in `video_id`, which is not necessarily the annotation's media path and is ambiguous for downstream video joins when one video has multiple questions. Rewrite each result through the fixed KPI annotations before generating the gap-analysis spec:

```bash
PREDICTIONS_JSON="$RUN_DIR/iter_${ITER}/gaps/predictions.json"

"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_gap_analysis_predictions.py" \
  --results-json "$PREVIOUS_RESULTS_JSON" \
  --annotations-json "$KPI_ANNOTATIONS_JSON" \
  --media-dir "$KPI_MEDIA_DIR" \
  --output-json "$PREDICTIONS_JSON"
```

Invoke `tao-analyze-gaps-vlm-bcq` with `predictions_json=$PREDICTIONS_JSON`, no `videos_dir` because the prepared ids are absolute paths, `results_dir=$RUN_DIR/iter_${ITER}/gaps`, and `output_spec=$RUN_DIR/iter_${ITER}/gaps/vlm_bcq_spec.yaml`. Let that skill use its bundled helper to generate the spec and run the action.

`KPI_ANNOTATIONS_JSON` is `kpi_dataset.annotations_path` from `workflow.yaml`. The preparation script preserves every prediction row and changes only `video_id`; multiple annotation ids may therefore resolve to the same video path while retaining their separate questions. After the submitted container exits successfully, count the valid JSON-object rows in `$RUN_DIR/iter_${ITER}/gaps/kpi_gaps.jsonl` and report that weak-sample count. A missing or empty file after successful completion means there are no weak samples. In that case, log `loop_stop` with reason `no_weak_samples`; otherwise continue with mining. On resume, `resume_position.py` performs this validation and count directly from `kpi_gaps.jsonl`.

2. **Prepare nearest-neighbor mining**:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_nearest_neighbor_mining.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER" \
  --gaps-jsonl "$RUN_DIR/iter_${ITER}/gaps/kpi_gaps.jsonl"
```

The command writes one `$RUN_DIR/iter_<N>/mining/target.parquet`. Gap-analysis `video_id` remains the external resolved media path; internally it joins to lookup `video_path`. Text target rows are failed `(video_path, question)` embeddings, and video target rows are unique failed-video embeddings. With `embeddings_modality: both`, both row types are appended to the same target. When `mining.mine_unique_only` is true or omitted, the single generated spec points at `$RUN_DIR/iter_<N>/mining/filtered_source.parquet`, which excludes source filepaths already recorded in `$RUN_DIR/mining/mined_paths_log.parquet`.

3. **Mine nearest neighbors**: use `tao-mine-nearest-neighbors` once with `$RUN_DIR/iter_<N>/mining/nearest_neighbors.yaml`. Every target row is queried independently against the combined text/video train source. The stage is complete when `$RUN_DIR/iter_<N>/mining/mined_neighbors.parquet` and its mining summary exist.

4. **Record mined paths**: when `mining.mine_unique_only` is true or omitted, update the cumulative mined-path log after the nearest-neighbor run completes.

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/record_mined_paths.py" \
  --mined-neighbors-parquet "$RUN_DIR/iter_${ITER}/mining/mined_neighbors.parquet" \
  --mined-log-parquet "$RUN_DIR/mining/mined_paths_log.parquet"
```

Future iterations filter their train/source embedding pool with the cumulative log. This does not remove records from the already assembled training annotation JSON.

When `mining.mine_unique_only` is false, do not run the command; append a `record_mined_paths` event with `status=skipped` so resume can advance unambiguously.

5. **Prepare Cosmos Reason training**:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_cosmos_reason_train.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER"
```

Nearest-neighbor output contains only selected source filepaths. The preparation script joins those paths back to the train embeddings parquet to recover each source row's modality, writes `$RUN_DIR/iter_<N>/mining/mined_train_annotations.json`, accumulates it into `$RUN_DIR/iter_<N>/train/train_annotations.json`, and writes `$RUN_DIR/iter_<N>/train/specs/train.toml`. A text source path selects its corresponding train question; a video source path selects every train annotation row whose `video_path` matches. Output records preserve the source `annotation_id` as their LLaVA `id` and are deduplicated by that stable id.

Iteration 1 trains on mined annotations only; `train_dataset.annotations_path` remains a mining source pool and is not inserted directly. Later iterations accumulate the previous iteration's assembled annotations. Iteration 1 starts from `cosmos_reason.baseline_model_path`. Later iterations use the checkpoint recorded in the previous iteration's generated evaluate TOML only when `cosmos_reason.continual_model: true`; otherwise they start from the baseline. Preparation counts the assembled annotations and calculates expected optimizer steps from `train.epoch`, `train.train_batch_per_replica`, `policy.parallelism.dp_shard_size`, `dp_replicate_size`, and `train.train_policy.dataloader_drop_last` (which defaults to `true` when absent). If the result is zero, stop before launch, report the values, and tell the user that the training parameters likely need tuning for that iteration's dataset size and GPU count.

6. **Train Cosmos Reason**: run the pinned `cosmos-rl --config <train.toml> /opt/cosmos_rl/tao_sft_example.py` command with `$RUN_DIR/iter_<N>/train/specs/train.toml`. Keep monitoring the submitted job until it reaches terminal success. Do not infer completion from checkpoint files appearing during training. After terminal success, require the final structured status or training log to report at least one completed optimizer step. If it reports zero, stop before checkpoint discovery or evaluation, tell the user the parameters likely need tuning, and ask how to proceed. Then restore host write access before checkpoint discovery or evaluation preparation:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/restore_docker_mount_permissions.py" \
  --path "$RUN_DIR/iter_${ITER}/train" \
  --docker-image "$DEFT_COSMOS_REASON_IMAGE"
```

Do not continue unless the helper succeeds.

7. **Prepare and run evaluation**: after the training job reaches terminal success, prepare evaluation:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_cosmos_reason_evaluate.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --iteration "$ITER"
```

The preparation command finds the latest `epoch_<N>` safetensors checkpoint under this iteration's completed train directory, prints the selected path, and writes it into `$RUN_DIR/iter_<N>/evaluate/specs/evaluate.toml`. If no checkpoint exists, stop before launching evaluation. Run the pinned `cosmos-rl-evaluate --config` command with that TOML. After the job exits successfully, restore host write access before result discovery:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/restore_docker_mount_permissions.py" \
  --path "$RUN_DIR/iter_${ITER}/evaluate" \
  --docker-image "$DEFT_COSMOS_REASON_IMAGE"
```

Do not continue unless the helper succeeds. Then locate the result and compute that iteration's metrics:

```bash
ITERATION_RESULTS_JSON="$("$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/find_cosmos_reason_results.py" \
  --evaluate-dir "$RUN_DIR/iter_${ITER}/evaluate")"

"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/compute_bcq_accuracy_metrics.py" \
  --results-json "$ITERATION_RESULTS_JSON" \
  --output-json "$RUN_DIR/iter_${ITER}/evaluate/bcq_accuracy_metrics.json"
```

Report the printed accuracy, balanced accuracy, false-positive count, false-negative count, and unparseable count in the iteration update. Log both files as `evaluate` artifacts. The evaluate stage is not complete until `bcq_accuracy_metrics.json` exists. Use `ITERATION_RESULTS_JSON` as the next iteration's `PREVIOUS_RESULTS_JSON`.

8. **Clean Cosmos Reason training checkpoints**: after evaluation and metric computation complete, remove the large resumable Cosmos-RL checkpoints while retaining the exported safetensors used by later iterations:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/cleanup_cosmos_reason_training.py" \
  --train-dir "$RUN_DIR/iter_${ITER}/train"
```

The command removes only each `<timestamp>/checkpoints/` directory and the train directory's `best/checkpoints` link. It also removes a legacy `<timestamp>/best/checkpoints` link when present. It preserves every `<timestamp>/safetensors/epoch_<N>` export, training logs, specs, and annotations. It refuses to delete anything unless at least one safetensors epoch export exists, and writes `$RUN_DIR/iter_<N>/train/checkpoint_cleanup.json`. Log that report as the `cleanup_cosmos_reason_training` artifact before starting the next iteration. If Docker ownership prevents cleanup, restore write access to the iteration train directory with `restore_docker_mount_permissions.py` and the Cosmos Reason image, then retry this stage.

## Final Accuracy Report

After the loop stops because it reached `max_iterations` or found no weak samples, generate the baseline/iteration report. Also generate it after a failed run when baseline metrics are available, so completed evaluations are not lost from the final account.

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/summarize_bcq_accuracy_metrics.py" \
  --run-dir "$RUN_DIR"
```

This writes `$RUN_DIR/bcq_accuracy_report.md` and `$RUN_DIR/bcq_accuracy_summary.json`. Read `bcq_accuracy_report.md` and include its full accuracy table in the agent's final response; providing only the artifact path is not sufficient. Log both report files as `loop_stop` artifacts together with the stop reason.

## Completion Criteria

| Stage | Complete when |
| --- | --- |
| `validate_workflow` | `verify_workflow_yaml.py` exits successfully. |
| `initialize_workflow` | `$RUN_DIR/workflow.yaml` and `$RUN_DIR/deft_state.json` exist. |
| `baseline_evaluate` | The evaluate job exits successfully, exactly one baseline `results.json` is found, and `baseline/evaluate/bcq_accuracy_metrics.json` exists. |
| `prepare_cosmos_embed_inference` | Each dataset has a lookup parquet and either all required Cosmos Embed specs or a staged combined embedding Parquet. |
| `cosmos_embed` | Every generated inference spec has a current `completion_validation.json`; validated exit `130` is recorded as a teardown warning, and all other nonzero exits fail. |
| `convert_embeddings` | `embedding_parquets/{kpi,train}/embeddings.parquet` exist with the required modalities. |
| `gap_analysis` | `$RUN_DIR/iter_<N>/gaps/predictions.json` exists and the container exits successfully. The workflow counts valid rows directly from `kpi_gaps.jsonl`; missing or empty output after successful completion means zero weak samples. |
| `prepare_nearest_neighbor_mining` | The target, optional filtered source, and nearest-neighbor spec exist. |
| `mine_nearest_neighbors` | One mined-neighbor parquet and mining summary exist. |
| `record_mined_paths` | The cumulative log exists when enabled, or a skipped event is logged when disabled. |
| `prepare_cosmos_reason_train` | Mined and accumulated LLaVA annotations plus `train/specs/train.toml` exist, with a positive expected optimizer-step count. |
| `train` | The Cosmos Reason job reaches terminal success, reports at least one optimizer step, and has writable timestamped output and checkpoint directories. |
| `evaluate` | Evaluation preparation finds the latest completed training checkpoint, the evaluation job exits successfully, exactly one iteration `results.json` is found, and its `bcq_accuracy_metrics.json` exists. |
| `cleanup_cosmos_reason_training` | `train/checkpoint_cleanup.json` exists, all timestamped `checkpoints/` directories and `best/checkpoints` links are absent, and the listed safetensors exports still exist. |
| `loop_stop` | Stop reason is logged; the run-level Markdown and JSON accuracy reports cover the baseline and every completed iteration. |
