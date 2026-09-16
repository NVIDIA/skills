---
name: tao-run-deft-cr-its-mining
description: >-
  Run the mining-based DEFT improvement workflow for ITS Cosmos-Reason binary video questions,
  focused on the non-reasoning classification/evaluation path. Use when the user asks for a
  DEFT CR ITS mining workflow, traffic-camera Cosmos Reason improvement loop, collision-identification workflow with data mining, or iterative Cosmos-RL refinement driven by gap analysis.
license: Apache-2.0
compatibility: Requires Docker with NVIDIA Container Toolkit, Python 3.11 with venv/ensurepip, numpy, pandas, pyarrow, PyYAML, and huggingface_hub, plus the selected platform CLI.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write Task
tags:
- application
- workflow
- deft
- its
- cosmos-reason
- cosmos-rl
- mining
---

# Skill: TAO Run DEFT CR ITS Mining

## Prerequisites

Before preflight, follow `references/host-prerequisites.md`; use its selected `DEFT_PYTHON` for every bundled helper and stop if the dependency probe fails.

## Bundled Resources

Resolve `DEFT_SKILL_ROOT` to the absolute directory containing this installed `SKILL.md`. The agent or plugin runtime resolves this path; it is not a user input. Invoke bundled helpers with `"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/<name>.py" ...`. Never require a skill-bank checkout or change the user's working directory to a repository root.

This workflow invokes `tao-finetune-cosmos-embed`, `tao-analyze-gaps-vlm-bcq`, `tao-mine-nearest-neighbors`, and the selected platform skill by registered skill name. Those skills own their credentials, actions, and bundled assets. Cosmos Reason is workflow-owned: resolve `images.tao_toolkit.deft_cosmos_reason`, generate TOMLs with this workflow's helpers and configured base templates, and submit the exact train/evaluate commands in `references/mining-loop.md` through the selected platform. Do not invoke `tao-finetune-cosmos-reason` for planning, templates, action bundles, commands, or hook resolution.

## User Inputs (DEFT Workspace and Workflow Configuration Yaml)

The user should provide a DEFT workspace absolute path that will be used for the entire run. This path should not be `/workspace` because that conflicts with the Cosmos Reason skills.

The layout of the DEFT workspace should be as follows:

```text
<deft_workspace>/
├── data/
├── hf_cache/
├── model/
├── specs/
└── results/
```

| Path | Purpose |
| --- | --- |
| `<deft_workspace>/data/` | User-provided KPI and training datasets, including LLaVA annotations and media directories. |
| `<deft_workspace>/hf_cache/` | Persistent Hugging Face cache used when workflow steps download or reuse HF models. |
| `<deft_workspace>/model/` | Local model/checkpoint inputs for the workflow, including the baseline Cosmos Reason checkpoint and any user-provided Cosmos Embed checkpoint. |
| `<deft_workspace>/specs/` | User-provided configuration files, including templates for Cosmos Reason train and evaluate, Cosmos Embed inference, and TAO Data Services mining. |
| `<deft_workspace>/results/` | All workflow-generated run artifacts, including state, logs, baseline evaluation, Cosmos Embed outputs, mining outputs, and iteration directories. |

Each workflow run writes under a run-specific directory inside `<deft_workspace>/results/`. If `run.name` is set in `workflow.yaml`, that exact value is used as the directory name; otherwise the workflow creates a timestamped directory. For example, `run.name: deft_cr_its_mining_test` writes baseline evaluation under `<deft_workspace>/results/deft_cr_its_mining_test/baseline/evaluate`.

The run directory layout is:

```text
<deft_workspace>/results/<run_name_or_timestamp>/
├── workflow.yaml
├── deft_state.json
├── loop_log.jsonl
├── bcq_accuracy_report.md
├── bcq_accuracy_summary.json
├── baseline/
│   └── evaluate/
│       └── bcq_accuracy_metrics.json
├── cosmos_embed_output/
│   ├── kpi/
│   └── train/
├── embedding_parquets/
│   ├── kpi/
│   │   └── embeddings.parquet
│   └── train/
│       └── embeddings.parquet
├── iter_1/
│   └── evaluate/
│       └── bcq_accuracy_metrics.json
├── iter_2/
└── ...
```

`workflow.yaml` is a copy of the high-level config that produced the run. `loop_log.jsonl` is the append-only event source for resume, and every stage-log append atomically refreshes `deft_state.json` as its current snapshot. `baseline/evaluate/` stores the initial evaluation of the baseline model. Each completed evaluation also stores `bcq_accuracy_metrics.json`. The run-level `bcq_accuracy_report.md` and `bcq_accuracy_summary.json` compare the baseline with every completed iteration. `cosmos_embed_output/kpi/` and `cosmos_embed_output/train/` store raw Cosmos Embed preparation and inference outputs. `embedding_parquets/kpi/` and `embedding_parquets/train/` store the fixed mining-ready parquet artifacts computed once from the KPI and train datasets. Each `iter_<N>/` directory stores the artifacts for one DEFT improvement iteration.

### Workflow Configuration

There should be a file called `<deft_workspace>/specs/workflow.yaml`, which will be the high-level workflow config for the full DEFT run:

```yaml
run:
  name: null
  max_iterations: 1

kpi_dataset:
  annotations_path: /abs/path/to/deft_workspace/data/kpi/annotations.json
  media_dir: /abs/path/to/deft_workspace/data/kpi/media

train_dataset:
  annotations_path: /abs/path/to/deft_workspace/data/train/annotations.json
  media_dir: /abs/path/to/deft_workspace/data/train/media

cosmos_reason:
  baseline_model_path: /abs/path/to/deft_workspace/model/reasoner_checkpoint
  base_evaluate_toml: /abs/path/to/deft_workspace/specs/cr_base_evaluate.toml
  base_train_toml: /abs/path/to/deft_workspace/specs/cr_base_train.toml
  continual_model: false

mining:
  embeddings_spec_template: /abs/path/to/deft_workspace/specs/cosmos_embed_inference_template.yaml
  embeddings_modality: both
  cosmos_embed_checkpoint_path: null
  embedding_parquets:
    kpi: null
    train: null
  mining_spec_template: /abs/path/to/deft_workspace/specs/mining_spec_template.yaml
  mine_unique_only: true
```

`run`, `kpi_dataset`, `train_dataset`, `cosmos_reason`, and `mining` are required; configured paths must be absolute inside `<deft_workspace>`. `cosmos_reason.baseline_model_path` must be a Cosmos Reasoner checkpoint. Convert a native Omni checkpoint first with `tao-finetune-cosmos-reason`. The optional `continual_model` defaults to `false`, which starts every iteration from the baseline. `embeddings_modality` selects KPI targets; train/source embeddings always include text and video. The optional `mine_unique_only` defaults to `true` and filters previously mined train/source paths from later iterations.

Write all workflow outputs under `<deft_workspace>/results`; do not add another output root. When `run.name` is set, use `<deft_workspace>/results/<run.name>` and explain that `baseline/`, `cosmos_embed_output/`, `embedding_parquets/`, and `iter_<N>/` are nested there. When it is `null`, create `<deft_workspace>/results/run_<YYYYMMDD_HHMMSS>` and record that path in `deft_state.json` so resume never recomputes it.

Before running any workflow stage, validate the config:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/verify_workflow_yaml.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

The validator checks required fields, workspace-contained absolute paths, `embeddings_modality`, path existence, Omni baselines, and compatibility between KPI/train LLaVA annotations and their `media_dir`. Every annotation needs a unique non-empty `id`; its `video` must resolve against that dataset's media directory. The Cosmos Embed template must declare a positive `inference.num_gpus`, which preflight prints for hardware verification. `mining.cosmos_embed_checkpoint_path` accepts `null`, an absolute workspace-local path, or a Hugging Face model id with an optional `hf_model://` prefix. Optional `mining.embedding_parquets.kpi` and `.train` values must be absolute existing mining-ready Parquet files inside the workspace with `filepath`, `embedding`, and `modality` columns. KPI must contain exactly the selected modalities; train must contain both text and video. The removed `text_embeddings` and `video_embeddings` fields are invalid. Each stage still follows its underlying skill, container, or platform runner's mount behavior.

If preflight rejects a Cosmos3 Omni baseline, tell the user that this workflow requires a Reasoner checkpoint, ask them to convert it first with `tao-finetune-cosmos-reason`, and exit without initializing the workflow.

If the user does not provide custom Cosmos Reason or Cosmos Embed templates, copy `$DEFT_SKILL_ROOT/assets/cr_base_evaluate.toml`, `$DEFT_SKILL_ROOT/assets/cr_base_train.toml`, and `$DEFT_SKILL_ROOT/assets/default_cosmos_embed_inference.yaml` into `<deft_workspace>/specs/`. Use `tao-mine-nearest-neighbors` to copy its bundled `assets/default_nearest_neighbors.yaml` when the user does not provide a mining template. Point `workflow.yaml` at the workspace copies. The bundled Cosmos Embed template requests 8 GPUs; do not launch until preflight confirms the selected platform can satisfy that request, or the user approves a modified template.

Initialize a run once after validation:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/initialize_workflow.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml"
```

Use the printed `run_dir` as `RUN_DIR` for every later command. If `run.name` is `null`, never ask another preparation script to derive the run directory again; pass `--run-dir "$RUN_DIR"` so all stages write into the initialized run.

Do not use `--force` for an ordinary resume. It rewrites the state/config snapshot but intentionally leaves `loop_log.jsonl` and all stage artifacts in place. Use it only to repair the snapshot for the same run; choose a new `run.name` for a clean restart. Before resuming, run `"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/resume_position.py" --run-dir "$RUN_DIR"` and continue from the reported stage.

## Baseline Evaluation

Run baseline evaluation once before the DEFT iteration loop. This evaluates `cosmos_reason.baseline_model_path` on the KPI dataset and produces the first `results.json` used by gap analysis. Immediately run `scripts/compute_bcq_accuracy_metrics.py` on that file and store `baseline/evaluate/bcq_accuracy_metrics.json`. Its parser recognizes `yes` and `no` in short or free-form responses, including capitalization and punctuation. It reports false positives, false negatives, accuracy, balanced accuracy, and unparseable predictions. Use `references/mining-loop.md` for the exact commands, completion check, and logging rule.

## Initialize Mining Embeddings

Initialize mining embeddings once before the DEFT iteration loop. The KPI and train datasets are fixed for the run, so their Cosmos Embed outputs are reusable across all iterations.

Use `scripts/prepare_cosmos_embed_inference.py` to prepare the Cosmos Embed inference specs from `workflow.yaml`. The KPI dataset requires the modality or modalities selected by `mining.embeddings_modality`; the train dataset always requires text and video. If the user provides a complete dataset Parquet under `mining.embedding_parquets.kpi` or `.train`, the preparation script stages it at `$RUN_DIR/embedding_parquets/<dataset>/embeddings.parquet` and generates no inference specs for that dataset. While staging, it remaps text embedding identifiers from the source run's question files to the current run's lookup by question content; embeddings and video identifiers are preserved. If the dataset value is `null` or omitted, the script generates every required modality spec for that dataset. Partial per-modality reuse is not supported. If `mining.cosmos_embed_checkpoint_path` is an absolute local path, the script uses it for generated specs. If it is a remote Hugging Face model id, with or without the `hf_model://` prefix, the script downloads or reuses it under `<deft_workspace>/hf_cache` and writes the local downloaded checkpoint path into the generated inference specs. If the field is `null`, the script uses `nvidia/Cosmos-Embed1-224p`.

The script downloads remote HF checkpoints before Cosmos Embed runs because Cosmos Embed startup can race when `inference.num_gpus > 1` and multiple workers try to download the same model at once. If both combined dataset Parquets are provided, the preparation script does not need the checkpoint and does not download it.

Use `scripts/prepare_cosmos_embed_inference.py` once. It prepares both the KPI dataset and the train dataset:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_cosmos_embed_inference.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR"
```

The preparation script writes only the missing embedding specs under the initialized `RUN_DIR`. Initialization owns the `$RUN_DIR/workflow.yaml` snapshot; this stage does not derive a new run or overwrite that snapshot.

For each dataset without a supplied combined Parquet, the preparation script writes all required modality-specific inference specs under `cosmos_embed_output/<dataset>/specs/`, creates raw Cosmos Embed result directories, stages text questions as files for stable joins, and writes `lookup.parquet`. After inference, the conversion script gathers those separate modality outputs into one mining-facing file per dataset. Each row contains `filepath`, `embedding`, and `modality`:

```text
$RUN_DIR/embedding_parquets/kpi/embeddings.parquet
$RUN_DIR/embedding_parquets/train/embeddings.parquet
```

The KPI parquet contains only the selected modality or modalities. The train parquet always contains both text and video rows.

### Run Cosmos Embed Inference

Use `tao-finetune-cosmos-embed` for each generated inference spec. Follow the mandatory terminal-exit validation, permission repair, and conversion procedure in `references/mining-loop.md`. Do not run inference or conversion for a dataset whose combined Parquet was supplied during preparation.

The generated lookup Parquet uses `annotation_id` for the original LLaVA `id` and `video_path` for resolved media. Do not rename either column to `video_id`; that external field belongs to Cosmos Reason and gap-analysis records.

## DEFT Mining Loop

Run iterations `1..run.max_iterations`. The loop is mining-only: no PAIDF or generated-video branch runs in this skill. Before gap analysis, each iteration rewrites Cosmos Reason result `video_id` values from LLaVA annotation ids to resolved KPI video paths and stores the prepared predictions in the iteration's `gaps/` directory. It then runs gap analysis, builds one selected-modality target parquet, mines it against the combined text/video train source in one `tao-mine-nearest-neighbors` run, converts mined rows to LLaVA annotations, merges annotations, trains Cosmos Reason, evaluates the trained checkpoint, computes that evaluation's BCQ accuracy metrics, and removes the resumable training checkpoints while preserving the exported safetensors. At loop termination, generate the run-level accuracy report and include its baseline/iteration table in the agent's final response. See `references/mining-loop.md` for exact commands and completion criteria.

## Completion Criteria

| Stage | Complete when |
| --- | --- |
| `validate_workflow` | `verify_workflow_yaml.py` exits successfully. |
| `initialize_workflow` | `$RUN_DIR/workflow.yaml` and `$RUN_DIR/deft_state.json` exist. |
| `baseline_evaluate` | The evaluate job exits successfully, exactly one baseline `results.json` is found, and `baseline/evaluate/bcq_accuracy_metrics.json` exists. |
| `prepare_cosmos_embed_inference` | Each dataset has a lookup parquet and either all required Cosmos Embed specs or a staged combined embedding Parquet. |
| `cosmos_embed` | Every generated inference spec has a current `completion_validation.json`; exit `0` is accepted after output validation, while exit `130` additionally records a teardown warning. |
| `convert_embeddings` | `embedding_parquets/{kpi,train}/embeddings.parquet` exist; train contains both modalities and KPI contains the selected modalities. |
| `gap_analysis` | `$RUN_DIR/iter_<N>/gaps/predictions.json` exists and the container exits successfully. The workflow counts valid rows directly from `kpi_gaps.jsonl`; missing or empty output after successful completion means zero weak samples. |
| `prepare_nearest_neighbor_mining` | One target parquet, optional filtered source parquet, and one nearest-neighbor spec exist. |
| `mine_nearest_neighbors` | One mined-neighbor parquet and mining summary exist. |
| `record_mined_paths` | When `mine_unique_only` is true, `$RUN_DIR/mining/mined_paths_log.parquet` exists; otherwise the stage is logged as skipped. |
| `prepare_cosmos_reason_train` | Mined and accumulated LLaVA annotations plus `train/specs/train.toml` exist, with a positive expected optimizer-step count. |
| `train` | The Cosmos Reason job reaches terminal success, reports at least one optimizer step, and has writable timestamped output and checkpoint directories. |
| `evaluate` | Evaluation preparation finds the latest completed training checkpoint, the evaluation job exits successfully, exactly one iteration `results.json` is found, and its `bcq_accuracy_metrics.json` exists. |
| `cleanup_cosmos_reason_training` | `train/checkpoint_cleanup.json` exists, raw checkpoint directories are gone, and its listed safetensors exports remain. |
| `loop_stop` | Stop reason is logged and `bcq_accuracy_report.md` plus `bcq_accuracy_summary.json` compare the baseline with every completed iteration. |

## Troubleshooting

**Prediction id does not match an annotation**: Confirm the evaluation used the same KPI annotations configured by `kpi_dataset.annotations_path`. Every Cosmos Reason result `video_id` must match a LLaVA annotation `id` or an already-resolved annotation `video` path.

**No target embeddings matched**: Check that `gaps/predictions.json` and gap-analysis `video_id` paths match the KPI LLaVA media paths used during embedding preparation. Text mode also requires the question text to match after removing `<video>` and trailing “Answer with yes or no.”

**Mined neighbors do not join to train lookup**: Confirm every mined `filepath` exists in `embedding_parquets/train/embeddings.parquet`. Its `modality` determines whether the path must match train `lookup.parquet` `filepath` for text or `video_path` for video.

**Run directory looks nested under `results/<run.name>`**: This is expected. `run.name` is the run directory name under `<deft_workspace>/results`; all baseline, embedding, and iteration artifacts are nested there.

**Docker output permissions or Cosmos Reason status paths fail**: Stop and follow the permission-repair and stage-local TAO status contracts in `references/mining-loop.md`.

**Unparseable prediction count is nonzero**: Report the count to the user and inspect the corresponding raw `results.json` responses. These predictions count as incorrect in accuracy and class recall; the metrics script does not silently drop them. An unparseable ground truth stops metric computation because the expected class is undefined.

**Cosmos Reason reports `No space left on device` for `/dev/shm/nccl-*`**: This is container shared memory, not ordinary disk capacity. For local Docker, run the Cosmos Reason action through `tao-run-on-docker` and require `--shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864`. Confirm these flags in the submitted job before retrying. Other platforms must provide equivalent shared-memory and memlock settings.
