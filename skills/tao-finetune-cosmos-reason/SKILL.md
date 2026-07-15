---
name: tao-finetune-cosmos-reason
description: Cosmos3-Nano video QA supervised fine-tuning with FSDP parallelism. Use when training or evaluating video
  question-answering models, fine-tuning Cosmos3-Nano or compatible Cosmos Reason models with SFT/LoRA, or working with
  Cosmos-RL. Trigger phrases include "fine-tune Cosmos", "Cosmos3 Nano Reasoner", "Cosmos-RL SFT",
  "video QA fine-tune", "Cosmos3-Nano training".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- video
- qa
- cosmos
- sft
- reasoning
- vlm
---

# Cosmos-RL

Supervised fine-tuning (SFT) of Cosmos Reason video QA models. The packaged
default base model is **hf_model://nvidia/Cosmos3-Nano**. Pretrained weights
are sourced from HuggingFace, not NGC. Gated HuggingFace models require
`HF_TOKEN`. Some Cosmos-RL images cannot load the native Cosmos3 Omni checkpoint
format directly; for those images, convert Cosmos3-Nano to a Qwen3-VL HF
safetensors directory before train/evaluate and use that converted directory as
the PTM path.

Uses FSDP-based parallelism with `dp_shard_size` for GPU count and `dp_replicate_size` for node count (not the standard `num_gpus`/`num_nodes`).

Requests for "Cosmos Reason 3", "Cosmos3 Nano Reasoner", or
`nvidia/Cosmos3-Nano` are handled by this skill. There is no separate Cosmos3
model directory in the skill bank; route those requests here. Override the base
HuggingFace model only when the user explicitly asks for a different model.

Deep detail lives in references; load the smallest one that matches the task:

- `references/cosmos-reason-launch.md` — launch intake, preflight, per-action dataset requirements, spec construction, typical overrides.
- `references/cosmos-reason-evaluate.md` — evaluate (flat TOML, task types, LoRA eval, selective download, results) and datasets.
- `references/cosmos-reason-automl.md` — AutoML/HPO policy and search-space guidance.
- `references/cosmos-reason-parameters.md` — important parameters, hardware, error patterns, DEFT/gap analysis, parent-model inference mappings.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, and `quantize` stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Credentials

- **HF_TOKEN** (required for gated models): HuggingFace access token. For the
  packaged default, the user must accept the model agreement at
  <https://huggingface.co/nvidia/Cosmos3-Nano> and provide a token with read
  access. If the user explicitly overrides the base model, they must accept
  that target model's agreement too. Passed to the container as a
  `docker_env_var`.

## Cosmos3 Checkpoint Conversion

When a selected image cannot load the native Cosmos3 checkpoint format
(`model_type="cosmos3_omni"` or `Cosmos3ForConditionalGeneration`), do not patch
QwenVL, Transformers, or vLLM first. Use the upstream Cosmos Framework VLM
conversion path to produce a Qwen3-VL HF safetensors directory, then point
Cosmos-RL specs at that converted directory.

The model skill packages a helper:

```bash
python skills/models/tao-finetune-cosmos-reason/scripts/prepare_cosmos3_vlm_checkpoint.py \
  --checkpoint-path /abs/path/Cosmos3-Nano \
  --output-path /abs/path/Cosmos3-Nano-VLM \
  --secrets-env ~/.tao/secrets.env \
  --validate-with-image <cosmos-rl-image>
```

After conversion, use the converted directory consistently as the PTM:

```text
train:    policy.model_name_or_path=/abs/path/Cosmos3-Nano-VLM
evaluate: model.model_name=/abs/path/Cosmos3-Nano-VLM
evaluate: model.base_model_path=/abs/path/Cosmos3-Nano-VLM
```

For local Docker, mount the converted directory read-only into the Cosmos-RL
container and set the spec to the container path. If a converted copy already
exists and validates, reuse it for PTM baseline evaluation, AutoML
recommendations, and final best-checkpoint evaluation rather than converting
again.

## Training Requirements

- **Dataset type:** vlm
- **Formats:** llava, daft
- **Accepted dataset intents:** training, evaluation, testing
- **Monitoring metric:** val/avg_loss, val/reward_avg, val/loss
- **Dataset URI examples:** `s3://bucket/cosmos/train`, `s3://bucket/cosmos/eval`, `/lustre/fsw/tao_datasets/cosmos_rl/train`, `/lustre/fsw/tao_datasets/cosmos_rl/eval`
- **Input modes:** accept either dataset roots or direct spec-key paths. Root mode maps `<root>/annotations.json` plus `<root>` as the media path. Direct spec mode is valid when annotations and media live in different locations, for example `custom.train_dataset.annotation_path=/lustre/.../train.json` and `custom.train_dataset.media_path=/lustre/.../videos.tar.gz`.
- **Media handling:** do not ask the user to choose `videos.tar.gz` vs `images.tar.gz` unless they are using direct spec mode or the model/action requires a single media archive. In root mode, pass the dataset root as the media path.
- **Annotation validation:** before launching train/AutoML/evaluate, verify the
  annotation JSON is readable and the referenced media path or archive is
  visible from the selected platform. Do not block, patch, or mutate
  annotations solely because optional fields are absent.
- **Per-record video FPS:** the packaged train template uses
  `custom.vision.nframes`, so per-record `video_fps` is not required by
  default. If the user switches to `custom.vision.fps`, selects a dataset
  profile that requires per-record timing, or uses an image/version that
  requires `video_fps`, make it a preflight requirement with
  `--json-required-field train_annotation=video_fps` and
  `--json-required-field val_annotation=video_fps` before any download or
  job launch.

## Spec construction

cosmos-rl is `mode: config`. **Always start from the packaged
`references/spec_template_<action>.yaml` for the requested action** — load it
as your base spec via `yaml.safe_load(...)` and apply user overrides on top.
Don't rebuild from scratch.

```python
import yaml
from pathlib import Path

skill = Path.home() / "tao-sdk/tao-skills-external/skills/models/tao-finetune-cosmos-reason"
action = "train"  # train, evaluate, inference, or quantize
specs = yaml.safe_load((skill / f"references/spec_template_{action}.yaml").read_text())
# Now apply your overrides on top of `specs`.
```

The reference TOML (and the spec the model actually consumes) is **nested
dicts**, not flat dotted keys. Dotted notation in override examples denotes
*paths into the nested spec* — walk the path and assign at the leaf. See
`skills/platform/tao-run-platform/SKILL.md`'s "spec is nested dicts" callout.
Data source overrides are **mandatory for every action**.

The packaged template keeps `custom.vision.nframes=8` for bounded 1-GPU memory;
switch to `fps` only after checking token budget and GPU memory, and delete
`custom.vision.nframes` from the spec when you do.

See `references/cosmos-reason-launch.md` for launch intake, the full
`check_tao_launch_preflight.py` slurm/local-Docker examples, the
`video_fps` preflight example, S3 staging, the GPU resource/architecture gate,
the per-action dataset requirements table, the `/workspace` mount caveat,
the quantize compatibility shim, and the full typical-overrides list.

## Critical Overrides (Train)

These are the keys whose template defaults are wrong or where omission flips the run into a different mode:

| Parameter | Template Default | Required Value | Why |
|---|---|---|---|
| `policy.model_name_or_path` | `hf_model://nvidia/Cosmos3-Nano` | Direct Docker: `nvidia/Cosmos3-Nano`, `hf_model://nvidia/Cosmos3-Nano`, or a local HF snapshot path. SDK/managed platform predownload: `hf_model://nvidia/Cosmos3-Nano`. | Keep the train and evaluate base model aligned. |
| `policy.model_max_length` | 40960 | Keep at 40960 or higher | Smaller than ~40k causes `vision_embeds` shape mismatch on video inputs |
| `train.train_batch_per_replica` | 32 | Any multiple of `train.train_policy.mini_batch` | Mismatch raises an immediate AssertionError |
| `train.train_policy.type` | `"sft"` | Keep as `"sft"` for SFT workflows | If dropped during agent regeneration, cosmos-rl flips to RL mode → rollout replica allocated → multi-node attempted → hostname errors when `num_nodes=1` |

## Evaluate

The evaluator reads a flat TOML config (`dataset`, `model`, `task`,
`evaluation`, `vision`, `generation`, `metrics`, `results`, `num_gpus`,
`results_dir`); the `actions.evaluate` block in `references/skill_info.yaml`
declares inputs and outputs. See `references/cosmos-reason-evaluate.md` for the
flat-TOML config detail, task types (`""` General Evaluator vs
`"its_directionality"`), LoRA evaluation via spec_overrides, selective download,
results/metrics, and the datasets section.

## AutoML / HPO Notes

The packaged default base model is `hf_model://nvidia/Cosmos3-Nano`; apply it
consistently to train (`policy.model_name_or_path`) and post-training evaluation
(`model.base_model_path`) unless the user provides a different model. See
`references/cosmos-reason-automl.md` for accuracy-vs-`val/avg_loss` objective
selection, the `eval_fn` per-recommendation evaluate flow, the knob mapping
(learning rate, batch size, epochs, weight decay, warmup ratio), example
`custom_param_ranges`, `train_sample_count` batch-size capping,
`ordered_int` requirements, and the pre-launch recommendation summary.

## Parameters, Hardware, Errors, DEFT, Inference

For parallelism, set `policy.parallelism.dp_shard_size` = GPUs per node and
`policy.parallelism.dp_replicate_size` = node count (1 for single node).
Cosmos-RL handles distributed init internally via FSDP and does not rely on
platform-level `MASTER_ADDR`/`WORLD_SIZE`; submit with
`gpu_count=<gpus_per_node>` and `num_nodes=<N>` and the spec keys drive
sharding. Cosmos-RL models are 8B parameters; recommended 8x A100 or H100
(80GB each).

See `references/cosmos-reason-parameters.md` for important parameters (training
loop, model/policy, parallelism incl. multi-node FSDP, optimization, vision
encoders, checkpointing incl. the `best` symlink/`epoch_*` resolution,
validation, logging), hardware sizing, the full error-pattern catalog (CUDA OOM,
LoRA-eval OOM, NaN loss, `vision_embeds` mismatch, quantize token mismatch,
batch-size divisibility and per-rank limits, stale cache, scheduler-None,
gated-repo `HF_TOKEN`, GPU resource/architecture gate, status-logging warnings),
DEFT support and `scripts/analyze_gaps.py` gap analysis, and the parent-model
inference mapping table.
