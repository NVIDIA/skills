# Cosmos-Reason Data And Specs

Dataset contracts, launch intake, spec construction, typical overrides, and train-critical overrides from the pre-refactor guide.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Dataclass Schemas
- Train Action Policy
- Credentials
- Training Requirements
- Launch Intake Reminder
- Per-Action Dataset Requirements
- Spec construction
- Now apply your overrides on top of `specs` (next section).
- Typical Spec Overrides
- Slurm/internal example:
- TRAIN_DATASET_URI = "/lustre/fsw/tao_datasets/cosmos_rl/train"
- EVAL_DATASET_URI = "/lustre/fsw/tao_datasets/cosmos_rl/eval"
- Direct spec-path example:
- TRAIN_ANNOTATION_PATH = "/lustre/fsw/.../annotations_train.json"
- TRAIN_MEDIA_PATH = "/lustre/fsw/.../videos_train.tar.gz"
- EVAL_ANNOTATION_PATH = "/lustre/fsw/.../annotations_eval.json"
- EVAL_MEDIA_PATH = "/lustre/fsw/.../eval_videos"
- Critical Overrides (Train)

# Cosmos-RL

Supervised fine-tuning (SFT) of Cosmos Reason video QA models. The packaged
default base model is **hf_model://nvidia/Cosmos3-Nano**. Pretrained weights
are sourced from HuggingFace, not NGC. Gated HuggingFace models require
`HF_TOKEN`.

Uses FSDP-based parallelism with `dp_shard_size` for GPU count and `dp_replicate_size` for node count (not the standard `num_gpus`/`num_nodes`).

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

## Training Requirements

- **Dataset type:** vlm
- **Formats:** llava
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

### Launch Intake Reminder

When prompting for Cosmos-RL train or AutoML data, list the actual spec keys as
an option. Users may provide roots, or they may directly provide:

- `custom.train_dataset.annotation_path`
- `custom.train_dataset.media_path`
- `custom.val_dataset.annotation_path`
- `custom.val_dataset.media_path`

For root mode, explain the automatic mapping: `train_root` maps to
`custom.train_dataset.annotation_path=train_root/annotations.json` and
`custom.train_dataset.media_path=train_root`; `eval_root` maps the same way for
`custom.val_dataset`.

Before train or AutoML runner generation, resolve the action=train container
image from `references/skill_info.yaml` and `versions.yaml` (or the packaged
`scripts/resolve_tao_image.py` helper), show the exact image to the user, and
ask whether to use it or override with `image=<override>`. Do not silently
launch on the default image. This skill does not package a
`skills/models/tao-finetune-cosmos-reason/config.json` file.

For launch preflight, pass the concrete annotation and media paths to the
shared helper:

```bash
scripts/check_tao_launch_preflight.py --platform slurm \
  --path train_annotation=/lustre/.../train/annotations.json \
  --path train_media=/lustre/.../train \
  --path val_annotation=/lustre/.../eval/annotations.json \
  --path val_media=/lustre/.../eval \
  --gpu-min-count 4 \
  --gpu-min-memory-gb 80 \
  --gpu-arch-allowlist cosmos_rl=sm_80,sm_90,sm_100,sm_120
```

For Cosmos-RL, count and memory are necessary but not sufficient. Treat the run
as launchable only when the target has at least 4 GPUs with 80GB-class memory or
higher, the GPU architecture is in the image-supported allowlist above, and the
normal Docker/platform, S3, and credential preflight checks pass. A remote image
manifest that advertises `linux/arm64` only proves CPU architecture support; it
does not prove CUDA SM support. Spark/GB10 `sm_121` must be blocked for this
image unless direct image introspection confirms `sm_121` support or the user
chooses a newer compatible image.

### Per-Action Dataset Requirements

The packaged Cosmos-RL model action metadata declares **train**, **evaluate**,
**inference**, and **quantize** (`references/skill_info.yaml` and
`schemas/manifest.json`). Do not advertise export, prune, deploy, or dataset
convert for Cosmos-RL unless those actions are added to the model metadata.

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | custom.train_dataset.annotation_path | train_datasets | annotations.json | No |
| train | custom.train_dataset.media_path | train_datasets | dataset root containing media payload | No |
| train | custom.val_dataset.annotation_path | eval_dataset | annotations.json | No |
| train | custom.val_dataset.media_path | eval_dataset | dataset root containing media payload | No |
| evaluate | dataset.annotation_path | eval_dataset | annotations.json | No |
| evaluate | dataset.media_dir | eval_dataset | dataset root containing media payload | No |
| inference | media | inference_dataset | one image/video or a media folder/archive | No |
| quantize | dataset.annotation_path | calibration_dataset | annotations.json | No |
| quantize | dataset.media_dir | calibration_dataset | dataset root containing media payload | No |

## Spec construction

cosmos-rl is `mode: config`. **Always start from the packaged
`references/spec_template_<action>.yaml` for the requested action** — load it
as your base spec via `yaml.safe_load(...)` and apply user overrides on top.
Don't rebuild from scratch. See `skills/platform/tao-run-platform/SKILL.md`'s "Constructing the
spec / args" section for the load-template-then-override pattern.

```python
import yaml
from pathlib import Path

skill = Path.home() / "tao-sdk/tao-skills-external/skills/models/tao-finetune-cosmos-reason"
action = "train"  # train, evaluate, inference, or quantize
specs = yaml.safe_load((skill / f"references/spec_template_{action}.yaml").read_text())
# Now apply your overrides on top of `specs` (next section).
```

The reference TOML (and the spec the model actually consumes) is **nested dicts**, not flat dotted keys. The dotted notation in the override examples below denotes *paths into the nested spec* — the agent must walk the path and assign at the leaf, not store the dotted string as a literal key. See `skills/platform/tao-run-platform/SKILL.md`'s "spec is nested dicts" callout.

### Typical Spec Overrides

These are the typical override **paths** to apply on top of the template (not the full spec). The agent reads each `key.subkey.leaf` as a dotted path and assigns the value at that nested location in the template-loaded `specs` dict.

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above.

For direct local Docker runs, mount user data somewhere other than
`/workspace` (for example `/tao-workspace`). The Cosmos-RL image keeps its
Python package under `/workspace/cosmos_rl`; bind-mounting over `/workspace`
hides the package and makes `cosmos-rl` fail with
`ModuleNotFoundError: No module named 'cosmos_rl'`.

```python
TRAIN_DATASET_URI = "s3://bucket/data/train"
EVAL_DATASET_URI = "s3://bucket/data/eval"
# Slurm/internal example:
# TRAIN_DATASET_URI = "/lustre/fsw/tao_datasets/cosmos_rl/train"
# EVAL_DATASET_URI = "/lustre/fsw/tao_datasets/cosmos_rl/eval"
# Direct spec-path example:
# TRAIN_ANNOTATION_PATH = "/lustre/fsw/.../annotations_train.json"
# TRAIN_MEDIA_PATH = "/lustre/fsw/.../videos_train.tar.gz"
# EVAL_ANNOTATION_PATH = "/lustre/fsw/.../annotations_eval.json"
# EVAL_MEDIA_PATH = "/lustre/fsw/.../eval_videos"
```

**train (mandatory data sources):**
```python
{
    "custom.train_dataset": {
        "annotation_path": f"{TRAIN_DATASET_URI}/annotations.json",
        "media_path": TRAIN_DATASET_URI,
    },
    "custom.val_dataset": {
        "annotation_path": f"{EVAL_DATASET_URI}/annotations.json",
        "media_path": EVAL_DATASET_URI,
    },
    "policy.model_name_or_path": "hf_model://nvidia/Cosmos3-Nano",
    "policy.model_max_length": 81920,
    "policy.parallelism.dp_shard_size": 4,
    "policy.parallelism.dp_replicate_size": 1,
    "policy.lora.lora_alpha": 256,
    "policy.lora.r": 16,
    "policy.lora.lora_dropout": 0.05,
    "train.epoch": 2,
    "train.train_batch_per_replica": 32,
    "train.optm_lr": 2e-5,
    "train.optm_impl": "fused",
    "train.deterministic": True,
    "train.ckpt.save_freq_in_epoch": 1,
    "train.ckpt.max_keep": 2,
    "train.train_policy.mini_batch": 1,
    "train.train_policy.dataset.test_size": 0,
    "train.train_policy.dataloader_num_workers": 4,
    "train.train_policy.dataloader_prefetch_factor": 4,
    "validation.freq_in_epoch": 1,
    "validation.batch_size": 1,
    "validation.enable_dataset_cache": False,
    # custom.vision.nframes defaults to 8 from the spec template for bounded
    # 1-GPU memory use. Switch to fps only when the GPU/context budget supports it.
    "custom.system_prompt": "You are a helpful assistant.",
    "logging.logger": ["console", "tao"],
}
```

`custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` are
valid train schema fields and are seeded in the packaged train template. Strict
validators must preserve those keys so AutoML can optimize against an explicit
validation set instead of silently falling back to training-only data.

**evaluate (mandatory data sources):**
```python
{
    "dataset.annotation_path": f"{EVAL_DATASET_URI}/annotations.json",
    "dataset.media_dir": EVAL_DATASET_URI,
    # vision.nframes defaults to 8 — see Vision Encoders for fps vs nframes.
    "model.enable_lora": True,
    "model.base_model_path": "hf_model://nvidia/Cosmos3-Nano",
}
```

**inference (mandatory media):**
```python
{
    "model_path": "hf_model://nvidia/Cosmos3-Nano",
    "media": "/tao-workspace/media/videos/example.mp4",
    "type": "video",
    "prompt": "Briefly describe this video.",
    "fps": 1,
    "total_pixels": 200704,
    "max_new_tokens": 32,
    "results_dir": "/results/inference",
    "enable_lora": False,
}
```

**quantize (mandatory calibration data):**
```python
{
    "model.model_path": "hf_model://nvidia/Cosmos3-Nano",
    "dataset.annotation_path": "/tao-workspace/calibration/annotations.json",
    "dataset.media_dir": "/tao-workspace/calibration/videos",
    "quantize.num_calibration_samples": 1,
    "quantize.max_sequence_length": 4096,
    "quantize.quantization_scheme": "W4A16",
    "quantize.skip_test_generation": True,
    "results_dir": "/results/quantize",
}
```

The quantize wrapper includes a compatibility shim for the current image's
`compressed_tensors`/`llmcompressor` import mismatch. Keep that shim in the
model-skill action metadata until the container packages matching versions.

## Critical Overrides (Train)

These are the keys whose template defaults are wrong or where omission flips the run into a different mode:

| Parameter | Template Default | Required Value | Why |
|---|---|---|---|
| `policy.model_name_or_path` | `hf_model://nvidia/Cosmos3-Nano` | Direct Docker: `nvidia/Cosmos3-Nano`, `hf_model://nvidia/Cosmos3-Nano`, or a local HF snapshot path. SDK/managed platform predownload: `hf_model://nvidia/Cosmos3-Nano`. | Keep the train and evaluate base model aligned. |
| `policy.model_max_length` | 40960 | Keep at 40960 or higher | Smaller than ~40k causes `vision_embeds` shape mismatch on video inputs |
| `train.train_batch_per_replica` | 32 | Any multiple of `train.train_policy.mini_batch` | Mismatch raises an immediate AssertionError |
| `train.train_policy.type` | `"sft"` | Keep as `"sft"` for SFT workflows | If dropped during agent regeneration, cosmos-rl flips to RL mode → rollout replica allocated → multi-node attempted → hostname errors when `num_nodes=1` |
