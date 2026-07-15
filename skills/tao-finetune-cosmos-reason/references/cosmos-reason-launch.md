# Cosmos-RL Launch Intake and Preflight

Load this only when `SKILL.md` points here. If this conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current/compact source wins.

## Launch Intake Reminder

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

For local Docker, pass the resolved Cosmos-RL image so preflight can enforce
NVIDIA runtime, host GPU memory, helper-container, and known image architecture
checks before any model/data download:

```bash
scripts/check_tao_launch_preflight.py --platform local-docker \
  --container-image <resolved-cosmos-rl-image> \
  --path train_annotation=/abs/path/train/annotations.json \
  --path train_media=/abs/path/train \
  --path val_annotation=/abs/path/eval/annotations.json \
  --path val_media=/abs/path/eval
```

For `s3://` paths, if this helper reports that `aws` is missing, ask for
approval and rerun the same command with `--install-missing-tools` so the helper
installs `awscli` and immediately verifies the dataset paths.

Cosmos-RL video datasets can include large `videos.tar.gz` archives. Before
AutoML, stage S3-backed media once to a platform-local/shared path and point
every recommendation at the staged directory or archive; do not let each trial
download the same large S3 object through the container. Prefer an extracted
directory when annotations reference individual files. Keep a
`<workspace>/evaluations/data_staging.json` record with the original S3 URI, the
staged path, and the command/log used to verify the copy.

For Cosmos-RL, count and memory are necessary but not sufficient. Treat the run
as launchable only when the target has at least 4 GPUs with 80GB-class memory or
higher, the GPU architecture is in the image-supported allowlist above, and the
normal Docker/platform, S3, and credential preflight checks pass. A remote image
manifest that advertises `linux/arm64` only proves CPU architecture support; it
does not prove CUDA SM support. Spark/GB10 `sm_121` must be blocked for this
image unless direct image introspection confirms `sm_121` support or the user
chooses a newer compatible image.

## Per-Action Dataset Requirements

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

For DAFT-style annotation files, use direct spec mode when the annotation file
name is not `annotations.json` or when media is not colocated with the
annotation file. Preserve the user's source files. Do not create compatibility
patches for optional annotation fields unless the user explicitly asks for that
dataset mutation.

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

These are the typical override **paths** to apply on top of the template. Treat
dotted notation as a path into the nested `specs` dict, not as a literal flat
key.

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above.

For direct local Docker runs, mount user data somewhere other than
`/workspace` (for example `/tao-workspace`). The Cosmos-RL image keeps its
Python package under `/workspace/cosmos_rl`; bind-mounting over `/workspace`
hides the package and makes `cosmos-rl` fail with
`ModuleNotFoundError: No module named 'cosmos_rl'`.

For root-style runs, map `TRAIN_DATASET_URI` and `EVAL_DATASET_URI` to:

- train: `custom.train_dataset.annotation_path=<train>/annotations.json`,
  `custom.train_dataset.media_path=<train>`,
  `custom.val_dataset.annotation_path=<eval>/annotations.json`,
  `custom.val_dataset.media_path=<eval>`.
- evaluate: `dataset.annotation_path=<eval>/annotations.json`,
  `dataset.media_dir=<eval>`.
- inference: set `media`, `type`, `prompt`, `model_path`, and `results_dir`.
- quantize: set `dataset.annotation_path`, `dataset.media_dir`,
  `model.model_path`, `quantize.num_calibration_samples`,
  `quantize.max_sequence_length`, and `quantize.quantization_scheme`.

For direct spec-path mode, set the annotation and media fields explicitly rather
than deriving them from a root.

Common train overrides: `policy.model_name_or_path`, `policy.model_max_length`,
`policy.parallelism.dp_shard_size`, `policy.parallelism.dp_replicate_size`,
LoRA settings, `train.epoch`, `train.train_batch_per_replica`,
`train.optm_lr`, `train.train_policy.mini_batch`, checkpoint retention,
validation frequency, and logging. The packaged template keeps
`custom.vision.nframes=8` for bounded 1-GPU memory; switch to `fps` only after
checking token budget and GPU memory.

Do not require per-record `video_fps` for the packaged `nframes` template. If a
run switches to `custom.vision.fps` or a selected dataset/image profile
requires per-record timing, validate the annotation files before launching:

```bash
scripts/check_tao_launch_preflight.py --platform <platform> \
  --path train_annotation=/path/to/train.json \
  --path val_annotation=/path/to/val.json \
  --json-required-field train_annotation=video_fps \
  --json-required-field val_annotation=video_fps
```

The packaged train/evaluate/inference/quantize templates default to
`hf_model://nvidia/Cosmos3-Nano` for base-model fields. Override that only when
the user provides a different HuggingFace model id, `hf_model://...` URI, or
cluster-local snapshot path.

`custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` are
valid train schema fields and are seeded in the packaged train template. Strict
validators must preserve those keys so AutoML can optimize against an explicit
validation set instead of silently falling back to training-only data.

The quantize wrapper includes a compatibility shim for the current image's
`compressed_tensors`/`llmcompressor` import mismatch. Keep that shim in the
model-skill action metadata until the container packages matching versions.
