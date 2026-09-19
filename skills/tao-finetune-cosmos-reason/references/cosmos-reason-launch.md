# Cosmos-RL Launch Intake and Preflight

Load this only when `SKILL.md` points here. If this conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current/compact source wins.

This is the Cosmos-RL backend supplement. First resolve the shared frontend
with `scripts/cosmos_workflow.py`; do not use this reference to select a
backend. Plain Nano SFT defaults to Cosmos-RL; Cosmos3-Edge and an explicit
Framework request resolve to Cosmos Framework.

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

Before train or AutoML runner generation, resolve the Cosmos-RL backend
contract and its pinned action image, show the exact image to the user, and ask
whether to use it or override with `image=<override>`. Do not read the shared
top-level image instead of the selected backend contract; the two pins match
only to preserve compatibility with legacy skill consumers. This skill does
not package a `skills/models/tao-finetune-cosmos-reason/config.json` file.

The same metadata declares model-level GPU host minimums. They override the
TAO-wide platform defaults for Cosmos-RL. On a self-managed Docker host, run:

```bash
bash skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh \
  --backend docker --check-only \
  --min-driver-version 580 \
  --min-cuda-version 13.0 \
  --min-container-toolkit-version 1.19.1
```

Use the same minimums when validating self-managed Kubernetes workers. On
administrator-managed SLURM or Kubernetes nodes, verify that the node image or
operator policy satisfies the profile before submission. These are minimum
bounds: newer compatible versions pass, while a driver below the CUDA 13.x
compatibility floor must fail.

For launch preflight, pass the concrete annotation and media paths to the
shared helper:

```bash
scripts/check_tao_launch_preflight.py --platform slurm \
  --path train_annotation=<TRAIN_ANNOTATION_PATH> \
  --path train_media=<TRAIN_MEDIA_ROOT> \
  --path val_annotation=<VALIDATION_ANNOTATION_PATH> \
  --path val_media=<VALIDATION_MEDIA_ROOT> \
  --gpu-min-total-memory-gb 256 \
  --gpu-arch-allowlist cosmos_rl=sm_80,sm_90,sm_100,sm_103,sm_103a,sm_120
```

For local Docker, pass the resolved Cosmos-RL image so preflight can enforce
NVIDIA runtime, host GPU memory, helper-container, and known image architecture
checks before any model/data download:

```bash
scripts/check_tao_launch_preflight.py --platform local-docker \
  --container-image <resolved-cosmos-rl-image> \
  --target-gpu-index 0 --target-gpu-index 1 \
  --target-gpu-index 2 --target-gpu-index 3 \
  --path results_dir=/abs/path/to/job-results \
  --min-free-disk-gb results_dir=384 \
  --path train_annotation=/abs/path/train/annotations.json \
  --path train_media=/abs/path/train \
  --path val_annotation=/abs/path/eval/annotations.json \
  --path val_media=/abs/path/eval \
  --gpu-min-total-memory-gb 256
```

Set `--target-gpu-index` to the exact indices passed to Docker's `--gpus`
allocation. This prevents an unallocated display or heterogeneous accelerator
from contaminating memory, architecture, and allocated-runtime checks.

The 384 GiB result-filesystem gate is mandatory for Cosmos-RL Nano training
with synchronous epoch checkpoints. A dense four-way sharded checkpoint plus
its optimizer state and Hugging Face safetensor export can consume roughly
115 GiB; retention briefly needs the new checkpoint and retained predecessors
at the same time. Check actual free bytes on the filesystem containing the
host-mounted result directory, not Docker's logical/reclaimable size. If this
gate fails, reclaim or relocate storage before launch; a PyTorch zip-writer
`unexpected pos` error during `torch.save` is a common ENOSPC symptom.

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

For Cosmos-RL SFT (`train.train_policy.type="sft"`), require at least 256 GB of
cumulative visible GPU memory. Do not impose a fixed device count or per-device
capacity: set `policy.parallelism.dp_shard_size` and the platform GPU request to
the actual visible GPU count, and set `policy.parallelism.dp_replicate_size=1`
for a single node. Workflows that allocate separate policy and rollout replicas
must still satisfy their explicit topology. In every case, each visible GPU
architecture must be in the image-supported allowlist above and the selected
image must pass the allocated-node CUDA-stack gate along with normal
Docker/platform, S3, and credential preflight checks. Architecture-specific
suffixes such as `a` and `f` are matched to the same base SM family by the
preflight helper.

The production recommendation remains at least 4 GPUs with 80GB-class memory.
A single high-memory GB300 is also supported when the selected image passes
architecture introspection and the spec sets
`policy.parallelism.dp_shard_size=1`; apply the single-GPU video guards in
`cosmos-reason-single-gpu-video.md` when that profile is selected. A remote image
manifest that advertises `linux/arm64` only proves CPU architecture support; it
does not prove CUDA SM support. `sm_121` must be blocked for this image unless
direct runtime validation confirms support or the user chooses a compatible
image.

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
Don't rebuild from scratch.

```python
import yaml
from pathlib import Path

skill = Path.home() / "tao-sdk/tao-skill-bank/skills/models/tao-finetune-cosmos-reason"
action = "train"  # train, evaluate, inference, or quantize
specs = yaml.safe_load((skill / f"references/spec_template_{action}.yaml").read_text())
# Now apply your overrides on top of `specs` (next section).
```

The reference TOML (and the spec the model actually consumes) is **nested dicts**, not flat dotted keys. The dotted notation in the override examples below denotes *paths into the nested spec* — the agent must walk the path and assign at the leaf, not store the dotted string as a literal key.

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

The DAFT hook also forwards FPS-only `min_frames` / `max_frames`, clip-time
`video_start` / `video_end`, paired `resized_height` / `resized_width`, and
`min_pixels` / `max_pixels` / `total_pixels`. Record all selected values in
the sealed plan and inherit them into linked evaluation.

Keep cadence epoch-based by default: use `train.epoch` for training duration,
`train.ckpt.save_freq_in_epoch=1` for checkpoints, and
`validation.freq_in_epoch=1` for validation. Do not select step-based
`train.ckpt.save_freq` or `validation.freq` because of a dataset, GPU SKU,
topology, or runtime image. Use step cadence only when the user explicitly
requests it.

Neither `nframes` nor `fps` sampling requires a per-record `video_fps` field.
The selected decoder reads the source frame rate from each media stream and
qwen-vl-utils uses it to resolve FPS sampling. Treat annotation-level `fps` or
`video_fps` as optional descriptive metadata and validate it when present;
never invent it or reject an otherwise valid dataset because it is absent.

The packaged train/evaluate/inference/quantize templates default to
the user-supplied immutable model URI or local path for base-model fields. Resolve it only when
the user provides a different HuggingFace model id, `hf_model://...` URI, or
cluster-local snapshot path.

`custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` are
valid train schema fields and are seeded in the packaged train template. Strict
validators must preserve those keys so AutoML can optimize against an explicit
validation set instead of silently falling back to training-only data.

The quantize wrapper includes a compatibility shim for the current image's
`compressed_tensors`/`llmcompressor` import mismatch. Keep that shim in the
model-skill action metadata until the container packages matching versions.
