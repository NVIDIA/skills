# Prepare-for-Inference

Final step of the DEFT loop. Produces two artifacts under `${RESULTS_DIR}/` so
downstream inference skills can consume the trained checkpoint without reading
`deft_state.json` or the training spec.

## Artifacts

| File | Role |
|---|---|
| `best_model.json` | Checkpoint plus customer metric contract/result and deployment metadata. |
| `best_model_inference_spec.yaml` | Ready-to-run TAO inference spec. The executable artifact. |

Both are written by `scripts/prepare_inference_spec.py`. At ordinary loop end,
use `"$PYTHON" scripts/finalize_run.py`; it writes these first and refuses the terminal
commit if either is absent. Never hand-edit either file.

### `best_model.json`

```json
{
  "checkpoint":     "/abs/path/to/best.pth",
  "threshold":      0.237481,
  "metric_contract": {
    "name": "weighted_escape_cost",
    "display_name": "Weighted escape cost",
    "operator": "<=",
    "target": 0.02,
    "unit": "cost/board"
  },
  "metric_result": {
    "name": "weighted_escape_cost",
    "value": 0.018,
    "unit": "cost/board",
    "passed": true
  },
  "iteration":      "iter1",
  "backbone":       "/abs/path/to/c_radio_v2_b.ckpt",
  "backbone_container_path": "/data/pretrained_models/c_radio_v2_b.ckpt",
  "backbone_type":  "c_radio_v2_vit_base_patch16_224",
  "backbone_frozen": false,
  "images_dir":     "/abs/path/to/workspace/images",
  "training_spec":  "/abs/path/to/baseline_spec.yaml"
}
```

| Field | Meaning |
|---|---|
| `checkpoint` | Best `.pth`: prefer candidates satisfying all constraints, then select lowest for `<`/`<=` or highest for `>`/`>=` according to the primary metric contract |
| `threshold` | Evaluator-selected operating threshold, or `null` when the customer metric intentionally uses the training spec's existing threshold |
| `metric_contract` | Approved customer target, direction, unit, evaluator, and constraints |
| `metric_result` | Best measured value, recomputed pass state, evidence, constraints, and optional diagnostics |
| `iteration` | Which iteration won (`baseline`, `iter1`, …) |
| `backbone` | Absolute path to the backbone `.ckpt` (mount this into the container) |
| `backbone_container_path` | Exact container destination matching the generated inference spec |
| `backbone_type` | Visual ChangeNet backbone type used by training |
| `backbone_frozen` | Whether the backbone was frozen during task-level training |
| `images_dir` | Path the model was evaluated against. Useful default for re-running on KPI data. |
| `training_spec` | Path to the training YAML used. Read this if you need fields the JSON doesn't expose. |

### `best_model_inference_spec.yaml`

Built by copying `model.*` and `dataset.classify.*` verbatim from the training
spec, then:

- Stripping `train_dataset`, `validation_dataset`, `test_dataset` from `dataset.classify`
- Setting `dataset.classify.infer_dataset.{csv_path,images_dir}` to empty (CONSUMER fills in)
- Setting `inference.checkpoint` to the best checkpoint
- Setting `model.classify.eval_margin` to the evaluator-selected threshold when one exists; otherwise retaining the training spec value
- Disabling augmentation (`augmentation_config.augment: false`)
- Carrying only `train.classify.loss` as a compatibility stub for the pinned
  TAO 7.1 image. It must match the checkpoint's training spec. TAO 7.2 images
  containing NVIDIA-TAO/tao-pytorch#107 derive the non-training criterion from
  the model architecture and no longer require this stub.

The consumer sets four things and runs:

1. `dataset.classify.infer_dataset.csv_path` — their inference CSV
2. `dataset.classify.infer_dataset.images_dir` — their images root
3. `inference.results_dir` — where outputs go
4. `results_dir` — top-level results dir (TAO requires it)

## Consumer Workflow

```bash
# 1. Read handoff metadata
jq . ${RESULTS_DIR}/best_model.json

# 2. Edit the spec to point at your data (or override on CLI)
cp ${RESULTS_DIR}/best_model_inference_spec.yaml /tmp/my_inference.yaml
# … set the four CONSUMER fields …

# 3. Pinned TAO pyt image URI (stamped from the release manifest).
TAO_PYT_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt  # versions-key: images.tao_toolkit.pyt

# 4. Run inference. Mount paths from best_model.json into the container.
HANDOFF="${RESULTS_DIR}/best_model.json"
BACKBONE=$(jq -er '.backbone | strings | select(length > 0)' "$HANDOFF")
BACKBONE_CONTAINER_PATH=$(jq -er '.backbone_container_path | strings | select(startswith("/"))' "$HANDOFF")
HOST_RESULTS=<output_dir>
mkdir -p "$HOST_RESULTS"
probe="$HOST_RESULTS/.tao-write-probe.$$"
(umask 077 && : >"$probe" && rm -f "$probe") || {
    echo "FATAL: $HOST_RESULTS is not writable by uid $(id -u)" >&2
    exit 2
}

docker run --pull=never --rm --gpus all --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v <your_csv_dir>:/data/infer \
    -v "$(jq -er .images_dir "$HANDOFF"):/data/images:ro" \
    -v "$(jq -er .checkpoint "$HANDOFF"):/model/best.pth:ro" \
    -v "${BACKBONE}:${BACKBONE_CONTAINER_PATH}:ro" \
    -v /tmp/my_inference.yaml:/specs/inference.yaml \
    -v "$HOST_RESULTS:/results" \
    "$TAO_PYT_IMAGE" \
    visual_changenet inference -e /specs/inference.yaml
```

The `--shm-size=8g` is required — TAO dataloaders crash with bus errors on the
default 64MB allocation.

The output tree is owned by the submitting host user, so it can be updated or
removed without sudo.

## Threshold Contract

When `threshold` is numeric, use it rather than an unrelated default. It is the
operating point selected by the approved evaluator and is already written to
the generated YAML.

When `threshold` is `null`, the customer metric did not select a new operating
point; the generated YAML retains the exact training spec value. Do not invent
one. If the selected evaluator requires a threshold, it must supply the
operating point used by evaluation.

Consumers who run the generated YAML as-is therefore get the same threshold
semantics used by evaluation.

## Silent-Failure Modes (Avoid These)

These are the four ways a config-mismatched inference run can produce
misleading or no output. The script prevents all of them by copying training
config verbatim, but if you build an inference spec by hand, watch out:

1. **`concat_type` mismatch (silent).** Training used `grid` 2×2, inference set
   to `linear`. Loads cleanly, produces wrong scores. Always copy `concat_type`
   and `grid_map` from the training spec.

2. **`difference_module` mismatch (cryptic).** Training used `euclidean`,
   inference set to `learnable`. Fails with `KeyError:
   model.backbone.radio.radio.radio.model.patch_generator.pos_embed` deep
   inside `load_state_dict`. The two architectures have different key
   nesting depths.

3. **`image_ext` mismatch (empty dataset).** Training used `.jpg`, inference
   set to `.png`. Dataloader finds zero rows; predict loop runs over 0 batches;
   no error. Verify `image_ext` matches actual files on disk.

4. **`loss` / `difference_module` pair (assertion on the pinned TAO 7.1 image).**
   Contrastive loss requires `difference_module: euclidean`; CE loss works with
   either. Copy `train.classify.loss` from the training spec until the documented
   image baseline includes NVIDIA-TAO/tao-pytorch#107. Do not copy other
   training-only keys.

## When to Re-Run

Re-run `prepare_inference_spec.py` whenever:

- The loop is about to finish (`finalize_run.py` handles this before
  `loop_stop`).
- A new iteration completes and you want to evaluate against the latest best.
  The script applies the metric contract's comparison direction, so calling it
  mid-loop gives the current customer-metric winner, not necessarily the final
  best.

Do **not** re-run after manually editing `deft_state.json`. Disk is canonical;
if state is stale, the artifact is wrong.
