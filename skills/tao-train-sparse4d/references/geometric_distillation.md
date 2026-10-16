# 2D-to-3D Geometric Distillation

Use this route for synchronized, calibrated real multi-camera scenes with 2D annotations or offline detector targets, while retaining 3D-labeled training data. **2D-to-3D** is the supervision direction; **3D-to-2D projection** maps predicted cuboids into camera images for the loss. This is the ordinary `sparse4d train` task, not a separate `distill` action.

For the full procedure and field descriptions, use the [canonical TAO Sparse4D guide](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/sparse4d/sparse4d.html). This reference supplies the skill's routing and artifact checks.

## Runtime and model selection

Use the model action's PyTorch image for training and the conversion action's Data Services image for preparation, resolved from `skill_info.yaml`. Verify the installed builds expose `model.head.loose_to_tight`, the co-training dataset fields, `annotations sparse4d_prepare`, and `aicity.load_annotations` / `aicity.fps`. An older image must be updated through the supported image-selection workflow before using missing features; do not silently install experimental patches.

Preserve the user's selected Sparse4D backbone and pretrained experiment specification. The released LTT checkpoint's bundle location does not determine that backbone. The released LTT route requires exactly this ordered taxonomy:

```yaml
classes: [person, gr1_t2, agility_digit, nova_carter, transporter, forklift, pallet_truck]
```

Use the same order in preparation `class_names` and training `dataset.classes`. Custom classes require compatible LTT weights; do not present a class-name edit or subset of the released weights as a supported adaptation. Fitting a new LTT model is outside this workflow.

## Prepare calibrated real scenes

Use the [maintained annotation-free conversion specification](https://github.com/NVIDIA-TAO/tao-data-services/blob/release/7.3.0/nvidia_tao_ds/annotations/experiment_specs/aicity2ovpkl_unlabeled.yaml). Copy it to the staged specifications directory and set its ordered `aicity.class_config.CLASS_LIST` to the selected model taxonomy. The complete specification sets `aicity.load_annotations: false`, disables recentering, selects JPEG input, and uses all calibrated cameras as one group.

Arrange `/data/aicity_root/train/<scene>/calibration.json` and `<camera>/rgb/000000000.jpg`, `000000001.jpg`, and so on. Calibration must contain separate camera intrinsics and world-to-camera poses. All camera sequences must be synchronized, contiguous from zero, nonempty, and equal in length. Set the actual capture rate, not an assumed 30 FPS:

```bash
annotations convert -e /specs/aicity2ovpkl_unlabeled.yaml \
  aicity.root=/data/aicity_root aicity.split=train aicity.fps=30 \
  results_dir=/results/real_infos
```

Here `30` is an example capture rate. HDF5 RGB input instead uses `aicity.rgb_format=h5` and `dataset.use_h5_file_for_rgb: true`; inspect the shipped specification and supported HWC frame layout. The `dataset.use_h5_file_for_rgb` setting applies to the combined training dataset: use `false` for JPEG PKLs and `true` for HDF5 PKLs. Inspect the actual image paths in every split and make their storage format consistent before mixing them; a pretrained specification's HDF5 setting must not be inherited for JPEG inputs. Keep the model's image dimensions and normalization unchanged.

Unlabeled conversion needs neither ground truth nor depth files. It emits info PKLs with `gt_boxes=None` and **does not generate anchors**. Reuse the selected pretrained model's anchors.

Verify image paths, camera names, source `frame_idx`, timestamps in seconds, and calibration before preparing teacher caches. The PKL's legacy `sensor2world_transform` name holds a world-to-camera transform for the unrecentered specification; do not invert it because of its name. For recentered BEV groups, it maps group-local coordinates into the camera frame.

## Use the released LTT weights

The [Sparse4D ResNet-50 trainable_v3.0 bundle](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/sparse4d_rn50/files?version=trainable_v3.0) includes the independent `_loose_to_tight_mlp.pth` artifact. After the asset download is authorized, the NGC CLI command is:

```bash
ngc registry model download-version nvidia/tao/sparse4d_rn50:trainable_v3.0 \
  --file _loose_to_tight_mlp.pth --dest /data
```

The resulting path is `/data/sparse4d_rn50_vtrainable_v3.0/_loose_to_tight_mlp.pth`. Preserve the leading underscore and mount the artifact inside the training runtime. Keep it for resume: it is frozen and omitted from Sparse4D model checkpoints. Evaluation, inference, and export do not need it.

## Prepare supervision and the mixed split

Use the [Data Services preparation specification](https://github.com/NVIDIA-TAO/tao-data-services/blob/release/7.3.0/nvidia_tao_ds/annotations/experiment_specs/sparse4d_prepare.yaml). Fill its paths, class order, teacher class aliases, scene name, and camera mapping before submitting the selected operation:

```bash
annotations sparse4d_prepare -e /specs/sparse4d_prepare.yaml operation=ltt_2dgt
annotations sparse4d_prepare -e /specs/sparse4d_prepare.yaml operation=rtdetr_2d
annotations sparse4d_prepare -e /specs/sparse4d_prepare.yaml operation=lazy_index
```

- `ltt_2dgt` is optional visible-2D supervision for 3D-labeled scenes. Its `box2` and `box3` are amodal and visible **2D** boxes, respectively.
- `rtdetr_2d` converts precomputed per-camera KITTI `labels.tar.gz` archives; it does not run the teacher. Generate detections on exactly the converted source frames. Explicitly map or drop teacher classes, and match scene/camera names to the PKLs. When using `dataset.rtdetr_2d_cache_dir`, set `rtdetr_2d.output_path` to `<scene>__rtdetr2d.npz` inside that directory, with `<scene>` matching the runtime scene name after the BEV-group separator and `rtdetr_2d_dedup_regex` normalization. For one cache with a different filename, use `dataset.rtdetr_2d_cache_path` explicitly. Keep these caches away from scenes intended to remain on the 3D route.
- `lazy_index` consumes a split with one container-visible PKL path per line, each included once. Include the 3D-labeled and real info PKLs. For a split named `mixed_train.txt`, it writes the required sibling `mixed_train_lazy_index.pkl` with embedded camera counts. Stage the split parent directory as a folder, preserving that adjacency and `_pkl_cam_counts.pkl` when used; fetching the split alone through S3/Kubernetes staging is insufficient. Keep referenced PKLs and images visible. Rebuild the index after changing the split or relocating source files. `dataset.pkl_cam_counts_path` does not replace the required index.

The bank's `dataset_convert` action is `annotations convert`. These preparation operations are separate Data Services commands, submitted through the selected platform's job contract; do not rename them to `sparse4d dataset_convert` or pretend they run during model training.

## Configure, train, and validate

Merge [the co-training fragment](geometric_distillation_train.yaml) into the selected model's complete train specification, preserving its backbone, anchors, preprocessing, optimizer, and held-out validation input. Paths and `RealWarehouse` are examples to replace. Leave `ltt_2dgt_sidecar_dir` unset when the optional visible-2D route is unused. Do not replace the complete specification with the fragment.

Both LTT `enable` and `pseudo_enable` are required. Each batch must have one supervision route. For distributed mixed training, keep `dataset.sync_route: true`, a positive `scene_switch_iters`, and scene keywords that include all real 2D-supervised scenes while excluding the 3D route. `real_block_prob: 0.5` is an example block probability. Leave `model.cotrain_param_touch: false` to use find-unused-parameters DDP; enabling it can change inactive parameters through optimizer state and weight decay. The route and class order are fixed inputs, not hyperparameters to silently change during HPO.

For a bounded wiring test, choose an explicit optimizer-step budget and a short positive `scene_switch_iters` (for example `1`). The training CLI currently computes total steps as `num_epochs * floor(num_frames * num_bev_groups / (num_nodes * num_gpus * batch_size))`. Verify that this is positive and reaches several route blocks. The production example's cadence of `100` is unsuitable for an eight-step smoke test. Route selection is probabilistic, so elapsed steps alone do not establish coverage; inspect the sampled routes and their losses. This budget adjustment is a test setting, not a memory optimization or production recommendation.

Check that a short run consumes both routes. Expect normal 3D losses on labeled data and `loss_box_2d_pseudo_*` / `loss_cls_pseudo_*` on valid real samples. A missing or mismatched cache can set `has_2d_pseudo=False` and skip supervision. A processed frame with no detections is distinct and can supervise background. Investigate zero losses and join warnings before a long run.

Evaluate baseline and adapted checkpoints on the same held-out real 3D/tracking labels, with matching thresholds and raw/EMA choice, and retain a synthetic holdout. Use `dataset.test_dataset.ann_file` for the standalone evaluation action. Report smoke validity separately from accuracy and training-budget equivalence; 2D agreement alone is not a 3D accuracy result.

If held-out real 3D/tracking labels are unavailable, evaluate on the available labeled holdout and explicitly report real-scene 3D accuracy as unverified. This does not prevent a wiring smoke test, but teacher-box agreement cannot fill the missing accuracy evidence.
