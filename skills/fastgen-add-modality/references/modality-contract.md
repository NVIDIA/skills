# Modality boundaries beyond the network

Read when adding a signal type rather than a backbone. The
[network contract](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-network/references/network-contract.md) owns
preprocessing, precision, conditioning and the module lifecycle.

- **State versus side input.** `FastGenModel.state_fields` lists the batch fields cast as
  method state (`real` by default); a loader key is neither state nor loss input until a
  method reads it. Follow the method's reads and the network's completed conditioning; a
  new condition may need no new state modality.
- **Identity and casts.** Every loader emits `idx` per sample and
  `FastGenModel._batch_size` reads it, also for modality mappings.
  `FastGenModule.cast_to_model_precision` takes a tensor, so dispatch over a mapping
  explicitly with the helpers in [tree_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/tree_utils.py).
- **Shapes.** `net.infer_shape` fills missing modality shapes, but consumers may read
  config shapes directly; audit both. `model_pixel_shape` in
  [config_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/config_utils.py) reads the `video` member of a
  mapping; do not invent a video field to satisfy it.
- **Clock.** `CompositeNoiseSchedule` in [noise_schedule.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/noise_schedule.py)
  dispatches state per modality but shares one timestep tensor, range, precision and
  student grid, chosen by its primary schedule. Different noise curves do not imply
  independent clocks; build sampler and training times in `t_precision` and adapt the
  upstream embedding dtype inside the network.
- **Loss balance.** Multimodal support is method-specific: SFT, PDD and DMD2 sum
  per-modality losses, while methods that index tensors directly stay unimodal. The
  reference's units, normalisation and reduction define the weighting; more elements do
  not weigh more when each modality is mean-reduced.
- **Serialization.** [save_media](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/media_utils.py) dispatches on tensor
  dimensionality and on image, video and audio keys, so other arrays can be misrouted.
  Add an explicit native writer and reader with shape and unit metadata instead of
  reshaping data to fit a media writer, and never let extra outputs silently drop out of
  joint-media export. Caller metadata must not overwrite authoritative tensor or schema
  fields; keep it separate or reject conflicting keys.
- **Inference and tracking.** The image and video scripts, `MediaEntry` and the writer
  dispatch in [wandb.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/callbacks/wandb.py) are not arbitrary-modality
  registries. Extend only the needed paths, keep sample identity and conditioning, and
  store native arrays next to rendered views; a `LogModule` adds derived tables but
  cannot decode unsupported media.

Test a complete raw sample through preprocessing, loss, sampling, serialization and
reload, together with an existing modality to catch changed dispatch. For lossless native
I/O, assert exact tensor values, dtype, shape, units and identity for every sample and
modality; numerical tolerances belong to lossy conversions. Image and video scores cannot
validate another modality's units, kinematics or alignment.
