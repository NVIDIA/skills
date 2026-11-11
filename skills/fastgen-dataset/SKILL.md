---
name: fastgen-dataset
description: >-
  Prepare, register or repair datasets for NVIDIA FastGen loaders: class-conditional ZIPs, WebDataset shards of images, prompts, latents, videos, audio or camera data, and KD pairs or trajectories; then verify decoded batches. For experiment configuration over already prepared data, use fastgen-configure.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Prepare a FastGen dataset

Apply [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md). The
[dataset README](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md) owns formats, preparation commands,
loader presets and key parameters; the loaders in
[wds_dataloaders.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/wds_dataloaders.py) and
[wds_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/wds_utils.py) document decoding, partitioning and
resume in their docstrings. Read [loader contracts](references/loader-contract.md) for
shard acceptance, coverage and resume accounting.

1. **Scope.** Resolve source, permitted use, destination, task, the method's supervision
   needs ([fastgen-configure](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) if unsettled), the
   network's preprocessing and the available sample count. Inspect existing inputs before
   downloading; keep provenance and counts. Follow the pretrained model's conditioning
   requirements; recaption prompts to match its reference style when needed, preserving meaning.
2. **Loader.** Choose the existing class-conditional or WebDataset loader. Shards need
   unique sample keys with the extensions named in `key_map`; constants come from
   `files_map` and `presets_map`. Raw media and prompts use the `*_raw` keys and cached
   latents and embeddings the plain keys; never label placeholders as model-space data,
   and record the model revision and preprocessing of every cache. Rebuild affected caches
   after edits.
3. **Pairing and geometry.** Keep task pairing (source and target for editing, reference
   frames for image-conditioned video, camera data, aligned audio). Derive frame counts,
   FPS, aspect ratio and temporal compression from the network; the representable frame
   count is not the nominal clip length.
4. **Splits and size.** Separate training from held-out evaluation inputs before deriving
   labels or trajectories. Estimate the unique examples needed from throughput, effective
   batch, updates and acceptable repetition; a tiny dataset validates plumbing only.
5. **Register.** Import and copy a preset from [data.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/data.py)
   rather than reproducing its loader fields; add a shared preset there when needed.
   Supply paths through the config or environment roots; keep data and preprocessing
   artifacts outside the source tree. For data-dependent fine-tuning or distillation, consider
   teacher-generated targets via [fastgen-inference](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-inference/SKILL.md).
   Save the method's required supervision, including
   [KD pairs or trajectories](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md#knowledge-distillation-loaders).
6. **Verify.** Iterate the real loader and a prepared batch (`prepare_batch`) in the
   proper runtime; check keys, shapes, dtypes, ranges, captions, pairing, negative
   conditions and modality lengths. Use enough batches and ranks to expose empty worker
   or shard splits, and keep a manifest of usable counts. Before prompt-only training,
   sample the teacher on a few real training prompts to confirm they are usable.

Report provenance, the split/count manifest, dataset location, data-config path and the
observed batch contract. For empty or short batches inspect tar members, `key_map`,
filters, shard/worker partitioning and `files_map` before touching the network; report
corrupt or excluded samples instead of counting them.
