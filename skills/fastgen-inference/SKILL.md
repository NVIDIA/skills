---
name: fastgen-inference
description: >-
  Generate and inspect NVIDIA FastGen teacher or distilled-student outputs with the inference scripts or ComfyUI nodes: config, checkpoint, prompts, conditioning media, seeds and sampling settings. Use it to prepare or execute sampling; use fastgen-validate to assess existing parity or quality evidence.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Run FastGen inference

Apply [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md). The
[inference README](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md) owns the scripts, arguments, modes, checkpoint
selection, guidance and FID evaluation; the [ComfyUI nodes](https://github.com/NVlabs/FastGen/blob/main/comfyui/README.md) provide
the interactive route. Read [weights and sampling](references/weights-and-sampling.md)
for checkpoint roles and reproducible comparisons.
For preparation-only requests, provide the command and pending checks without launching.

1. **Resolve.** Config, teacher versus student (or both), weight source, prompts or class
   labels, conditioning media, seed, sampling settings and destination. Verify that the
   checkpoint matches the network, method and branch, and never present an initialised
   student as distilled ([checkpoint selection](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md#checkpoint-selection)).
2. **Entry point.** Pick the script or node that supports the task and preserves every
   output modality; inspect its `--help` and module docstring. Teacher steps come from
   `--sample_steps`, student steps from the scheduler's `model.student_grid_cfg.steps`
   ([key arguments](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md#key-arguments)); methods can add constraints, so
   verify the effective number of network evaluations.
3. **Inputs.** Check list lengths and pairing: class labels within the configured count,
   source images for editing, image, video or camera inputs for video modes, and the
   network's negative conditioning ([guidance](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md#guidance)). Short
   lists repeat their last entry, so count the effective inputs.
4. **Run.** Use explicit output paths and the requested checkpoint within the resource
   budget. For parity or acceptance evidence follow
   [fastgen-validate](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md).
5. **Inspect.** Open the media and native outputs; check dimensions, temporal alignment,
   conditioning and sample count, and record the loaded weight source. Local files do not
   prove tracker uploads.

State whether inference was prepared or executed; report artifact paths, sampling branch
and checkpoint, seed, grid, guidance and input settings. For memory failures inspect
resident networks and offload options before changing the workload; for bad samples
compare preprocessing and schedule against the reference before blaming the weights
([inference ignores a checkpoint](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#inference-appears-to-ignore-a-checkpoint)).
