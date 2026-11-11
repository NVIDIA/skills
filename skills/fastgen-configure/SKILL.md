---
name: fastgen-configure
description: >-
  Create or adapt an NVIDIA FastGen experiment config for a network, task, dataset and training objective, and pick a distillation method when none was chosen. Use it whenever a FastGen run needs a new or modified config file or non-trivial overrides.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Configure a FastGen experiment

Read the [configuration README](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md) for composition, override
grammar and run directories, and the [method index](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/README.md) for the
chosen method's documentation. Read [config contracts](references/config-contract.md)
when changing inherited values, geometry or supervision. For an algorithm the library
does not implement, use [fastgen-add-method](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-method/SKILL.md).

1. **Task.** Infer the complete task from request, network and data: conditioning type
   (none, class, text, image, video, camera, audio), generation versus editing, image or
   video, joint audio-video, causal generation; checkpoint variant, resolution,
   duration/FPS and target sampling budget (NFE). A model family name does not establish
   that every mode is supported; check the
   [supported networks](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#supported-networks) and the existing
   experiment configs.
2. **Method.** Keep an explicitly chosen method. Otherwise use SFT for fine-tuning and the
   [baseline guidance](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/README.md#choosing-a-distillation-baseline) for
   distillation, including data-free training and the
   [causal pipeline](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/README.md#causal-distillation-pipeline). Check the
   method's prerequisite checkpoints and supervision in its category README.
3. **Compose.** Start with hyperparameters from the experiment config closest in network
   architecture, task and method. If none is compatible, compose the method config's
   `create_config()` with compatible presets. Copy network, data, optimizer and callback
   presets from `fastgen/configs/*.py` and derive loader geometry from the network where needed
   (`model_pixel_shape`). Change only justified values; do not carry precision, guidance,
   schedule, learning-rate or batch settings over from an unrelated recipe.
4. **Supervision.** Decide data needs from the active losses, not the method name: SFT
   needs targets, KD needs cached pairs or trajectories in the
   [KD loader format](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md#knowledge-distillation-loaders),
   backward simulation replaces clean targets in student inputs but not in other enabled
   losses, and image- or video-conditioned tasks always need their conditioning media.
   Route missing data to [fastgen-dataset](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-dataset/SKILL.md).
5. **Alignment.** Match loader keys and latent geometry to the network; align teacher and
   student weights, prediction type, noise schedule, guidance and negative conditioning;
   check that every enabled loss has its modules and trainable parameters, routing
   missing capabilities to [fastgen-add-network](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-network/SKILL.md).
   Set iterations, optimizer, effective batch, logging, checkpoints and
   [validation limits](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/references/run-contract.md#what-an-iteration-does)
   to the budget.
6. **Resolve.** Consult [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md) when writing
   the command, including its override syntax. Run `--dryrun` and inspect the resolved
   values for placeholder paths, unintended weights, auxiliary networks and conflicting
   distributed modes. Hand execution to
   [fastgen-train](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md); a config-only task ends
   with the reviewed config.

Report the config path, departures from the nearest recipe, the method and data rationale
and how far the config was validated. For `Unknown config fields`, inspect the typed
schema before adding keys; for unsupported method/network combinations explain the
missing capability instead of silently disabling a loss.
