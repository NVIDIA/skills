---
name: fastgen-add-network
description: >-
  Integrate a new diffusion or flow model, or a task variant such as I2V, editing or causal generation, into NVIDIA FastGen as a FastGenNetwork adapter with preprocessors, configs, method capabilities, reference parity and trainability. Use it whenever a model family or checkpoint is not yet supported.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Add a FastGen network

Apply [fastgen-develop](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-develop/SKILL.md)
and [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md). Start with the
[network README](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md) and the interface docstrings in
[network.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/network.py) and [module.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/module.py),
then inspect the closest maintained adapter for the same task. Read
[network contracts](references/network-contract.md) before implementing and load its
conditional sections only for the capabilities being added. For a new signal type also
use [fastgen-add-modality](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-modality/SKILL.md).

1. **Source.** Verify the upstream model, its revision, code, weights, access and
   supported modes. Diffusers implementations are reusable, but check the authors'
   reference for precision and preprocessing defaults, and whether the checkpoint's
   precision or quantisation supports training. Never substitute a similarly named model.
2. **Trace upstream.** Preprocessing, tokenizer and chat templates, conditioning and
   masks, VAE scaling and packing, position encodings, time convention, scheduler grid
   and guidance recipe, plus modality geometry and duration/FPS where relevant.
   Distinguish native edit or reference conditioning from initialising noisy latents.
3. **Implement.** Add the adapter under `fastgen/networks/<Name>/` and its presets in
   [net.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/net.py). Follow the module lifecycle, preprocessing,
   conditioning, output-transformation and sampling contracts; share task-independent
   components such as text encoders and VAEs, and do not embed an inference pipeline as
   the training forward. Load weights through the documented
   [pretrained-model conventions](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#pretrained-models) so
   local mirrors and offline mode keep working. Audit precision, buffer reconstruction,
   meta loading and sharding for the network and every preprocessor.
4. **Capabilities.** Trace the requested methods' forward and loss calls; add feature and
   discriminator gradients, second-time inputs, PDD, LoRA, multimodal or causal support
   as needed and check each independently. SFT support implies none of them.
   Add and run small forward, conditioning and capability regressions following
   `tests/test_network*.py` and `tests/test_conditioning.py` before expensive parity or
   training runs.
5. **Configs.** Use [fastgen-configure](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) for SFT and
   the requested method configs, with negative-prompt presets where the upstream guidance
   needs them and none where it does not.
6. **Parity.** Follow [validate a new network](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#validate-a-new-network)
   with [fastgen-validate](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md) and
   [fastgen-inference](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-inference/SKILL.md), and resolve unexplained
   differences before training.
7. **Train and document.** Use [fastgen-train](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md) for a
   short SFT run with media every iteration, then the requested methods; compare outputs
   before and after and reload a checkpoint. Update the README tables (network list,
   pretrained models).

Report implementation and config paths and per-task parity, training and checkpoint
results. If the loss decreases while outputs degrade, inspect input normalisation,
prediction and time convention and guidance before tuning the optimizer; mark
unsupported capabilities explicitly.
