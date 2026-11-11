---
name: fastgen-add-modality
description: >-
  Extend NVIDIA FastGen to a new generated or conditioning modality beyond the existing image, video, audio, camera and text paths: representation, loader keys, model state, noise-schedule clock, losses, sampling, serialization and validation. Use it when a new signal type must cross the data, model and output interfaces.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Add a FastGen modality

Apply [fastgen-develop](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-develop/SKILL.md)
and [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md). Read the
[network](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md) and [dataset](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md)
READMEs, then [modality boundaries](references/modality-contract.md). Use
[fastgen-add-network](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-network/SKILL.md) alone for another model of
an already supported task.

1. **Role.** Establish whether the modality is generated state, conditioning or control,
   or a derived training target, from the upstream model and its source. Record
   representation, units, normalisation, dimensions, temporal alignment, masks and decode
   semantics; a checkpoint name or an extra channel is not evidence of support.
2. **Trace.** Follow one sample through loader, `prepare_batch`, network, method loss,
   generator and output writer; note which boundaries already handle it and which assume
   image, video, audio or a single tensor. Preserve raw-versus-encoded distinctions and
   upstream packing, and do not turn a conditioning signal into diffusion state to fit an
   existing multimodal example.
3. **Model.** Use [fastgen-add-network](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-network/SKILL.md) for encoder
   and backbone changes; add only the needed shape inference, schedule dispatch and
   conditioning paths. For joint generation check whether modalities share a clock, grid
   and batch and keep the upstream relationship.
4. **Data.** Use [fastgen-dataset](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-dataset/SKILL.md) for acquisition,
   serialization, pairing and loader registration; version learned normalisation with the
   model artifacts and verify encode/decode and alignment before large preprocessing.
5. **Methods.** Use [fastgen-configure](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) for SFT and
   the requested distillation recipes; trace each method's tensor operations, since a
   tree utility accepting a key does not make the method support it. Reject unsupported
   combinations explicitly.
6. **Outputs.** Extend an inference entry point and serializer where the existing ones
   assume conventional media; keep native arrays in a usable format with shape and unit
   metadata, separate from rendered previews, tracker logging and metrics.
7. **Validate.** Use [fastgen-validate](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md) for parity and
   reviewable comparisons, then [fastgen-train](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md) for short
   SFT and the requested method; check per-modality gradients, conditioning sensitivity,
   serialization of every generated modality, checkpoint reload and that existing modes
   still work.

Report supported input and output modes, implementation, data and config paths, native
artifacts and evidence per tested combination. For plausible but wrong outputs check
units, packing, alignment and normalisation before tuning losses, and state whether the
result is a new conditioning path or a newly generatable modality.
