<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Co-DETR — pairing a checkpoint with its spec

A spec whose architecture does not match the checkpoint does not raise. `codetr
inference` loads what fits, discards what does not, prints `Execution status: PASS`,
and exits 0 — having written one label file per image with **no boxes in it**. The
layers the checkpoint could not populate stay randomly initialised, so nothing clears
`conf_threshold`. Anything reading those labels sees an image set with no objects in
it and proceeds normally.

The only signal is one line in the load log, among thousands:

```
Skipping size-mismatched key ... patch_embed.proj.weight: ckpt [1024,3,16,16] vs model [192,3,4,4]
```

For the ViT-Large COCO-80 checkpoint (`zongzhuofan/co-detr-vit-large-coco`):

| field | this checkpoint | TAO schema default | wrong value fails |
|---|---|---|---|
| `model.backbone` | `vit_large_codetr` | `swin_large_patch4_window7_224` | silently |
| `model.num_queries` | 1500 | 900 | silently |
| `model.num_feature_levels` | 5 | 4 | silently |
| `dataset.num_classes` | 80 | 91 | silently |
| `model.return_interm_indices` | `[0, 1, 2, 3, 4]` | `[1, 2, 3, 4]` | loudly |
| `dataset.augmentation.fixed_random_crop` | 1536 | `null` | loudly |

The last two are consequences of the first four rather than independent choices.
`return_interm_indices` must hold exactly `num_feature_levels` entries — TAO raises
`num_feature_levels: 5 does not match the size of return_interm_indices` — and its
values index the stride map `{0: 4, 1: 8, 2: 16, 3: 32, 4: 64}`, so five levels
means strides 4 through 64. `fixed_random_crop` becomes the transformer's
`lsj_resolution`, and `vit_large_codetr` refuses a null one outright. Both fail at
startup, so they cost seconds rather than a whole pass.

For a **different** checkpoint, derive the values rather than guessing — and because
these are architecture, not preference:

* `patch_embed.proj.weight` of shape `[D, 3, P, P]` gives embed dim `D` and patch
  size `P`. `1024/16` here means ViT-L/16, which is `vit_large_codetr`. The patch-14
  DINOv2 backbone (`pretrained_dinov2_classification_imagenet:vit_large_patch14_dinov2`)
  cannot load into it.
* `query_embed`'s first dimension gives `num_queries`.
* The classification head's output width gives `num_classes` — COCO-80, not COCO-91.

A correctly paired load reports missing keys but **no** unexpected ones; the missing
keys are the training-only collaborative heads.
