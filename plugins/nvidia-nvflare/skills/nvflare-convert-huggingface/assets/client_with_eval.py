# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged Hugging Face Client API conversion template: patch + Trainer loop.

Copy and adapt this into a generated ``client.py``. ``trainer_factory`` builds
the source project's model, tokenizer/processor, datasets, collator, arguments,
callbacks, and Trainer once after FLARE initialization. The patched Trainer
owns model exchange; do not add a manual ``FLModel`` receive/send path.

Keep ``evaluate_before_train=True`` when source-backed per-round evaluation or
best-model selection is required. Set it to ``False`` only for a valid
train-only source path.

The entry is rankless for CPU, single-GPU, and other single-process training.
The Hugging Face Client API resolves an initialized process-group rank or global
``RANK`` and rejects an unresolved multi-process launch. For a preserved
multi-GPU launch, follow
``references/huggingface-state-and-distributed.md``.
"""

import nvflare.client.hf as flare


def make_hf_argument_parser(dataclass_types):
    """Create a strict parser for generated clients that use Hugging Face dataclasses."""
    from transformers import HfArgumentParser

    return HfArgumentParser(dataclass_types, allow_abbrev=False)


def main(trainer_factory, *, evaluate_before_train=True):
    """Run one persistent patched Trainer in single-process or distributed mode."""
    flare.init()
    trainer = trainer_factory()
    flare.patch(trainer)

    while flare.is_running():
        if evaluate_before_train:
            trainer.evaluate()
        trainer.train()
