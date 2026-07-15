---
name: nemo-mbridge-perf-sequence-packing
description: Validate and use packed sequences and long-context training in Megatron-Bridge, distinguishing offline packed SFT for LLMs from in-batch packing for VLMs, and applying the right CP constraints.
license: Apache-2.0
when_to_use: Enabling sequence packing or long-context SFT, or investigating a commit that broke sequence packing or changed packing behavior; 'packed sequences', 'sequence packing', 'PackedSequenceSpecs', 'enable_in_batch_packing', 'CP with packing'.
---

# Sequence Packing Skill

For stable background and recommendation level, see:

- @docs/training/packed-sequences.md
- @skills/nemo-mbridge-perf-sequence-packing/card.yaml

## Enablement

Offline packed SFT for LLM finetuning:

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

cfg.train.micro_batch_size = 1
cfg.dataset.seq_length = 4096
cfg.model.seq_length = 4096
cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
cfg.dataset.enable_offline_packing = True
cfg.dataset.offline_packing_specs = PackedSequenceSpecs(
    packed_sequence_size=4096,
    pad_seq_to_mult=1,
)
```

If CP is enabled:

```python
cfg.model.context_parallel_size = 2
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False
cfg.dataset.offline_packing_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

# Offline packing is not finalized by ConfigContainer. If sequence_parallel is
# also enabled, align offline samples to both constraints explicitly:
# import math
# cfg.dataset.offline_packing_specs.pad_seq_to_mult = math.lcm(2 * CP, CP * TP)
# ConfigContainer computes this CP/SP LCM automatically for in-batch packing only.
```

If CUDA graphs are enabled for this packed path:

```python
cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
cfg.dataset.dataset_kwargs["pad_to_max_length"] = True
```

**Note:** `pad_cu_seqlens = True` also requires a metadata JSON file alongside
the packed dataset (asserted in `src/megatron/bridge/data/datasets/sft.py`).
Custom packed datasets that omit the metadata file will hit an assertion at
dataset initialization.

In-batch packing for VLM finetuning:

```python
cfg.dataset.enable_in_batch_packing = True
cfg.train.micro_batch_size = 2
```

Long-context baseline:

```python
cfg.model.seq_length = 16384
cfg.dataset.seq_length = 16384
cfg.model.context_parallel_size = 2
```

## Code Anchors

LLM packed SFT config surface:

```128:143:src/megatron/bridge/recipes/utils/dataset_utils.py
dataset_kwargs = {}
offline_packing_specs = None
if enable_offline_packing:
    dataset_kwargs["pad_to_max_length"] = True
    offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

return _text_hf_dataset_config(
    source=HFDatasetSourceConfig(dataset_name="squad"),
    preprocessing=PromptCompletionSFTPreprocessingConfig(separator=" "),
    seq_length=seq_length,
    enable_offline_packing=enable_offline_packing,
    offline_packing_specs=offline_packing_specs,
    dataset_kwargs=dataset_kwargs,
    val_proportion=0.1,
    num_workers=1,
)
```

Bridge validation:

```1220:1248:src/megatron/bridge/training/config.py
enable_in_batch_packing = getattr(self.dataset, "enable_in_batch_packing", False)
enable_offline_packing = getattr(self.dataset, "enable_offline_packing", False)
offline_packing_specs = getattr(self.dataset, "offline_packing_specs", None)

if enable_offline_packing and enable_in_batch_packing:
    raise ValueError("enable_offline_packing and enable_in_batch_packing are mutually exclusive.")
if enable_offline_packing and offline_packing_specs is None:
    raise ValueError("offline_packing_specs must be set when enable_offline_packing=True.")
...
if enable_in_batch_packing:
    ...
    cp_multiple = 2 * cp_size if cp_size > 1 else 1
    sp_multiple = cp_size * tp_size if has_sp and tp_size > 1 else 1
    self.dataset.in_batch_packing_pad_to_multiple_of = math.lcm(cp_multiple, sp_multiple)
```

```1400:1442:src/megatron/bridge/training/config.py
if self.model.context_parallel_size > 1:
    assert self.model.seq_length % (self.model.context_parallel_size * 2) == 0, ...
    if isinstance(self.dataset, FinetuningDatasetConfig):
        assert self.model.calculate_per_token_loss, ...
        assert not self.ddp.average_in_collective, ...
...
if enable_offline_packing and self.train.micro_batch_size > 1:
    raise ValueError(...)
...
if enable_in_batch_packing and self.train.micro_batch_size == 1:
    raise ValueError(...)
```

Collate-time in-batch runtime used by VLM providers:

```397:449:src/megatron/bridge/data/sequence_batching.py
def prepare_padded_or_packed_sequence_batch(
    batch,
    *,
    sequence_length,
    ...
    enable_in_batch_packing=False,
    in_batch_packing_pad_to_multiple_of=1,
    ...
):
    ...
    if enable_in_batch_packing:
        pack_right_padded_sequence_batch_to_mcore_thd(
            batch,
            sequence_length=sequence_length,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            ...
        )
        return
```

Packed THD runtime constraint:

```94:108:src/megatron/bridge/training/gpt_step.py
if batch.get("cu_seqlens_q") is not None:
    cu_seqlens = batch.get("cu_seqlens_q_padded")
    if cu_seqlens is None:
        cu_seqlens = batch["cu_seqlens_q"]
    if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
        raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
    return cu_seqlens.squeeze()

cu_seqlens = batch["cu_seqlens"]
if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
    raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
```

## Pitfalls

1. Offline packed SFT and VLM in-batch packing are different features with opposite micro-batch rules.
2. When CP is enabled, packed sequence lengths must respect `2 * context_parallel_size` divisibility.
3. For finetuning with CP, `calculate_per_token_loss=True` and `ddp.average_in_collective=False` are required.
4. `pad_cu_seqlens=True` also requires `pad_to_max_length=True`.
5. Packing support is model-family-specific. `Qwen3-Next`, `GLM-4.5`, and `Qwen3.5-VL` contain explicit opt-outs in different paths.
6. MTP finetuning is documented as incompatible with packed sequences.
7. Synthetic padding rows, including negative indices remapped through `samples_mapping`, must retain an all-zero loss mask.

## Verification

Use the checked-in unit coverage:

```bash
uv run python -m pytest tests/unit_tests/training/utils/test_packed_seq_utils.py -v && \
uv run python -m pytest tests/unit_tests/training/test_config.py -k "packed_sequence or enable_in_batch_packing or offline_and_in_batch_packing_are_mutually_exclusive or context_parallel_seq_length_divisibility or context_parallel_finetuning_validations" -v && \
uv run python -m pytest tests/unit_tests/data/packing/test_in_batch.py -v && \
uv run python -m pytest tests/unit_tests/training/test_vlm_step.py -k "deferred_in_batch_packing or packed_metadata" -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_packed_parquet.py -k "negative_index_zeroes_loss_mask" -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_sft.py -k "mapped_padding_rows_do_not_contribute_to_loss" -v
```

Success criteria:

- all selected tests pass
- offline and in-batch configuration validation remains mutually exclusive
- packed metadata reaches the training step in MCore THD form
- mapped padding rows do not contribute to loss
