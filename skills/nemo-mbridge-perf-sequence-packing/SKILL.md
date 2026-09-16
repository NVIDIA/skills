---
name: nemo-mbridge-perf-sequence-packing
description: Validate and use packed sequences and long-context training in Megatron-Bridge, including offline LLM packing, collate-time VLM packing, Energon online packing, and CP constraints.
license: Apache-2.0
---

# Sequence Packing Skill

For stable background and recommendation level, see:

- @docs/training/packed-sequences.md
- @skills/nemo-mbridge-perf-sequence-packing/card.yaml

## Enablement

Offline packed SFT for LLM finetuning:

```python
import math

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

cfg.train.micro_batch_size = 1
cfg.train.global_batch_size = 8
cfg.dataset.seq_length = 8192
cfg.model.seq_length = 8192
cfg.dataset.enable_offline_packing = True

cp_size = cfg.model.context_parallel_size
tp_size = cfg.model.tensor_model_parallel_size
cp_multiple = 2 * cp_size if cp_size > 1 else 1
sp_multiple = cp_size * tp_size if cfg.model.sequence_parallel and tp_size > 1 else 1
cfg.dataset.offline_packing_specs = PackedSequenceSpecs(
    packed_sequence_size=8192,
    pad_seq_to_mult=math.lcm(cp_multiple, sp_multiple),
)
```

### Choose the offline pack length

For text-only LLM SFT and PEFT verification, start with an 8192-token offline
pack when the model context limit, memory, and model-family support allow it.
Benchmark pack lengths at equal token slots per optimizer step:

```text
token_slots_per_step = packed_sequence_size * global_batch_size
```

For example, 2K/GBS32, 4K/GBS16, and 8K/GBS8 each expose 65,536 token slots per
step. Longer packs aggregate more source examples into each physical MBS1 row
and can reduce gradient accumulation and per-step overhead. They also increase
activation memory and may expose kernel-width constraints, so select the
largest measured configuration that fits rather than assuming longer is
always faster.

Offline packing requires MBS1. Require `global_batch_size % data_parallel_size
== 0` and `global_batch_size >= data_parallel_size`; an 8K/GBS8 workload
therefore needs DP no larger than 8. Keep `model.seq_length`,
`dataset.seq_length`, and `packed_sequence_size` equal, use a fresh packed-data
output root after changing any of them, and inspect the resolved post-setup
configuration.

Equal token slots do not make different pack lengths numerically identical:
the longer target changes truncation and pack membership. Rerun finite-loss,
no-skip/NaN, and convergence sentinels before replacing verified evidence.

For finetuning with CP enabled:

```python
cfg.model.context_parallel_size = 2
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False
```

Use the same alignment formula for SFT and PEFT. It produces 1 for TP1/CP1 with
SP disabled and 4 for TP4/CP1 with SP enabled. Offline packing does not derive
the value automatically, so pin it explicitly and rebuild packed data after a
topology change.

If a dispatcher or kernel requires a fixed final token width:

```python
cfg.dataset.dataset_kwargs = {
    **(cfg.dataset.dataset_kwargs or {}),
    "pad_to_max_length": True,
}
```

Choose `packed_sequence_size` to satisfy the kernel multiple. For example,
HybridEP with a 128-token combine chunk requires a width divisible by 128.
This is separate from `pad_seq_to_mult`, which aligns each constituent
sequence for CP/SP.

If CUDA graphs are enabled for this packed path, fixed token width is required
and packed metadata must also have a static shape:

```python
cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
cfg.dataset.dataset_kwargs["pad_to_max_length"] = True
```

**Note:** `pad_cu_seqlens = True` also requires a metadata JSON file alongside
the packed dataset (asserted in `src/megatron/bridge/data/datasets/sft.py`).
Custom packed datasets that omit the metadata file will hit an assertion at
dataset initialization.

In-batch packing for GPT SFT and supported VLM finetuning:

```python
cfg.dataset.enable_in_batch_packing = True
cfg.dataset.dataloader_type = "single"
cfg.train.micro_batch_size = 4
```

For local or materialized GPT-SFT JSONL, this keeps the existing mmap-backed
dataset and performs tokenization lazily. Both prompt/completion
(`GPTSFTDataset`) and chat (`GPTSFTChatDataset`) preserve their loss-mask
semantics. Use `dataloader_type="single"` or `"cyclic"` so every DataLoader
yield is one logical microbatch; GPT-SFT in-batch packing does not support the
global-batch `"batch"` dataloader.

Energon online packing for Qwen-VL uses Energon's per-worker candidate buffer
instead of limiting selection to one collator micro batch:

```python
cfg.dataset.packing_buffer_size = 16
cfg.dataset.micro_batch_size = 1
cfg.train.micro_batch_size = 1
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False
```

`packing_buffer_size` is the sole native-packing selector; leave the legacy
collator- and step-owned packing flags at their defaults. Use `vlm_step`. The
buffer size counts prepared candidate samples per worker, not bytes or packed
tokens. Since prepared image/video patch tensors remain in
host memory until selection, start at 8-16 for high-resolution or video data
and measure worker RSS, first-batch latency, and bin fill before increasing it.
This path does not write offline packs; the source WebDataset shards remain
unchanged. It supports eager Qwen-VL with MBS1 and rejects MTP, CUDA graphs,
Qwen3-VL DistTrain, and PP. Requested MoE expert-parallel communication overlap
is disabled with a warning. Standard eager `alltoall` EP has functional coverage
for Qwen3.6-35B-A3B at TP1/PP1/EP8 with overlap disabled; this is not performance
evidence. Other EP dispatchers are accepted with fixed-width native packs but do
not yet have equivalent runtime evidence. The Qwen-VL model derives a MoE
padding mask from logical and physical THD boundaries so fixed-width gaps do not
enter auxiliary-loss, z-loss, or expert-bias statistics. Current MCore may still
dispatch padded positions; expert-capacity/token-dropping configurations lack
native-packing runtime coverage.

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

The shared text-dataset helper currently opts into fixed-width packs. Treat
that as a helper default, not a universal offline-packing runtime requirement;
preserve it when the selected dispatcher, kernel, or CUDA-graph path requires
static width.

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

GPT-SFT direct-row packing:

```627:671:src/megatron/bridge/data/datasets/gpt_sft.py
def _collate_in_batch(self, batch):
    ...
    return build_mcore_thd_sequence_batch_from_rows(...)
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

1. Offline packed SFT, runtime in-batch packing, and Energon online packing are different features. Offline and Energon packing use physical MBS1; runtime in-batch packing uses MBS greater than one.
2. GPT-SFT in-batch packing requires `dataloader_type="single"` or `"cyclic"`; it does not support `"batch"`.
3. When CP is enabled, packed sequence lengths must respect `2 * context_parallel_size` divisibility.
4. For finetuning with CP, `calculate_per_token_loss=True` and `ddp.average_in_collective=False` are required.
5. `pad_cu_seqlens=True` also requires `pad_to_max_length=True`.
6. Packing support is model-family-specific. `Qwen3-Next`, `GLM-4.5`, and `Qwen3.5-VL` contain explicit opt-outs in different paths.
7. MTP finetuning is documented as incompatible with packed sequences.
8. Synthetic padding rows, including negative indices remapped through `samples_mapping`, must retain an all-zero loss mask.
9. `global_batch_size` must be divisible by and no smaller than data parallel size when offline packing uses MBS1.
10. Derive `pad_seq_to_mult` from CP/TP/SP for both SFT and PEFT; do not hardcode different values by workload type.
11. `pad_to_max_length` controls final pack width and is conditional on fixed-shape execution requirements.
12. Energon `packing_buffer_size` is per worker and also affects validation; global/eval batch counts refer to physical packs rather than source conversations.
13. Exact Energon loader resume requires unchanged shards/splits, DP world size, worker counts, shuffle settings/seed, processor, sequence length, topology, and packing-buffer size.

## Verification

Use the checked-in unit coverage:

```bash
uv run python -m pytest tests/unit_tests/training/utils/test_packed_seq_utils.py -v && \
uv run python -m pytest tests/unit_tests/training/test_config.py -k "packed_sequence or enable_in_batch_packing or offline_and_in_batch_packing_are_mutually_exclusive or context_parallel_seq_length_divisibility or context_parallel_finetuning_validations" -v && \
uv run python -m pytest tests/unit_tests/data/packing/test_in_batch.py -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_gpt_sft.py -k "in_batch_packing" -v && \
uv run python -m pytest tests/unit_tests/data/builders/test_gpt_sft_config.py -v && \
uv run python -m pytest tests/unit_tests/training/test_vlm_step.py -k "deferred_in_batch_packing or packed_metadata" -v && \
uv run python -m pytest tests/unit_tests/models/qwen_vl/data/test_energon.py tests/unit_tests/data/builders/test_energon_builder.py -v && \
uv run python -m pytest tests/unit_tests/tutorials/test_multimodal_data_tutorials.py -k "native_packing_loader" -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_packed_parquet.py -k "negative_index_zeroes_loss_mask" -v && \
uv run python -m pytest tests/unit_tests/data/datasets/test_sft.py -k "mapped_padding_rows_do_not_contribute_to_loss" -v
```

Success criteria:

- all selected tests pass
- offline and in-batch configuration validation remains mutually exclusive
- packed metadata reaches the training step in MCore THD form
- GPT-SFT in-batch packing rejects the global-batch `"batch"` dataloader
- native Energon packing restores pending groups exactly and flushes finite partial buffers without dropping samples
- mapped padding rows do not contribute to loss
