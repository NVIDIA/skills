# Network integration contracts

Read before an adapter's first parity or training test. The docstrings own the details:
the module lifecycle in [module.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/module.py), the forward and
conditioning contracts in [network.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/network.py), output
conditioning and guidance in the
[network README](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#output-conditioning-and-guidance) and
[guidance.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/guidance.py), and registered state in the
[README](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#registered-state) with
[utils/module.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/module.py), [fsdp.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/distributed/fsdp.py)
and [checkpointer.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/checkpointer.py). Read the conditional sections
only as needed.

## Construction and ownership

Trace `FastGenModel.build_model`, `on_train_begin`, `init_preprocessors` and
`validate_modules` in [model.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/model.py).

- Constructors build the registered tree; preparing a root prepares **all** nested
  `FastGenModule`s once. `_record_precision` broadcasts the root request; express local
  precision through overrides before that preparation, not a second child preparation.
  Attach heads and embedders first and construct fresh instances after a failed
  preparation. Never deepcopy a live sharded model to make a teacher.
- `preprocessors` is a **plain dict** outside the network's state dict.
  `init_preprocessors` constructs `FastGenModule` wrappers lazily and the model registers
  them as independent frozen roots, not children prepared again under the backbone.
  Encode and decode work must go through their `forward` so hooks run. A network that
  supplies a VAE also implements the `init_vae` hook, which
  `FastGenNetwork.vae_context` checks even for an installed VAE.
- Preparation sets `train(trainable)` and re-enables gradients on trainable roots;
  override `requires_grad_` (and the mode policy) to keep submodules frozen or in eval
  mode, then verify the optimizer parameter set.
- Weights loaded by the constructor, `model.pretrained_*_path` initialisation and
  training checkpoints are distinct paths; confirm which one supplies each root and
  compare missing keys against deliberate new heads. Test repository-ID and
  local-directory loading for every root and preprocessor through the helpers in
  [env.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/env.py).

## Raw batches become model inputs once

Trace `prepare_batch` and `net_preprocessors` in `model.py`, then `encode_real`,
`build_conditioning`, `_resolve_conditions` and `validate_conditioning` in `network.py`.

- Placement, encoding and conditioning belong to `prepare_batch`: `real` is model-space
  state, `real_raw` is encoded only when `real` is absent, and `build_conditioning` runs
  even when real data is not materialised, so exercise it for prompt-only training and
  for inference.
- Plain cached conditions win over raw counterparts, and completed rich conditioning
  mappings are builder outputs, not loader inputs. Do not mutate the input mapping, and
  keep absent keys distinct from present-but-`None` ones.
- Trace the arguments each method passes to `forward`: batch fields reach the network
  only through the completed `condition`. Keep positive and negative branches
  geometrically aligned, integer ids and masks in their dtypes, state-joining latents in
  state precision and backbone-only inputs in model precision.
- Match the reference tokenizer, chat template, hidden-state selection and padding on the
  real corpus; compare mixed-length batches against per-example execution when the
  backbone ignores attention masks.
- The preprocessor host is the student (`net`, then `teacher`, then EMA), so a separate
  teacher must accept the same representation; different tokenizers or VAEs need an
  explicit design. For condition dropout see `SFTModel._mix_condition` in
  [sft.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/fine_tuning/sft.py) and implement `mix_condition` for
  structured conditions.

## Time, prediction, precision and guidance

- Establish `alpha(t) * real + sigma(t) * noise`, the native prediction type and the
  upstream time parameter before mapping tensors; convert timesteps inside the network
  (`rescale_t`, shifts, reversal) so FastGen's clock stays the schedule's. Check
  intermediate and near-endpoint times and the actual sampling grids
  ([noise schedules](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#noise-schedules)).
- Every forward ends in `transform_output`; honour `fwd_pred_type`,
  `apply_output_conditioning=False` and `output_dtype=None`, and never apply guidance or
  clean-frame constraints inside the native path, because guidance composes native
  branches and transforms once and DMD2 caches them. Embedded guidance, negative-prompt
  CFG and separate unconditional backbones are different recipes to express through
  `GuidanceConfig`. Parameterize tests over the cross-product of compatible return modes
  and prediction types, checking values, conditioning and dtype in every combination.
  Use differing model and state dtypes to expose unwanted casts when checking
  `output_dtype=None`; forwarded-argument assertions alone miss transformation errors.
- Keep state, model, optimizer and timestep precision separate: use
  `cast_to_model_precision`, pin reference FP32 state through
  `_fixed_precision_overrides`, give preprocessors their reference `precision_model`
  policy (a strict region also controls autocast), and compare operator arithmetic, not
  only output dtypes. `set_cuda_backend` in [scripts.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/scripts.py)
  sets TF32 and cuDNN flags that affect parity. Where configuration cannot reproduce the
  reference's per-module dtypes, bind a scoped policy to the instance rather than
  switching a global default.
- `sample()` serves teacher and SFT generation with the reference scheduler and guidance;
  the generic `generator_fn` follows the student grid without CFG. Validate forward
  arguments with `_validate_forward_kwargs` and sampler options with
  `_validate_sample_kwargs`; a standalone forward does not exercise what guidance and
  sampling pass.

## Loss paths need more than ordinary backward

- Adversarial student updates need input gradients through frozen teacher features.
  Freeze parameters in the regression and require finite, nonzero input gradients from
  each supported block family. Exercise early and prediction-plus-features returns and
  the empty `feature_indices` schema documented in `FastGenNetwork.forward`.
- Second-time (`r`) inputs and JVP or finite-difference objectives
  ([mean_flow.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/trajectory_matching/mean_flow.py)) need a derivative
  path that attention kernels, checkpointing and in-place operations can break; exercise
  it before declaring support.
- Attention backends can change backward finiteness without changing forward parity
  ([NaN guidance](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#loss-or-gradients-become-nan)); scope a
  workaround to the instance and keep it through deepcopy and checkpoint recompute
  (`attention_forward_without_cudnn` in [utils/module.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/module.py)).
- Distillation drives more graph variants than inference or SFT (several roots, masks and
  recompute), so with `model.torch_compile_mode` (`apply_torch_compile`) expect
  recompilation; scope compile settings to the kernel and compare against eager.

## Conditional capabilities

- **Multimodal.** State is a tensor or a flat modality mapping and conditioning may nest;
  [tree_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/tree_utils.py) dispatches both. A decoder accepting a
  dict does not make a method multimodal.
- **PDD.** `make_pdd` and `PDDMixin` in [pdd_mixin.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/utils/pdd_mixin.py)
  document the projection and schedule contract their validator enforces; heads index
  the anchored base grid, not the runtime NFE.
- **LoRA.** [lora_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/lora_utils.py) documents adapter
  injection, the shared-base teacher and the distribution, EMA and meta-init
  restrictions; prove that teacher outputs stay fixed after adapter updates.
- **Causal.** [network_causal.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/network_causal.py),
  [causal_sampling.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/causal_sampling.py) and the
  [KV-cache README](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/kv_cache/README.md) define contexts, caches and
  masks; validate teacher-forcing masks separately from cache reads and writes and
  chunk offsets.

## Distributed state and reload

- Meta initialisation loads weights on rank zero only; audit **all** non-persistent
  buffers and unregistered caches, because `materialize_meta_module` calls only the
  root's `reset_parameters` (call the superclass hook for schedule tables), which must
  rebuild in place without random draws or collectives.
- Storage pins do not set FSDP compute precision: `_validate_fsdp_pins` and
  `make_fully_shard_kwargs` isolate pinned parameters, and `fully_shard` wraps inner
  modules after activation checkpointing. Preprocessors shard only when listed in
  `model.extra_fsdp_modules` and do not inherit the meta context.
- All ranks execute sharded calls in the same order, including optional preprocessor
  branches, media callbacks and routed experts with empty local batches.
- Reload the actual format and root set (DCP shards with their marker, adapter-only
  state), and verify that inference followed by backward preserves training behavior.
  Distinguish initialisation from resume and use
  [fastgen-validate](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md) for the evidence. Start
  diagnosis at [sharded results differ](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#sharded-results-differ-or-meta-initialization-fails).
