# CLIP LoRA and preservation regularization

The training spec has top-level `peft` and `regularization` blocks. The
legacy tower contract uses
`peft.enabled: true`, `peft.method: lora`, and separate
`peft.vision.enabled` / `peft.text.enabled` booleans. Enable at least one
tower for LoRA. Both tower flags default to `false`, as does `peft.enabled`;
leaving them at their defaults preserves full fine-tuning. Each enabled tower
has `target_modules`, `num_last_blocks` (3; 0 means all blocks), `rank` (8),
`alpha` (16; scale is alpha/rank), and `dropout` (0.05). Set target modules
for the selected backbone: SigLIP2 uses `q_proj`, `k_proj`, `v_proj`,
`out_proj`; RADIO uses `qkv`, `proj`, so override the SigLIP2 target defaults
for RADIO. Legacy config metadata also lists `qkv`, `proj` for OpenCLIP;
that metadata alone does not establish working LoRA injection.

Under the tower `mode` contract, OpenCLIP's `nn.MultiheadAttention` towers
reject `mode: lora`. Use `full` or `frozen` for OpenCLIP, or select a
supported SigLIP2/RADIO configuration for LoRA. Changing target-module names
does not remove this restriction; verify the selected runtime's support.

For each LoRA tower, use an integer `rank >= 1`, an integer `alpha >= 1`,
and finite `0 <= dropout < 1`. Defaults are rank 8, alpha 16, and dropout
0.05. `num_last_blocks` must be an integer from 0 through the number of
transformer blocks in that tower; 0 selects all blocks. Check these bounds
when preparing specs and AutoML search spaces. Rank 0 makes alpha/rank
undefined, while alpha 0 or dropout 1 disables the adapter's contribution.
Do not use those values to freeze a tower; use `mode: frozen` under the
newer contract. See `references/error-patterns.md` for failures from older
runtimes that do not enforce the numeric bounds.

LoRA trains a small adapter parameter set, but the documented training
checkpoint remains a full CLIP checkpoint; do not budget storage as though it
were an adapter-only file. `regularization.enabled` defaults to `false`.
Enabling it creates a frozen teacher copy of the full model and adds
embedding MSE (weight 0.05), cosine (0.05), and image-text similarity
preservation (0.10) losses. Budget memory for that second model copy.

Preservation regularization has not been fully validated for SigLIP2
fine-tuning on PAS and is not currently recommended for that workflow. Keep
`regularization.enabled: false` in the recommended PAS configuration.
Enabling it keeps the frozen teacher on the GPU and adds a teacher forward
pass, increasing VRAM use and step time. The teacher is excluded from saved
checkpoints; its extra memory cost does not increase checkpoint size.

A config field alone does not establish that LoRA injection works in an
image. Verify the selected image exposes the required fields and injection
path before launching LoRA; keep PEFT disabled if it does not. Use the schema
of the selected runtime and apply the migration below when it uses tower
`mode` fields.

#### Checkpoint actions

For a LoRA checkpoint, recover the training spec from the parent job and
preserve its model settings and complete top-level `peft` block in PyTorch
`evaluate`, `inference`, and `export` specs. Keep the per-tower `enabled` or
`mode` settings, `target_modules`, `num_last_blocks`, `rank`, `alpha`, and
`dropout`, plus `method` and `train_logit_calibration` when present. Use the
selected runtime's contract; apply the migration below if it requires modes.

Packaged action templates and evaluation/export schemas omit PEFT, so merge
the recovered block explicitly into the nested action spec. The checkpoint
loader constructs the model using that spec; disabling PEFT or changing the
adapter layout can cause missing or unexpected state-dict keys. If the
training configuration is unavailable, recover it or request the training
spec before proceeding rather than guessing adapter settings.

ONNX export loads the adapters before merging them into the base weights.
Subsequent TensorRT engine generation and TensorRT evaluation/inference use
the exported model and do not need a training `peft` block.

#### Migrating PEFT specs from 7.2 to 7.3

The 7.3 tower contract replaces each tower's `enabled` boolean with `mode`.
Apply the following mapping independently to `vision` and `text`:

| 7.2 tower setting | 7.3 tower setting |
|---|---|
| `peft.<tower>.enabled: true` | `peft.<tower>.mode: lora` |
| `peft.<tower>.enabled: false` | `peft.<tower>.mode: frozen` |

Remove the old tower `enabled` keys; keep the top-level `peft.enabled` and
`peft.method` keys. A tower's `mode` defaults to `frozen`, and `full` is an
additional option for training all its parameters. Existing `target_modules`,
`num_last_blocks`, `rank`, `alpha`, and `dropout` settings remain under their
respective towers.

The new `peft.train_logit_calibration` flag defaults to `true`, allowing
`logit_scale` and optional `logit_bias` to train while PEFT is enabled. Set it
to `false` to freeze those parameters. A migrated spec fragment for LoRA on
both SigLIP2 towers is:

```yaml
peft:
  enabled: true
  method: lora
  train_logit_calibration: true
  vision:
    mode: lora
  text:
    mode: lora
```

`ConfigKeyError: Key 'enabled' not in 'CLIPLoRATargetConfig'` with
`full_key: peft.vision.enabled` or `peft.text.enabled` means the runtime
expects this newer contract. Update both tower blocks before retrying; the
error occurs during config merge, before model construction. See
`references/error-patterns.md`. Apply this migration when selecting a runtime
with the 7.3 contract; the packaged templates and schema describe the legacy
boolean surface and must be adapted for runtimes that require tower modes.

#### Encoder freeze flags and PEFT precedence

Under the tower `mode` contract, `peft.enabled: true` makes
`peft.vision.mode` and `peft.text.mode` control encoder trainability.
PEFT first freezes all model parameters, then applies each tower's mode:

| Tower mode | Result when PEFT is enabled |
|---|---|
| `frozen` | All parameters in that encoder remain frozen. |
| `full` | All parameters in that encoder become trainable. |
| `lora` | Injected adapter parameters are trainable; backbone parameters remain frozen. |

These modes override the earlier `model.freeze_vision_encoder` and
`model.freeze_text_encoder` settings in both directions. For example,
`model.freeze_vision_encoder: true` with `peft.vision.mode: full` trains the
vision encoder; `model.freeze_text_encoder: false` with
`peft.text.mode: frozen` freezes the text encoder. With `peft.enabled: false`,
the model freeze flags apply normally. Logit calibration parameters are
controlled separately by `peft.train_logit_calibration` in this contract.

Use tower modes as the source of trainability when PEFT is enabled, remove
conflicting model freeze flags, and verify the per-tower trainable-parameter
counts in the launch logs, even if no conflict warning is emitted.
These `mode` fields belong to the
newer contract and must not be mixed with the older per-tower `enabled`
booleans above; the packaged templates describe that older config surface.

