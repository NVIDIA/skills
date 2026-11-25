# PyTorch-Family Model Exchange

Use this reference only for PyTorch-family skills: plain PyTorch, PyTorch
Lightning, and Hugging Face Trainer. Do not load it from TensorFlow, sklearn,
XGBoost, or other non-PyTorch framework skills.

## Tensor Payload Rule

For PyTorch Client API training, outbound `FLModel(params=...)` must contain
`torch.Tensor` values from the trained model state. Do not convert outbound
weights to NumPy before sending.

The manual `flare.send` snippet below applies only to plain PyTorch, where
client code builds the payload itself:

```python
params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
assert all(isinstance(v, torch.Tensor) for v in params.values())
flare.send(
    flare.FLModel(
        params=params,
        metrics=metrics,
        meta={**meta, MetaKey.NUM_STEPS_CURRENT_ROUND: round_num_steps},
    )
)
```

`round_num_steps` is the positive number of optimizer steps completed during
the current round. For a manually generated plain-PyTorch client, count it in
the source-backed training loop and never omit it: FedAvg uses
`NUM_STEPS_CURRENT_ROUND` as its aggregation weight. The patched Lightning and
Hugging Face integrations populate this metadata themselves; do not add a
second manual payload there.

For PyTorch Lightning and Hugging Face Trainer, the patched trainer builds and
sends the payload, so do not write this snippet in patched client code. The
tensor-not-NumPy rule still applies to the PyTorch family, but patched Trainer
integrations enforce it through recipe/job configuration rather than manual
client payload construction.

## Related Recipe Construction

After `recipe show`, follow `pytorch-family-recipe-construction.md`. That
reference owns capability-gated tensor transport, decomposer, transfer-mode,
best-model, and execution-mode settings. It separately owns
`enable_tensor_disk_offload` as a server memory optimization; disk offload is
not part of the model payload or exchange-format contract. Do not copy settings
from a different recipe or move decomposer registration into a
framework-neutral executor or generated trainer code.

## State-Dict Compatibility

The server-side initial model and the site-side training model must have
compatible constructor arguments, state-dict keys, and tensor shapes. If the
model constructor needs values such as input dimension, vocabulary size, number
of classes, hidden size, or dropout, make those values explicit in both the
server recipe/job config and the client model construction path.

Pay special attention to data-derived arguments, such as a `vocab_size` built
from training data. Pin them to a shared value so the server and every site
construct the same architecture.

For a vocabulary the mapping matters, not just the size: use one shared
vocabulary/tokenizer definition so every token resolves to the same ID at every
site. FedAvg averages embedding rows by position, so a per-site token-to-ID
mapping built independently from local data would silently blend unrelated
tokens even when `vocab_size` matches.

That shared vocabulary/tokenizer must be public or pre-provided, or come from
an explicitly user-authorized cross-site statistics workflow under
`conversion-common.md` ("Preprocessing Data Locality"). Never create it by
pooling site records implicitly.

Pin only architecture or state-dict compatibility values this way. Do not treat
training-policy values or label/data-derived loss statistics as model
constructor values that must be globally pinned for exchange compatibility.
Examples include class-imbalance weights, sample weights, batch weights,
thresholds, sampler weights, and optimizer state. These values should be
computed from each site's local training partition or passed as an explicit
user-requested training policy outside the exchanged model state.

Constructing the model the same way on both sides guarantees matching keys and
shapes, so do not read NVFLARE exchange source to determine which subset of keys
is serialized. A state-dict key or tensor-shape mismatch means the server and
site constructions diverged, usually through a missing or data-derived argument.
Treat it as a conversion bug, and do not change the user model architecture to
hide the mismatch without user approval.

## Scope

This reference covers PyTorch-family tensor/state-dict exchange only. Framework
skills still own their training-loop pattern:

- plain PyTorch owns manual `nvflare.client` receive/load/train/send code;
- PyTorch Lightning owns patched `Trainer` and callback-driven model exchange;
- Hugging Face owns patched `transformers.Trainer` / TRL `SFTTrainer` exchange
  through `nvflare.client.hf.patch(trainer)` and its conversion reference.
