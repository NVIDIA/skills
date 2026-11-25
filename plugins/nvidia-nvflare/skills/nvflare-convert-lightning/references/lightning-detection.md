# Lightning Detection

Use this reference to decide whether to apply the Lightning conversion pattern or
hand off to another framework skill. It covers **how to use and when to override**
`nvflare agent inspect source`, not a second copy of its detection rules.

## Default Evidence Source

`nvflare agent inspect source <path> --format json` is the default detection
source. It reports supported direct Trainer ownership and routing; direct
reading supplies imports and module evidence outside those closed forms. Trust
its result by default.

At a high level, a Lightning project uses a `LightningModule` /
`LightningDataModule` and a `Trainer` fit/validate/test loop. Do not maintain a
separate list of exact import or alias forms here; `inspect` owns that and is the
tested source of truth.

`inspect` is advisory, not a hard gate. Override its default only with explicit
code or user evidence, as described below.

## Plain PyTorch Versus Lightning

Route to `nvflare-convert-pytorch` instead of this skill when the project uses a
plain `torch.nn.Module` with a manual training loop (an explicit
`for batch in loader:` loop calling `loss.backward()` and `optimizer.step()`)
and there is no `LightningModule`, `LightningDataModule`, or `Trainer`.

A project may contain both a plain `nn.Module` and a `LightningModule`. Choose
the skill from the training entry point the user asks to federate:

- if the user federates the `Trainer` fit/validate/test loop, use this skill;
- if the user federates a manual PyTorch loop and only borrows an `nn.Module`,
  use `nvflare-convert-pytorch`.

## Hugging Face Trainer Ownership

Hugging Face models and tokenizers inside a Lightning-owned training loop do
not change ownership; keep the Lightning conversion. A Hugging Face
Trainer-only entrypoint routes to `nvflare-convert-huggingface`. When inspection
finds active Lightning and Hugging Face Trainer entrypoints in the same
project, route to `nvflare-orient` and ask the user to select one owner or split
the entrypoints/jobs. Never patch both Trainers in one federated round loop.

## Lightning Trainer Wrappers

Some ecosystems build the trainer through a wrapper or factory, for example
`nl.Trainer(...)` from `nemo.lightning`. `inspect` intentionally does not promote
wrapper imports alone to Lightning, so this is a case where you confirm with
extra evidence. Use this skill only when the user explicitly asks for Lightning
conversion, canonical PyTorch Lightning evidence is also present, or an
existing/verified `nvflare.client.lightning.patch(trainer)` compatibility signal
shows that the wrapper trainer can use the Lightning Client API. The definitive
converted-state signal is a patched trainer, not the exact constructor module.

## Negative Handoff

When no Lightning evidence exists, do not force the Lightning patch pattern.
State that the project is not Lightning and route to `nvflare-convert-pytorch`
for plain PyTorch loops, or to the appropriate framework conversion skill for
TensorFlow, XGBoost, scikit-learn, or Hugging Face code.
