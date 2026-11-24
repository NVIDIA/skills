# Smoke overrides for Cosmos-Embed1 actions

For a small functional check, keep the same specs and override the expensive knobs on the command line:

```bash
train.max_iter=1
train.validation_iter=2
train.checkpoint_iter=1
train.optim.optim=adamw
train.optim.warmup_steps=0
train.optim.lr_decay_iters=1
dataset.train_dataset.batch_size=1
dataset.val_dataset.batch_size=1
dataset.train_dataset.workers=0
dataset.val_dataset.workers=0
```

When shortening the cosine scheduler for smoke runs, keep `train.optim.lr_decay_iters` greater than `train.optim.warmup_steps`, or set `train.optim.warmup_steps=0` as shown above. The scheduler divides by `lr_decay_iters - warmup_steps`, so equal values fail before the checkpoint is written.

If no local Cosmos-Embed1 pretrained checkpoint is available, set `model.pretrained_model_path=null` for a plumbing-only smoke train. The model quality is meaningless in that mode, but the train/evaluate/inference/export action paths can still be exercised. The Q-Former path can still fetch `google-bert/bert-base-uncased`; provide `HF_TOKEN` or a mounted HuggingFace cache for fresh ephemeral containers.

For the LoRA recipe, a 20-iteration run with `train.validation_iter=20 train.checkpoint_iter=20` is enough to produce a checkpoint, run the `lora_B` check, and exercise HF and ONNX export on it (see `lora-finetune-physicalai.md`).

For evaluation and inference smoke tests on a tiny subset:

```bash
evaluate.callbacks.embedding_visualization=false
evaluate.callbacks.max_eval_samples=8
dataset.test_dataset.batch_size=1
dataset.test_dataset.workers=0
inference.k=2
dataset.inference_dataset.batch_size=1
dataset.inference_dataset.workers=0
```
