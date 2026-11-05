"""Training step for a token-classification model; called once per iteration.

`batch` comes from a DataLoader(pin_memory=True). `class_weights` is a Python
list of floats (one per class). The training loop logs the returned dict.
"""
import torch
import torch.nn.functional as F

IGNORE = -100


def train_step(model, optimizer, batch, step, class_weights, clip=1.0):
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    weight = torch.tensor(class_weights, device="cuda")

    logits = model(input_ids)                       # [B, T, C]
    flat_logits = logits.flatten(0, 1)
    flat_labels = labels.flatten()
    valid = flat_labels != IGNORE
    per_tok = F.cross_entropy(flat_logits, flat_labels.clamp(min=0), weight=weight, reduction="none")
    loss = per_tok[valid].mean()
    print(f"step {step}: loss={loss.item():.4f}")   # debug print left in

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):   # skip steps with bad gradients
        optimizer.zero_grad(set_to_none=True)
        return None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    preds = flat_logits.argmax(-1)
    acc = (preds[valid] == flat_labels[valid]).float().mean()
    n_tokens = int(valid.sum())
    return {"loss": loss.item(), "grad_norm": grad_norm.item(), "acc": acc.item(), "tokens": n_tokens}
