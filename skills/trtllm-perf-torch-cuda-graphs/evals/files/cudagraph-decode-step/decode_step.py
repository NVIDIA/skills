"""Token-by-token decode loop that we tried to speed up with a CUDA graph.

It runs, but the generated text is wrong (and with the EOS check enabled the
capture fails). `model` is a decoder with a static, pre-allocated KV cache whose
forward(tokens) takes int64 token ids of shape [B, 1] and returns logits [B, 1, V].
"""
import torch


def sample(logits, temperature):
    # Gumbel-max sampling: argmax(logits / T + Gumbel noise)
    noise = -torch.log(-torch.log(torch.rand_like(logits)))
    return torch.argmax(logits / temperature + noise, dim=-1, keepdim=True)


def generate(model, prompt_ids, max_new_tokens=128, temperature=0.8, eos_id=2, stop_on_eos=True):
    tokens = prompt_ids[:, -1:].contiguous()          # [B, 1]
    outputs = []

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):                         # captured on first use
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)[:, -1, :].float()  # [B, V]
        next_tok = sample(logits, temperature)        # [B, 1]
        if stop_on_eos and (next_tok == eos_id).all().item():
            next_tok.fill_(eos_id)

    for step in range(max_new_tokens):
        g.replay()
        outputs.append(next_tok)
        tokens = next_tok                             # feed the sampled token back in
        temperature = max(0.3, temperature * 0.98)    # anneal sampling temperature
    return torch.cat(outputs, dim=1)
