"""Top-k post-processing for our ranking service (runs once per request batch).

A colleague "removed the syncs" here by adding non_blocking=True to both GPU->CPU
copies. torch.cuda.set_sync_debug_mode("warn") no longer complains and p50
latency dropped, but every few thousand requests we now return nonsense token ids
and probabilities.
"""
import torch


def postprocess(scores_gpu: torch.Tensor, k: int, vocab: list[str]):
    topk = torch.topk(scores_gpu, k, dim=-1)
    ids_cpu = topk.indices.to("cpu", non_blocking=True)
    probs_cpu = topk.values.softmax(dim=-1).to("cpu", non_blocking=True)

    results = []
    for row_ids, row_probs in zip(ids_cpu.tolist(), probs_cpu.tolist()):
        results.append([(vocab[i], round(p, 4)) for i, p in zip(row_ids, row_probs)])
    return results
