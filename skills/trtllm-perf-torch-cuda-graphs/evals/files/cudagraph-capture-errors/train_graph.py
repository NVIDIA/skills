"""Excerpt of our training script with manual full-step CUDA graph capture (PyTorch 2.5)."""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def build(mode, model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    if mode == "ddp":
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        model = DDP(model.cuda(), device_ids=[local_rank])        # built on the default stream
    return model, loader


def capture(model, optimizer, criterion, static_x, static_y, ema_model=None):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):                                     # warmup on a side stream
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            criterion(model(static_x), static_y).backward()
    torch.cuda.current_stream().wait_stream(s)

    aux = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_loss = criterion(model(static_x), static_y)
        static_loss.backward()
        if ema_model is not None:                                  # run 3 only
            aux.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(aux):
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(0.999).add_(p.detach(), alpha=0.001)
    return g, static_loss


def train(model, optimizer, loader, g, static_x, static_y):
    for x, y in loader:                                            # DataLoader keeps pinning in the background
        static_x.copy_(x, non_blocking=True)
        static_y.copy_(y, non_blocking=True)
        g.replay()
        optimizer.step()
