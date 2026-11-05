# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
import os

import torch


def get_sm_version() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def maybe_compile(func=None, **compile_kwargs):
    """torch.compile `func` unless TRTLLM_DISABLE_TORCH_COMPILE=1."""

    def decorator(f):
        if os.environ.get("TRTLLM_DISABLE_TORCH_COMPILE", "0") == "1":
            return f
        return torch.compile(f, **compile_kwargs)

    return decorator(func) if func is not None else decorator


@maybe_compile(dynamic=True)
def maybe_compiled_cat(tensors, dim: int):
    return torch.cat(tensors, dim)


@maybe_compile(dynamic=True)
def maybe_compiled_copy_(dst, src):
    dst.copy_(src)
