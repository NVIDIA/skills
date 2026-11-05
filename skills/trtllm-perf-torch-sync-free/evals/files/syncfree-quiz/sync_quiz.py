# Every numbered line below runs inside a hot loop on the default CUDA stream.
#   x_gpu, y_gpu       : float32 CUDA tensors of shape [4096]
#   mask_gpu           : bool CUDA tensor of shape [4096]
#   idx_gpu            : int64 CUDA tensor of indices
#   repeats_gpu        : int64 CUDA tensor of per-row repeat counts (sum is 4096)
#   i_gpu, j_gpu       : 0-dim int64 CUDA tensors computed earlier on the GPU
#   idx_list           : Python list of ints
#   ev                 : torch.cuda.Event recorded earlier on another stream
import torch

a = torch.zeros(4096, device="cuda")                                       # L1
b = x_gpu[idx_list]                                                        # L2
c = torch.where(mask_gpu, x_gpu, 0)                                        # L3
d = torch.where(mask_gpu)                                                  # L4
e = x_gpu[idx_gpu]                                                         # L5
x_gpu.masked_fill_(mask_gpu, -1)                                           # L6
f = x_gpu[i_gpu:j_gpu]                                                     # L7
g = x_gpu.type(torch.LongTensor)                                           # L8
h = x_gpu.type(torch.long)                                                 # L9
m = max(x_gpu[0], y_gpu[0])                                                # L10
n = torch.max(x_gpu[0], y_gpu[0])                                          # L11
r = torch.repeat_interleave(y_gpu, repeats_gpu, dim=0)                     # L12
s = torch.repeat_interleave(y_gpu, repeats_gpu, dim=0, output_size=4096)   # L13
u = torch.unique(idx_gpu)                                                  # L14
ev.wait()                                                                  # L15
ev.synchronize()                                                           # L16
