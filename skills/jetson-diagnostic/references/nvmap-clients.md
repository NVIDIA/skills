# Reading `/sys/kernel/debug/nvmap/iovmm/clients`

> **Which source to trust for per-process GPU memory on Jetson**
>
> Run once: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`.
> - If that prints a plain integer, the device is on the unified `nvidia.ko`
>   compute-driver stack. `nvidia-smi --query-compute-apps=pid,process_name,used_memory`
>   is the authoritative per-process GPU-memory source. Under that stack
>   `iovmm/clients` will typically show `0K` for every row even under heavy CUDA
>   load, because CUDA allocations bypass nvmap.
> - If that prints `[N/A]`, `Not Supported`, or is empty, the device is on the
>   `nvgpu` in-tree driver. `nvidia-smi` is a stub and cannot report
>   Memory-Usage or compute processes; **this file** is the authoritative
>   per-process GPU-memory source.
>
> The skill's `scripts/snapshot.sh` picks the right source automatically with
> that one capability probe; no SKU or BSP check.

NvMap is the kernel allocator that backs CUDA, multimedia buffers (NVENC/NVDEC, V4L2, libargus), and accelerator surfaces on Jetsons running the `nvgpu` driver. Per-process accounting is exposed at:

```bash
sudo cat /sys/kernel/debug/nvmap/iovmm/clients
```

## Example

```
CLIENT                        PROCESS                PID    SIZE
user                          gnome-shell           1543    220K
user                          Xorg                  1234    180K
user                          vlm-server            4321    524288K
total                                                       524688K
```

## Field meanings

| Column     | Meaning                                                                          |
|------------|----------------------------------------------------------------------------------|
| `CLIENT`   | NvMap client class. `user` = user-space process. Kernel clients also appear.     |
| `PROCESS`  | Process name (kernel `comm`).                                                    |
| `PID`      | Process ID.                                                                      |
| `SIZE`     | Bytes (suffixed `K`/`M`) currently allocated through NvMap.                      |
| `total`    | Sum across all clients. The most useful single number on this page.              |

## Why NvMap matters

On Jetson, **CPU-side and GPU/multimedia memory share the same physical pool**. A process that allocates a giant CUDA tensor or a chain of NVDEC surfaces will not show that memory in `/proc/[pid]/status` (RSS) or `top`, but it _will_ show it here. If `free -m` says memory is full and `top` cannot account for it, NvMap is the next stop.

## Quick triage

- **Total NvMap >= 50% of system memory** → graphics / multimedia / CUDA is the dominant consumer; see [tegrastats-fields.md](tegrastats-fields.md) `GR3D_FREQ`.
- **GUI processes (`gnome-shell`, `Xorg`) holding non-trivial NvMap** → headless mode would reclaim it. See `jetson-headless-mode`.
- **One inference process holding most of the total** → likely model weights + KV cache. Tune the runtime (smaller batch, smaller context, lower precision). See `jetson-inference-mem-tune`.
- **A dead container or stale `python3` holding NvMap** → reap it; container teardown sometimes leaks GPU surfaces.

## Permissions

`debugfs` is root-readable. Agents running unprivileged will see "Permission denied". Either run with `sudo` or add a small `sudo`-rules entry for the diagnostic command.

## Per-client view

For a richer breakdown (per-allocation, with backtraces) see:

```bash
sudo cat /sys/kernel/debug/nvmap/iovmm/allocations
```

Verbose; use only when triaging a specific leak.
