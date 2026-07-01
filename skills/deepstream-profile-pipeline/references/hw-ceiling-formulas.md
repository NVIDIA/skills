# Hardware ceilings from `nvidia-smi`

How to turn `nvidia-smi` output into theoretical ceilings the skill uses as "is my measured
FPS realistic?" reference in Stage 5. All numbers are upper bounds — real workloads typically
reach 30–70% of these.

## 1. Identity and memory

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,\
clocks.max.sm,clocks.max.memory \
--format=csv,noheader,nounits
```

Keys out of this:

- `name` — e.g. `NVIDIA L40S`, `NVIDIA A100-SXM4-80GB`, `Orin (nvgpu)`
- `compute_cap` — e.g. `8.9` (used below for ops-per-clock)
- `memory.total` — used to bound `batch-size × model-memory`
- `clocks.max.sm` (MHz) — feed into the compute ceiling
- `clocks.max.memory` (MHz) — feed into the memory-bandwidth ceiling

## 2. SM count (compute cap lookup)

`nvidia-smi` doesn't print SM count directly. Resolve from `compute_cap` + GPU name:

| Compute cap | SMs typical for card | Note |
|-------------|----------------------|------|
| 7.5 (T4) | 40 | Turing |
| 8.0 (A100 40/80GB SXM/PCIe) | 108 | Ampere |
| 8.6 (A10 / A40 / RTX 30xx) | 72 / 84 / varies | Ampere |
| 8.7 (Orin AGX) | 16 | Jetson Ampere |
| 8.9 (L4 / L40 / L40S / RTX 40xx) | 60 / 142 / 142 / varies | Ada |
| 9.0 (H100 SXM/PCIe) | 132 / 114 | Hopper |
| 10.0 (Thor) | TBD | Blackwell |
| 10.0 (B200 / GB200) | varies | Blackwell |

If uncertain, read from `/proc/driver/nvidia/gpus/0/information` or
`cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` via a tiny CUDA probe; for the MVP
the table above is enough.

## 3. NVDEC / NVENC count

`nvidia-smi` does not expose engine count. **Source of truth:** the NVIDIA Video
Encode/Decode Support Matrix at <https://developer.nvidia.com/video-encode-decode-support-matrix>
— the matrix lists per-GPU NVDEC/NVENC counts and per-codec support (YES/NO).
Re-verify the table when DS major versions change; NVIDIA updates the matrix periodically.

NVDEC counts (from the matrix as of 2026-04):

| GPU | NVDEC | NVDEC generation | Notes |
|---|---|---|---|
| V100 (Volta) | 1 | 3rd gen | H264 / H265 8-bit only |
| T4 (Turing) | 2 | 4th gen | adds AV1 8-bit decode |
| A2 (Ampere) | 1 | 5th gen | low-end Ampere |
| A10 (Ampere) | 2 | 5th gen | |
| A40 (Ampere) | 2 | 5th gen | |
| A100 (Ampere) | 5 | 4th gen | |
| RTX A6000 (Ampere) | 2 | 5th gen | workstation |
| L4 (Ada) | 4 | 5th gen | inference-tuned |
| L40 / L40S (Ada) | 3 | 5th gen | |
| H100 SXM/PCIe (Hopper) | 7 | 4th gen | |
| RTX 4090 (Ada consumer) | 1 | 5th gen | |
| RTX 5090 (Blackwell consumer) | 2 | 6th gen | adds H265 10/12-bit decode |
| B200 / GB200 (Blackwell datacenter) | 7 | 6th gen | |
| Orin AGX (Jetson Ampere) | 2 | 5th gen | |
| Thor / T5000 (Jetson Blackwell) | 2 | 6.1 gen | full codec support incl. H265 12-bit |

NVENC counts vary; consult the matrix when needed. Most pipelines do not use NVENC unless
they encode output for streaming/recording.

**Decode ceiling (fps) ≈ NVDEC_count × per-unit_fps_for_codec_res** using:

| Codec × resolution | fps per NVDEC (Turing+) | fps per NVDEC (Ada / Hopper / Blackwell) |
|---|---|---|
| H264 1080p | ~1000 | ~1400 |
| H265 1080p | ~900 | ~1300 |
| H264 4K | ~250 | ~350 |
| H265 4K | ~220 | ~340 |

> **Note on the per-NVDEC fps table.** NVIDIA's matrix only publishes codec support
> (YES/NO) and engine counts — **not per-codec fps**. The numbers above are
> conservative midpoints from informal benchmarks; real workloads typically realize
> 50–70% of these. The Stage 3 microbench overrides the table when they conflict —
> measurement is always authoritative.

Example formula application: a card with N NVDECs running H265 1080p decode has roughly
N × per_unit_fps total decode headroom. If the user wants K × 1080p30 streams (= K × 30 fps),
compare K × 30 to that headroom: well below ⇒ decode is not the wall, plenty above ⇒ decode
is the wall.

## 4. Compute ceiling (TOPS)

Closed form per precision:

```
TOPS_fp16  = SM_count × clock_GHz × ops_per_clock_fp16
TOPS_int8  = SM_count × clock_GHz × ops_per_clock_int8
```

`ops_per_clock` by compute cap, assuming Tensor Cores (the path DS/TRT actually uses):

| Compute cap | FP16 (TC) ops/clock/SM | INT8 (TC) ops/clock/SM |
|---|---|---|
| 7.5 (T4) | 512 | 1024 |
| 8.0 (A100) | 1024 | 2048 |
| 8.6 / 8.7 (A10/A40/Orin) | 512 | 1024 |
| 8.9 (L4/L40/L40S) | 1024 | 2048 |
| 9.0 (H100) | 2048 | 4096 |
| 10.0 (B200/GB200) | 4096 | 8192 |

Example for L40S at 2.52 GHz max SM clock, 142 SMs, INT8:
`142 × 2.52 × 2048 ≈ 733,000 Gops ≈ 733 TOPS` (matches the L40S datasheet ~733 TOPS INT8
sparse or ~367 TOPS dense — the formula above is the dense number).

Use as: peak inference FPS ≤ `TOPS_at_precision / (model_GOPs × batch_efficiency)`, where
`batch_efficiency` ≈ 0.4 for batch=1, ≈ 0.7 for batch=16 on common detectors. This is why
Stage 3's micro-benchmark finds a plateau.

## 5. Memory-bandwidth ceiling (GB/s)

```
BW_GB_s = 2 × (mem_clock_MHz × 1e6) × (bus_width_bits / 8) / 1e9
```

`bus_width_bits` from the card datasheet (or compute cap + GPU name):

| GPU | Bus width (bits) |
|---|---|
| T4 | 256 |
| A100 (HBM2) | 5120 |
| A10 | 384 |
| L4 | 192 |
| L40 / L40S | 384 |
| H100 (HBM3) | 5120 |
| Orin AGX (LPDDR5) | 256 |
| Thor (LPDDR5X) | 256 |
| B200 (HBM3e) | 8192 |

Example: L40S at 9001 MHz GDDR6 mem clock × 384 bits → 2 × 9e9 × 48 B = 864 GB/s (matches
datasheet).

Rule of thumb: if the model's weight bytes × inference fps > 0.7 × BW, it's memory-bound;
INT8 weights help more than batching. `nsys stats --report gpu_metric_gpu_util_sum` reports
DRAM throughput directly in Stage 5.

## 6. PCIe (memcpy) ceiling

```bash
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current \
--format=csv,noheader,nounits
```

Gen/width → practical H2D bandwidth:

| Gen × width | Theoretical (GB/s) | Practical (~80%) |
|---|---|---|
| Gen3 x16 | 15.75 | 12 |
| Gen4 x16 | 31.5 | 25 |
| Gen5 x16 | 63 | 50 |

Only matters if NVMM zero-copy is broken (H2D/D2H memcpy should be <5% of wall time in a
correct DS pipeline). If Stage 5 shows memcpy >15%, check the pipeline for a CPU-memory
`videoconvert` and set `nvbuf-memory-type=0` on the source.

## Quick helper snippet

A one-liner the skill can drop into the report:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,\
clocks.max.sm,clocks.max.memory,utilization.decoder,utilization.encoder,\
pcie.link.gen.current,pcie.link.width.current \
--format=csv,noheader
```

Combined with the tables above, this gives all four ceilings (decode, compute, memory
bandwidth, memcpy) with no additional probing.
