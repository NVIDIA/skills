# Parabricks Alignment Performance Guidance

Shared BWA tuning guidance for `fq2bam`, `fq2bam_meth`, and the `germline`,
`deepvariant_germline`, and `somatic` pipelines.
(Giraffe uses `--nstreams`; see `pbrun-giraffe.md`.)

## BWA Streams

Prefer automatic stream selection for general commands: leave `--bwa-nstreams`
unset, or set `--bwa-nstreams auto` only when making the default explicit.
Parabricks picks the number of BWA streams from the GPU's device memory.

Use integer `--bwa-nstreams` values only for benchmark-driven tuning or
memory-pressure troubleshooting after confirming the selected Parabricks
version's docs. More streams increase device memory use, so fixed stream counts
should not be part of conservative default command templates.

## Optional fq2bam / fq2bam_meth controls

- `--max-read-length-on-gpu`: keep the default 512 bp unless many reads are
  longer than 512 bp. Valid values are 512 or 1024. Reads above the selected
  value still use CPU recovery. Raising GPU processing above 512 bp also
  requires raising `--max-read-length`.
- `--run-partition`: try on systems with 4 or more GPUs. It forks one worker
  process per partition. `--bwa-gpu-num-per-partition` controls GPUs per
  partition (default: 2) and must divide `--num-gpus` evenly. The total
  `--bwa-primary-cpus` must also divide evenly across the partitions.
- `--bwa-primary-cpus` and `--bwa-cpu-thread-pool` tune CPU-stage threading.
  `fq2bam`, `fq2bam_meth`, and the `germline`, `deepvariant_germline`, and
  `somatic` pipelines default to `auto` (16 primary threads per GPU) and a pool
  of 1. A manual `--bwa-primary-cpus` is the run total, not per GPU; keep its
  product with `--bwa-cpu-thread-pool` within the host's CPU threads.

Official performance guidance (4.8.0):
<https://docs.nvidia.com/clara/parabricks/v4.8.0/get-started/getting-the-best-performance>
