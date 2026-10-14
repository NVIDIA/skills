# Parabricks Alignment Performance Guidance

Shared BWA-stream tuning guidance for the `fq2bam` and `fq2bam_meth` commands.
(Giraffe uses `--nstreams`; see `pbrun-giraffe.md`.)

## BWA Streams

Prefer the documented automatic stream selection for general commands: leave
`--bwa-nstreams` unset, or set `--bwa-nstreams auto` only when making the
default explicit. NVIDIA Parabricks 4.7.0 documentation says Parabricks
automatically chooses an optimal number of BWA streams from the GPU device
memory specification.

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
  partition (default: 2) and must divide `--num-gpus` evenly.
- For `fq2bam_meth`, `--bwa-primary-cpus` and `--bwa-cpu-thread-pool` tune
  CPU-stage threading (`--bwa-cpu-thread-pool` defaults to 1; check the selected
  release before reusing an older fq2bam default).

Official performance guidance:
<https://archive.docs.nvidia.com/clara/parabricks/4.7.0/GettingStarted/BestPerformance.html>
