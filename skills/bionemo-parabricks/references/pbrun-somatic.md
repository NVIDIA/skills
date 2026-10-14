# Parabricks somatic

## First Steps

1. Confirm tumor-normal or tumor-only workflow support for the selected version.
2. Collect required inputs:
   - Reference FASTA.
   - Tumor BAM/CRAM or FASTQ inputs as supported.
   - Normal BAM/CRAM or FASTQ inputs when applicable.
   - Output VCF or output directory.
   - Somatic resources such as germline resource, panel of normals, known-sites,
     or intervals when required by the workflow.
3. Ask for tumor and normal sample names explicitly.

## Command Shape

```bash
pbrun somatic \
  --ref /workdir/<reference.fa> \
  <version-specific-tumor-normal-inputs> \
  <version-specific-output-options>
```

Verify exact input mode, resources, output, interval, and filtering flags
against the selected version.

Optional performance controls, inherited from the alignment and Mutect stages
of the pipeline:

- `--max-read-length-on-gpu`: keep the default 512 bp unless many reads are
  longer than 512 bp. Valid values are 512 or 1024. Reads above the selected
  value still use CPU recovery, and processing longer reads at all also requires
  raising `--max-read-length`.
- `--run-partition`: genome/Mutect partitions, not fq2bam GPU-process partitioning.
- `--bwa-gpu-num-per-partition` (default: 2): BWA GPUs per worker when
  `--run-partition` is set; must divide `--num-gpus`.
- `--gpu-num-per-partition`: Mutect GPUs per partition.
- `--num-streams-per-gpu` (default: 1): Mutect streams, not a partition flag.
- `--mutect-low-memory` and `--low-memory` for memory-pressure troubleshooting.

## Somatic Pipeline Option Mapping

Use this mapping when translating a baseline somatic workflow to
`pbrun somatic`. Depending on the selected version and requested caller, the
baseline may include BWA-MEM/GATK preprocessing plus Mutect2-style calling or a
DeepSomatic-style caller. Treat this as a pipeline mapping, not a one-to-one
single-tool mapping.

| Baseline option | `pbrun somatic` equivalent | Notes |
| --- | --- | --- |
| Reference FASTA | `--ref` | Required reference FASTA path. |
| Tumor FASTQ inputs | Tumor FASTQ input flags documented for the selected version | Verify exact tumor FASTQ flag names before finalizing. |
| Normal FASTQ inputs | Normal FASTQ input flags documented for the selected version | Verify exact normal FASTQ flag names before finalizing. |
| Tumor BAM/CRAM input | Tumor BAM input flag documented for the selected version | Use prepared-alignment mode only when supported. |
| Normal BAM/CRAM input | Normal BAM input flag documented for the selected version | Use paired mode when required by the workflow. |
| Tumor sample name | Tumor sample-name flag documented for the selected version | Must match BAM read group sample names where applicable. |
| Normal sample name | Normal sample-name flag documented for the selected version | Must match BAM read group sample names where applicable. |
| BWA-MEM read group/options | Read group flags and `--bwa-options` where documented | Do not invent sample or read group values. |
| GATK/Picard duplicate metrics | `--out-duplicate-metrics` or selected metrics flag | Verify output support for the selected version. |
| Mutect2 `--germline-resource` | `--mutect-germline-resource` | Germline resource VCF when used. |
| Mutect2 `--panel-of-normals`, `--pon` | No direct `pbrun somatic` equivalent | Use a `mutectcaller` PON workflow with `prepon`/`postpon` when required. |
| Mutect2/DeepSomatic output VCF | `--out-vcf` | Output somatic VCF. |
| Intervals/regions | `--interval` or `--interval-file` | Parabricks separates inline intervals from interval files where documented. |
| Baseline options not listed here | No direct equivalent | Not exposed by current Parabricks docs for `pbrun somatic`. |

## somatic Options Without Baseline Somatic Equivalents

| `pbrun somatic` option | Why it has no direct baseline equivalent |
| --- | --- |
| End-to-end tumor/normal input-mode flags | Parabricks combines preprocessing and somatic calling stages. |
| `prepon`/`postpon` PON handoff flags | Parabricks-specific decomposition of PON processing. |
| `--max-read-length`, `--min-read-length`, `--max-read-length-on-gpu` | Parabricks FASTQ filtering/alignment bounds and GPU read-length cap (512 or 1024 bp). |
| `--run-partition` | Genome/Mutect partitions (one process per partition). Not fq2bam GPU-process partitioning. |
| `--bwa-gpu-num-per-partition` | BWA GPUs per worker when `--run-partition` is set (default: 2; must divide `--num-gpus`). |
| `--gpu-num-per-partition` | Mutect GPUs per partition. |
| `--num-streams-per-gpu` | Mutect streams (default: 1). Not a partition flag. |
| `--mutect-low-memory` | Parabricks low-memory mode for the Mutect stage. |
| GPU alignment/calling stream, queue, and memory controls | Parabricks GPU runtime tuning. |
| GPU write/sort/storage controls | Parabricks accelerated output pipeline. |
| Caller-specific TensorRT/model flags where documented | Parabricks model deployment differs from CPU/GATK or Google runtimes. |

## Validation

- Tumor/normal labels and sample relationships are correct.
- Reference and resources match the same build.
- Output VCF or output directory exists.

## Guardrails

- Do not use for germline calling.
- Do not invent tumor/normal relationships, panel-of-normals, or resource files.
- If the user asks for a specific caller, consider `mutectcaller` or
  `deepsomatic` instead.
