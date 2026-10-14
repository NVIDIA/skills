# Parabricks deepvariant

## First Steps

1. Confirm the input is prepared aligned reads. If starting from FASTQ and the
   user wants an end-to-end germline pipeline, consider
   `pbrun-deepvariant_germline.md`.
2. Collect required inputs:
   - Reference FASTA.
   - Input BAM/CRAM.
   - Output VCF and optional gVCF.
   - Model type or model file/resource when required.
3. Ask for intervals, sample name, `--haploid-contigs`, and logs only when
   relevant. Do not pass `--sample-sex`.

## Command Shape

```bash
pbrun deepvariant \
  --ref /workdir/<reference.fa> \
  --in-bam /workdir/<input.bam> \
  --out-variants /outputdir/<sample.vcf.gz>
```

Verify exact model, gVCF, interval, and output flags against the selected
version.

## Gotchas

Parabricks DeepVariant `--gvcf` option actually produces both .g.vcf and .vcf files with the same name. Do not generate separate commands for gvcf and vcf outputs.

The small model is enabled by default. Do not add `--enable-small-model`; that
flag is not in the current tool reference. For WES, pass `--disable-small-model`
together with `--use-wes-model` when matching Google DeepVariant.

With both `--proposed-variants` and `--gvcf`, Parabricks warns and ignores
`--run-partition`; the [4.7.1 manual](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/deepvariant)
also warns of a slowdown for that combination. Preserve the requested candidate
input and gVCF output, and omit the ignored partition flag. Removing it does not
prove a speedup, and increasing GPU streams cannot restore partitioning. Measure
performance with the required output contract before making throughput claims.

## DeepVariant Option Mapping

Use this when translating Google DeepVariant `run_deepvariant` to
`pbrun deepvariant`. Core DeepVariant mappings (--ref, `--in-bam`, `--out-variants`,
model modes, intervals, gVCF, models, and shard/stream controls) for
`deepvariant` are in [shared-options.md](shared-options.md). The small model is
enabled by default; variants it calls include an additional `MID` VCF field.

| Google DeepVariant option | `pbrun deepvariant` equivalent | Notes |
| --- | --- | --- |
| `--proposed_variants` | `--proposed-variants` | Candidate/importer VCF input. |
| `--make_examples_extra_args` for supported candidate/pileup/read controls | Matching explicit Parabricks flags such as `--vsc-*`, `--alt-aligned-pileup`, `--variant-caller`, `--min-*`, `--channel-*` | Parabricks exposes many make-examples options as first-class flags. |
| Google DeepVariant options not listed here or in shared-options | No direct equivalent | Not exposed for `deepvariant` in current docs. |

## deepvariant Options Without DeepVariant Equivalents

GPU runtime tuning for `deepvariant` is in [shared-options.md](shared-options.md)
(**Execution controls**).

| `pbrun deepvariant` option | Why it has no Google DeepVariant equivalent |
| --- | --- |
| `--keep-legacy-allele-counter-behavior` | Compatibility flag tied to a specific upstream behavior change. |
| `--max-read-size-512`, `--prealign-helper-thread`, `--filter-reads-too-long` | Read-size and helper-thread controls. |
| `--haploid-contigs` | Haploid-contig handling convenience. |

## Validation

- BAM/CRAM and reference build match.
- Model/resource selection matches sequencing technology and assay.
- Output VCF/gVCF exists and is indexed when requested.

## Guardrails

- Do not use this as the end-to-end FASTQ pipeline unless the selected version
  documents that mode.
- Do not infer model type from filename alone.
