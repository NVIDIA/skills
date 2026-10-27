# Parabricks deepsomatic

## First Steps

1. Confirm the somatic analysis design: tumor-normal or tumor-only if supported
   by the selected version.
2. Collect required inputs:
   - Reference FASTA.
   - Tumor BAM/CRAM.
   - Normal BAM/CRAM when applicable.
   - Output VCF/gVCF or output directory.
   - Model or resource bundle when required by the selected version.
3. Ask for intervals, sample names, optional candidate resources, and logs only
   when relevant.

## Command Shape

```bash
pbrun deepsomatic \
  --ref /workdir/<reference.fa> \
  <version-specific-tumor-normal-inputs> \
  <version-specific-output-options>
```

Verify exact input, model, output, interval, and sample-label flags against the
selected version before finalizing.

## Gotchas

Parabricks `deepsomatic` has no `--gvcf` option; write the VCF with
`--out-variants`.

## DeepSomatic Option Mapping

Use this when translating Google DeepSomatic to `pbrun deepsomatic`. Shared
DeepSomatic mappings (--ref, `--out-variants`, model modes, intervals, models,
and shard/stream controls) for `deepsomatic` are in
[shared-options.md](shared-options.md).

| Google DeepSomatic option | `pbrun deepsomatic` equivalent | Notes |
| --- | --- | --- |
| `--reads_tumor` | `--in-tumor-bam` | Required tumor BAM/CRAM input. |
| `--reads_normal` | `--in-normal-bam` | Required normal BAM/CRAM for paired mode. |
| `--make_examples_extra_args` for supported candidate/pileup/read controls | Matching explicit Parabricks flags such as `--vsc-*`, `--alt-aligned-pileup`, `--min-mapping-quality`, and `--channel-*` | Parabricks exposes many make-examples options as first-class flags. |
| Google DeepSomatic options not listed here or in shared-options | No direct equivalent | Not exposed for `deepsomatic` in current docs. |

Parabricks GPU runtime and wrapper controls for `deepsomatic` are in
[shared-options.md](shared-options.md) and
[command-conventions.md](command-conventions.md#shared-wrapper-controls).

## Validation

- Tumor and normal sample roles are explicit and correct.
- BAM/CRAM, reference, indexes, model/resources, and intervals match the same
  reference build.
- Output VCF/gVCF or output directory exists.

## Guardrails

- Do not substitute `deepsomatic` for germline DeepVariant.
- Do not invent tumor/normal relationships or model files.
- Do not promise exact sensitivity/specificity without comparable validation.
