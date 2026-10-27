# Parabricks indexgvcf

## First Steps

1. Confirm the input is a GVCF that needs indexing. `haplotypecaller` and
   `deepvariant` already write `<output>.tbi` for `.gz` output, so check for
   that index before planning an `indexgvcf` run.
2. Collect required inputs:
   - Input GVCF path, and whether its directory is writable inside the container.
   - Where the index must end up. There is no documented output-path option; the
     index is written next to the input (see Output Naming below).
3. Ask whether the output will be consumed by `genotypegvcf` or another tool.

## Command Shape

```bash
pbrun indexgvcf \
  --input /results/<sample.g.vcf.gz>
```

The GVCF must be on a writable mount because the index is created beside it.
Verify exact flags against the selected version.

## Output Naming and Read-Only Inputs

- The index name is the input filename with `.tbi` appended for a `.gz` input
  (`sample.g.vcf.gz` → `sample.g.vcf.gz.tbi`), or `.idx` for an uncompressed
  `.g.vcf`. Do not use or invent an `--out` flag; none is documented for
  `indexgvcf`.
- If the original GVCF is on a read-only mount (for example `/data`), indexing it
  in place fails. Index a **writable copy** instead and leave the original
  untouched.
- Before copying, check for existing destination files so nothing is silently
  overwritten:

  ```bash
  [ ! -e /results/<sample.g.vcf.gz> ] && [ ! -e /results/<sample.g.vcf.gz>.tbi ] \
    || { echo "destination exists; choose another name or confirm overwrite"; exit 1; }
  cp -p /data/<sample.g.vcf.gz> /results/<sample.g.vcf.gz>
  ```

  For an uncompressed `.g.vcf`, check for `<sample.g.vcf>.idx` instead of the
  `.tbi`. Ask the user before replacing an existing GVCF or index, and make
  clear the commands are a plan until the user runs them.

## IndexFeatureFile Option Mapping

Use this mapping when translating a GATK `IndexFeatureFile` command for GVCF
indexing to `pbrun indexgvcf`.

| GATK option | `pbrun indexgvcf` equivalent | Notes |
| --- | --- | --- |
| `--input`, `-I` | `--input` | Required gVCF/gVCF.GZ input. |
| Implicit output index path | Implicit `.tbi` (`.gz` input) or `.idx` (`.g.vcf`) next to input | Not configurable; see Output Naming and Read-Only Inputs. |
| GATK output path override, cloud flags, `--arguments_file`, and Java options | No direct equivalent | Not exposed by current Parabricks docs for `indexgvcf`. |

## Validation

- Output index file is created.
- Index path is accessible to the downstream tool.

## Guardrails

- Index only GVCFs and ordinary `.vcf.gz` files; other inputs are not
  documented.
- Do not run genotyping here; route to `pbrun-genotypegvcf.md`.
- Do not invent output index suffixes; verify version behavior.
- Do not overwrite an existing GVCF or index (`.tbi`, or `.idx` for an
  uncompressed `.g.vcf`), and do not index a read-only original in place; use
  a writable copy after checking the destination.
