# Parabricks starfusion

## First Steps

1. Confirm the user has STAR `Chimeric.out.junction` input. If the user only
   has raw RNA FASTQs, route to `pbrun-rna_fq2bam.md` or ask how they will
   produce STAR-Fusion-compatible chimeric junction input.
2. Collect required inputs:
   - Chimeric junction input path.
   - STAR-Fusion genome resource library directory.
   - Output directory.
3. Ask about optional version-specific filters, logs, and runtime settings only
   when relevant.

## Command Shape

Fusion detection from chimeric junction input:

```bash
pbrun starfusion \
  --chimeric-junction /workdir/<chimeric_junction.out> \
  --genome-lib-dir /workdir/<genome_resource_library>/ \
  --output-dir /outputdir/<starfusion_output>/
```

Verify exact input, genome resource, output, filter, and log flags against the
selected version.

## STAR-Fusion Option Mapping

Use this mapping when translating an upstream `STAR-Fusion` command to
`pbrun starfusion`. Parabricks exposes a focused subset of the
STAR-Fusion CLI: the chimeric junction input, CTAT genome resource library,
output directory, output prefix, and worker thread count. Many upstream
STAR-Fusion filtering, STAR-alignment, FusionInspector, shared-memory, and
single-cell options are not exposed by `pbrun starfusion`; if the selected
Parabricks version does not document a flag, do not pass it through.

If a requested upstream filter or FusionInspector mode is mandatory, explain
that the supported `pbrun starfusion` command is only a partial migration.
Retain the upstream stage, or plan a separately validated downstream stage;
do not claim that dropping an unsupported option preserves the analysis.
Upstream FusionInspector also needs the read inputs, not just the junction
file. The CTAT/STAR-Fusion resource library is distinct from a plain STAR
alignment index, even when both come from the same reference build.

| STAR-Fusion option | `pbrun starfusion` equivalent | Notes |
| --- | --- | --- |
| `--chimeric_junction`, `-J` | `--chimeric-junction` | Required path to `Chimeric.out.junction` from STAR. |
| `--genome_lib_dir` | `--genome-lib-dir` | Required CTAT/STAR-Fusion genome resource library directory. |
| `--output_dir`, `-O` | `--output-dir` | Required output directory. |
| `--out_prefix` | `--out-prefix` | Same output-prefix role. |
| `--CPU` | `--num-threads` | Same worker-thread role; Parabricks default is version-specific. |
| `--tmpdir` | `--tmp-dir` | Same temporary-directory role, but Parabricks treats it as a wrapper/runtime path. |
| `--version` | `--version` | Same version-reporting role. |
| `--verbose_level` | `--verbose` | Partial equivalent only: Parabricks exposes a boolean verbose flag, not STAR-Fusion's numeric verbosity level. |
| `--left_fq` | No direct equivalent | `pbrun starfusion` starts from a STAR chimeric junction file, not raw FASTQ. Use `pbrun-rna_fq2bam.md` or STAR first to produce compatible junction input. |
| `--right_fq` | No direct equivalent | No raw paired FASTQ mode in current Parabricks `starfusion` docs. |
| `--samples_file` | No direct equivalent | No upstream SmartSeq2/multi-sample file mode is documented for `pbrun starfusion`. |
| `--min_junction_reads` | No direct equivalent | Upstream fusion filtering option not exposed by current Parabricks docs. |
| `--min_sum_frags` | No direct equivalent | Upstream fusion filtering option not exposed by current Parabricks docs. |
| `--max_promiscuity` | No direct equivalent | Upstream fusion filtering option not exposed by current Parabricks docs. |
| `--min_pct_dom_promiscuity` | No direct equivalent | Upstream promiscuity-filter option not exposed by current Parabricks docs. |
| `--min_spanning_frags_only` | No direct equivalent | Upstream fusion-support filter not exposed by current Parabricks docs. |
| `--min_pct_MM_nonspecific`, `-M` | No direct equivalent | Upstream multimapping specificity filter not exposed by current Parabricks docs. |
| `--require_LDAS` | No direct equivalent | Upstream long double-anchor support filter not exposed by current Parabricks docs. |
| `--min_novel_junction_support` | No direct equivalent | Upstream novel-junction support filter not exposed by current Parabricks docs. |
| `--min_alt_pct_junction` | No direct equivalent | Upstream alternate-junction support filter not exposed by current Parabricks docs. |
| `--aggregate_novel_junction_dist` | No direct equivalent | Upstream novel-junction aggregation option not exposed by current Parabricks docs. |
| `--min_FFPM` | No direct equivalent | Upstream FFPM threshold not exposed by current Parabricks docs. |
| `--no_filter` | No direct equivalent | Upstream all-filter bypass not exposed by current Parabricks docs. |
| `--no_annotation_filter` | No direct equivalent | Upstream annotation-filter bypass not exposed by current Parabricks docs. |
| `--no_RT_artifact_filter` | No direct equivalent | Upstream RT-artifact filter bypass not exposed by current Parabricks docs. |
| `--no_single_fusion_per_breakpoint` | No direct equivalent | Upstream breakpoint-filter bypass not exposed by current Parabricks docs. |
| `--max_sensitivity` | No direct equivalent | Upstream preset that changes multiple filters; not exposed by current Parabricks docs. |
| `--full_Monty` | No direct equivalent | Upstream high-sensitivity/filter-relaxing preset not exposed by current Parabricks docs. |
| `--skip_EM` | No direct equivalent | Upstream EM assignment skip option not exposed by current Parabricks docs. |
| `--skip_FFPM` | No direct equivalent | Upstream FFPM skip option not exposed by current Parabricks docs. |
| `--examine_coding_effect` | No direct equivalent | Upstream downstream coding-effect analysis not exposed by current Parabricks docs. |
| `--extract_fusion_reads` | No direct equivalent | Upstream extraction of supporting fusion reads not exposed by current Parabricks docs. |
| `--FusionInspector` | No direct equivalent | Upstream FusionInspector integration not exposed by current Parabricks docs. |
| `--denovo_reconstruct` | No direct equivalent | Upstream Trinity reconstruction mode not exposed by current Parabricks docs. |
| `--misc_FI_opts` | No direct equivalent | Upstream FusionInspector pass-through option not exposed by current Parabricks docs. |
| `--run_STAR_only` | No direct equivalent | `pbrun starfusion` does not run STAR alignment; it consumes STAR chimeric junction output. |
| `--STAR_PATH` | No direct equivalent | Parabricks does not expose a STAR executable path for `pbrun starfusion`. |
| `--STAR_twopass` | No direct equivalent | STAR-alignment option not exposed because this Parabricks command consumes existing junctions. |
| `--STAR_max_mate_dist` | No direct equivalent | STAR-alignment option not exposed by `pbrun starfusion`. |
| `--STAR_SJDBoverhangMin` | No direct equivalent | STAR-alignment option not exposed by `pbrun starfusion`. |
| `--STAR_SortedByCoordinate` | No direct equivalent | STAR-alignment output option not exposed by `pbrun starfusion`. |
| `--STAR_limitBAMsortRAM` | No direct equivalent | STAR sorting option not exposed by `pbrun starfusion`. |
| `--STAR_peOverlapNbasesMin` | No direct equivalent | STAR paired-end overlap option not exposed by `pbrun starfusion`. |
| `--STAR_peOverlapMMp` | No direct equivalent | STAR paired-end overlap option not exposed by `pbrun starfusion`. |
| `--STAR_chimMultimapScoreRange` | No direct equivalent | STAR chimeric-multimap option not exposed by `pbrun starfusion`. |
| `--STAR_chimMultimapNmax` | No direct equivalent | STAR chimeric-multimap option not exposed by `pbrun starfusion`. |
| `--STAR_chimNonchimScoreDropMin` | No direct equivalent | STAR chimeric scoring option not exposed by `pbrun starfusion`. |
| `--STAR_outSAMattrRGline` | No direct equivalent | STAR read-group pass-through not exposed by `pbrun starfusion`. |
| `--STAR_use_shared_memory` | No direct equivalent | Upstream STAR shared-memory mode not exposed by `pbrun starfusion`. |
| `--STAR_LoadAndExit` | No direct equivalent | Upstream STAR shared-memory load mode not exposed by `pbrun starfusion`. |
| `--STAR_Remove` | No direct equivalent | Upstream STAR shared-memory cleanup mode not exposed by `pbrun starfusion`. |
| `--outTmpDir` | No direct equivalent | Upstream STAR temporary directory pass-through, distinct from Parabricks `--tmp-dir`. |
| `--DEVEL_STAR` | No direct equivalent | Upstream development option not exposed by `pbrun starfusion`. |
| `--show_full_usage_info` | No direct equivalent | Upstream help/usage option not exposed by `pbrun starfusion`. |
| `--help`, `-h` | No direct equivalent | Use `pbrun starfusion --help` or the selected Parabricks tool reference instead. |

## starfusion Options Without STAR-Fusion Equivalents

These options are Parabricks runtime, container, logging, or filesystem wrapper
options and are not STAR-Fusion CLI options already covered in the mapping
above.

| `pbrun starfusion` option | Why it has no STAR-Fusion equivalent |
| --- | --- |
| `--logfile` | Parabricks wrapper log file path; upstream STAR-Fusion writes to standard output/error and its output directory. |
| `--keep-tmp` | Parabricks temporary-file retention. Upstream STAR-Fusion has `--tmpdir`, but not this wrapper cleanup flag. |

## Validation

- Input format is compatible with STAR-Fusion expectations.
- Output directory contains expected STAR-Fusion result files.

## Guardrails

- Do not treat `starfusion` as a raw FASTQ aligner.
- Do not invent genome resource library paths or compatibility.
- Do not over-interpret fusion calls without project-specific filtering and
  validation criteria.

## Key References

- <https://raw.githubusercontent.com/STAR-Fusion/STAR-Fusion/master/STAR-Fusion>
