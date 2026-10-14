# Command conventions

Read this reference once when preparing or reviewing a Parabricks command.
Individual tool references contain biological constraints, concrete examples,
and unique options. Shared option mappings are linked from SKILL.md; filter
their Tools column for the selected command.

## Container image and mounts

Tool examples give the command inside the container. Add one Docker envelope
only when requested; preserve a requested bare `pbrun` command or argument array.
Use an exact NVIDIA release tag (4.7.1 uses `4.7.1-1`), never `latest` or an
implicit tag. For reproducible execution, resolve that release to its trusted
NGC manifest digest and record the digest with the run. An image's local
`RepoDigests` after an authorized pull can identify the downloaded manifest;
do not invent a digest or silently switch releases.

This is a template: replace the digest and paths with verified values before
execution. Image pulls and container launches require the user's execution
authorization. Mount only the required inputs, outputs and scratch directories:

```bash
docker run --rm --gpus all \
  --volume /host/input:/workdir:ro \
  --volume /host/output:/outputdir \
  --workdir /outputdir \
  nvcr.io/nvidia/clara/clara-parabricks@sha256:<verified-digest> \
  pbrun <selected-tool> <tool-specific-options>
```

If a tool writes beside an input, use a writable working copy rather than making
the original input directory writable. Keep sample data and registry credentials
out of documentation lookups; request only public command/version information.

## Preparing inputs

Confirm the target release, execution host or launcher, assay, sequencing
technology, reference build, sample relationships, and desired outputs. Use
explicit host mounts and container paths for inputs, indexes, output, and
scratch space. Collect optional intervals, logs, worker counts, and metrics
only when relevant. Leave unknown values as visible placeholders; obtain sample
names, read groups, reference resources, and model or graph bundles from the user.

For runtime or installation questions, use the runtime reference linked from
SKILL.md. Probe the intended execution
host; local hardware does not describe a remote target. Follow the skill's probe
consent rules. Preparing a workload command does not authorize executing it.

## Translating options

The mapping tables primarily describe Parabricks 4.7.1. Check each selected
release's NVIDIA tool manual before finalizing syntax, defaults, optional flags,
input modes, or output naming. An upstream option without a documented mapping
has no assumed Parabricks equivalent; do not pass it through or borrow an option
from another `pbrun` tool. Flag availability does not establish biological parity.
In option tables, **—** in the baseline column denotes a Parabricks-only control.
If an upstream option is absent from both the tool-specific table and its
scoped rows in `shared-options.md`, treat it as unmapped until the selected
release's official tool reference confirms support. Keep it visible for review.

Preserve an upstream option's value only when its meaning, units, and scope
match the Parabricks option. For partial mappings, omit the setting from the
base command or present it as a separate tuning choice with its own rationale.
An explanatory caveat does not make a copied value equivalent; check that the
generated command follows the mapping explanation.

GATK and Picard wrapper translations shared by `applybqsr`, `bam2fq`,
`bammetrics`, `bamsort`, `bqsr`, `collectmultiplemetrics`, `fq2bam`,
`fq2bam_meth`, `genotypegvcf`, `giraffe`, `haplotypecaller`, `indexgvcf`,
`markdup`, `minimap2`, and `mutectcaller` are:

| Baseline option | Parabricks option | Interpretation |
| --- | --- | --- |
| `--TMP_DIR` or `--tmp-dir` | `--tmp-dir` | Wrapper scratch directory; mount it explicitly. This translation also applies to `germline`. |
| `--VERBOSITY` or `--verbosity` | `--verbose` | Partial mapping: a boolean, not the upstream log-level enum. |
| `--version` | `--version` | Report compatible software versions. |

The RNA references retain their STAR-specific logging and scratch-directory
comparisons. For GPU, threading, memory, and algorithm controls, combine the
selected tool's table with its entries in the shared mappings. Respect the
listed tool scope because meanings and availability vary by command.

For BWA stream defaults on `fq2bam` and `fq2bam_meth`, use the performance
reference linked from SKILL.md. Giraffe `--nstreams` tuning stays in
`pbrun-giraffe.md`.

## Shared wrapper controls

These controls recur across the tool references. Include them only for an
identified need, after checking support in the chosen tool and release.

| Option | Role |
| --- | --- |
| `--logfile` | Wrapper log destination. |
| `--x3` | Display the full underlying command-line arguments. |
| `--with-petagene-dir` | PetaGene integration directory. |
| `--keep-tmp` | Retain temporary files. |
| `--no-seccomp-override` | Change the wrapper's Docker seccomp behavior; it is not a routine performance flag. |
| `--preserve-file-symlinks` | Control wrapper handling of symlinked paths. |

## Validating a run

Before presenting a command, check its required arguments as a complete set;
translating the user's upstream flags alone may omit required wrapper inputs
or outputs. For example, `rna_fq2bam` needs both `--output-dir` for generated
STAR outputs and `--out-bam` for the final BAM. A Docker output mount does not
supply either argument. Retain explicit placeholders when a required path is
unknown. Check this from the command text without launching a container.

Before execution, check that every input and required index resolves through
the mounts and that reference, known-sites, model, and graph resources are
compatible with the assay and each other. CRAM decoding needs the appropriate
reference. Confirm writable output and scratch directories with adequate space.

Treat inputs as read-only and never overwrite existing data. When a tool writes
beside its input (for example `indexgvcf` writes the `.tbi` next to the GVCF) and
the input mount is read-only, plan a copy to a writable mount and index the
copy. Check that destination files do not already exist before copying, ask
before replacing anything, and keep commands labeled as a plan until the user
runs them.

After an authorized run, verify completion and the requested output artifacts,
including optional metrics, recalibration reports, and indexes. An existing
directory alone is not evidence of success. Review logs for malformed inputs,
reference or resource mismatches, missing indexes, container mount failures,
CUDA errors, and exhausted memory or storage. Apply the selected tool's
additional biological checks, and report unverified checks as pending.

Do not claim exact performance, caller accuracy, or CPU/GPU output equivalence
without a comparable benchmark or validation run.
