---
name: parabricks
description: >-
  Plan NVIDIA Parabricks pbrun commands, translate CPU options and check runtime
  readiness. Use for individual tools, including planning-only questions.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "Angel Pizarro <apizarro@nvidia.com>"
  version: "1.3.0"
  tags:
    - parabricks
    - genomics
    - nvidia
---

# Parabricks

## Purpose

Use this skill to discover the right NVIDIA Parabricks `pbrun` command, assess
runtime readiness, and generate version-aware command guidance for individual
tools.

Do **not** use this skill for whole-workflow inspection, acceleration planning,
or wiring optional GPU branches. For pipeline-level work, use
`genomics-workflow-acceleration`.

## When to Use This Skill

- Which `pbrun` tool fits the user's data and goal
- GPU, driver, Docker, container, storage, or installation readiness
- Command shape, flags, and validation for a specific Parabricks tool
- Reference/index compatibility and CPU-to-Parabricks option translation
- Troubleshooting a single Parabricks command or tool family
- Planning-only or "do not run" questions about a named `pbrun` tool (for
  example reference/index compatibility, output naming, or flag validity). Use
  this skill's references to answer; do not skip it because nothing will be
  executed.

## Prerequisites

- **Command guidance:** No Parabricks/NGC credential or local GPU/container
  runtime is required to read references or generate commands.
- **Readiness helper:** Requires Bash and GNU coreutils `timeout` (or `gtimeout`)
  on the host being inspected. Missing runtime components are reported as
  readiness gaps; without a timeout utility it exits before running probes.
- **STAR translation helper:** Requires Python 3.7 or newer, using only the
  standard library. It does not require a GPU or a Parabricks installation.
- **Execution:** The documented Docker commands require a supported Linux host,
  an NVIDIA GPU with sufficient memory, a compatible NVIDIA driver, Docker,
  NVIDIA Container Toolkit, and access to the selected Parabricks image.
  Pulling the image may require an NGC API key for authentication to `nvcr.io`.
  See [runtime-environment.md](references/runtime-environment.md) for
  version-specific requirements and checks.

Tool scope: read this skill's references and user-supplied manifests; write only
requested command artifacts. Use the bundled helpers for translation or an
authorized host inspection. Documentation lookups use `docs.nvidia.com`; image
access uses the selected registry only for authorized container execution.
These instructions do not grant tool permissions; the host's permission policy
still applies to shell, file, network and optional documentation MCP access.

## Inputs

Use the user's explicit arguments and supplied manifests as the source of
truth; use prior context only where it does not conflict. Read a named manifest
before translating it. For command guidance, identify the assay, input paths,
reference/index provenance, outputs and target release. Ask only for missing
facts that change the answer; label unresolved template values as placeholders.
Required inputs are the selected tool's read/reference and output arguments.
Optional inputs include intervals, resource bundles, logs and tuning parameters
when supported by that tool; do not invent them to fill a template.
Runtime-only questions do not need sample metadata. Supplied deployment paths
are not permission to probe local data or hardware.

If the user is unsure which tool applies, scan the **Tool Reference Index**
below, then load the matching `references/pbrun-<tool>.md` file.


## Limitations

This skill routes and guides Parabricks commands. It does not install
Parabricks, infer missing sample metadata, guarantee output parity, provide
clinical interpretation, or promise exact runtime without benchmark data.

## Instructions

1. Identify the requested release. The option references cover v4.7.1; examples
   from another release are labeled. A software version and image tag can
   differ: NVIDIA's 4.7.1 image is tagged `4.7.1-1`. Preserve the user's verified
   image selection; use [command-conventions.md](references/command-conventions.md)
   for the shared launch and validation contract.
2. Classify the request:
   - **Runtime** → [runtime-environment.md](references/runtime-environment.md)
   - **Tool discovery** → **Tool Reference Index** (below)
   - **Specific command** → matching `references/pbrun-<tool>.md`. Read it
     **before** any web search or MCP lookup; use those only to verify
     version-sensitive details the reference does not settle.
3. Combine the tool-specific mapping with [shared-options.md](references/shared-options.md)
   rows whose **Tools** column names the selected command. For STAR argument
   arrays or multi-lane batches,
   use the local [translation helper](references/rna-translation.md); it preserves
   tokens and reports unmapped semantics without executing a workload.
4. Collect missing biological and filesystem context before generating commands.
5. Return the requested command or artifact with unresolved assumptions. For
   planning-only work, check command text against the references and supplied
   facts. Filesystem, container and output checks apply only to an authorized
   execution. Do not turn command preparation into a readiness inspection.

## Available Scripts

Run these from the skill directory containing this file; paths are relative to
that directory. See the helper prerequisites above.

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/check_parabricks_runtime.sh` | Inspect the execution host; report gaps without installing anything | `--path DIR` (repeatable), `--format text\|json`, `--timeout SECONDS`; container probes require `--run-container-check` or `--run-parabricks-check` with `--parabricks-version TAG` / `--parabricks-image IMAGE` |
| `scripts/translate_star_to_parabricks.py` | Convert explicit STAR argument arrays to reviewed command drafts | `--input FILE` or `--input -`; [request and output schema](references/rna-translation.md) |

## Output Format

For explanations, give the decision, complete command, and remaining checks.
For file requests, write the specified path and schema, then read back and
validate the artifact against the input manifest. Report all job IDs, omitted
options and unresolved output contracts. Do not replace supplied values with
illustrative examples. The runtime helper's JSON has `schema_version`, `checks`
and readiness results; the translator emits `status`, `draft_argv`, `command`,
`notes`, `issues` and `untranslated` for each request.

## Tool Reference Index

If the command is named, open `references/pbrun-<command>.md` directly.
For tool selection and official manual links, use [tool-index.md](references/tool-index.md).
Common routes:

| Request | Reference |
| --- | --- |
| DNA FASTQ alignment / RNA STAR translation | [fq2bam](references/pbrun-fq2bam.md) / [rna_fq2bam](references/pbrun-rna_fq2bam.md) |
| Existing BAM germline calls / tumor-normal calls | [haplotypecaller](references/pbrun-haplotypecaller.md) / [mutectcaller](references/pbrun-mutectcaller.md) |
| Giraffe graph alignment | [giraffe](references/pbrun-giraffe.md) |
| STAR-Fusion migration | [starfusion](references/pbrun-starfusion.md) |
| Recalibration / apply a recalibration report | [bqsr](references/pbrun-bqsr.md) / [applybqsr](references/pbrun-applybqsr.md) |
| BAM metrics / multiple metric families | [bammetrics](references/pbrun-bammetrics.md) / [collectmultiplemetrics](references/pbrun-collectmultiplemetrics.md) |
| GVCF indexing / joint genotyping | [indexgvcf](references/pbrun-indexgvcf.md) / [genotypegvcf](references/pbrun-genotypegvcf.md) |
| Panel-of-normals preparation / annotation | [prepon](references/pbrun-prepon.md) / [postpon](references/pbrun-postpon.md) |
| BWA performance options | [performance](references/performance.md) |

## Examples

### Runtime Readiness

For GPU, driver, Docker, container, storage, or installation questions, read
[runtime-environment.md](references/runtime-environment.md) and prefer:

```bash
bash scripts/check_parabricks_runtime.sh --format json
```

Add `--path <dir>` for known input/output/tmp paths. Run container probes only
with user consent.

### STAR Translation

```bash
python3 scripts/translate_star_to_parabricks.py --input requests.json
```

Prepare the request using the [translation schema](references/rna-translation.md).
Review `issues` and `untranslated` before using the draft. Tool references show
the `pbrun` portion; the shared container envelope is documented once in
[command-conventions.md](references/command-conventions.md).

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| Multiple plausible tools | Data type or goal underspecified | Ask for assay, inputs, caller preference, desired output; use Tool Reference Index |
| Exact flag requested | Options are version-sensitive | Check the selected tool reference, NVIDIA docs, or `pbrun <tool> --help` |
| Runtime question | GPU, Docker, drivers, or storage | Use runtime-environment reference and diagnostic script |
| Wrong tool family | Assay or input type unclear | Confirm DNA/RNA/methylation/long-read/pangenome before routing |
| CUDA or memory failure | Runtime not ready or GPU memory constrained | Assess runtime before tuning command flags |
| Documentation MCP unavailable | Connection, timeout or server error | Use the bundled release reference first; for an unresolved fact, fall back to the official versioned manual and mark any remaining uncertainty |

## Guardrails

- Treat command availability and options as version-sensitive.
- Do not infer exact flags from command names alone.
- Do not collapse standalone tools and full pipelines when explaining tradeoffs.
- Do not substitute DNA `fq2bam` for RNA, or germline for somatic callers.
- Do not invent sample names, read groups, reference builds, known-sites files,
  model files, graph resources, container tags, or output paths.
- Do not install, upgrade, or modify packages. Label setup commands as user-run.
- Do not claim CPU execution of Parabricks tools.
- Do not claim biological or VCF parity without a comparison run.
- Use the optional documentation MCP server only for details the bundled
  references do not settle. If its connection fails, avoid repeated discovery
  or lookup loops; use the official manual. Never claim a failed lookup verified
  an option. Documentation access does not require genomic data or credentials.

## Key References

- Parabricks documentation MCP server:
  <https://docs.nvidia.com/clara/parabricks/_mcp/server>
- Parabricks tool index:
  <https://docs.nvidia.com/clara/parabricks/tool-reference>
- Output accuracy and compatible CPU software versions:
  <https://docs.nvidia.com/clara/parabricks/about-parabricks/software-overview/output-accuracy-and-compatible-cpu-software-versions>
- Getting started:
  <https://docs.nvidia.com/clara/parabricks/get-started>
- Overview:
  <https://docs.nvidia.com/clara/parabricks/about-parabricks>
