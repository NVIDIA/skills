# Parabricks prepon

Prepare an existing panel of normals (PON) for `mutectcaller --pon`.
This is a somatic-calling resource step; it does not need a pangenome graph
and does not create a PON from normal-sample reads.

## PON prerequisites

The input is a bgzip-compressed PON VCF with a matching tabix index. Its header
must give each contig's name and length. Use a reference `.fai` matching the
PON assembly if those header fields need repair. Preserve the original, write
a new VCF, and generate a fresh tabix index for that rewritten file before
running prepon. Do not reuse the original index for changed compressed content.

## Command and handoff

```bash
pbrun prepon --in-pon-file /workdir/<prepared-pon.vcf.gz>
```

Run in a writable working area for generated resources. `prepon`
exposes no `--out` option; retain the prepared PON and generated companion
resources for the caller. Supply that same PON path to `mutectcaller --pon`.
Then use postpon to annotate the resulting calls. Keep all three commands as
a plan when execution was not requested.

The only required tool input is `--in-pon-file`; runtime controls follow the
shared command contract. Check prepon completion and the caller's access to its
resources before calling. Do not infer graph or cohort inputs from the name.

Source: [NVIDIA prepon manual](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/prepon).
