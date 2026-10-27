# Parabricks postpon

Annotate Mutectcaller variants against an existing panel of normals (PON),
updating their INFO fields. This is the postprocessing step after a call using
`mutectcaller --pon`; it does not operate on pangenome graph intermediates.

## Annotation command

```bash
pbrun postpon \
  --in-vcf /workdir/<mutect-calls.vcf> \
  --in-pon-file /workdir/<prepared-pon.vcf.gz> \
  --out-vcf /outputdir/<annotated-calls.vcf>
```

All three paths are required. Use the same assembly-compatible PON as
in prepon and mutectcaller, including its matching tabix index. Preserve the
unannotated calls by selecting a distinct output path. Verify the annotated
VCF and intended INFO changes after an authorized run; an output directory
alone is insufficient. PON annotation does not establish somatic truth or
replace project-specific filtering.

Runtime controls follow the shared command contract. Read the prepon reference
for header repair and PON preparation; ordinary GATK engine flags do not have
assumed Parabricks equivalents.

Source: [NVIDIA postpon manual](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/postpon).
