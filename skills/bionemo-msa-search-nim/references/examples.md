# MSA-Search Examples

Use these compact patterns when the activated skill needs more examples.

## Fast Local Deployment (recommended)

When deploying the MSA-Search NIM locally for a paired/complex workflow, prefer the fast
path: download only the UniRef30 database in parallel, then launch against it. A plain
`docker run` uses the NIM's built-in downloader (>80 min for UniRef30); the parallel path
is ~14 min. Example prompt:

> Deploy the MSA-Search NIM locally using the fast path: download only the UniRef30
> database (`databases:uniref30` profile) in parallel with aria2c, then start the NIM
> against those files with `NIM_MODEL_NAME`. Look up the current profile hash with
> `list-model-profiles` rather than hardcoding it. If the endpoint is already running with
> the database loaded, skip the redeploy. Then run a paired MSA search for chains A and B
> using `Uniref30_2302` only.

See the "Recommended For Large Profiles: Parallel Download" section of `SKILL.md` for the
exact aria2c + `NIM_MODEL_NAME` commands.

## Hosted Standard MSA

Run `scripts/hosted_search.py` from the skill root to save the response and A3M
files. The request payload is:

```python
payload = {
    "sequence": sequence,
    "databases": ["Uniref30_2302", "colabfold_envdb_202108"],
    "e_value": 0.0001,
    "output_alignment_formats": ["a3m"],
}
```

## Hosted Paired MSA

```python
url = "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/paired/predict"
payload = {
    "sequences": [chain_a, chain_b],
    "e_value": 0.0001,
    "max_msa_sequences": 500,
}
```

## Local Template Search

```python
url = "http://localhost:8000/biology/colabfold/msa-search/structure-templates/predict"
payload = {
    "sequence": sequence,
    "structural_template_databases": ["pdb70_220313"],
    "max_structures": 20,
    "max_msa_sequences": 500,
}
```

## Save Standard Alignments

```python
for db_name, formats in result["alignments"].items():
    for fmt_name, data in formats.items():
        path = f"msa_{db_name}.{fmt_name}"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(data["alignment"])
```
