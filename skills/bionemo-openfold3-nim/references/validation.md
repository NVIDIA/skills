# OpenFold3 Validation Reference

## Request Checks

- Hosted requests must include `Authorization: Bearer $NGC_API_KEY`.
- Local prediction requests must not include hosted auth after the container is
  healthy.
- Hosted prediction URL includes `/v1/`; local prediction URL does not.
- Top-level payload contains `inputs` with exactly one item.
- `molecules` contains 1-32 objects.
- Each molecule has a valid `type`: `protein`, `dna`, `rna`, or `ligand`.
- Protein/RNA sequences use valid biological alphabets for the intended task.
- DNA sequences use ATCG; RNA sequences use AUCG.
- Ligand objects use exactly one of `smiles` or `ccd_codes`.
- MSA `alignment` strings start with a FASTA header such as `>query\n`.
- `diffusion_samples` is an integer from 1 to 5.

## Response Checks

- Response JSON contains `outputs`.
- First output contains `structures_with_scores`.
- Each sample contains `structure`, `format`, and score fields when returned.
- `format` matches the requested `output_format` when requested.
- Structure text is non-empty and starts like the selected format:
  - PDB commonly includes records such as `ATOM`, `HETATM`, `MODEL`, or `HEADER`.
  - CIF commonly includes `data_` blocks and mmCIF fields.
- Score values should be finite numbers. Do not assume all scores are bounded
  identically across all fields; document outliers instead of hiding them.

## Artifact Checks

Save these together:

- Request payload JSON.
- Response JSON.
- One `.pdb` or `.cif` file per returned structure sample.
- Score summary table.
- Environment metadata: hosted/local mode, endpoint, image tag for local runs,
  and timestamp.

## Scientific Sanity Checks

- Confirm the returned chain/entity count matches the intended assembly.
- Inspect ligand presence and rough placement when ligand co-folding is used.
- Inspect interfaces for implausible chain separation, severe clashes, or
  obvious missing components.
- Treat low confidence, low interface confidence, or high predicted distance
  error as a warning that downstream conclusions need extra review.
- For toy eval inputs, label the output as an API/format smoke test rather than
  a scientifically useful prediction.

## Blocking Conditions

Treat these as blocking for "validated" claims:

- Hosted API returns auth or validation errors.
- Local container is not healthy.
- Response lacks a parseable structure artifact.
- Returned arrays/files are empty.
- Required artifacts are not saved.
- Scientific validation reports `FAIL` without an explicit residual-risk note.
