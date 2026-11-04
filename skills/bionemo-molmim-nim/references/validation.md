# MolMIM Validation Guide

## Request Checks

- `smi` or `sequences` contains valid SMILES.
- `algorithm` is `"CMA-ES"` or `"none"`.
- `property_name` is `"QED"` or `"plogP"`.
- `num_molecules`, `iterations`, `particles`, `min_similarity`, and
  `scaled_radius` are within documented ranges.
- Hosted requests use Bearer auth.
- Local requests do not include auth headers.
- Local Docker mounts `LOCAL_NIM_CACHE` to `/home/nvs/.cache/nim`.

## Response Checks

For `/generate`:

- Hosted responses may use `molecules`, a JSON-encoded list of `{sample, score}`
  objects; local responses may use `generated`. Extract SMILES from both shapes.
- Each generated SMILES parses with RDKit when available.
- Generated list is unique or duplicates are reported.
- For seed-constrained generation, compute similarity to the seed if RDKit is
  available.
- Save `molmim_response.json` and `molmim_generated.smi`.

For `/embedding`:

- `embeddings` exists.
- Embeddings are numeric arrays with finite values.
- Number of embeddings matches number of input `sequences`.

For `/hidden`:

- `hiddens` and `mask` exist.
- Numeric hidden values are finite.
- The mask shape is compatible with the hidden representation.

For `/decode`:

- `generated` exists.
- Decoded SMILES parse with RDKit when available.

## Suggested Live Validation

This portable skill collection does not include live hosted/local validation harnesses. Save request and response artifacts from your own NIM calls and apply the checks above.

## Scientific Caveats

Passing API validation is not medicinal-chemistry validation. For prioritized
lead lists, add RDKit property checks, diversity/novelty checks, PAINS or alert
filters, seed-similarity checks, and downstream docking/affinity assessment.
