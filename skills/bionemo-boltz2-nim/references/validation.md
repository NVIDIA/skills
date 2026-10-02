# Boltz2 Validation

Validate both the API mechanics and the scientific shape of the result before
calling a run successful.

## Response Checks

- `structures` exists and has one entry per requested `diffusion_samples`.
- Each structure has non-empty `structure` text and `format` equal to `mmcif`.
- `confidence_scores` exists and has one score per returned structure.
- If affinity was requested, `affinities` is present and keyed by ligand ID.
- Affinity entries include `affinity_pic50`, `affinity_pred_value`, and
  `affinity_probability_binary`, usually as lists.

## Artifact Checks

- Save every returned structure as `.cif`.
- Keep the raw JSON response next to generated structures when possible.
- Name files with the chain or sample number when multiple samples are returned.
- Do not overwrite previous samples in loops.

## Scientific Sanity Checks

- Confirm the response used the intended molecule types and sequences.
- Inspect confidence scores; low confidence should be called out, not hidden.
- Treat affinity predictions as ranking or triage signals, not measured IC50.
- For ligand requests, confirm the ligand identifier in the affinity map matches
  the requested ligand.
- For MSA-assisted requests, verify the MSA begins with a FASTA header and uses
  the same protein sequence as the polymer.
