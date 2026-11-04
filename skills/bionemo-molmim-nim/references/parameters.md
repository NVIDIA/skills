# MolMIM Parameter Guide

## Generate Endpoint

- `smi`: seed SMILES. Use chemically valid, canonicalizable SMILES.
- `algorithm`: `"CMA-ES"` for guided optimization, `"none"` for unguided
  sampling around the seed.
- `num_molecules`: requested number of molecules, 1-100 for `/generate`.
- `property_name`: `"QED"` or `"plogP"`.
- `minimize`: `false` to maximize property, `true` to minimize.
- `min_similarity`: lower values allow more exploration away from the seed;
  higher values enforce more seed similarity.
- `particles`: CMA-ES population size; higher values may explore more but cost
  more runtime.
- `iterations`: CMA-ES optimization iterations; higher values may improve the
  property objective but cost more runtime and can over-optimize proxies.
- `scaled_radius`: latent sampling radius, 0-2. Larger values produce more
  diverse but potentially less seed-like molecules.

## Local Latent Endpoints

- `/embedding` maps SMILES to fixed-length embeddings for similarity or
  clustering.
- `/hidden` returns hidden states and `mask`; keep both together.
- `/decode` reconstructs/generates SMILES from `hiddens` and `mask`.
- `/sampling` uses `sequences`, `beam_size`, `num_molecules`, and
  `scaled_radius`; local docs cap `num_molecules` at 10 for sampling.

## Tuning Guidance

- For conservative analog generation: `algorithm="CMA-ES"`, high
  `min_similarity`, modest `iterations`, and small `num_molecules`.
- For broader exploration: `algorithm="none"` or local `/sampling` with larger
  `scaled_radius`.
- For hit-to-lead ideation: generate more molecules than needed, then filter
  with RDKit and downstream assays/models.
- For property optimization: report both the objective direction and the
  limitations of proxy properties.

## Common Mistakes

- Using `smiles` instead of `smi` for `/generate`.
- Treating hosted latent endpoints as available.
- Forgetting that local `/decode` requires both `hiddens` and `mask`.
- Optimizing plogP or QED without chemical sanity checks.
