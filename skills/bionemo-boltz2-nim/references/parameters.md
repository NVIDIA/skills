# Boltz2 Parameter Guidance

Use conservative defaults for examples unless the user asks to tune runtime or
sampling depth.

## Core Runtime Parameters

- `recycling_steps`: default `3`. More recycles may improve consistency but add
  runtime.
- `sampling_steps`: default `50`. Higher values increase diffusion work and may
  improve difficult cases; lower values are faster for smoke tests.
- `diffusion_samples`: default `1`. Increase when the user needs alternative
  structures or uncertainty across samples.
- `step_scale`: default `1.638`. Treat as an advanced sampling parameter; do
  not tune casually in basic examples.
- `output_format`: use `"mmcif"`. Save returned structures as `.cif`.

## Affinity Parameters

- Set `predict_affinity: true` on exactly one ligand.
- `sampling_steps_affinity`: default `200`; reduce for quick examples only when
  runtime matters more than affinity quality.
- `diffusion_samples_affinity`: default `5`; use `1` for fast smoke tests.
- Report `affinity_pic50`, `affinity_pred_value`, and
  `affinity_probability_binary`; affinity fields are lists.

## Input Parameters

- `polymers[].molecule_type` is `protein`, `dna`, or `rna`.
- Protein, DNA, and RNA sequences should contain valid residue/base characters
  for their molecule type.
- Ligands use exactly one of `smiles` or `ccd`.
- Pocket constraints can guide a ligand toward known residues, but they do not
  replace experimental evidence of a binding site.

## MSA Shape

For protein MSAs, use the validated nested record:

```json
{
  "msa_search": {
    "a3m": {
      "alignment": ">query\nSEQUENCE",
      "format": "a3m",
      "rank": 0
    }
  }
}
```

Use `alignment`, not the stale `data` field.
