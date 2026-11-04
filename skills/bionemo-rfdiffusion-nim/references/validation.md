# RFDiffusion Validation

## Response Checks

- `output_pdb` exists and is non-empty.
- Save the PDB artifact with a descriptive filename.
- Record `elapsed_ms`, contigs, hotspots, diffusion steps, and random seed.

## PDB Checks

- Output contains ATOM records.
- Chain IDs and residue numbering are plausible for the requested design mode.
- For motif scaffolding, confirm preserved residue references match the input.
- For binders, record target chain and hotspot residues.

## Scientific Checks

- Do not treat a generated backbone as a complete protein design; it still
  needs downstream sequence design and fold validation (see the handoff
  pipeline in `science.md`).
- Generate multiple candidates for serious design work.
