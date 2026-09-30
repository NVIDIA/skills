# RFDiffusion Science Notes

RFDiffusion generates protein backbone structures from design constraints. It
is a backbone generator, not a sequence designer or final structure validator.

## Best-Fit Uses

- De novo backbone generation over a target length range.
- Motif scaffolding while preserving residues from an input structure.
- Binder backbone generation against a target and hotspot residues.
- Generating PDB backbones for downstream ProteinMPNN sequence design.

## Scientific Limits

- Outputs are backbone structures; they need sequence design before expression
  or fold validation.
- Contig syntax must match chain IDs and residue numbers in `input_pdb`.
- Fewer diffusion steps are faster but can reduce quality.
- Binder designs require downstream interface and fold validation.

## Handoffs

- ProteinMPNN designs sequences for generated backbones.
- OpenFold3 or Boltz2 can validate whether designed sequences recover the
  intended backbone or interface.
