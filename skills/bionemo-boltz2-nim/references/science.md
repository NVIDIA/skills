# Boltz2 Science Notes

Boltz2 predicts biomolecular complex structures and can also estimate ligand
binding affinity. It is most useful when the user has a defined molecular
assembly to model: a protein, a protein complex, a protein-ligand complex, or a
protein-nucleic-acid complex.

## Best-Fit Uses

- Predict a structure for a protein sequence when an experimental structure is
  unavailable.
- Model a protein-ligand complex from a protein sequence plus a SMILES string
  or CCD ligand code.
- Include DNA or RNA chains in the same structural prediction.
- Request affinity estimates for one ligand in a protein-ligand request.
- Use a precomputed protein MSA, for example from MSA-Search NIM, when the user
  already has one.

## Scientific Limits

- Boltz2 is a structure and affinity predictor, not a full docking search
  engine. If the user only wants pose ranking against an existing protein
  structure, DiffDock may be the better NIM.
- Affinity prediction supports one affinity ligand per request. Treat pIC50 and
  binding probability as model estimates that need orthogonal validation.
- The service returns structures and model confidence, not experimental proof of
  binding, function, or biological activity.
- Very short toy sequences are useful for API smoke tests, but usually not for
  meaningful structural biology.

## Handoffs

- MSA-Search can provide A3M alignments for protein polymers. Use the validated
  Boltz2 MSA shape with `alignment`, `format`, and `rank`.
- GenMol can generate candidate ligands, but Boltz2 inputs still require valid
  SMILES or CCD ligand identifiers.
- DiffDock can be useful when the user wants docking against a known protein
  structure rather than joint structure prediction from sequence.
