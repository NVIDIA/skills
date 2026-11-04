# DiffDock Science Notes

DiffDock predicts protein-ligand binding poses with a diffusion model. It is a
blind docking tool: it can search the receptor surface without a user-specified
binding pocket, but its confidence scores are pose-ranking signals, not binding
free energies.

## Best-Fit Uses

- Rapidly generate candidate binding poses for one ligand against one receptor.
- Rank multiple poses for visualization and downstream structural review.
- Screen small batches of ligands when the receptor is already prepared.
- Provide pose hypotheses for follow-up physics, medicinal chemistry, or assay
  planning.

## Scientific Limits

- Confidence scores are useful for ranking poses within a response, but they are
  not calibrated affinities and should not be reported as pIC50, Kd, or Delta G.
- DiffDock does not replace receptor preparation. Protonation, missing residues,
  cofactors, waters, alternate conformations, and biological assembly choices can
  change docking outcomes.
- The NIM expects receptor ATOM records. HETATM records, ligands, waters, and
  headers should be stripped unless they are intentionally encoded in the receptor
  representation accepted by the service.
- SMILES inputs rely on conformer generation. For controlled 3D ligand geometry,
  use SDF or MOL2 and review `skip_gen_conformer` behavior.
- Pose plausibility still needs visual inspection for clashes, unrealistic
  orientation, missing pocket interactions, and chemically impossible contacts.

## Handoffs

- GenMol can propose ligands before DiffDock pose generation.
- OpenFold3 or Boltz2 can help model or validate receptor structures before
  docking, but experimentally determined receptor structures are preferred when
  available.
- Medicinal chemistry review should follow docking to assess synthetic,
  property, and SAR plausibility.
- For virtual screening, combine pose confidence with orthogonal filters rather
  than treating the score as a final ranking alone.
