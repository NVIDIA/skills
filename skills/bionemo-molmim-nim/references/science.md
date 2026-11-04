# MolMIM Science Notes

MolMIM is a probabilistic autoencoder for small molecules. It learns a
clustered latent space over SMILES strings, enabling embeddings, hidden-state
manipulation, decoding, sampling, and property-guided molecule generation.

## Best Uses

- Generate analogs around a seed molecule.
- Optimize generated molecules for QED or penalized LogP (`plogP`).
- Explore local chemical neighborhoods through latent-space sampling.
- Compute embeddings or hidden states for clustering, interpolation, or local
  optimization workflows.
- Feed generated SMILES into downstream filters, docking, affinity prediction,
  or medicinal-chemistry triage.

## Not For

- Proving biological activity by itself.
- Enforcing synthesis feasibility, toxicity, PAINS, selectivity, or binding.
- Preserving a scaffold exactly unless downstream checks confirm preservation.
- Direct docking or affinity scoring. Use DiffDock, Boltz2 affinity, or other
  scoring workflows after chemistry validation.

## Interpreting Generated Molecules

MolMIM returns candidate SMILES. Validate them before prioritizing:

- RDKit parseability and canonicalization.
- Uniqueness and duplicate removal.
- Similarity to the seed when `min_similarity` or `scaled_radius` is used.
- QED/plogP direction: `minimize=False` means maximize the property.
- Basic drug-likeness and property filters.

High property scores from a generative model are not proof of useful leads.
Use generated molecules as proposals for a larger screening funnel.

## Hosted Versus Local Scientific Surface

Hosted `/generate` is enough for simple seed-based generation and CMA-ES
optimization. Local Docker is needed for embeddings, hidden states, decoding,
sampling, interpolation, and custom guided optimization loops.

## Useful Handoffs

- MolMIM -> DiffDock: dock generated SMILES into a protein binding site.
- MolMIM -> Boltz2 affinity: evaluate protein-ligand complexes and pIC50-like
  affinity outputs where applicable.
- MolMIM -> medicinal chemistry filters: RDKit property filters, PAINS/alerts,
  novelty, diversity, and synthetic-feasibility screens.
- GenMol versus MolMIM: GenMol uses SAFE notation and fragment masks; MolMIM
  uses SMILES seeds and latent-space sampling/optimization.
