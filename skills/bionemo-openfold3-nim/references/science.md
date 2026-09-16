# OpenFold3 Science Reference

## Purpose

OpenFold3 predicts 3D structures for biomolecular assemblies containing
proteins, DNA, RNA, and small-molecule ligands. Use it when the primary task is
structure prediction or co-folding, not sequence search, de novo backbone
generation, inverse folding, or docking-only pose refinement.

## Strong Use Cases

- Protein monomer or multimer structure prediction from sequence.
- Protein-DNA and protein-RNA complex prediction when a plausible binding
  sequence is known.
- Protein-ligand co-folding when the ligand can be represented by SMILES or a
  CCD code.
- Template-guided predictions when relevant structure templates are available.
- Generating candidate structures for downstream visualization, review,
  docking, design, or manual scientific inspection.

## What The Output Means

OpenFold3 returns one or more structure samples plus confidence-like scores.
Treat the scores as model confidence and ranking aids, not experimental
measurements.

- `confidence_score`: overall ranking score for a sample.
- `complex_plddt_score`: local structure confidence; higher is generally better.
- `complex_pde_score`: predicted distance error; lower is generally better.
- `ptm_score`: predicted global fold quality.
- `iptm_score`: predicted interface quality; most useful for complexes.

## Limits And Gotchas

- Predictions can be wrong even when scores look good. Validate geometry and
  biological plausibility before using results downstream.
- Toy sequences and very short examples are good API smoke tests but rarely
  meaningful biology.
- MSAs can improve protein and RNA predictions when they contain real
  homologous context; a single-sequence `>query` MSA is only a minimal fallback.
- Ligand predictions depend on correct chemistry representation. Invalid
  SMILES, wrong protonation, ambiguous tautomers, or an inappropriate CCD code
  can make the structure scientifically misleading.
- OpenFold3 is not a docking search engine. For pose ranking or blind docking,
  DiffDock may be the better NIM.
- OpenFold3 is not an inverse-folding or protein-design model. For sequence
  design against a backbone, use ProteinMPNN; for de novo backbone generation,
  use RFDiffusion.

## Handoff Guidance

- Use MSA Search before OpenFold3 when the user needs evolutionary context for a
  protein sequence and does not already have an MSA.
- Use OpenFold3 output structures as inputs to visualization, docking,
  structure-quality checks, or design review workflows.
- Keep all generated structures, request payloads, and score summaries together
  so downstream interpretation is reproducible.
