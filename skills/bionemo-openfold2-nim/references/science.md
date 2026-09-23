# OpenFold2 Science Notes

OpenFold2 is a PyTorch reimplementation of AlphaFold2-style monomer protein
structure prediction. It predicts a single protein chain from a sequence and
can use evolutionary information from A3M MSAs plus structural information from
mmCIF templates.

## Best Uses

- Monomer protein structure prediction from an amino-acid sequence.
- Structure prediction when the user already has a meaningful A3M alignment.
- Homology/template-informed modeling when trustworthy mmCIF templates are
  available.
- Fast comparison of selected AlphaFold2/OpenFold parameter sets.
- Downstream handoff into ProteinMPNN, DiffDock, or visualization tools after
  structure sanity checks.

## Not For

- Protein-ligand, protein-DNA, protein-RNA, or multi-chain complex prediction.
  Use OpenFold3 or Boltz2 for biomolecular complexes.
- Direct ligand docking or affinity ranking.
- Proteins whose biological state depends on disorder, conformational switching,
  membrane context, post-translational modification, or binding partners not
  represented in the monomer input.
- Claiming experimental validation from toy sequences or shallow alignments.

## MSA And Template Context

MSA depth and template relevance strongly affect scientific usefulness. A
single-sequence A3M can validate endpoint shape, but it is weak evidence for a
production-quality fold. Prefer MSA Search or another trusted homology search
to create A3M alignments, then preserve the query sequence, gap semantics, and
database identity when building the OpenFold2 payload.

Templates can improve predictions when the template is homologous and the
aligned region matches the target biology. They can also mislead the model when
the template is from the wrong conformational state, wrong domain, or wrong
oligomeric context. Current OpenFold2 docs favor explicit mmCIF templates.

## Interpreting Results

Save the full JSON response and all structure artifacts. Inspect:

- Whether structure text is parseable as PDB or mmCIF.
- Whether the residue count roughly matches the input sequence length.
- Whether confidence/ranking fields exist in the live response.
- Whether low-confidence/disordered regions are biologically expected.
- Whether selected models agree or diverge substantially.

The response includes one prediction per selected model parameter set, ordered
by confidence per NVIDIA build-page wording. Treat model disagreement as a
reason to inspect MSAs/templates and avoid overclaiming one structure.

## Useful Handoffs

- MSA Search -> OpenFold2: generate A3M alignments for better monomer inputs.
- OpenFold2 -> ProteinMPNN: design sequences for a validated monomer backbone.
- OpenFold2 -> DiffDock: dock small molecules into a predicted monomer after
  checking the structure is plausible for the binding site.
- OpenFold2 -> OpenFold3/Boltz2: switch when the scientific task involves
  complexes, ligands, nucleic acids, or multimeric assemblies.
