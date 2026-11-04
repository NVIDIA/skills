# GenMol Science Notes

GenMol generates molecules in SAFE notation and returns standard SMILES with a
score. It is a molecular design NIM, not a structure predictor or experimental
activity assay.

## Best-Fit Uses

- De novo generation from a masked SAFE length token such as `[*{20-30}]`.
- Scaffold decoration by appending a masked fragment to a known scaffold.
- Motif extension by placing masked fragments around a core motif.
- Lead optimization by encoding a hit molecule as SAFE and masking a fragment
  to vary.
- Early ideation where QED or LogP ranking is useful for triage.

## Scientific Limits

- QED and LogP are heuristic properties, not direct binding, activity, or
  toxicity measurements.
- Returned molecules may be fewer than `num_molecules` because invalid or
  duplicate molecules can be filtered after generation.
- SAFE conversion can fail for simple ring-only scaffolds with no useful
  fragmentation points. Use the documented fallback and explain it.
- Generated molecules still need medicinal chemistry review, synthetic
  feasibility checks, liability checks, and downstream modeling before use.

## Handoffs

- Boltz2 or DiffDock can evaluate generated ligands in a protein context when
  the user supplies a target.
- RDKit is useful for validity, deduplication, visualization, and additional
  descriptors.
- A standalone client or helper script may be justified if repeated SAFE
  conversion and result filtering dominates generated answers.
