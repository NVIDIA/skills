# MSA-Search Science Notes

MSA-Search finds evolutionary homologs and builds multiple sequence alignments
for protein structure prediction and comparative sequence analysis.

## Best-Fit Uses

- Generate A3M or FASTA alignments for a protein sequence.
- Search UniRef30 and ColabFold environmental databases.
- Build paired MSAs for protein complexes.
- Search structural templates locally when a structure prediction workflow needs
  template mmCIF files and M8 hit tables.

## Scientific Limits

- MSA depth and quality depend on database coverage and sequence family.
- Very short, low-complexity, transmembrane, or disordered sequences may produce
  shallow or biased alignments.
- Paired MSAs depend on taxonomic pairing and can be weak for poorly annotated
  chains.
- Hosted template search was not exposed on `health.api` during May 2026
  validation; use local Docker for template search.

## Handoffs

- OpenFold3 and Boltz2 can consume A3M alignments when the downstream payload
  uses the model-specific MSA shape.
- Template mmCIF and M8 outputs should be saved separately and tracked with the
  sequence and database versions used to generate them.
