# Evo 2 Science Notes

Evo 2 is a genomic foundation model for DNA sequence generation and local
representation extraction. Use it for sequence continuation, exploratory
genomic design prompts, and local tensor analysis when the user has a running
container.

## Best-Fit Uses

- Continue a DNA prompt with generated nucleotide sequence.
- Inspect sampled probabilities and timing for a hosted generation request.
- Capture local forward-pass layer outputs for downstream analysis.
- Smoke-test local genomic model deployment with a short DNA sequence.

## Scientific Limits

- Hosted docs expose generation only. Local Docker also documents `/forward`.
  Do not invent hosted layer-output support.
- Generated DNA is model output, not a validated promoter, gene, regulatory
  element, or organismal design.
- Short toy prompts are useful for API validation but weak scientific evidence.
- `random_seed` supports development reproducibility only; it does not make a
  sequence biologically certain.
- Ambiguous bases, extreme GC content, long homopolymers, or low complexity
  should be flagged.

## Handoffs

- Use local `/forward` outputs for downstream embedding or representation work
  only after decoding and validating finite arrays.
- Use external genomics tools for motif, regulatory, coding, or safety analysis
  before treating generated sequences as biologically meaningful.
