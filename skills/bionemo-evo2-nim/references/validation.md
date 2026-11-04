# Evo 2 Validation

Validate Evo 2 outputs mechanically and scientifically before presenting them
as useful.

## Generation Checks

- Response has a non-empty `sequence`.
- Generated sequence uses the intended DNA alphabet. For basic examples, accept
  only A/C/G/T unless ambiguous bases were requested deliberately.
- Save raw request and response JSON.
- Save generated DNA as FASTA.
- Record length, GC fraction, ambiguous-base fraction, and longest homopolymer.
- If `sampled_probs` is requested, every value should be numeric, finite, and
  between 0 and 1.
- If timing is requested, elapsed fields should be non-negative.

## Warning Conditions

- Extreme GC fraction.
- Long homopolymer runs.
- Low-complexity or repetitive sequence.
- Unexpected ambiguous bases.
- Duplicate generated regions when multiple runs are compared.

## Forward Checks

- Decode `data` as base64 NPZ bytes.
- Save the exact NPZ file.
- Load with `numpy.load(..., allow_pickle=False)`.
- Print tensor names, shapes, dtypes, and finite-value summaries.
- Treat non-finite arrays as hard failures.

## Local Deployment Checks

- Confirm GPU support before local work. Evo 2 local requires FP8-capable GPUs.
- Default 40B requires 2x H100 80GB or 1x H200 141GB.
- Use 7B with supported hardware when 40B memory is unavailable.
