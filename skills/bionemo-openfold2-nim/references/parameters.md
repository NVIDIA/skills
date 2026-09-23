# OpenFold2 Parameter Guide

## Required Input

- `sequence`: amino-acid sequence. Use standard protein IUPAC letters and avoid
  whitespace or FASTA headers in the value.

## Identifiers

- `input_id`: optional request label. Use stable names such as `P12345_domain1`
  or `7WBN_A` so output files can be traced to inputs.

## Alignments

- `alignments`: optional A3M MSA object. Use database names such as `uniref90`
  or `small_bfd` as keys, with nested `a3m` objects containing `alignment` and
  `format`.
- A3M should start with a query header such as `>query`.
- Deeper, biologically relevant MSAs generally improve structure confidence.
- Shallow or mismatched MSAs can produce plausible-looking but unreliable
  structures.

## Models

- `selected_models`: list of OpenFold/AlphaFold parameter set IDs 1-5.
- Default is all five. Use `[1]` or `[1, 2]` for smoke tests or latency-sensitive
  examples. Use all five for stronger production confidence comparison.
- The response includes one prediction per selected model, ordered by
  confidence.

## Relaxation

- `relax_prediction`: boolean. Relaxation can improve local geometry but adds
  runtime. Keep `False` for quick API smoke tests; consider `True` for final
  artifacts intended for downstream modeling.

## Templates

- `use_templates`: boolean. Set `True` only when providing relevant templates.
- `explicit_templates`: list of mmCIF template objects with `structure`,
  `format`, `name`, and `source`.
- Current docs state that OpenFold2 2.0.0 and later support mmCIF-based
  templates; avoid new HHR-template examples.

## Runtime Tradeoffs

- Longer sequences increase runtime and memory use.
- More selected models increase runtime but give better model agreement
  evidence.
- Relaxation adds runtime.
- Templates add payload complexity and require careful biological alignment.
