# ProteinMPNN Science Notes

ProteinMPNN performs inverse protein folding: given a backbone structure, it
designs sequences expected to be compatible with that backbone.

## Best-Fit Uses

- Generate candidate sequences for a fixed backbone.
- Redesign selected chains while preserving other chains.
- Exclude amino acids globally or by position.
- Use soluble-model bias for soluble protein design.
- Explore diversity with multiple sampling temperatures.

## Scientific Limits

- ProteinMPNN designs sequence compatibility with a backbone; it does not prove
  folding, expression, stability, binding, or function.
- Backbone quality strongly controls output usefulness.
- Higher sampling temperature increases diversity but may reduce confidence.
- Designed sequences should be checked by a structure predictor such as
  OpenFold3 or Boltz2 and, when relevant, by experimental or biophysical tools.

## Handoffs

- Use OpenFold3 or Boltz2 to predict whether designed sequences recover the
  intended fold.
- Use MSA-Search only after sequence design if downstream structure prediction
  benefits from evolutionary context.
