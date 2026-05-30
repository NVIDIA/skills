# LLM Judge Notes

Use `nemo evaluator evaluate explain` to inspect the current Evaluator plugin spec schema before creating an LLM-judge run.

When configuring an LLM judge, verify:

1. The judge model authentication reference matches the execution mode. See [Evaluator API Auth](api-auth.md).

2. The judge model name is the API model ID expected by the endpoint, not an entity display name.

3. The metric prompt and parser match the output you expect from the judge model.

For local iteration, keep the metric and dataset in a spec file and run:

```bash
nemo evaluator evaluate run --spec-file evaluation-spec.json
```

For cluster execution, submit the same spec:

```bash
nemo evaluator evaluate submit \
  --spec-file evaluation-spec.json \
  --workspace default \
  --profile default
```

Prefer `--spec-file` over inline `--spec` for LLM-judge metrics because prompts and score definitions quickly become hard to audit as shell-escaped JSON.
