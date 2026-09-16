# AutoML common pitfalls

- Do not expect `~/tao-core` at runtime. Schemas and templates must be packaged
  inside the model skill.
- Do not infer dataset URIs from previous runs.
- Do not precompute SDK-managed output paths; non-URI output values are routed
  by the SDK.
- For SLURM, stage large datasets on Lustre rather than burning GPU allocation
  time on large S3 downloads.
- For gated HuggingFace models, verify `HF_TOKEN` is set without reading it.
- If all recommendations fail, stop and summarize the shared root cause instead
  of launching more trials.
- Do not disable `automl_delete_intermediate_ckpt` by default. Keeping every
  trial can consume one full distributed checkpoint set per recommendation.
- Do not bypass a retention-preflight failure by disabling cleanup unless the
  user has explicitly accepted external ownership and manual lifecycle
  management for every trial artifact.
