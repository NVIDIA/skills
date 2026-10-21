# Stage configuration guardrails

Shared rules for every production stage page under this directory. Each stage
reference links here and keeps only stage-specific notes inline.

- Stage references return configuration only and do not execute. Running a
  stage in Docker via `workflow-runner` (and its approval) is the operator
  skill's job.
- Do not put secrets, tokens, or absolute user home paths in committed
  cookbooks. Use placeholders such as `<model-cache>` for local mounts and
  environment variables for endpoint keys.
