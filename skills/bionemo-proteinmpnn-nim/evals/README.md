# ProteinMPNN evaluations

The default `config.yml` selects the hosted request in `evals.json`. It uses
`NGC_API_KEY`, stages `files/1R42.pdb`, and requires an executed inference request,
saved designed sequences, and response-derived scores.

## Local GPU task

`harbor/proteinmpnn-local-design` is a separate native Harbor task with a custom
grader. It requires the ProteinMPNN NIM image, a GPU, its model download
credentials, and a running NIM server in the task container. A shared Astra
sandbox using the generic agent template ignores the task image and cannot
satisfy its loopback health check.

For a dedicated local GPU evaluation, use a separate checkout and replace
`evals/config.yml` with:

```yaml
schema_version: 1
harbor:
  task_source: native_harbor
  custom_dockerfile_mode: preserve
  base_image_mode: disabled
  n_attempts: 1
  pass_threshold: 0.8
  stop_on_pass: false
  n_concurrent: 1
  runtime_env:
    - NGC_API_KEY
grading:
  mode: custom_only
```

Run on a GPU-capable Docker host with `--env-mode docker`, or a dedicated sandbox
template configured with the NIM image, GPU, credentials, and server startup.
The native task's readiness check must pass before an agent starts. Keep the
hosted configuration as the shared CI default.
