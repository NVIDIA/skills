# Evaluator API Auth

Read this file before configuring a model, agent, remote metric, LLM judge, or
durable platform job.

## Match the secret reference to the execution mode

`api_key_secret` is a reference, never the credential value. It resolves to a different value depending on execution mode: standalone and plugin submission.

| Execution | `api_key_secret` resolves to |
| --- | --- |
| Standalone SDK | Environment-variable name in the calling process, such as `NVIDIA_API_KEY` |
| Plugin `submit` | NeMo Platform secret name in the target workspace, such as `nvidia-api-key` |

A remote job cannot read the submitting shell's environment variables. Before
submitting, verify the `api_key_secret` is in the list of secrets:

```bash
nemo secrets list
```

Create it through the supported secrets CLI for the installed NeMo Platform
version. Do not put the key directly in a spec, command line, log, or committed
file.

```bash
printf '%s' "$NVIDIA_API_KEY" | nemo secrets create nvidia-api-key --from-file -
nemo secrets list
```

## Adapt the local-first spec for platform submission

The checked `llm_as_judge.json` uses the local environment variable
`NVIDIA_API_KEY`. Create a platform copy that points both the generation target
and the judge's environment binding at the workspace secret:

```bash
jq --arg platform_secret "nvidia-api-key" '
  .target.api_key_secret = $platform_secret
  | .metrics[0].secrets.NVIDIA_API_KEY = $platform_secret
' skills/nemo-evaluator-plugin/assets/specs/llm_as_judge.json \
  > llm_as_judge.platform.json
```

The bundle key `NVIDIA_API_KEY` remains the environment-variable name expected
by the judge; its value becomes the platform secret name. Do not edit
`metrics[*].payload` or its digest will no longer describe the inline metric.

## Diagnose remote 409 responses

Do not assume HTTP 409 means a duplicate job. Inspect the response body. The
Jobs service can return 409 when a referenced platform secret does not exist
or is inaccessible, for example:

```text
Unable to create job because one or more referenced secrets were not found or
are not accessible.
```

The response intentionally may not identify the secret. Verify every referenced
workspace and secret name, then retry the submission.

## Follow security best practices

- Print secret names only, never values.
- Redact authorization headers and provider responses that echo credentials.
- Use placeholders such as `<platform-secret-name>` in shared examples.
- Do not copy `.env` files into job artifacts.
