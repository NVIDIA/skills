# Docs to read on every run

Read this in the `docs` step, after render and before validate. Fetch the pages listed for the
run's target with `curl -sL URL` (or your web fetch tool; if that is denied, use `curl`), read
only the facts named here, then record them:

`python3 "<SKILL_DIR>/scripts/nimdeploy.py" docs-check RUN --url URL --url URL --doc-version V --note "port=...; env=..."`

`--doc-version` is the version the pages state (for example `2.1.4`, or `2.1.x`). `--note` holds the
facts you took, so the approval summary can show them. If a page names a port or an override
variable that differs from the run's `container_port`, pass it with
`nimdeploy.py set RUN container_port=N --source docs`. Never rely on remembered values: the
`references/*.md` facts were fetched on the dates noted there and can be old.

If a page cannot be fetched, say so and ask the user. Only after the user agrees, record
`nimdeploy.py docs-check RUN --offline --reason TEXT`; the summary then says DOCS NOT CHECKED.
A docs check is valid for 60 minutes and is cleared when the image changes.

## docker (NIM LLM)
Base: `https://docs.nvidia.com/nim/large-language-models/latest/`
| page | extract |
|---|---|
| `get-started/quickstart.html` | the run example: required env vars, cache path, shm, GPU flag, port, health paths, the docs version |
| `reference/architecture.html` | the API port and its override variable, the health port, probe paths |
| `reference/environment-variables.html` | every `NIM_*` variable the run needs or may set (checklist below) and its default |
| `reference/advanced-configuration.html` | the priority order, the `NIM_*` to engine-flag mapping, the `NIM_PASSTHROUGH_ARGS` rules |
| `deployment/model-free-nim.html` | only for a model-free image: `NIM_MODEL_PATH` sources and credentials |

### Environment variables to confirm
Check each row against the pages above and put what you took into `--note`. The rows come from the
2.1.4 docs and can change in a newer version. Set a value only when the user asks for it or the docs
say the image needs it. Secrets are passed by name only, never as values.

| variable | what it does (docs 2.1.4) | how this skill sets it |
|---|---|---|
| `NGC_API_KEY` | needed only for production-branch (`-pb`) models, NIMs older than 2.0.10 and gated artifacts; the quickstart leaves it out when keyless access works | by name: `-e NGC_API_KEY` on Docker, the `ngc_secret` Secret on Kubernetes |
| `HF_TOKEN` | private or gated `hf://` models | by name: `pass_hf_token=true` on Docker, `hf_token_secret` on Kubernetes |
| `NIM_CACHE_PATH` | model cache inside the container, default `/opt/nim/.cache` | the host `cache_dir` is mounted there; it must be writable by group 0 |
| `NIM_SERVER_PORT` | external API port, default 8000 | read into `container_port` |
| `NIM_HEALTH_PORT` | port of the health endpoints, default `NIM_SERVER_PORT` | not set by this skill; `verify.py ready` checks the API port, so note a separate health port |
| `NIM_MODEL_PROFILE` | model profile, chosen from the GPU by default | `model_profile` |
| `NIM_MODEL_PATH` | model source (`hf://`, `ngc://`, `modelscope://`, a local folder) for a model-free image | `model_path` |
| `NIM_SERVED_MODEL_NAME` | the name shown by `/v1/models` | `served_model_name` |
| `NIM_MAX_MODEL_LEN`, `NIM_TENSOR_PARALLEL_SIZE`, `NIM_PIPELINE_PARALLEL_SIZE`, `NIM_KVCACHE_PERCENT` | context length, parallelism and GPU memory share; the profile sets the defaults | `nimdeploy.py set RUN env.NAME=value` |
| `NIM_PASSTHROUGH_ARGS` | extra engine flags; setting it replaces the image's own value instead of adding to it | never through `env`: Helm `customArgs`, Operator `spec.args` |

Priority, highest first: container arguments, `NIM_PASSTHROUGH_ARGS`, `NIM_*` variables,
`runtime_config.json`, profile tags, internal values, defaults. `NIM_HTTP_API_PORT` is not on the
2.1.4 environment page; some images set it, so keep reading the port from the image (`container_port`).

## helm (nim-llm chart)
| page | extract |
|---|---|
| `https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/helm-k8s.html` | chart values (image, model, persistence, resources, customArgs, env), service name and port, the chart version |
| `https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html` | env vars and the API port override |

## nim-operator
| page | extract |
|---|---|
| `https://docs.nvidia.com/nim-operator/latest/service.html` | NIMService fields (`image`, `authSecret`, `storage.nimCache`, `expose`, probes), apiVersion, the Operator version |
| `https://docs.nvidia.com/nim-operator/latest/cache.html` | NIMCache fields (`source.ngc`, `model` filters, `storage.pvc`) |
| `https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/nim-operator-deployment.html` | the LLM example, probe defaults, the minimum Operator version |
