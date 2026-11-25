# Docker target

Read this when the target is a single GPU host, or when a NIM container answers wrong. The agent
never types the start or stop command: `scripts/nimdeploy.py status RUN` prints it as
`facts.next_command`, and the guard accepts only that form.

## What the docker runner produces
The docker target is started by a runner (`scripts/runners/docker.py`). It writes one file into the
run folder, `docker-run.sh`, and the apply step is `bash <run folder>/docker-run.sh`. The rendered
script looks like this (this is output to review, not a command to type):

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${NGC_API_KEY:?export NGC_API_KEY before running this script}"
mkdir -p "$HOME/.cache/nim"
docker run -d \
  --name nim-llama --gpus all --shm-size=16g -u "$(id -u):0" \
  -e NGC_API_KEY \
  -v "$HOME/.cache/nim:/opt/nim/.cache" -p 8000:8000 --restart unless-stopped \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8
```

Only `image` is required. A regular NIM image already contains its model, so there is no model input; a
model-free image also needs `model_path`. Values the user did not give are recorded with source
`default`, listed in `facts.defaulted` and shown in the approval summary. Inputs the runner reads
(`nimdeploy.py set RUN key=value`):

| input | meaning | default |
|---|---|---|
| `image` | NIM image with a tag | ask |
| `name` | container name | image name without registry path, tag and digest |
| `container_port` | the port the container serves on | the image metadata (see below), else 8000 flagged as unverified |
| `port` | host port mapped to `container_port` | the container port |
| `gpus` | `all`, a count, or device ids such as `0,1` | `all` |
| `cache_dir` | host model cache | `~/.cache/nim` |
| `shm_size` | `/dev/shm` size | `16g` |
| `model_profile` | sets `NIM_MODEL_PROFILE` | unset: NIM picks |
| `model_path` | sets `NIM_MODEL_PATH` for a model-free image | unset |
| `served_model_name` | sets `NIM_SERVED_MODEL_NAME` | unset |
| `pass_hf_token` | also pass `HF_TOKEN` by name | false |
| `env` | extra non-secret variables | none |
| `run_as_host_user` | add `-u "$(id -u):0"` | true |
| `runner` | which runner starts the container | `NIM_DEPLOY_RUNNER`, else docker |

## Facts to rely on
- Readiness is `GET /v1/health/ready`: 503 while the model loads, 200 when it is ready. Liveness is
  `GET /v1/health/live`. Both are served by the nginx proxy on `NIM_SERVER_PORT` (documented
  default 8000), or on `NIM_HEALTH_PORT` when that is set. The engine itself listens on loopback
  `NIM_BACKEND_PORT` (default 8001) and is not exposed.
- The model cache `/opt/nim/.cache` must be writable by group 0, so the container runs as
  `-u $(id -u):0`. A root-owned cache folder gives permission errors at download time.
- `NGC_API_KEY` is passed by NAME only (`-e NGC_API_KEY`); the value stays in the user's shell. A
  key is optional for eligible public-catalog images and required for production-branch images
  (tag suffix `-pb<version>`) and releases before 2.0.10. Never write the value anywhere.
- Shared memory is 16g (`--shm-size=16g`); a smaller `/dev/shm` ends in "Bus error" or NCCL failures.
- NIM picks a profile for the detected GPUs. Override with `NIM_MODEL_PROFILE` (the `model_profile`
  input). To see the profiles, ask the user to list them with the engine image's
  `list-model-profiles` command, or use the read-only form the guard allows: `docker run --rm
  --gpus all -e NGC_API_KEY IMAGE list-model-profiles` with no `-d`, `-p` or extra mounts.
- Model-free images (`nvcr.io/nim/nvidia/model-free-nim`) need `NIM_MODEL_PATH`: `hf://`,
  `ngc://`, `s3://`, `gs://` or `modelscope://` (or an absolute local folder). Gated Hugging Face
  models also need `HF_TOKEN` (set `pass_hf_token=true`, token exported by the user). S3 needs the
  `AWS_*` variables, GCS needs `GOOGLE_APPLICATION_CREDENTIALS`.
- `NIM_PASSTHROUGH_ARGS` replaces the value baked into the image; it does not append. Do not set it
  through `env`. For extra engine flags use `customArgs` on Helm or `spec.args` on the Operator
  (`references/kubernetes.md`).
- The served model name is `GET /v1/models` `.data[].id`; smoke tests use it.

## Before the start
`validate.py` checks: Docker answers, an NVIDIA runtime exists, the host port is free and no
container has the same name. A busy port or name is a question for the user, not something to
remove or kill.

## Which container port, and why
The port is not assumed. Order, recorded as `port_source` in the approval summary:
1. the user's `container_port`, or one the docs-check step passed with `--source docs`;
2. the image's own metadata, read without pulling: `docker image inspect` (local image), else
   `docker buildx imagetools inspect --format '{{json .Image}}'`, else `skopeo inspect --config`.
   An override variable in the image `Env` (`NIM_HTTP_API_PORT`, `NIM_SERVER_PORT`) wins; else the
   one exposed tcp port; several ports are a question for the user. Registry reads need the user's
   registry login; a failed read is reported and the next source is tried;
3. the documented NIM default 8000, labelled "not verified for this image".
After the start, `verify.py ready` compares `docker port NAME` with the run, so a wrong port shows
up as "image exposes X, you mapped Y".

## Doc sources of these facts
All from the NIM LLM docs, version 2.1.4, fetched 2026-10-22. The docs step re-reads the pages in
`references/docs-sources.md` on every run; these notes only say where each fact came from.

| fact | source_url |
|---|---|
| ports, startup order, health paths (`NIM_SERVER_PORT` 8000, `NIM_HEALTH_PORT`, `NIM_BACKEND_PORT` 8001) | https://docs.nvidia.com/nim/large-language-models/latest/reference/architecture.html |
| `NIM_SERVER_PORT`, `NIM_HEALTH_PORT` definitions | https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html |
| run example, keyless access, health responses | https://docs.nvidia.com/nim/large-language-models/latest/get-started/quickstart.html |
| model-free sources and credentials | https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-free-nim.html |
| profile selection, `list-model-profiles` | https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html |
| `NIM_HTTP_API_PORT` | not in these pages; some images set it in their `Env` |
