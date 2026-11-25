# Troubleshooting

Read this in `diagnose`, after `verify.py triage` printed matches (or none), and before any retry.
`scripts/log-triage.json` holds the patterns; `triage` prints the matching `id`, `meaning`, `fix`
and the log line. Show them to the user, then ask which fix to take. Change an input only with
`nimdeploy.py set RUN key=value` after `diagnose` > `render`; the old approval does not carry over.

| id | log sign | meaning | fix |
|---|---|---|---|
| `cuda-oom` | `CUDA out of memory`, `OutOfMemoryError`, `No available memory for the cache blocks` | model, KV cache or activations do not fit | lower the context length (`NIM_MAX_MODEL_LEN`, `--max-model-len`), lower GPU memory use, pick a smaller or quantized profile, or add GPUs (tensor parallel) |
| `ngc-401` | `401 Unauthorized`, `Invalid NGC_API_KEY` | NGC rejected the credentials for the image or the model | the user exports a valid `NGC_API_KEY` (never in a file) and logs in to `nvcr.io`; on Kubernetes recreate the pull secret and the key secret |
| `no-gpu` | `could not select device driver ... [[gpu]]`, `No CUDA GPUs are available`, `unknown or invalid runtime name: nvidia` | the container cannot see a GPU | the user installs the NVIDIA Container Toolkit and configures the docker runtime; check with an `nvidia-smi` container |
| `profile-not-found` | `No compatible profile`, `NIM_MODEL_PROFILE ... not valid` | no profile fits the GPUs, or the named profile is not in this image | stop and ask; list profiles (`references/docker.md`). A profile from nim-select is never dropped: select again for this image. Else, with the user, choose a compatible one or change the GPU count |
| `port-in-use` | `Address already in use`, `port is already allocated` | the host port is taken | ask for a free port; the owner shows with `ss -ltnp` or `docker ps --filter publish=PORT` |
| `hf-token` | `GatedRepoError`, `Cannot access gated repo`, `Invalid user token` | the Hugging Face model is gated or the token is missing | the user provides a token with access, exported (docker, `pass_hf_token=true`) or in a secret (`hf_token_secret`); never in a rendered file |
| `image-pull-backoff` | `ImagePullBackOff`, `ErrImagePull`, `manifest unknown` | the image cannot be pulled: wrong tag or missing pull secret | check the tag exists, check the namespace has an `nvcr.io` pull secret and the manifest references it |
| `insufficient-gpu` | `Insufficient nvidia.com/gpu` | no node has enough free GPUs | lower `gpus`, `tp` or `replicas`, or ask the user to free GPUs (`kubectl describe nodes --context CTX`) |
| `shm-too-small` | `Bus error`, `/dev/shm`, `unable to allocate shared memory` | shared memory too small for NCCL or the data loader | docker: `shm_size=16g` or more; Kubernetes: memory-backed `/dev/shm` |

## Other signs
- Readiness stays 503 for a long time: large models load slowly. Docker: wait and read the log.
  Operator: raise `startup_failure_threshold` (default 120 x 10 s = 20 minutes).
- `profile` mismatch on SGLang vs vLLM images: the profile name starts with the engine
  (`vllm-...`, `sglang-...`).
- Smoke test returns 404 for the model: use the id from `GET /v1/models` (`--model ID`).

## Rules
Look at the triage output and this page before a retry. Three visits to a state is the limit; after
that stop and show the user every error so far.
