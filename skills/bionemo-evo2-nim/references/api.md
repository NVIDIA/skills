# Evo 2 NIM - API Reference

This reference captures the Evo 2 NIM API facts used by the skill. Re-check the
official NVIDIA docs before critical use because hosted and container surfaces
can change independently.

## Endpoints

| Mode | Method | URL |
|---|---|---|
| Hosted generation | POST | `https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate` |
| Local generation | POST | `$EVO2_NIM_URL/biology/arc/evo2/generate` |
| Local forward/layer outputs | POST | `$EVO2_NIM_URL/biology/arc/evo2/forward` |
| Local health check | GET | `$EVO2_NIM_URL/v1/health/ready` |

Honor `NIM_API_MODE=hosted|local` when supplied. For local mode, set
`EVO2_NIM_URL` to the base URL visible to the caller; default to
`http://localhost:8000` only when no URL is configured. A separate task
container should use the NIM's service DNS name rather than `localhost`. Never
silently fall back from the selected mode to another endpoint.

Hosted requests require `Authorization: Bearer $NGC_API_KEY`.
Local inference requests use no auth header; `NGC_API_KEY` is passed to Docker
for supported startup, entitlement/resource checks, and first-run model
download.

The hosted API reference provided for this skill documents generation. The
local Evo 2 NIM docs document both generation and `/forward`.

## Generate request body schema

| Field | Type | Required | Default | Range / Values | Notes |
|---|---|---:|---|---|---|
| `sequence` | string | yes | none | length >= 1 | DNA prompt to continue. |
| `num_tokens` | integer or null | no | `100` | >= 1 | Number of tokens to generate. |
| `temperature` | number or null | no | `0.7` | hosted docs show <= `1.3` | Sampling temperature. |
| `top_k` | integer or null | no | `3` | `0` to `6` | `0` considers all tokens. |
| `top_p` | number or null | no | `0.0` | `0` to `1` | `0.0` disables nucleus sampling. |
| `random_seed` | integer or null | no | not specified | integer | Development reproducibility only. |
| `enable_logits` | boolean | no | `false` | true / false | Adds logits to response; can be large. |
| `enable_sampled_probs` | boolean | no | `false` | true / false | Adds sampled token probabilities. |
| `enable_elapsed_ms_per_token` | boolean | no | `false` | true / false | Adds per-token timing. |

## Generate example

```json
{
  "sequence": "ACTGACTGACTGACTG",
  "num_tokens": 8,
  "top_k": 1,
  "enable_sampled_probs": true
}
```

Hosted curl:

```bash
curl -sS -X POST \
  https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate \
  -H "Authorization: Bearer $NGC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACTGACTGACTGACTG","num_tokens":8,"top_k":1,"enable_sampled_probs":true}'
```

Local curl:

```bash
evo2_nim_url="${EVO2_NIM_URL:-http://localhost:8000}"
curl -sS -X POST \
  "${evo2_nim_url%/}/biology/arc/evo2/generate" \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACTGACTGACTGACTG","num_tokens":8,"top_k":1,"enable_sampled_probs":true}'
```

## Generate response schema

```json
{
  "sequence": "ACTGACTG",
  "logits": null,
  "sampled_probs": [0.98, 0.97],
  "elapsed_ms": 123,
  "elapsed_ms_per_token": null
}
```

| Field | Type | Present when | Notes |
|---|---|---|---|
| `sequence` | string | always on success | Generated DNA sequence text. |
| `logits` | array or null | `enable_logits=true` | Documented shape is `[num_tokens, 512]`. |
| `sampled_probs` | array or null | `enable_sampled_probs=true` | Probability per sampled token. |
| `elapsed_ms` | integer | always on success | Server-side elapsed milliseconds. |
| `elapsed_ms_per_token` | array or null | `enable_elapsed_ms_per_token=true` | Per-token timing values. |

Validation errors are reported by the hosted API as HTTP `422`.

## Forward request body schema

Forward is documented for local Docker.

| Field | Type | Required | Notes |
|---|---|---:|---|
| `sequence` | string | yes | DNA sequence. |
| `output_layers` | array of strings | yes | Layer names to capture. |

Example:

```json
{
  "sequence": "ACTGACTGACTG",
  "output_layers": [
    "output_layer",
    "decoder.layers.3.self_attention",
    "decoder.layers.20.mlp.linear_fc2"
  ]
}
```

## Forward response schema

```json
{
  "data": "<base64-encoded NPZ bytes>",
  "elapsed_ms": 123
}
```

Decode with:

```python
import base64
import io

import numpy as np

npz_bytes = base64.b64decode(response_json["data"])
arrays = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
print(arrays.files)
```

## Layer notes

- 7B model: layer indices `0` to `31`.
- 40B model: layer indices `0` to `49`.
- 7B Transformer layers: `3`, `10`, `17`, `24`, `31`.
- 40B Transformer layers: `3`, `10`, `17`, `24`, `31`, `35`, `42`, `49`.
- Other indexed decoder layers are Hyena layers.
- Model-level names include `embedding`, `decoder.final_norm`, and `output_layer`.
- The `embedding` layer is rarely useful for downstream analysis; prefer
  context-dependent intermediate or final layer outputs when possible.

## Logits and vocabulary notes

- Evo 2 vocabulary size is `512`.
- DNA base logit indices map to ASCII values: `A=65`, `C=67`, `G=71`, `T=84`.
- `top_k` accepts values up to `6`, but pretrained DNA generation should focus
  on `A`, `C`, `G`, and `T`.
- Generated sequences can occasionally contain ambiguous characters; flag those
  during validation instead of silently accepting them as high-quality output.

## Docker run reference

```bash
set -a
[ -f .env ] && . ./.env
set +a

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  export NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY in the environment or repo-root .env}"
: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE in the environment or repo-root .env}"

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# 40B default: use 0,1 for 2x H100 80 GB; set NIM_TEST_GPUS=0 for a single H200.
export NIM_TEST_GPUS="${NIM_TEST_GPUS:-0,1}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod 700 "${LOCAL_NIM_CACHE}"   # owner-only; if the NIM runs as a different UID, add -u "$(id -u)" to docker run

# Optional: export NIM_VARIANT=7b and add `-e NIM_VARIANT` for the 7B model.
docker run --rm -it --name evo2-nim \
  --runtime=nvidia \
  --gpus "\"device=${NIM_TEST_GPUS}\"" \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/arc/evo2:2
```

| Flag / env | Purpose |
|---|---|
| `--runtime=nvidia` | Enable the NVIDIA container runtime. |
| `--gpus "\"device=${NIM_TEST_GPUS}\""` | Select one or more GPUs; default helper value is `0,1` for 40B on 2x H100, `0` for single-H200 40B or 7B. |
| `NGC_API_KEY` | Required for supported startup, entitlement/resource checks, and model/resource download. |
| `NVIDIA_API_KEY` | Optional local fallback copied to `NGC_API_KEY` by repo scripts. |
| `LOCAL_NIM_CACHE` | Host cache directory mounted into the container. |
| `/opt/nim/.cache` | Container cache target. |
| `NIM_VARIANT=7b` | Optional 7B model variant; default is 40B. |
| `nvcr.io/nim/arc/evo2:2` | Current Evo 2 NIM image. |

## Hardware and runtime notes

- The default local container variant is 40B; set `NIM_VARIANT=7b` for 7B.
- Evo 2 local deployment requires FP8-capable GPUs. The release notes state
  that older GPUs are not currently supported.
- Current prerequisites list 40B on H100 80 GB x2 or H200 141 GB x1.
- Current prerequisites list 7B on H100 80 GB, H200 141 GB, RTX 6000 Ada
  48 GB, or L40S 48 GB.
- Current prerequisites list 110 GB disk for 40B, 50 GB disk for 7B, and
  16 GB CPU RAM for both variants.
- The 40B local command is multi-GPU by default in this repo via
  `NIM_TEST_GPUS=0,1`; for a single H200 40B host, set `NIM_TEST_GPUS=0`;
  for 7B, set `NIM_VARIANT=7b` and use one selected GPU such as
  `NIM_TEST_GPUS=0`.
- If the host cannot meet the 40B memory requirement, use `NIM_VARIANT=7b`
  with a supported 7B GPU instead of trying to force the 40B model.
- RTX PRO 6000 Blackwell Workstation is not the same SKU as RTX 6000 Ada in
  the current support matrix. On a single RTX PRO 6000 Blackwell Workstation
  validation host, the 7B image and assets downloaded but warmup failed before
  readiness with `No dot product attention backend is available`; NVTE debug
  showed no usable Flash/Fused/Unfused dot-product attention backend.
- A 2026-05-14 retest after pulling the current `nvcr.io/nim/arc/evo2:2`
  image found `:latest` and `:2` resolving to the same manifest, and the 7B
  Blackwell workstation warmup still failed. Logs reported
  `flash-attn 2.5.9.post1` while Transformer Engine expected
  `flash-attn >=2.7.3, <=2.7.4.post1` for that attention path.
- For deliberate local experiments on a GPU outside the documented matrix, set
  `EVO2_EXPERIMENTAL_LOCAL_GPU=1` before running the repo local Docker runner.
  Treat any success as evidence to re-check docs/release notes before updating
  support claims.
- The cache target is `/opt/nim/.cache`; keep the cache on local storage when
  possible and ensure enough free disk before pulling the image.
- On an A100 80 GB validation host, the default local container pulled
  successfully but failed during warmup with `Device compute capability 8.9 or
  higher required for FP8 execution`. Treat local Evo 2 validation as
  hardware-conditional even when the image and cache are available.
