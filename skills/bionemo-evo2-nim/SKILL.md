---
name: evo2-nim
description: >
  Generate and analyze DNA sequences using NVIDIA's Evo 2 BioNeMo NIM microservice. Use for Evo2/Evo 2, DNA generation, genomic sequence generation, hosted generation, local Docker deployment, local forward passes, layer outputs, logits, sampled probabilities, and BioNeMo NIM workflows.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "requests>=2.28; numpy>=1.24"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# Evo 2 NIM

Use Evo 2 for DNA generation and, locally, layer-output extraction. Load
supplemental files only when needed:

- `references/api.md`: exact schemas, layer names, Docker flags, hardware notes.
- `references/science.md`: genomic use cases, limits, and interpretation.
- `references/parameters.md`: generation/forward parameter effects.
- `references/validation.md`: DNA, probability, timing, and tensor checks.
- `references/examples.md`: compact hosted/local request patterns.

## Instructions

For generation, use `scripts/generate.py` to execute the request, validate the
response, and save its artifacts. Resolve the script path relative to this
skill's directory and choose an output directory in the user's workspace.
Use the user's sequence and requested parameters; the example below is only
a smoke test.

1. Select the requested mode. For hosted generation, go directly to the
   generation example; Docker setup and local forward passes are separate tasks.
2. When the user asks to run generation, execute the client and inspect its
   exit status and result. Writing a script alone does not complete that request.
3. Report the generated DNA (or its file for long sequences), actual
   `elapsed_ms`, sampled-probability summary, seed, and artifact paths from the
   successful run. Read the saved response or metrics if any result is unclear.

If the request or validation fails, report the actual failure and any diagnostic
files. Do not replace an unavailable API response with example values. For a
code-only request, provide the command without making an inference call.

## Choose Mode

Honor `NIM_API_MODE` when it is set. Accepted values are `hosted` and `local`.
If it is unset, treat an explicit `EVO2_NIM_URL` as local; otherwise ask when
the requested mode is unclear:

> Hosted NVIDIA API or local Docker Evo 2 NIM?

- Hosted generation: `https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate`
- Local base URL: `$EVO2_NIM_URL`, falling back to `http://localhost:8000`
- Local generation: `$EVO2_NIM_URL/biology/arc/evo2/generate`
- Local forward/layer outputs: `$EVO2_NIM_URL/biology/arc/evo2/forward`

Always resolve local health and inference routes from `EVO2_NIM_URL` when it
is present. `localhost` works only when the caller and NIM share a network
namespace; a caller in a separate container usually needs a service URL such
as `http://evo2-nim:8000`. Do not silently switch modes when the selected
endpoint is unavailable. Report the failed endpoint and fix its configuration.

The hosted docs expose generation. `/forward` is documented for local Docker;
do not invent a hosted `/forward` endpoint. Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Supported local Docker
startup uses `NGC_API_KEY` (or `NVIDIA_API_KEY` via the preflight) for
registry login, entitlement checks, and first-run model downloads; pass it
into the container with `-e NGC_API_KEY`. Local inference requests use no
auth header after readiness. Warm-cache key-free startup varies by
image/version and should not be assumed.

## Examples

Normalize prompts before sending. Use A/C/G/T unless ambiguous bases are a
deliberate modeling choice and clearly reported.

For a hosted generation request, run the bundled client with the user's inputs
(the script path below is relative to the skill directory):

```bash
python scripts/generate.py \
  --mode hosted \
  --sequence ACTGACTGACTGACTG \
  --num-tokens 64 --seed 1 \
  --temperature 0.7 --top-k 3 --top-p 0.0 \
  --output-dir /path/to/workspace/evo2-output
```

For an already-ready local NIM, use `--mode local`; the client resolves
`EVO2_NIM_URL` and sends no Authorization header. It never switches endpoints
after a failed request. Set `--timeout` for a longer read if the user requests
a larger generation; failed requests are not automatically resubmitted.

The client saves `request.json`, the actual `response.json`, `generated.fasta`,
and `metrics.json` in the chosen output directory. It also saves the exact
response body in `response.raw` before checking HTTP status or parsing JSON,
so diagnostics survive malformed JSON and non-finite probability/timing values.
It validates the requested number of generated bases, A/C/G/T alphabet, finite sampled probabilities in
`[0, 1]`, and nonnegative timing before printing a successful summary. Existing
directories are never reused, even if empty. Choose an output directory that
does not exist; the client creates it atomically so concurrent runs cannot
overwrite each other's artifacts.
The FASTA contains generated bases only, not the input prompt prepended again.

`sampled_probs` is requested by the client and summarized with count/min/max/mean;
the full values stay in the saved response. A missing or malformed probability
array is a validation failure, not permission to invent confidence values.
Only request `enable_logits` in a custom request when needed; logits can make
responses large. See `references/api.md` for custom payloads.
`random_seed` supports development reproducibility, not biological certainty.

## Local Docker Requirements

Evo 2 local deployment requires FP8-capable GPUs. Do not present A100 as
compatible; A100 can pull the image but fails warmup because FP8 requires
compute capability 8.9 or higher.

- Default 40B: 2x H100 80 GB or 1x H200 141 GB. Use `NIM_TEST_GPUS=0,1` for
  2x H100, or `NIM_TEST_GPUS=0` for one H200.
- 7B fallback: set `NIM_VARIANT=7b`; supported GPUs include H100, H200,
  RTX 6000 Ada, and L40S.
- Approximate disk: 110 GB for 40B, 50 GB for 7B.

Use shell env first; source repo-root `.env` only if present. Do not invent a
cache default or drop the `NVIDIA_API_KEY` fallback.

```bash
set -a
[ -f .env ] && . ./.env
set +a

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  export NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY}"
: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE}"

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# 40B default: 0,1 for 2x H100; set 0 for a single H200.
export NIM_TEST_GPUS="${NIM_TEST_GPUS:-0,1}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod 700 "${LOCAL_NIM_CACHE}"   # owner-only; if the NIM runs as a different UID, add -u "$(id -u)" to docker run

# For 7B: export NIM_VARIANT=7b; export NIM_TEST_GPUS="${NIM_TEST_GPUS:-0}"
docker run --rm -it --name evo2-nim \
  --runtime=nvidia \
  --gpus "\"device=${NIM_TEST_GPUS}\"" \
  -e NGC_API_KEY \
  -e NIM_VARIANT \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/arc/evo2:2
```

Readiness:

```bash
evo2_nim_url="${EVO2_NIM_URL:-http://localhost:8000}"
until curl -sf "${evo2_nim_url%/}/v1/health/ready"; do sleep 10; done
```

If RTX PRO 6000 Blackwell Workstation fails with no Transformer Engine
attention backend, treat it as outside the current validated matrix and rerun
on a documented GPU/runtime.

## Local Forward Pass

Forward returns base64-encoded NPZ tensors.

```python
import base64
import io
import os
import numpy as np
import requests

mode = os.getenv("NIM_API_MODE", "local")
if mode != "local":
    raise RuntimeError("Evo 2 /forward is available only in local mode")
nim_url = os.getenv("EVO2_NIM_URL", "http://localhost:8000").rstrip("/")
sequence = "ACTGACTGACTG"  # Replace with the user's DNA sequence.
sequence = "".join(sequence.upper().split())
if not sequence or set(sequence) - set("ACGT"):
    raise ValueError("Expected nonempty A/C/G/T DNA")
payload = {
    "sequence": sequence,
    "output_layers": ["output_layer", "decoder.layers.3.self_attention"],
}
response = requests.post(
    f"{nim_url}/biology/arc/evo2/forward",
    headers={"Content-Type": "application/json"},
    json=payload,
    timeout=300,
)
response.raise_for_status()
npz_bytes = base64.b64decode(response.json()["data"])
with open("evo2_forward_outputs.npz", "wb") as handle:
    handle.write(npz_bytes)
arrays = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
for name in arrays.files:
    arr = arrays[name]
    print(name, arr.shape, arr.dtype, bool(np.isfinite(arr).all()), float(arr.mean()))
```

## Validate And Report

Save request/response JSON, generated FASTA, and a metrics JSON with sequence
length, GC fraction, ambiguous-base fraction, homopolymer length, sampled-prob
checks, and elapsed timing. Treat invalid schema or alphabet as hard failures;
treat extreme GC, low complexity, duplicates, and missing motifs as warnings.
For deeper checks, read `references/validation.md`.

Key fields: `sequence`, `num_tokens`, `temperature`, `top_k` (0-6), `top_p`
(0-1), `random_seed`, `enable_sampled_probs`, `enable_elapsed_ms_per_token`,
and optional `enable_logits`.

## Troubleshooting

- `401/403`: hosted key missing/expired or not sent as Bearer token.
- `422`: wrong field names such as `max_tokens` instead of `num_tokens`.
- Local endpoint confusion: print `NIM_API_MODE` and `EVO2_NIM_URL`; do not
  replace a configured service URL with `localhost`.
- Local auth confusion: do not send `Authorization` to local inference.
- Local startup: first run downloads model assets; poll
  `$EVO2_NIM_URL/v1/health/ready` before inference.
- FP8 failure: use hosted, 7B on a supported FP8 GPU, or documented 40B GPUs.
