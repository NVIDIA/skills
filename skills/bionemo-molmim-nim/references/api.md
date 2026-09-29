# MolMIM NIM API Reference

## Endpoints

Hosted:

- `POST https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate`

Local Docker:

- `POST http://localhost:8000/embedding`
- `POST http://localhost:8000/hidden`
- `POST http://localhost:8000/decode`
- `POST http://localhost:8000/sampling`
- `POST http://localhost:8000/generate`
- `GET http://localhost:8000/v1/health/ready`

Hosted requests require `Authorization: Bearer $NGC_API_KEY`. Local inference
requests use no auth header after readiness.

Hosted/local difference: the hosted API reference exposes `/generate`; local
docs expose the full latent-space surface. Do not invent hosted `/embedding`,
`/hidden`, `/decode`, or `/sampling` endpoints.

## Hosted `/generate` Request

Fields:

- `smi` string: seed SMILES.
- `algorithm` string enum: `"CMA-ES"` or `"none"`. Default: `"CMA-ES"`.
- `num_molecules` integer: 1-100. Default: 10.
- `iterations` integer: 1-1000. Default: 10.
- `property_name` string enum: `"QED"` or `"plogP"`. Default: `"QED"`.
- `particles` integer: 2-1000. Hosted API default: 20. Local docs commonly use 30.
- `minimize` boolean: default `false`.
- `min_similarity` number: 0-1 in hosted API docs. Default: 0.7.
- `scaled_radius` number: 0-2. Default: 1.

CMA-ES guided optimization:

```json
{
  "smi": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  "algorithm": "CMA-ES",
  "num_molecules": 5,
  "property_name": "QED",
  "minimize": false,
  "min_similarity": 0.4,
  "particles": 8,
  "iterations": 3
}
```

Unguided generation through `/generate`:

```json
{
  "smi": "CC(Cc1ccc(cc1)C(C(=O)O)C)C",
  "algorithm": "none",
  "num_molecules": 10,
  "particles": 20,
  "scaled_radius": 1.0
}
```

Response:

- Hosted live validation returned `molecules` as a JSON-encoded string
  containing objects with `sample` SMILES and `score`, plus `score_type`.
- Local `/generate`, `/decode`, and `/sampling` examples may return `generated`
  arrays instead. Robust clients/scripts should save the full JSON and extract
  SMILES from hosted `molecules[*].sample` or local `generated`.
- Hosted API docs list JSON `200` and `422`; save the full response because
  live service details can change.

## Local Latent Endpoints

`/embedding`:

- Request: `{"sequences": ["<SMILES>", "..."]}`
- Response: `{"embeddings": [[...], ...]}`

`/hidden`:

- Request: `{"sequences": ["<SMILES>", "..."]}`
- Response: `{"hiddens": [[[...]]], "mask": [[...]]}`

`/decode`:

- Request: `{"hiddens": ..., "mask": ...}`
- Response: `{"generated": ["<SMILES>", ...]}`

`/sampling`:

- Request fields: `sequences`, `beam_size`, `num_molecules`, `scaled_radius`.
- `beam_size`: 1-10, default 1.
- `num_molecules`: 1-10, default 1.
- `scaled_radius`: 0-2, local docs default 0.7.
- Response: `{"generated": [["<SMILES>", ...], ...]}`

`/generate`:

- Same conceptual generation surface as hosted, local path has no `/v1`.

## Local Docker

Image:

- `nvcr.io/nim/nvidia/molmim:1.0.0`

Startup:

```bash
set -a
[ -f .env ] && . ./.env
set +a

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  export NGC_API_KEY="$NVIDIA_API_KEY"
fi
if [ -z "${NGC_CLI_API_KEY:-}" ] && [ -n "${NGC_API_KEY:-}" ]; then
  export NGC_CLI_API_KEY="$NGC_API_KEY"
fi
: "${NGC_CLI_API_KEY:?Set NGC_API_KEY, NVIDIA_API_KEY, or NGC_CLI_API_KEY}"
: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE}"

echo "$NGC_CLI_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

export NIM_TEST_GPU="${NIM_TEST_GPU:-0}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

docker run --rm -it --name molmim \
  --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES="${NIM_TEST_GPU}" \
  -e NGC_CLI_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/home/nvs/.cache/nim" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/molmim:1.0.0
```

Cache target is `/home/nvs/.cache/nim`.

## Hardware And Runtime

Current support matrix:

- Single GPU.
- Minimum GPU memory: 3 GB.
- Compute capability > 7.0.
- Tested configurations include L40 48 GB, A100 40-80 GB, and A10 24 GB.
- Minimum system hardware: 4 CPU cores, 16 GB RAM, 50 GB NVMe SSD storage.
- Docker >=23.0.1, NVIDIA driver >=535, NVIDIA Container Toolkit >=1.13.5.

## Guided Optimization Example

The `digital-biology-examples` MolMIM guided optimization package is a local
FastAPI wrapper around a locally hosted MolMIM NIM. It accepts `n`, `smiles`,
`scores`, and `sigma`, then uses MolMIM endpoints to update latent
representations and decode optimized molecules. Treat it as an advanced local
workflow reference, not part of this skill bundle and not a standalone client
to vendor here.
