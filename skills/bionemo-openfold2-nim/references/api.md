# OpenFold2 NIM API Reference

## Endpoints

Hosted:

- `POST https://health.api.nvidia.com/v1/biology/openfold/openfold2/predict-structure-from-msa-and-template`

Local Docker:

- `POST http://localhost:8000/biology/openfold/openfold2/predict-structure-from-msa-and-template`
- `GET http://localhost:8000/v1/health/ready`

Hosted requests require `Authorization: Bearer $NGC_API_KEY`. Local inference
requests use no auth header after the container reports ready. The local
prediction path does not include the hosted `/v1` prefix.

## Request Schema

Required:

- `sequence` string: amino-acid sequence using valid IUPAC protein symbols.
  Hosted API docs list length 1-1000. Current local support-matrix docs state
  sequences up to 2048 amino acids on supported hardware.

Optional:

- `input_id` string or null: request label, length <=128.
- `alignments` object or null: multiple sequence alignments in A3M format.
- `templates` object or null: legacy/internal template object. Current docs
  recommend explicit mmCIF templates instead of HHR examples.
- `selected_models` array or null: parameter set IDs, defaults to `[1,2,3,4,5]`.
- `relax_prediction` boolean or null: run structural relaxation after prediction.
- `use_templates` boolean or null: use provided templates as model features.
- `explicit_templates` array or null: user-supplied structural templates in
  mmCIF format.

Alignment object pattern:

```json
{
  "alignments": {
    "uniref90": {
      "a3m": {
        "alignment": ">query\nMTEYK...",
        "format": "a3m"
      }
    }
  }
}
```

Explicit template pattern:

```json
{
  "use_templates": true,
  "explicit_templates": [
    {
      "structure": "data_TEMPLATE\n...",
      "format": "mmcif",
      "name": "template_1",
      "source": "user_provided"
    }
  ]
}
```

## Response Handling

The build-page deploy example states that the response includes one prediction
for each selected model parameter set, ordered by confidence. The API docs list
JSON `200` and `422` responses but do not expose a detailed response schema.

Robust client/script behavior:

1. Save the full JSON response as `openfold2_response.json`.
2. Recursively search the JSON for string fields that look like PDB (`ATOM`)
   or mmCIF (`data_`) structure text and save them as `.pdb` or `.cif`.
3. Print any numeric confidence/ranking fields present, without assuming a
   field name until live validation confirms the shape.
4. Keep request metadata: sequence length, `input_id`, selected models, MSA
   source, template names, and whether relaxation was enabled.

## Local Docker

Image:

- `nvcr.io/nim/openfold/openfold2:latest`

Current docs also show the latest Enroot/HPC image tag as
`nvcr.io/nim/openfold/openfold2:2.4.0`; keep Docker examples on `:latest`
unless the user asks for a pinned version.

Startup:

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

export NIM_TEST_GPU="${NIM_TEST_GPU:-0}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

docker run --rm --name openfold2 \
  --runtime=nvidia \
  --gpus "device=${NIM_TEST_GPU}" \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/openfold/openfold2:latest
```

The cache target is `/opt/nim/.cache`. First startup downloads model
parameters if absent.

## Hardware And Runtime

Current latest support-matrix facts:

- Single-GPU NIM.
- BF16 precision.
- Supported examples include A100 80 GB, H100 80 GB, H200, L40S, RTX 6000 Ada,
  RTX PRO 6000 Blackwell Workstation, B200, GB10, GB200, and GH200.
- Recommended disk: at least 80 GB. Container is roughly 55 GB; model
  parameters are roughly 10 GB.
- System memory: at least 64 GB RAM.
- CPU: at least 8 available cores.
- OpenFold2 current docs support input sequences up to 2048 amino acids locally;
  hosted API docs list 1-1000.

## Hosted/Local Caveats

- Do not add `/v1` to the local prediction path.
- Do not send `Authorization` to local inference endpoints.
- Do not use current OpenFold3 payload fields such as top-level `inputs` or
  `molecules`; OpenFold2 is a monomer model with direct `sequence`.
- Do not use HHR as the preferred new template path. Current docs state that
  OpenFold2 2.0.0 and later support mmCIF-based explicit templates.
