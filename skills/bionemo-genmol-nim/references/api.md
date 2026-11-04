# GenMol NIM — Full API Reference

## Endpoints

| Mode | Method | URL |
|---|---|---|
| Hosted | POST | `https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate` |
| Local | POST | `http://localhost:8000/generate` |
| Health check (local) | GET | `http://localhost:8000/v1/health/ready` |
| List models (local) | GET | `http://localhost:8000/v1/models` |

Auth for hosted: `Authorization: Bearer $NGC_API_KEY` header on every request.
Auth for local: none on inference requests after readiness. Authenticate image pulls with `docker login nvcr.io` on the host; pass `-e NGC_API_KEY` to the container for entitlement/resource checks and first-run model downloads.

---

## Request body schema

| Field | Type | Required | Default | Range / Values | Notes |
|---|---|---|---|---|---|
| `smiles` | string | yes | — | Valid SAFE notation | Despite the name, accepts SAFE notation with `[*{min-max}]` masked fragments |
| `num_molecules` | integer | no | 30 | 1–1000 | Actual output count may be lower — invalid molecules are filtered post-generation |
| `temperature` | **string** | no | `"1"` | `"0.01"`–`"10"` | **Must be a string**, not a float — API quirk |
| `noise` | **string** | no | `"1"` | `"0"`–`"2"` | **Must be a string**, not a float — API quirk |
| `step_size` | integer | no | 1 | 1–10 | Diffusion step size; larger values may reduce quality |
| `scoring` | string | no | `"QED"` | `"QED"` or `"LogP"` | QED = drug-likeness (0–1, higher better); LogP = lipophilicity |
| `unique` | boolean | no | false | true / false | When true, deduplicate output before returning |

---

## SAFE input format

GenMol uses **SAFE (Sequential Attachment-based Fragment Embedding)** notation. SAFE represents
molecules as fragments connected by `.` (period), where ring closures encode attachment points.
Masked (unknown) fragments use `[*{min-max}]` where min and max are atom count bounds.

### Pattern examples by use case

**De novo generation** (no scaffold):
```
[*{20-30}]
```
Generates a molecule of approximately 20–30 heavy atoms from scratch.

**Scaffold decoration** (extend a known scaffold):
```python
import safe as sf
try:
    safe_str = sf.encode("C1CC(=O)NC1")   # pyrrolidinone scaffold
except sf.SAFEFragmentationError:
    safe_str = "C1CC(=O)NC1"              # ring-only fallback
masked = f"{safe_str}.[*{{10-15}}]"   # append 10–15 atom fragment
```

**Motif extension** (grow from both ends):
```
[*{5-10}].<core_in_safe>.[*{5-10}]
```

**Lead optimization** (vary one fragment):
```python
import safe as sf
safe_str = sf.encode("CC1=CC=C(C=C1)NC(=O)C")  # encode hit molecule
# Replace last fragment with masked position
masked = safe_str.rsplit(".", 1)[0] + ".[*{5-12}]"
```

### safe-mol package

```bash
pip install safe-mol
```

```python
import safe as sf

sf.encode("C1CC(=O)NC1")   # SMILES → SAFE string
sf.decode(safe_str)         # SAFE string → SMILES
```

Required for conditioned generation and lead optimization. Not needed for pure de novo.

---

## Response schema

```json
{
  "status": "success",
  "molecules": [
    {
      "smiles": "OC[C@@H]1O[CH][C@@H](O)[C@H]1O",
      "score": 0.397
    }
  ]
}
```

On failure:
```json
{
  "status": "failed",
  "error": "<error message string>"
}
```

| Field | Type | Present when | Description |
|---|---|---|---|
| `status` | string | always | `"success"` or `"failed"` |
| `molecules` | list | success | Array of generated molecules |
| `molecules[].smiles` | string | success | Standard SMILES (converted from SAFE internally) |
| `molecules[].score` | float | success | QED or LogP score depending on `scoring` param |
| `error` | string | failure | Error description |

**Note:** `molecules` may have fewer entries than `num_molecules` — invalid or (if `unique: true`) duplicate molecules are silently removed after generation.

---

## Local container startup

Passing `-e NGC_API_KEY` to the container does not authenticate the image pull.
The commands below log in to NGC using the key on stdin, then start the container
only if login succeeds. The unauthenticated API is bound to `127.0.0.1`.
Provide the key and cache path through the environment before running this
example; it does not load credential files. Keep shell tracing disabled.
Registry login authenticates to `nvcr.io` using the key; the container also uses
it for entitlement checks and model downloads, which need about 20 GB of cache.

```bash
set +x

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY in the environment}"
export NGC_API_KEY

: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE in the environment}"
export NIM_TEST_GPU="${NIM_TEST_GPU:-0}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

printf '%s\n' "$NGC_API_KEY" | \
  docker login nvcr.io --username '$oauthtoken' --password-stdin && \
docker run --rm -it --name genmol-nim \
  --runtime=nvidia \
  --gpus=all \
  -e NVIDIA_VISIBLE_DEVICES="${NIM_TEST_GPU}" \
  --shm-size=2G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 127.0.0.1:8000:8000 \
  nvcr.io/nim/nvidia/genmol:1.0.1
```

| Flag | Value | Purpose |
|---|---|---|
| `--runtime=nvidia` | — | Enable NVIDIA container runtime |
| `--gpus=all` | — | Expose all host GPUs |
| `NVIDIA_VISIBLE_DEVICES` | `NIM_TEST_GPU` / `0` | Single GPU only — GenMol uses one GPU regardless |
| `--shm-size` | `2G` | Shared memory for model inference |
| `--ulimit memlock` | `-1` | Unlimited locked memory |
| `--ulimit stack` | `67108864` | 64 MB stack |
| `-e NGC_API_KEY` | — | Passed through for model weight download on first run |
| `-v ... :/opt/nim/.cache` | local path | Cache dir for downloaded model weights (~20 GB) |
| `-p 127.0.0.1:8000:8000` | — | Bind the HTTP API to the host's loopback address |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NIM_TEST_GPU` | 0 | GPU index used to set `NVIDIA_VISIBLE_DEVICES` |
| `NGC_API_KEY` | — | Required for registry login and model download |
| `NVIDIA_API_KEY` | — | Optional fallback source if `NGC_API_KEY` is absent |
| `LOCAL_NIM_CACHE` | — | Required host cache path for downloaded model weights |
| `NIM_LOG_LEVEL` | INFO | Verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL |

---

## Hardware requirements

- **Supported GPUs**: H100 (80GB), A100 (40/80GB), L40S (48GB), A10G (24GB), A6000 (48GB)
- **Minimum GPU memory**: 16 GB
- **Minimum disk**: 20 GB (container + model weights)
- **CUDA Compute Capability**: ≥ 7.0
- **NVIDIA driver**: 535.104.05 or above
- **Software**: Docker + NVIDIA Container Toolkit

### Performance (wall-time for 1000 molecules)

| GPU | motif extension | scaffold decoration |
|---|---|---|
| H100 | 1.7s | 1.0s |
| A100 | 2.4s | 1.7s |
| L40S | 1.9s | 1.2s |
| A10G | 2.5s | 1.8s |
