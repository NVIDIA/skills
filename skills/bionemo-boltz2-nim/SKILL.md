---
name: boltz2-nim
description: >
  Use Boltz2 NIM for biomolecular structure prediction and binding affinity. Invoke for Boltz2, protein structures, protein-ligand/DNA/RNA complexes, SMILES or CCD ligands, pIC50/IC50 affinity scoring, mmCIF output, hosted NVIDIA API calls, or local Docker deployment.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "requests>=2.28"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# Boltz2 NIM

Predict biomolecular structures and optional ligand affinity. Use this
guide for first-pass hosted/local usage; load supplemental files only when
needed:

- `references/api.md`: exact endpoints, schemas, Docker flags, response fields.
- `references/science.md`: purpose, strengths, limitations, and handoffs.
- `references/parameters.md`: prediction, sampling, MSA, template, affinity tuning.
- `references/validation.md`: mmCIF, confidence, affinity, and chemistry checks.
- `references/examples.md`: compact hosted/local payload patterns.

## Instructions

Read credentials from the environment only when needed. Check presence with
`bool(os.getenv("NGC_API_KEY"))`; keep key values and Authorization headers out of
terminal output, logs, saved artifacts, and the final response. Avoid environment
dumps when diagnosing authentication. If the hosted key is absent, report the
missing variable before submitting a request.

### Choose Mode

Ask only when context is unclear:

> Hosted NVIDIA API or local Docker NIM?

- Hosted: `https://health.api.nvidia.com/v1/biology/mit/boltz2/predict`
- Local: `http://localhost:8000/biology/mit/boltz2/predict`

Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Supported local Docker
startup uses `NGC_API_KEY` (or `NVIDIA_API_KEY` via the preflight) for
registry login, entitlement checks, and first-run model downloads; pass it
into the container with `-e NGC_API_KEY`. Local inference requests use no
auth header after readiness. Warm-cache key-free startup varies by
image/version and should not be assumed.

## Local Docker

For local setup answers, copy the preflight below before `docker login`,
`docker run`, readiness, and the no-auth local request. Do not invent a cache
default or drop the `.env` load or `NVIDIA_API_KEY` fallback.

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

mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

docker run --rm --name boltz2 --gpus all \
  --shm-size=16G \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/mit/boltz2:1.6.0
```

Readiness:

```bash
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done
```

First startup downloads about 30 GB of model weights.

## Examples

### Prediction Request

```python
import os
import requests

HOSTED = True
url = (
    "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
    if HOSTED else "http://localhost:8000/biology/mit/boltz2/predict"
)
headers = {"Content-Type": "application/json"}
if HOSTED:
    api_key = os.getenv("NGC_API_KEY")
    if not api_key:
        raise SystemExit("NGC_API_KEY is required for the hosted API")
    headers["Authorization"] = f"Bearer {api_key}"

payload = {
    "polymers": [{
        "id": "A",
        "molecule_type": "protein",
        "sequence": "MTEYKLVVVGACGVGKSALTIQLIQNHFVDEYDPT",
    }],
    "recycling_steps": 3,
    "sampling_steps": 50,
    "diffusion_samples": 1,
    "step_scale": 1.638,
    "output_format": "mmcif",
}
response = requests.post(url, headers=headers, json=payload, timeout=300)
response.raise_for_status()
result = response.json()
```

Payload essentials:

- Protein polymer: `{"molecule_type": "protein", "sequence": "..."}`.
- DNA/RNA polymer: add another polymer with `molecule_type` `"dna"` or `"rna"`.
- Ligand by SMILES: `{"id": "L1", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}`.
- Ligand by CCD: `{"id": "L1", "ccd": "ATP"}`.
- Affinity: set `"predict_affinity": True` on exactly one ligand; report
  `affinity_pic50`, `affinity_pred_value`, and `affinity_probability_binary`.
- Precomputed A3M MSA goes under the protein polymer. The A3M record uses
  `alignment`, `format`, and `rank`; do not use a stale `data` field.

```python
protein_with_msa = {
    "id": "A",
    "molecule_type": "protein",
    "sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT",
    "msa": {"msa_search": {"a3m": {
        "alignment": ">query\nMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT",
        "format": "a3m",
        "rank": 0,
    }}},
}
```

## Save And Report Output

Save every `.cif` artifact and read the confidence/affinity fields using the
snippet in [`references/examples.md`](references/examples.md) under **Save
Structures And Affinity**. Visualize in PyMOL, ChimeraX, or UCSF Chimera. For
confidence/affinity sanity checks, read `references/validation.md`.

## Limits And Troubleshooting

- Polymers/request: 12. Ligands/request: 20. Chain length: 4096 residues.
- Affinity prediction supports one ligand per request and adds runtime.
- `422`: invalid sequence, invalid CCD/SMILES, malformed MSA, or multiple
  affinity ligands.
- Local URL/auth: local path has no hosted auth header; wait on `/v1/health/ready`.
- Local startup: use `--gpus all`, `--shm-size=16G`, and the `/opt/nim/.cache` mount.
