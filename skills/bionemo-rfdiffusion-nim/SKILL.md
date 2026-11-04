---
name: rfdiffusion-nim
description: >
  Run RFDiffusion protein backbone design via NVIDIA NIM. Use for de novo protein backbones, motif scaffolding, binder design, hotspot residues, contigs syntax, diffusion steps, hosted NVIDIA API calls, local Docker deployment, and PDB backbone outputs for ProteinMPNN sequence design.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "requests>=2.28"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# RFDiffusion NIM

Design protein backbone PDBs for de novo proteins, motif scaffolds, and binders.
Use this guide for first-pass hosted/local usage; load supplemental files
only when needed:

- `references/api.md`: exact endpoints, schemas, Docker flags, response fields.
- `references/science.md`: design modes, strengths, limits, and handoffs.
- `references/parameters.md`: contigs, hotspots, steps, and seeds.
- `references/validation.md`: PDB, contig, and artifact sanity checks.
- `references/examples.md`: compact hosted/local request patterns.

## Choose Mode

Ask only when context is unclear:

> Hosted NVIDIA API or local Docker NIM?

- Hosted: `https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate`
- Local: `http://localhost:8000/biology/ipd/rfdiffusion/generate`

Local inference paths do not include `/v1/`. Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Supported local Docker
startup uses `NGC_API_KEY` (or `NVIDIA_API_KEY` via the preflight) for
registry login, entitlement checks, and first-run model downloads; pass it
into the container with `-e NGC_API_KEY`. Local inference requests use no
auth header after readiness. Warm-cache key-free startup varies by
image/version and should not be assumed.

## Local Docker

For local setup answers, copy the preflight below exactly before `docker login`,
`docker run`, readiness, and the no-auth local request. Do not replace it with a
simple `: "${NGC_API_KEY:?Set NGC_API_KEY}"` check, do not invent a cache
default, and do not drop the `NVIDIA_API_KEY` fallback. Default setup is single
GPU `device=0`.

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

docker run -it \
  --runtime=nvidia \
  --gpus "device=0" \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/ipd/rfdiffusion:2
```

Readiness:

```bash
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done
```

## Contigs DSL

`contigs` defines what to keep and what to generate. For the full pattern
syntax (fixed length, ranges, kept chain segments, chain breaks), see
[`references/api.md`](references/api.md) under **Contigs Language Reference**.

Design modes:

- De novo: `contigs="80-120"`; live hosted validation requires a non-empty
  `input_pdb` or `input_pdb_asset`, so inline requests should include the dummy
  PDB below.
- Motif scaffolding: read `target.pdb`, pass `input_pdb`, use a contig like
  `"A25-35/0 50-80"`.
- Binder design: pass target `input_pdb`, contig with target and binder segment,
  and `hotspot_res=["A50", "A51", ...]` in ChainResidue string format.

```python
DUMMY_PDB = (
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)
```

## Request Pattern

```python
import os
from pathlib import Path
import requests

HOSTED = True
url = (
    "https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate"
    if HOSTED else "http://localhost:8000/biology/ipd/rfdiffusion/generate"
)
headers = {"Content-Type": "application/json"}
if HOSTED:
    headers["Authorization"] = f"Bearer {os.getenv('NGC_API_KEY')}"

payload = {
    "input_pdb": DUMMY_PDB,
    "contigs": "80-120",
    "diffusion_steps": 50,
}
response = requests.post(url, headers=headers, json=payload, timeout=300)
response.raise_for_status()
result = response.json()
Path("designed_backbone.pdb").write_text(result["output_pdb"])
```

Motif scaffold:

```python
payload = {
    "input_pdb": Path("target.pdb").read_text(),
    "contigs": "A25-35/0 50-80",
    "diffusion_steps": 50,
}
```

Binder design:

```python
payload = {
    "input_pdb": Path("target.pdb").read_text(),
    "contigs": "A1-100/0 50-100",
    "hotspot_res": ["A50", "A51", "A52", "A53", "A54"],
    "diffusion_steps": 50,
}
```

## Save And Interpret Output

Save `result["output_pdb"]` as a PDB artifact and report `elapsed_ms` when
present. Generated backbones are not final proteins; feed them to ProteinMPNN
for sequence design, then validate sequences/structures with Boltz2 or
OpenFold3. For PDB and contig checks, read `references/validation.md`.

## Limits And Troubleshooting

- `diffusion_steps`: 1-50; 50 is maximum quality, fewer is faster.
- Single GPU; minimum GPU VRAM is about 12 GB.
- `hotspot_res` uses strings like `"A50"`, not tuples.
- `422` usually means chain IDs in `contigs`/`hotspot_res` do not match
  `input_pdb`, a malformed contig, or omitted `input_pdb` for hosted de novo.
- Local URL 404 usually means an accidental `/v1/` prefix.
