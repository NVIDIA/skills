---
name: openfold3-nim
description: >
  Use this skill for OpenFold3, NVIDIA's BioNeMo NIM microservice for biomolecular structure prediction. Invoke whenever the user mentions OpenFold3 or needs protein, protein-ligand, protein-DNA/RNA, or multi-chain complex prediction with the hosted NVIDIA API or local Docker NIM. Covers endpoint choice, auth, request payloads, output artifacts, confidence scores, and local container setup.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "requests>=2.28"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# OpenFold3 NIM

Predict biomolecular structures with OpenFold3. It supports proteins, DNA, RNA,
small-molecule ligands, and multi-entity assemblies. Use this guide for
basic hosted and local NIM use; load supplemental files only when the task needs
deeper context:

- `references/api.md`: exact endpoints, schemas, Docker flags, response fields.
- `references/science.md`: purpose, strengths, limitations, and model handoffs.
- `references/parameters.md`: molecule fields, MSAs, templates, samples, tuning.
- `references/validation.md`: artifact checks and scientific sanity checks.
- `references/examples.md`: compact hosted and local request patterns.

## Choose Mode

Ask only when context is unclear:

> Hosted NVIDIA API or local Docker NIM?

- Hosted URL: `https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict`
- Local URL: `http://localhost:8000/biology/openfold/openfold3/predict`
- Local readiness: `http://localhost:8000/v1/health/ready`

Mode difference: the local prediction path has no `/v1/` prefix. Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Supported local Docker
startup uses `NGC_API_KEY` (or `NVIDIA_API_KEY` via the preflight) for
registry login, entitlement checks, and first-run model downloads; pass it
into the container with `-e NGC_API_KEY`. Local inference requests use no
auth header after readiness, so bind the host port to loopback with
`-p 127.0.0.1:8000:8000`. Warm-cache key-free startup varies by image version
and should not be assumed.

## Auth And Environment

Use credentials already supplied in the environment or injected by a secret
manager. Do not load credential files, print keys, or enable shell tracing.
Confirm keys exist with shell tests.

Hosted needs `NGC_API_KEY` in the request header. Local startup needs
`NGC_API_KEY`, or `NVIDIA_API_KEY` as a fallback, plus `LOCAL_NIM_CACHE`.

## Local Docker

Use the official OpenFold3 NIM image and mount `LOCAL_NIM_CACHE` at
`/opt/nim/.cache`. Before executing setup, explain that registry authentication
sends the key to the NVIDIA registry at https://nvcr.io and first startup
downloads about 10–15 GB of model weights into the cache. Run deployment only
when requested; for a setup guide, provide the commands without running them.

When writing local setup commands, copy the preflight below exactly. Do not
replace it with a simple `: "${NGC_API_KEY:?Set NGC_API_KEY}"` check, do not
drop `NVIDIA_API_KEY`, and do not invent a default `LOCAL_NIM_CACHE`; those
lines are the repo's local NIM env contract. The default single-GPU launch
should show the literal `--gpus "device=0"`; choose a different device only
when the user asks.

```bash
set +x

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY}"
export NGC_API_KEY
: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE}"

mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

printf '%s\n' "$NGC_API_KEY" | \
  docker login nvcr.io --username '$oauthtoken' --password-stdin && \
docker run --rm --name openfold3 \
  --runtime=nvidia \
  --gpus "device=0" \
  --shm-size=16g \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 127.0.0.1:8000:8000 \
  nvcr.io/nim/openfold/openfold3:latest
```

Readiness check:

```bash
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done
```

## Request Pattern

Use `requests.post(..., json=payload, timeout=300)`. For local Docker tasks,
set `hosted = False` after the readiness check passes.

```python
import os
import requests

hosted = True
url = (
    "https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict"
    if hosted
    else "http://localhost:8000/biology/openfold/openfold3/predict"
)
headers = {"Content-Type": "application/json"}
if hosted:
    headers["Authorization"] = f"Bearer {os.getenv('NGC_API_KEY')}"

seq = "MKTVRQERLKSIVR"
payload = {
    "inputs": [{
        "input_id": "prediction_1",
        "output_format": "pdb",
        "molecules": [{
            "type": "protein",
            "id": "A",
            "sequence": seq,
            "diffusion_samples": 1,
            "msa": {
                "main": {
                    "a3m": {
                        "alignment": f">query\n{seq}",
                        "format": "a3m"
                    }
                }
            }
        }]
    }]
}

response = requests.post(url, headers=headers, json=payload, timeout=300)
response.raise_for_status()
result = response.json()
```

Payload gotchas:

- Top level is `{"inputs": [...]}` and OpenFold3 accepts exactly one input.
- `molecules` can contain 1-32 objects with `type`: `protein`, `dna`, `rna`,
  or `ligand`.
- Protein/RNA MSAs are optional but, when supplied, `alignment` must start with
  a FASTA header such as `>query\nSEQUENCE`.
- Ligands use either `smiles` or `ccd_codes`, for example
  `{"type": "ligand", "id": "L", "ccd_codes": "ATP"}`.
- DNA/RNA entities use `sequence`, for example
  `{"type": "dna", "id": "B", "sequence": "ATCGATCG"}`.
- `diffusion_samples` is 1-5. `output_format` is `pdb` or `cif`.

## Save And Interpret Output

Save every returned structure as a scientific artifact. Main response path:
`result["outputs"][0]["structures_with_scores"]`.

```python
output = result["outputs"][0]
for i, sample in enumerate(output["structures_with_scores"], start=1):
    fmt = sample["format"]
    with open(f"openfold3_structure_{i}.{fmt}", "w", encoding="utf-8") as fh:
        fh.write(sample["structure"])
    print("confidence_score", sample.get("confidence_score"))
    print("complex_plddt_score", sample.get("complex_plddt_score"))
    print("ptm_score", sample.get("ptm_score"))
    print("iptm_score", sample.get("iptm_score"))
    print("complex_pde_score", sample.get("complex_pde_score"))
```

Higher `confidence_score`, `complex_plddt_score`, `ptm_score`, and `iptm_score`
are generally better; lower `complex_pde_score` is generally better. Treat toy
or very short sequences as API smoke tests, not meaningful structural biology.
For why and when OpenFold3 is scientifically appropriate, read
`references/science.md`.

## Common Limits

- Inputs per request: 1.
- Molecules per input: 1-32.
- Diffusion samples: 1-5.
- TensorRT path supports shorter sequences; PyTorch path can support longer
  sequences, but long inputs need much more GPU memory.
- Sequences over roughly 1800 residues require at least 80 GB GPU memory.
- Local NIM is single-GPU only; choose the target device in the Docker flag.

## Troubleshooting

- `401`: missing, expired, or unauthorized NGC API key.
- `422`: invalid molecule type, invalid sequence characters, bad MSA shape, or
  `diffusion_samples` outside 1-5.
- MSA errors: ensure the alignment starts with `>query\n`.
- Local `404`: remove `/v1/` from the prediction URL.
- Local startup stalls: first run may be downloading 10-15 GB of model weights
  into `LOCAL_NIM_CACHE`.
- Memory errors: shorten the sequence, reduce samples, or use a larger GPU.
