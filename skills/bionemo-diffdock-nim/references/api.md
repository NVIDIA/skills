# DiffDock NIM — API Reference

## Endpoints

| Mode | Method | URL |
|---|---|---|
| Hosted | POST | `https://health.api.nvidia.com/v1/biology/mit/diffdock` |
| Local Docker | POST | `http://localhost:8000/molecular-docking/diffdock/generate` |
| Health (local) | GET | `http://localhost:8000/v1/health/ready` |

**IMPORTANT**: Hosted and local paths differ. Hosted uses `/v1/biology/mit/diffdock`; local uses `/molecular-docking/diffdock/generate` (no `/v1/` prefix).

---

## Request Body Schema

| Field | Type | Required | Default | Constraints | Notes |
|---|---|---|---|---|---|
| `protein` | string | Yes | — | — | PDB file content, **ATOM records only** |
| `ligand` | string | Yes | — | — | Ligand file content with `\n` line endings |
| `ligand_file_type` | string | Yes | — | `"mol2"`, `"sdf"`, `"txt"` | Use `"txt"` for SMILES — NOT `"smiles"` |
| `num_poses` | integer | No | ~10 | ≤100 | Number of docking poses to generate |
| `time_divisions` | integer | No | 20 | ≤20 | Diffusion time divisions |
| `steps` | integer | No | 18 | ≤18 | Diffusion steps |
| `save_trajectory` | boolean | No | false | — | Include trajectory frames in response |
| `skip_gen_conformer` | boolean | No | false | — | Skip 3D conformer generation for ligand |
| `is_staged` | boolean | No | false | — | Staging flag |

**No string-typed numeric fields** — all numeric fields use proper integer/boolean types.

---

## Response Schema

| Field | Type | Description |
|---|---|---|
| `status` | string | Result status |
| `details` | string | Human-readable result details |
| `protein` | string | Input protein (echoed back) |
| `ligand` | string | Input ligand (echoed back) |
| `ligand_positions` | list[string] | Ranked SDF-format docking poses; `[0]` is rank 1 (best) |
| `position_confidence` | list[float] | Numeric confidence scores; higher values rank better within a response and stay parallel to `ligand_positions` |
| `trajectory` | list[string] | Only present if `save_trajectory=true` |

### Accessing poses

```python
for i, (pose, conf) in enumerate(zip(result["ligand_positions"], result["position_confidence"])):
    print(f"Pose {i+1}: confidence={conf:.4f}")
    # pose is a complete SDF string — save directly to .sdf file
```

---

## Input Preparation

### Protein: ATOM records only

```python
raw_pdb = Path("protein.pdb").read_text()
protein = "\n".join(line for line in raw_pdb.splitlines() if line.startswith("ATOM"))
```

Bash equivalent:
```bash
grep -E '^ATOM' protein.pdb | sed -z 's/\n/\\n/g'
```

### Ligand as SMILES (`ligand_file_type: "txt"`)

```python
# One SMILES per line; for a single ligand:
ligand = "CC(=O)OC1=CC=CC=C1C(=O)O"   # no escaping needed in Python
ligand_file_type = "txt"
```

### Ligand as SDF

```python
ligand = Path("ligand.sdf").read_text()
ligand_file_type = "sdf"
```

---

## Docker Reference

```bash
set -a
[ -f .env ] && . ./.env
set +a

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  export NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY in the environment or repo-root .env}"

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE in the environment or repo-root .env}"
export NIM_TEST_GPU="${NIM_TEST_GPU:-0}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod u=rwx,go=rx "${LOCAL_NIM_CACHE}"

docker run --rm -it --name diffdock-nim \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="${NIM_TEST_GPU}" \
  --shm-size=2G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/mit/diffdock:2.2.0
```

| Flag | Value | Notes |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `NIM_TEST_GPU` / `0` | Single GPU only — env var, not `--gpus` flag |
| `--shm-size` | `2G` | Required |
| `--ulimit memlock` | `-1` | Required |
| `--ulimit stack` | `67108864` | Required |
| Cache mount | `/opt/nim/.cache` | ~40 GB |
| Image | `nvcr.io/nim/mit/diffdock:2.2.0` | Pinned version tag |

---

## Annotated Example Request

```json
{
  "protein": "ATOM      1  N   ALA A   1 ...\nATOM      2  CA  ALA A   1 ...",
  "ligand": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "ligand_file_type": "txt",
  "num_poses": 10,
  "time_divisions": 20,
  "steps": 18,
  "save_trajectory": false
}
```

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| Minimum GPU VRAM | 24 GB |
| GPU count | 1 (single GPU only) |
| CPU | 4 cores |
| RAM | 8 GB |
| Storage | 40 GB NVMe SSD |
| Driver | ≥535.104.05 |

Tested GPUs: H100, A100, L40S, A6000, A10G (24 GB min).
