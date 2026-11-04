# ProteinMPNN NIM — API Reference

## Endpoints

| Mode | Method | URL |
|---|---|---|
| Hosted | POST | `https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict` |
| Local Docker | POST | `http://localhost:8000/biology/ipd/proteinmpnn/predict` |
| Health (local) | GET | `http://localhost:8000/v1/health/ready` |

**IMPORTANT**: Local path has no `/v1/` prefix.

The local URLs above use the default base URL. Set `NIM_API_MODE=local` and,
when the client is not in the NIM container, set `PROTEINMPNN_NIM_URL` to the
container-reachable base URL and append the same endpoint paths. Do not use the
client container's `localhost` for a NIM running in a separate container. Local
inference requests do not use an authorization header.

## Data Handling and Permissions

`SKILL.md` declares `network` for inference HTTP requests and `env` for reading
`NIM_API_MODE`, `PROTEINMPNN_NIM_URL`, and the hosted `NGC_API_KEY`. The existing
`Read` and `Write` tool declarations cover the user's PDB and saved artifacts.

- **Hosted:** the full PDB content and design parameters leave the user's
  environment in a JSON POST to
  `https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict`. The API key
  is sent only as an HTTPS Bearer authorization header, not in the JSON body or
  saved request. The client does not follow redirects.
- **Local:** the same input is sent to the user-selected `PROTEINMPNN_NIM_URL`
  (default `http://localhost:8000`) with no authorization header. Use an approved
  NIM deployment for confidential structures; setting a remote URL still sends
  the structure to that machine. Registry authentication and model downloads
  during Docker setup are separate from inference.
- **Authorization:** disclose the hosted upload before execution. An explicit
  request to process the PDB with the hosted API or prior approval authorizes
  that transfer; otherwise obtain confirmation first. Never silently fall back
  from local to hosted processing.
- **Artifacts:** `request.json` retains the full input PDB; response, FASTA,
  and summary files retain the returned sequences and scores. Choose an output
  location suitable for this data, and never save or print API credentials.

---

## Request Body Schema

All fields are optional (minimum: provide `input_pdb`).

| Field | Type | Default | Range | Notes |
|---|---|---|---|---|
| `input_pdb` | string | null | — | Raw PDB file content as inline string |
| `input_pdb_asset` | string | null | — | NVCF Asset ID for large PDB files; use with original filename in `input_pdb` |
| `input_pdb_chains` | array[string] | null | — | Chains to design e.g. `["A", "B"]`; if null, all chains are designed |
| `ca_only` | boolean | false | — | Enable alpha-carbon-only backbone model |
| `use_soluble_model` | boolean | false | — | Use soluble protein model variant |
| `random_seed` | integer | null | — | Set for reproducibility |
| `num_seq_per_target` | integer | 1 | 1–100 | Sequences generated per structure |
| `sampling_temp` | array[number] | null | 0.0–1.0 | Temperature array; recommended 0.1–0.3; controls diversity |
| `pssm_jsonl` | string | null | — | PSSM file content (JSONL); evolutionary information |
| `pssm_multi` | number | 0.0 | 0.0–1.0 | Balance between PSSM and model predictions |
| `pssm_threshold` | number | 0.0 | any | Filter amino acids by PSSM score |
| `pssm_bias_flag` | boolean | false | — | Enable PSSM-based bias |
| `pssm_log_odds_flag` | boolean | false | — | Transform PSSM to log-odds |
| `fixed_positions_jsonl` | string | null | — | JSONL specifying positions that stay unchanged |
| `omit_AAs` | array[string] | null | — | One-letter codes to exclude globally e.g. `["C", "M"]` |
| `omit_AA_jsonl` | string | null | — | Chain-specific amino acid exclusions (JSONL) |
| `bias_AA_jsonl` | string | null | — | Global amino acid composition biasing (JSON) |
| `bias_by_res_jsonl` | string | null | — | Position-specific biasing (JSONL) |
| `tied_positions_jsonl` | string | null | — | Enforce identical residues across positions (JSONL) |

**No string-typed numeric fields** — all numeric parameters use proper types.

---

## Response Schema

| Field | Type | Description |
|---|---|---|
| `mfasta` | string | Multi-FASTA string with all designed sequences |
| `scores` | array[float] | Returned sequence scores; match to designed records and preserve the values |
| `probs` | array | Per-position amino acid probabilities |

The [NIM endpoint documentation](https://docs.nvidia.com/nim/bionemo/proteinmpnn/latest/endpoints.html)
describes the JSON scores as log-probabilities. The original ProteinMPNN FASTA
[`score` and `global_score` fields](https://github.com/dauparas/ProteinMPNN#readme)
are negative log-probabilities (lower is better); `score` covers designed
residues, while `global_score` covers all residues. Keep the score source explicit
and verify the served version's convention before ranking across these fields.
Scores are not calibrated folding or binding probabilities.

### Example mfasta output

```
>T=0.1, score=1.2345, seq=0
MKTVRQERLKSIVRGPKRAKELMSIQRQAPQTTQNLDLWLQAAQDL
>T=0.1, score=1.1987, seq=1
MKTVRQERLKSIVQGPKRAKELMSIQRQAPQTTQNLDIWLQAAEDL
```

---

## Docker Reference

The default setup binds port 8000 to the host's loopback address because local
inference is unauthenticated. See [Docker's port publishing guidance](https://docs.docker.com/engine/network/port-publishing/).
Registry login uses `--password-stdin`; the `&&` starts the container only after
login succeeds.

```bash
set -a
[ -f .env ] && . ./.env
set +a

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  export NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY in the environment or repo-root .env}"

: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE in the environment or repo-root .env}"
export NIM_TEST_GPU="${NIM_TEST_GPU:-0}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

printf '%s\n' "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin && \
docker run -it \
  --runtime=nvidia \
  --gpus "device=${NIM_TEST_GPU}" \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/home/nvs/.cache/nim" \
  -p 127.0.0.1:8000:8000 \
  nvcr.io/nim/ipd/proteinmpnn:latest
```

| Flag | Value | Notes |
|---|---|---|
| `--gpus` | `device=${NIM_TEST_GPU}` | Single GPU only |
| Port binding | `127.0.0.1:8000:8000` | Host loopback for unauthenticated local inference |
| Cache mount | `/home/nvs/.cache/nim` | **Different from other NIMs** — NOT `/opt/nim/.cache` |
| Image | `nvcr.io/nim/ipd/proteinmpnn:latest` | v1.1.0 as of 2025 |
| No `--shm-size` | — | Not required for this NIM |

---

## Annotated Example Request

```json
{
  "input_pdb": "<full PDB file content>",
  "num_seq_per_target": 10,
  "sampling_temp": [0.1],
  "use_soluble_model": false,
  "ca_only": false,
  "omit_AAs": ["C"],
  "input_pdb_chains": ["A"]
}
```

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| Minimum GPU VRAM | 3 GB |
| Compute capability | >7.0 |
| CPU | 4 cores |
| RAM | 8 GB |
| Storage | 10 GB |
