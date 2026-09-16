---
name: proteinmpnn-nim
description: >
  Run ProteinMPNN inverse folding via NVIDIA NIM to design protein sequences for a target backbone. Sends user-provided PDB files and design parameters to NVIDIA's hosted API, authenticated with an environment API key, or to a user-selected local NIM. Use for sequence design, backbone redesign, fixed chains and residues, omit_AAs, sampling temperature, soluble model, local Docker, and multi-FASTA output.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "Python >=3.10; requests>=2.28"
allowed-tools: Bash, Read, Write, AskUserQuestion
permissions:
  - network
  - env
---

# ProteinMPNN NIM

<!-- nv-carps: dummy edit to trigger NIM skill validation. -->

Design protein sequences for a supplied backbone PDB. Use this guide for
first-pass hosted/local usage; load supplemental files only when needed:

- `references/api.md`: exact endpoints, schemas, Docker flags, response fields.
- `references/science.md`: inverse-folding uses, limits, and validation.
- `references/parameters.md`: design controls, fixed positions, sampling.
- `references/validation.md`: FASTA, score, and structure checks.
- `references/examples.md`: compact hosted/local request patterns.

## Choose Mode

Honor the user's explicit mode; otherwise use the configured runtime. `NIM_API_MODE=local` selects
the local service at `PROTEINMPNN_NIM_URL`; the URL defaults to
`http://localhost:8000` for a NIM running in the same host or container. Ask only
when neither the environment nor the user's request makes the mode clear:

> Hosted NVIDIA API or local Docker NIM?

- Hosted: `https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict`
- Local: append `/biology/ipd/proteinmpnn/predict` to `PROTEINMPNN_NIM_URL`
  (default base URL: `http://localhost:8000`).

Local inference paths do not include `/v1/`. Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Supported local Docker
startup uses `NGC_API_KEY` (or `NVIDIA_API_KEY` via the preflight) for
registry login, entitlement checks, and first-run model downloads; pass it
into the container with `-e NGC_API_KEY`. Local inference requests use no
auth header after readiness. Warm-cache key-free startup varies by
image/version and should not be assumed.

## Data Transfer and Authorization

Before a hosted request, tell the user that the **entire PDB file and design
parameters will be uploaded to NVIDIA's hosted API** at the endpoint above.
Proceed if the user has explicitly requested hosted processing of that PDB or
already approved the transfer; otherwise ask for confirmation before submitting.
For confidential structures, recommend a local NIM in the user's approved
environment. A configured local URL may point to another machine; use only the
configured or user-selected destination. Do not switch from local to hosted processing without
the user's authorization.

The client reads `NIM_API_MODE`, `PROTEINMPNN_NIM_URL`, and, for hosted mode,
`NGC_API_KEY` from the environment. It sends the key only in the HTTPS
Authorization header to the hosted endpoint; local inference sends no key.
Keep credentials out of logs and saved artifacts. The output directory contains
the full input PDB in `request.json` and the returned sequences and scores, so
use a location appropriate for the input's sensitivity. See
[`references/api.md`](references/api.md) for endpoint and data-handling details.

## Local Docker

For local setup, run the full sequence — env preflight, `docker login`,
`docker run`, readiness loop, then the no-auth localhost request; do not answer
with only a localhost Python request. For the exact preflight (`.env` sourcing,
`NGC_API_KEY`/`NVIDIA_API_KEY` handling, and the `docker run` for
`nvcr.io/nim/ipd/proteinmpnn:latest`), copy the command block in
[`references/api.md`](references/api.md) under **Docker Reference** verbatim.
This NIM's cache mount is `/home/nvs/.cache/nim`, not `/opt/nim/.cache`.
When `PROTEINMPNN_NIM_URL` is supplied, the service is already managed elsewhere;
use that URL and do not start another Docker container.

Readiness:

```bash
proteinmpnn_nim_url="${PROTEINMPNN_NIM_URL:-http://localhost:8000}"
until curl -sf "${proteinmpnn_nim_url%/}/v1/health/ready"; do sleep 5; done
```

## Instructions

For a request to execute a design, run [`scripts/design.py`](scripts/design.py)
and inspect its results. Writing a request script alone does not complete an
execution request. If the user asks only for code or setup instructions, provide
those without submitting an inference request.

1. Use the user's PDB path and requested sequence count. The client reads the
   entire PDB into `input_pdb`; do not replace or truncate the supplied backbone.
2. Select `--mode hosted` or `--mode local` and follow **Data Transfer and
   Authorization** above before submitting. Hosted mode uploads the PDB to the
   documented NVIDIA endpoint and requires `NGC_API_KEY` in the environment.
   Check only whether the key is set; do not print it, dump the environment, or
   save authentication headers. Local inference sends no authorization header.
3. Choose a new `--output-dir` for each request. The client reserves it before
   submitting, preserves the raw response for diagnostics, and validates the
   designed sequence count and score alignment before reporting completion.
4. Read `summary.json` and report the actual results described below. If the
   request or validation fails, report the failure and diagnostic path; do not
   substitute example sequences or repeatedly resubmit the same request.

## Examples

Run from this skill's directory, or use an absolute path to `scripts/design.py`.
Substitute the user's input path and a new output directory:

```bash
python scripts/design.py --mode hosted \
  --pdb /path/to/backbone.pdb --num-sequences 10 \
  --temperature 0.1 --output-dir /path/to/new-design-run
```

For a running local NIM, use `--mode local`; the client honors
`PROTEINMPNN_NIM_URL`. To design only chain A, exclude cysteine, or request the
soluble model, add `--chains A`, `--omit-aas C`, or `--soluble` respectively.
`--seed` sets `random_seed`; `--ca-only` selects the CA-only model. The helper
uses one temperature per request; run separate output directories for a
temperature sweep. For advanced JSONL controls or a custom batch request,
use [`references/api.md`](references/api.md) and the post-response example in
[`references/examples.md`](references/examples.md).

## Save And Report Output

The client writes `request.json`, `response.raw`, `response.json`,
`designed_sequences.fa`, and `summary.json` into the requested output directory.
The FASTA preserves the complete returned `mfasta`, including a native/WT entry
when present. The summary contains only designed sequences, each paired with
its actual score, and records whether scores came from the JSON array or FASTA
headers. It is also printed after the artifacts are saved and checked.

In the final response, report:

- The number of **designed** sequences, excluding the native/WT reference.
- Each design's identifier and actual returned score, plus its sequence (for
  long sequences, give a clearly labelled preview and link to the full FASTA).
- The saved FASTA and summary paths, and the raw response path for provenance.
- That these are inverse-folding candidates, with no fold-back validation
  performed unless it was actually requested and run.

Do not treat a score as proof that a sequence folds or binds. Further validation
with Boltz2 or OpenFold3 is an optional next step. For FASTA/score sanity checks,
read [`references/validation.md`](references/validation.md).

## Limits And Troubleshooting

- Minimum GPU VRAM: about 3 GB.
- `sampling_temp` must be a list, even for one value.
- Empty `mfasta`: check non-empty `input_pdb` and `num_seq_per_target >= 1`.
- PDB parse errors: use valid PDB ATOM records.
- Local URL 404 usually means an accidental `/v1/` prefix.
- Cache mount error: use `/home/nvs/.cache/nim` inside the container.
