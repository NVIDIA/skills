---
name: codonfm-setup
description: Set up the public CodonFM v1 repository and download public Encodon checkpoints. Use for requests to build or launch the CodonFM development container, configure local data/checkpoint mounts, verify GPU access, or download public Encodon 80M, 600M, 1B, or Cdwt-1B weights. Do not use for Decodon, Encodon 5B/10B, missense-aggregation, or codon-optimization setup because those implementations are not in the public repository.
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
---

# CodonFM public setup

Operate from the public CodonFM repository root. Support only the checked-in
public v1 code and public Encodon checkpoints.

## Instructions

1. Determine whether the user wants instructions, a downloaded checkpoint, a
   working model environment, or a combination of these.
2. For setup instructions or runtime work, inspect the supplied files and
   configuration directly: the [runner](../../src/runner.py),
   [model configuration](../../src/config.py), [Dockerfile](../../Dockerfile),
   [launcher](../../run_dev.sh), and [requirements](../../requirements.txt).
   Runtime setup requires a checkout; a supplied source archive is sufficient
   for preparing instructions.
3. Check the [public-v1 boundaries](#public-v1-boundaries). For an unsupported
   request, inspect `MODEL_ARCHITECTURES` in `src/config.py`, the model modules,
   and any requested script before explaining the boundary and ending that
   path. Runner argument choices alone do not establish implementation support.
4. Reuse available environments and checkpoints, and choose explicit paths
   from the user's project.
5. Follow only the requested paths below. Environment setup alone does not
   require a checkpoint download; download weights only when the request needs
   them and a suitable local checkpoint is unavailable.

| Requested scope | Action and completion condition |
| --- | --- |
| Instructions only | Inspect the supplied source/configuration, provide the commands described under Reporting setup instructions, then stop. No installation or GPU verification is required. |
| Checkpoint only | Follow Download a checkpoint, check the downloaded files, report their paths, then stop. No Docker, GPU, or model runtime is required. |
| Working model environment | Follow Runtime preflight, choose the container or direct-host path, then Verify the runtime. Report the checks performed and any remaining limitations. |

For supplied source archives, inspect selected files with the available Python
3 standard library (`zipfile.ZipFile.namelist()` and `read()`) without extracting
the whole archive. If extraction is needed, use a fresh directory from
`mktemp -d` or `tempfile.mkdtemp()`. Preserve existing checkouts and temporary
directories; do not delete or overwrite them to prepare a source inspection.

The runner's optional `--dryrun` requires the ML dependencies to be installed
already. It constructs runtime configuration, then stops before execution.
It does not install packages, validate CSV data, or load weights. Preparing
setup instructions does not require running it.

## Reporting setup instructions

For instruction requests, put complete commands for the requested setup path
early in a compact, self-contained answer, even when also writing a guide file.

- For downloads, use supplied checkpoint metadata for the exact repository,
  revision, weight filename, and `config.json`. Show the destination directory
  and keep the weights and configuration together.
- For containers, state Docker/GPU prerequisites, explain existing-container
  replacement before the launcher command, and show explicit host data and
  checkpoint paths and the checkpoint mount at `/data/checkpoints`.
- For direct-host setup, include `python3.11 -m venv`,
  `python -m pip install -r requirements.txt`, a writable `MPLCONFIGDIR`,
  `torch.cuda.is_available()` verification, and explicit host checkpoint paths.
- State which checks actually ran and what remains unverified before model
  execution. Written instructions alone do not establish a working environment.

For a compatibility-only question, give the source-backed availability answer
without adding an unrelated installation procedure.

## Runtime preflight

For a working environment, check hardware before installing the runtime: use
`nvidia-smi` if available, or check CUDA through an existing PyTorch installation.
Actual model execution requires the ML dependencies and a compatible NVIDIA GPU.
Compare the driver with the CUDA version required by the selected runtime using
[NVIDIA's compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
For the Dockerfile's `nvcr.io/nvidia/pytorch:24.10-py3` base, also check the
[24.10 driver requirements](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html#driver-requirements).
If a prerequisite is missing, follow Failure handling below.

### Container preflight

1. Confirm `Dockerfile`, `run_dev.sh`, and `src/runner.py` exist.
2. Confirm `docker info` succeeds. Docker must have NVIDIA Container Toolkit
   configured for `--gpus all`; host GPU visibility alone does not establish
   container GPU access. Verify access in the launched container below.
3. Run `bash -n run_dev.sh` before launching it.
4. Resolve existing absolute host paths for data and checkpoints. Always pass
   both path flags to the launcher rather than relying on `/data/codonfm`
   defaults. Create missing project directories only as needed for the request.
5. Check for an existing container before launch:

```bash
docker ps -a --filter name='^/codon-fm-dev-container$'
```

If an exact-name container is running, `run_dev.sh` stops and removes it; tell
the user before replacement. If it is stopped, the script cannot reuse the
name, so obtain confirmation before removing it with
`docker rm codon-fm-dev-container`. If removal is declined, preserve the
container, skip this launch, and report the name conflict.

The public script uses host networking/IPC and mounts the user's SSH directory
read-only; disclose this before execution. It has no opt-out flags for these
settings. If they conflict with the user's constraints, use the direct-host
path when feasible; otherwise report that container launch remains blocked.

## Build and launch

Set `CODONFM_REPO_DIR`, `CODONFM_DATA_DIR`, and `CODONFM_CHECKPOINT_DIR` to
existing absolute paths chosen for the project.

```bash
cd "${CODONFM_REPO_DIR:?Set the repository path}"
bash run_dev.sh \
    --data-dir "${CODONFM_DATA_DIR:?Set the host data path}" \
    --checkpoints-dir "${CODONFM_CHECKPOINT_DIR:?Set the host checkpoint path}"
```

The host checkpoint directory is mounted at `/data/checkpoints` inside the
container. The image is `codon-fm-dev`; the container is
`codon-fm-dev-container`.

Use only the checked-in public code and the dependency versions declared in
its `Dockerfile` and `requirements.txt`. Continue to Verify the runtime after
launch; checkpoint downloads are a separate requested action.

## Run directly without Docker

Use this path when the user prefers host execution or Docker is unavailable.
It requires a compatible NVIDIA driver, Python 3.11 for the commands below,
and a writable checkout. Confirm `python3.11 --version` succeeds before
installation. Reuse a compatible project environment; otherwise create a
dedicated virtual environment. Set `CODONFM_CACHE_DIR` to a writable cache
directory before running these commands:

```bash
cd "${CODONFM_REPO_DIR:?Set the repository path}"
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p "${CODONFM_CACHE_DIR:?Set a writable cache path}/matplotlib"
export MPLCONFIGDIR="$CODONFM_CACHE_DIR/matplotlib"
python -c "import sys, torch; available = torch.cuda.is_available(); \
print(available, torch.cuda.get_device_name(0) if available else 'CUDA unavailable'); \
sys.exit(0 if available else 1)"
```

The last command is the direct-host GPU verification; interpret it as described
under Verify the runtime. The requirements file configures the CUDA 12.4
PyTorch index for xFormers. Use explicit host paths in subsequent runner
commands; no `/data/checkpoints` mount is created on this path.

## Download a checkpoint

Run only for a requested checkpoint. Reuse a suitable local copy first.
Check `hf --help` and `hf download --help` in the environment that will perform
the download. If the CLI is missing, use a separate download virtual environment
and `python -m pip install huggingface_hub`; preserve the model environment's
dependency versions. The [CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
describes installation and supported options. Public ungated downloads do not
require `hf auth login`.

Set `CODONFM_CHECKPOINT_DIR` to an absolute writable directory in the environment
running `hf`: the chosen host checkpoint root on the host, or `/data/checkpoints`
inside the launched container. Host shell variables are not automatically set
inside the container. Use supplied metadata for exact filenames and revisions;
keep the weights and `config.json` together.

For the public 1B checkpoint:

```bash
hf download nvidia/NV-CodonFM-Encodon-1B-v1 \
    NV-CodonFM-Encodon-1B-v1.safetensors config.json \
    --local-dir "${CODONFM_CHECKPOINT_DIR:?Set the checkpoint root}/encodon-1b"
```

### Small checkpoint example

For a small demonstration, prefer the original public Encodon 80M weights:

```bash
hf download nvidia/NV-CodonFM-Encodon-80M-v1 \
    NV-CodonFM-Encodon-80M-v1.safetensors config.json \
    --revision 399ca9fe17b57941a7bebc6788033919b417413c \
    --local-dir "${CODONFM_CHECKPOINT_DIR:?Set the checkpoint root}/encodon-80m"
```

The [checkpoint](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1/tree/main)
is publicly accessible without a gated-model approval, and the weight file is
307,351,588 bytes. It need not be mirrored to GitHub LFS. The `-TE-` model IDs
use TransformerEngine in `bionemo-recipes`; use the original model IDs with this
public CodonFM codebase. Download only the weights and `config.json`, and reuse
an existing local checkpoint.

Other supported public model IDs are:

- `nvidia/NV-CodonFM-Encodon-80M-v1`
- `nvidia/NV-CodonFM-Encodon-600M-v1`
- `nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1`

Use `--model_name encodon_80m`, `encodon_600m`, or `encodon_1b` according to
architecture size. Cdwt-1B uses `encodon_1b` because Cdwt is a checkpoint
training property, not a separate architecture.

For `.safetensors`, keep `config.json` in the same directory as the model
file. Never invent a Decodon or undocumented checkpoint path.

After a successful download, confirm the expected files exist, `config.json`
parses, and any supplied byte size or checksum matches. Report the absolute
file paths and revision. A checkpoint-only request ends here; it does not
continue to GPU verification. For a combined request, continue only the other
requested path.

## Verify the runtime

This section applies only to working-environment requests. For direct-host
execution, use the GPU check at the end of Run directly without Docker in the
model's activated environment. For a running container, use a host terminal:

```bash
docker exec codon-fm-dev-container python -c \
    "import sys, torch; available = torch.cuda.is_available(); \
print(available, torch.cuda.get_device_name(0) if available else 'CUDA unavailable'); \
sys.exit(0 if available else 1)"
```

Expect `True`, a GPU name, and exit status zero. `False` or an exception means
runtime verification failed; report the missing prerequisite or error. A CUDA
check establishes GPU access, not successful checkpoint loading or model
execution. Finish the environment request by reporting the verified runtime,
available checkpoint paths, and any checks that remain unperformed.

## Failure handling

- If a command fails, diagnose the reported cause. Retry an unchanged command
  at most once for a transient failure, such as a download timeout. For a
  persistent failure, stop that path and report the error and needed fix.
- For failed downloads, preserve the cache and partial files, retry the same
  supported command when appropriate, and report which files remain missing
  or unverified. Do not invent retry flags or claim an incomplete download
  succeeded.
- If runtime prerequisites or container constraints cannot be met, complete
  independent work within the request: inspect supplied source/configuration,
  prepare setup commands, or download a requested checkpoint when possible.
  Report completed work and the unmet prerequisites; do not claim the runtime
  is working.

## Public-v1 boundaries

- Supported: Encodon 80M, 600M, 1B, and Cdwt-1B.
- Not supported: Decodon, Encodon 5B/10B, sequence generation, specialized
  missense aggregation/fine-tuning, and `scripts/codon_optimize.py`.
- CodonFM consumes coding sequences. It is not a variant caller, aligner, GTF
  annotator, or general VCF analysis tool.
