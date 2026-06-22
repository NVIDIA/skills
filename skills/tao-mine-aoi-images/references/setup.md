# Setup and Environment

The mining and embedding tasks live inside the `tao_toolkit.data_services` image declared in `versions.yaml`. Resolve the concrete URI once at the top of the run, then confirm Docker, the NVIDIA container toolkit, and a GPU are present before doing anything else:

```bash
# Resolve tao_toolkit.data_services → concrete nvcr.io/... URI from versions.yaml
DS_IMAGE=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")
echo "DS_IMAGE=$DS_IMAGE"

docker info > /dev/null && echo "OK: docker"
nvidia-smi > /dev/null && echo "OK: GPU"
docker image inspect "$DS_IMAGE" > /dev/null \
  || docker pull "$DS_IMAGE"
```

`TAO_SKILL_BANK_PATH` is exported by the plugin's `session_start` hook. If it is unset (e.g. running outside the Claude Code plugin), point it at the skill-bank repo root before resolving.

A GPU is required for both the encoder forward pass and the cuML/cuDF k-NN search; both steps will fail without CUDA.

**Path mounting.** Every host path the container reads or writes — input parquets, output dirs, and the source-pool image root — must be bind-mounted. The simplest and most predictable approach is to mount the workspace root with **identical paths** inside and outside the container so the absolute paths in the parquet args resolve the same way on both sides:

```bash
WORKSPACE=<absolute path that contains all parquets, outputs, and the source-pool images>
DOCKER="docker run --gpus all --rm --ipc=host -v $WORKSPACE:$WORKSPACE -w $WORKSPACE $DS_IMAGE"
```

Do **not** pass `--user $(id -u):$(id -g)`. The `data_services` image imports `transformers` at startup, which calls `getpass.getuser()` → `pwd.getpwuid(os.getuid())`. With a non-root host UID and no matching `/etc/passwd` entry inside the container, this raises `KeyError: 'getpwuid(): uid not found: <uid>'` before any embedding or mining work starts. The container runs as root; chown outputs back to the host UID afterward if needed:

```bash
docker run --rm -v "$WORKSPACE:/w" alpine chown -R "$(id -u):$(id -g)" "/w/<results_subdir>"
```

Reuse `$DOCKER` for the three invocations in the Method section.

If the source pool is provided only as a CSV, convert it to a parquet up front:

```python
import pandas as pd
pd.read_csv(source_pool_csv).to_parquet(source_pool_parquet, index=False)
```

The conversion must preserve the `filepath` column verbatim (and `label` if present). Do not add a path prefix — the container reads input parquets as-is, and the `$WORKSPACE` mount keeps host and container paths identical.

**Author the two spec files once per iteration.** Both files live under `$WORKSPACE` so the `-e` argument resolves on both sides of the mount. Per-run values stay out of the spec and are passed as Hydra overrides at invocation time.

```bash
cat > "$WORKSPACE/embedding_spec.yaml" <<'EOF'
model: SigLIP                                # CLIP, SigLIP, or a TAO checkpoint
model_path: google/siglip-base-patch16-224   # HF id, local HF dir, or .pth/.ckpt
# model_config_path: <train_spec.yaml>       # required only when model_path is a TAO checkpoint
batch_size: 64
EOF

cat > "$WORKSPACE/mining_spec.yaml" <<'EOF'
topn: 5
knn_metric: cosine                           # cosine for SigLIP/CLIP; euclidean/manhattan otherwise
filter_by_label: "false"                     # quoted — the schema reads it as a string
EOF
```

Any field in either spec can still be overridden inline at the CLI (e.g. `topn=10`) — Hydra applies CLI overrides on top of the spec.
