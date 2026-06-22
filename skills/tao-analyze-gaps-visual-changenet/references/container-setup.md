# Container Setup and Path Mounting

The threshold sweep, weakness ranking, and per-lighting expansion all run inside the `tao_toolkit.data_services` image declared in `versions.yaml`. Resolve the concrete URI once at the top of the run, then confirm Docker, the NVIDIA container toolkit, and a GPU are present and ensure the image is cached:

```bash
# Resolve tao_toolkit.data_services → concrete nvcr.io/... URI from versions.yaml
DS_IMAGE=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")
echo "DS_IMAGE=$DS_IMAGE"

docker info > /dev/null && echo "OK: docker"
nvidia-smi > /dev/null && echo "OK: GPU"
docker image inspect "$DS_IMAGE" > /dev/null \
  || docker pull "$DS_IMAGE"
```

`TAO_SKILL_BANK_PATH` is usually exported by the installed skill bank. If it is unset, point it at the skill-bank repo root before resolving.

A GPU is required (the same image is used across the AOI loop and other actions assume CUDA is present). Aborting early on a GPU-less host saves a confusing late error.

**Path mounting.** Every host path the container reads or writes — `inference.csv`, the train YAML, the dataset image root, and the output dir — must be bind-mounted. The simplest pattern is to mount the workspace root with **identical paths** inside and outside the container so absolute paths in args resolve the same on both sides:

```bash
WORKSPACE=<absolute path that contains inference.csv, train YAML, dataset images, and the output dir>
DOCKER="docker run --gpus all --rm --ipc=host -v $WORKSPACE:$WORKSPACE -w $WORKSPACE $DS_IMAGE"
```

**Do not pass `--user $(id -u):$(id -g)`.** The container imports `transformers` at startup, which lazy-loads a module path that ends up in `getpass.getuser()` → `pwd.getpwuid(os.getuid())`. With a non-root host UID and no matching `/etc/passwd` entry inside the container, this raises `KeyError: 'getpwuid(): uid not found: <uid>'` *before* `gap_analysis` runs. Drop the flag and `chown` the outputs back to the host UID afterwards if you need them host-writable:

```bash
docker run --rm -v "$WORKSPACE:/w" alpine chown -R $(id -u):$(id -g) /w/<results_subdir>
```

If `inference.csv`, the train YAML, and the dataset images live in different roots, pass multiple `-v` flags — but every absolute path you pass in args must resolve inside the container.

**`-e <spec>` is required, not optional.** Older revisions of this doc described `-e <spec>` as a convenience and Hydra CLI overrides as sufficient on their own. In current `tao_toolkit.data_services` images (verified on `7.0.0-rc-180-multiarch` and later) the entrypoint hard-requires the flag and exits with `ValueError: The subtask vcn_aoi requires the following argument: -e/--experiment_spec_file` before parsing any CLI override. Always pass a minimal spec file — even a thin YAML containing just `min_recall` / `top_k_per_label` / `threshold` is enough; the CLI overrides win at merge time. The `## Reference invocation` block writes this file once per run.
