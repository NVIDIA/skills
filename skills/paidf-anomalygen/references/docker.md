# Running in Docker — container launch, mounts & permissions

The `paidf-anomalygen` image runs as a **non-root baked-in user**
(`USER anomalygen`, `uid=10000`), independent of your host uid. Docker does
**not** remap uids on bind mounts by default, so a host directory owned by
your uid is **not writable by uid 10000**. The container fails the instant it
tries to *create* a file/subdir inside such a mount — kernel checks write+exec
on the **parent** dir. This surfaces early at Phase 0 (HF writes
`stored_tokens` into `HF_HOME`) and again at Phase 2 (AMP creates
`<name>/amp/<sample>/…` under `ag_inference/`). Any single host-owned ancestor
in a runtime-created subtree breaks creation.

**Recommended: run as your host uid (`--user`).** Outputs end up owned by you,
no host `chmod` needed. Two companions are mandatory for *this* image:

```bash
WORK=/abs/path/to/run        # holds home/, checkpoints/, results/, ag_inference/, ag_configs/
mkdir -p "$WORK"/{home,checkpoints,results,ag_inference,ag_configs}
docker run -d --name agrun --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \   `# ① uid must resolve to a name`\
  -e HOME=/work/home -e HF_HOME=/work/home/hf \                 `# ② redirect all caches to a writable mount`\
  -e XDG_CACHE_HOME=/work/home/.cache -e TRITON_CACHE_DIR=/work/home/.triton \
  -e TORCHINDUCTOR_CACHE_DIR=/work/home/.inductor -e MPLCONFIGDIR=/work/home/.mpl \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$WORK/home:/work/home" \
  -v "$WORK/checkpoints:/workspace/paidf-anomalygen/checkpoints" \
  -v "$WORK/results:/workspace/paidf-anomalygen/results" \
  -v "$WORK/ag_inference:/workspace/paidf-anomalygen/ag_inference" \
  -v "$WORK/ag_configs:/workspace/paidf-anomalygen/ag_configs" \
  <image> sleep infinity
```

Both companions are load-bearing — verified by smoke test:

| Flag | Omit it and… |
|---|---|
| `--user $(id -u):$(id -g)` | container stays uid 10000 → can't write host mounts (the original error). |
| `-v /etc/passwd:/etc/passwd:ro` (+`/etc/group`) | your uid has no name in the image → **Phase 4 eval crashes** in `torch.compile` (`getpass.getuser()` → `KeyError: getpwuid(): uid not found`). |
| `-e HOME=…` + cache vars | real `HOME` (`/home/anomalygen`) is uid 10000's → HF / triton / inductor / matplotlib caches hit a fresh `EACCES`. |

**Fail-fast preflight** (run before Phase 0 — catches the mismatch in seconds
instead of mid-Phase 2):

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$WORK/ag_inference:/mnt" <image> \
  bash -lc 'mkdir -p /mnt/.wtest/a/b && rm -rf /mnt/.wtest && echo OK \
            || { echo "mount not writable by uid $(id -u)"; exit 1; }'
```

**Fallback (only when you cannot use `--user`** — e.g. shared host where output
files *must* stay owned by uid 10000): keep the container at uid 10000 and make
each **mount root** writable by it — `sudo chown -R 10000:10000 "$WORK"` (or
`chmod -R 777 "$WORK"`). Apply it to the mount **root**, never just leaf output
dirs: the pipeline creates deep subtrees (`amp/<sample>/`, `rounds/round_NN/`,
`regens/regen_NN/`) at runtime, and one host-owned ancestor anywhere breaks it.
