# Troubleshooting

Symptoms an agent is likely to hit while routing, and what to do about
them. Anything specific to a sibling skill's own commands is covered in
that skill, not here.

| Error / symptom | Likely cause | Solution |
|-----------------|--------------|----------|
| `nurec-skills` clone missing or empty | Upstream not fetched yet | Walk the local lookup order in the router's "Locate and fetch the upstream skills" section, then ask consent and run the clone block |
| `test -f .../.agents/skills/SKILL.md` fails | Wrong upstream path — the index lives at `skills/nurec-index/` | Use the `skills/nurec-index/` path (or the `.agents/skills/` symlink alias) |
| `403`/`401` pulling `nvidia/PhysicalAI-*` from HF | Gated license not accepted, or `HF_TOKEN` unset / wrong scope | Accept the gated license on Hugging Face, then `hf auth login` with a token that has `read` access |
| `denied: requested access to the resource is denied` from `nvcr.io/nvidia/nre/*` | Missing or expired NGC key | `docker login nvcr.io` with `$oauthtoken` / `${NGC_CLI_API_KEY:-$NGC_API_KEY}`; rotate at `org.ngc.nvidia.com/setup/api-key` if needed |
| `manifest unknown` / `not found` pulling an NRE image | Pulling the legacy un-suffixed name, or a release *name* used as a tag (e.g. `:release_26.04` — the GA channel publishes `26.04.01`, `26.04`, `26`, `latest`) | Pull the GA names with a published tag: `nvcr.io/nvidia/nre/nre-ga:26.04.01` (or `:latest`) and `nvcr.io/nvidia/nre/nre-tools-ga:latest`; list tags on the NGC catalog page for `nvidia/nre/nre-ga` |
| `--renderer` or `export-custom-rig-trajectory` rejected as unknown | Cached image is older than `26.04` / `26.03` | Pull a `26.04+` GA image; `--image-format jpeg` works on every family, so don't fall back to PNG |
| NRE refuses to load a clip ("not valid NCore V4") | Recording was not converted | Run the `ncore` skill before invoking `nre` |
| `serve-grpc` cold-start latency dominates a Python loop | One-shot Docker invocation per render | Use the `nre` warm `serve-grpc` + thin Python client (`batch_render_rgb`) recipe; the warm fast path needs a `26.04+` image |
| Output files are owned by `root` after a `docker run` | `-u $(id -u):$(id -g)` was missing | `sudo chown -R "$(id -u):$(id -g)" <output_dir>`; add the `-u` flag next time |
| Frames have ghosting / floaters / flicker after rendering | Inline cleanup not enabled | Re-render with `nre --enable-difix`, or post-process with `nurec-fixer` (DiffusionHarmonizer) |
| Stale names (`ncore-data-conversion`, `nvidia/Fixer`, `nvidia/DiffusionHarmonizer` weights) in agent output | Out-of-date cached skill | Update to `ncore` and `nurec-fixer`; the model now lives at `nvidia/Harmonizer` — see [`maintenance.md`](maintenance.md) |
| Bash anti-pattern `${HF_TOKEN:+yes}${HF_TOKEN:-no}` echoed token value | Misuse of bash parameter expansion | Rotate the token; use `hf auth whoami` or length-only checks (see [`secrets-handling.md`](secrets-handling.md)) |
