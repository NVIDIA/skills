# Changelog

All notable changes to `deepstream-import-vision-model` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and the skill uses
[Semantic Versioning](https://semver.org/).

## [1.5.2] — 2026-08-05

### Changed
- Added `scripts/model/resolve-engine.sh` so Steps 6–7 and Step 8 share one engine
  resolver instead of each inlining the same glob, empty check, and `MAX_BS` parse.

### Fixed
- Six defects found running the skill end-to-end on an H100 rather than by review.

## [1.5.1] — 2026-08-04

### Changed
- De-duplicated ONNX label extraction into a single helper shared by the HuggingFace and
  NGC routes.

## [1.5.0] — 2026-08-04

### Fixed
- Cleared the HIGH defects reported by SkillCritic across both DeepStream skills.

## [1.4.3] — 2026-08-04

### Added
- The missing `## Examples` section, with three concrete invocations: the default
  end-to-end run, a SafeTensors export showing the real dynamo → TorchScript fallback,
  and a revision-pinned build.

## [1.4.2] — 2026-08-04

### Fixed
- Cleared the remaining Tier-1 Code Risk Analysis findings, including B615
  `huggingface_unsafe_download` — every `from_pretrained` / `snapshot_download` call now
  pins a revision.

## [1.4.1] — 2026-08-04

### Fixed
- Neutralised the NGC images' global `PIP_CONSTRAINT`, whose tested pins conflicted with
  this skill's dependency set and made pip fail with `ResolutionImpossible`.
- Repaired transformers 5.x breakage in the export path.

## [1.4.0] — 2026-08-04

### Changed
- **Replaced `optimum-cli` with `torch.onnx.export`.** `optimum[exporters]` pinned
  transformers below 4.54.0, but the two HIGH RCE advisories (GHSA-29pf-2h5f-8g72,
  GHSA-fgcw-684q-jj6r) are only fixed in 5.3.0 and 5.5.0, so the pin was unfixable while
  optimum stayed. optimum 2.1.0 had also dropped the `onnx` subcommand, making that path a
  dead end regardless. The exporter now uses the dynamo backend with a TorchScript
  fallback and verifies the batch dimension stayed dynamic.

## [1.3.6] — 2026-08-04

### Fixed
- Cleared the NVSkills-Eval Tier-1 high-risk and Tier-2 findings that blocked the content
  gate, including the Agent Snooping findings in `install.sh`.

## [1.3.5] — 2026-07-29

### Added
- Explicit model-choice intake: the skill now always offers the validated default model
  and a custom object-detection model, and never silently substitutes one.

## [1.3.4] — 2026-07-28

### Fixed
- Updated the full workflow and container references to DeepStream 9.1 and CUDA 13.2.
- Made Bash and PowerShell reinstallation safe when source and destination are identical.
- Reconciled the README, phase references, tests, and eval expectations with the consolidated single-skill architecture.
- Pinned the Python dependency stack used by the in-container setup.
- Added standard Codex UI metadata and repository-required SPDX identifiers.

## [1.3.3] — 2026-07-20

### Fixed
- **`bc` dependency removed — timing/throughput math silently returned empty.** The DeepStream
  container has no `bc`, so every `$(echo "$A - $B" | bc)` produced an empty string with no error,
  leaving all pipeline-timing (and some throughput) values blank. Installing `bc` via `setup.sh`
  would not help — apt installs do not persist across the ephemeral `--rm` phase containers. All
  reference-doc timing now uses `python3 -c "print(round(...))"`; the helper scripts
  (`benchmark-ds.sh`, `ds-sweep.sh`, `benchmark-trtexec.sh`) use `awk "BEGIN{printf ...}"`. Both
  `python3` and `awk` are always present in the image.
- **Step 8 PDF generation failed on a fresh run.** `setup.sh` installs `wkhtmltopdf` via apt inside
  an ephemeral `--rm` container, so the binary is gone by the time Step 8 runs in a later container
  (only the `/work`-mounted venv persists). `scripts/report/md-to-html-pdf.py` now self-heals via an
  `ensure_wkhtmltopdf()` helper that installs it if missing before rendering the PDF. The HTML
  (charts base64-inlined) was already unaffected.

### Changed
- **Real-time stream selection now converges instead of halving.** When DS Run 2 came in marginally
  under 30 fps/stream, the old fallback *halved* RT_STREAMS (e.g. 38 streams @ 29.6 fps → 19),
  discarding ~half the GPU's real capacity and reporting a misleadingly low real-time count. Step 7
  now recomputes the target from the measured throughput (`floor(TOTAL_FPS_RUN2 / 30)`) and steps
  down one stream at a time, landing on the true ceiling (e.g. 37) in 1–2 short retries.

## [1.3.2] — 2026-07-17

### Removed
- **`.claude-plugin/plugin.json`** — the skill now ships as a plain skill (like the sibling
  `deepstream-eval-and-finetune`), consistent with how it is installed by `install.sh` (whole-directory
  copy into `.claude/skills/` and `.cursor/skills/`) and used both standalone and bundled with other
  skills. The manifest was not referenced by `install.sh` and the skill was not registered in the repo
  marketplace, so removal has no effect on standalone or bundled use. This also lets NVCARPS nv-base
  classify the directory as `Type: skill` and run its Tier-3 live agent-eval (producing `BENCHMARK.md` +
  an `AGENT_EVAL` result), which the content gate requires and which the `Type: plugin` path skipped.
  To publish it later as a standalone marketplace plugin, re-add the manifest and run the signed
  marketplace flow.

## [1.3.1] — 2026-07-16

### Added
- **`install.ps1`** — native-Windows (PowerShell) installer twin of `install.sh`, with the identical
  sequence and flags (`-Target`=`--target`, `-NoCursor`=`--no-cursor`, `-DryRun`=`--dry-run`). Copies
  the skill into `<project>\.claude\skills\` (and `.cursor\skills\`). PowerShell 5.1+ compatible.

### Changed
- `.gitattributes` forces LF on `*.ps1`; `references/windows.md` install note now points to `install.ps1`.

## [1.3.0] — 2026-07-16

### Changed
- **Runs entirely through Docker — no host packages.** Every step (venv/ONNX export, TensorRT engine
  build, nvinfer parser compile, DeepStream run, PDF report) now executes INSIDE the DeepStream
  container. The host needs only Docker + the NVIDIA driver, so the skill runs identically on Linux,
  **Windows** (Docker Desktop + WSL2 backend), and macOS.
- Removed the host-native toolchain assumptions (host `trtexec`/`nvidia-smi`/`dpkg`/`make`/host venv/
  `apt-get`); `wkhtmltopdf` + the export venv (`build/.venv_optimum`) are provisioned in-container by
  the new `setup.sh`.
- **Reversed the "always build engines on the host" guidance** — build and run now share one image, so
  there is no TensorRT build-vs-runtime version skew (the exact failure the old rule tried to avoid).
- `install.sh` now installs the **whole self-contained skill dir** (SKILL.md + references + scripts +
  setup.sh) into `.claude/skills/…`, dropping the separate `scripts/` tree and the `ln -sf` symlink path.

### Added
- `setup.sh` (in-container bootstrap: venv + deps + wkhtmltopdf), `scripts/preflight.sh` (GPU + venv +
  trtexec, with container-mode), `scripts/requirements.txt`, `scripts/dsrun.sh` (docker wrapper),
  `.gitattributes` (LF), and `references/windows.md` (cross-platform runbook).

### Fixed
- `scripts/model/safetensors-to-onnx.sh` no longer runs `python3 -m venv` (fails on the container
  python, which lacks ensurepip) — it reuses the virtualenv built by `setup.sh`.

## [1.2.2] — 2026-05-19

### Changed
- **Skill renamed** from `deepstream-byovm` to `deepstream-import-vision-model` across
  all files: `name:` in `SKILL.md` and `.claude-plugin/plugin.json`, package directory
  (`team-skills/deepstream-sdk/deepstream-import-vision-model/`), installed skill
  directories (`.claude/skills/deepstream-import-vision-model`,
  `.cursor/skills/deepstream-import-vision-model`), runtime scripts path
  (`scripts/deepstream-import-vision-model/`), invocation hints, eval prompts, tag
  (`byovm` → `import-vision-model`), README/title (`DS BYOVM` →
  `DeepStream Import Vision Model`), and cross-skill references in
  `team-skills/deepstream-sdk/README.md`,
  `team-skills/deepstream-sdk/deepstream-profile-pipeline/SKILL.md`, and
  `team-skills/deepstream-sdk/deepstream-profile-pipeline/README.md`. Body content
  (SKILL.md sections, `references/*.md`, and 5 differing scripts) was also resynced
  with the upstream `ds-copilot/skills/deepstream-import-vision-model` source
- **Encoder fallback**: replaced `x264enc` fallback with `theoraenc + oggmux` (LGPL,
  outputs `.ogv`). `x264enc` and `openh264enc` are now prohibited (Rule 10). When
  neither NVENC nor `theoraenc`/`oggmux` is available, single-stream capture is
  skipped gracefully (`DS_SINGLE_STREAM_MODE=skipped`)
- **Video source**: enforced `sample_720p.mp4` (1280×720) as the mandatory default;
  custom paths only via explicit `DS_VIDEO` (Rule 11)
- **Performance measurement**: switched DS multi-stream benchmark from
  `gst-launch-1.0 ! fpsdisplaysink` (parsing `Current FPS:`) to `deepstream-app -c …
  enable-perf-measurement=1` (parsing `**PERF:` log lines) via the new
  `scripts/deepstream/ds-perf-run.sh` wrapper. Removes runtime
  dependency on `gstreamer1.0-plugins-bad`
- **Media probing**: replaced `ffprobe` / `gst-discoverer` calls in
  `benchmark-ds.sh` and `ds-sweep.sh` with `mediainfo` (with safe fallbacks)
- **Pipeline NVENC primary**: switched `nvvideoconvert` output format from `I420`
  to `NV12` ahead of `nvv4l2h264enc`
- **Puppeteer sandbox**: split into two vetted configs — `mermaid-puppeteer.json`
  (sandboxed; non-root) and `mermaid-puppeteer-root.json` (sandbox disabled; only
  selected when `uid == 0`). `render-mermaid-for-pdf.py` auto-selects the right
  one and refuses any user-supplied config that does not resolve to one of these
  two shipped files (blocks `--remote-debugging-port`, `--load-extension`, etc.)
- **`benchmark-trtexec.sh` interface**: replaced fixed `b1 b16 b32 b64` positional
  args with variadic `<bs:engine> [<bs:engine> …] [duration]`
- **`ds-sweep.sh`**: input shape and tensor name are now derived dynamically via
  `inspect-onnx.py` instead of hardcoded `inputs` + `640×640` (fixes YOLOv8 /
  RT-DETR / DETR / non-YOLOX models). Power-law batch-size prediction is guarded
  against α≈0 (flat curves)
- **Frame extraction**: `extract-frame.sh` now auto-detects `.mp4` vs `.ogv` and
  routes through the matching demux+decoder chain
- **Custom parser filenames**: introduced `MODEL_NAME_SAFE = tr -c 'A-Za-z0-9' '_'`
  for `.cpp`/`.so` filenames so models like `rtdetr-l` produce a consistent
  `libnvdsinfer_rtdetr_l_parser.so`
- **nvinfer config**: moved `cluster-mode` inline `#` comments to their own
  lines in both heredocs (GKeyFile rejects inline `#`)
- **`make-static-batch-onnx.py`**: use `onnx.numpy_helper.to_array` /
  `from_array` for Reshape initializer patching (the old raw-bytes path silently
  skipped `int64_data` initializers, leaving `batch=1` baked in)
- **Report verification**: replaced `>500 KB` heuristic with deterministic
  `grep -o 'data:image/png' benchmark_report.html | wc -l == 5`
- **System tools**: pre-flight now installs `mediainfo` and checks for
  `deepstream-app` (instead of `gstreamer1.0-plugins-bad`)

### Added
- `scripts/deepstream/ds-perf-run.sh` — wraps `deepstream-app` with
  `enable-perf-measurement=1`, emits `**PERF:` log lines for the report parser
- `scripts/report/mermaid-puppeteer-root.json` — vetted root-only Puppeteer config
- New SKILL.md table rows: `ds-perf-run.sh`, `md-to-pdf.sh`,
  `mermaid-puppeteer-root.json`

### Fixed
- `ds-kitti-dump.sh`: added `set -euo pipefail`, replaced manual `rm -f` with
  trap-based cleanup, guarded `timeout` pipeline with `set +o pipefail` to
  preserve `PIPESTATUS`
- `safetensors-to-onnx.sh`: added missing `set -euo pipefail`
- `generate-benchmark-charts.py`: removed unused `import math`

## [1.2.1] — 2026-04-24

### Changed
- **Skill renamed** from `ds-byovm` to `deepstream-byovm` across all files:
  `name:` in `SKILL.md` and `plugin.json`, installed skill directory
  (`.claude/skills/deepstream-byovm`), runtime scripts path
  (`scripts/deepstream-byovm/`), invocation hints, eval prompts, and
  all cross-references in `references/*.md`

## [1.2.0] — 2026-04-24

### Changed
- **References pattern**: removed 4 standalone sub-skills (`nv-model-acquire`,
  `nv-engine-build`, `ds-run-pipeline`, `nv-byovm-report`); their content is now
  in `skills/ds-byovm/references/` (4 .md files) matching the ds-copilot
  `deepstream-dev` convention of single skill + reference documents
- **Single skill dir**: `SKILL.md` moved from package root into
  `skills/ds-byovm/SKILL.md` (lean ~170 lines); root `SKILL.md` removed
- **plugin.json**: `"skills": "./"` → `"skills": "skills/ds-byovm/"` to point
  at the skill directory instead of the package root
- **install.sh**: creates one symlink (`skills/ds-byovm/` → `.claude/skills/ds-byovm`
  and `.cursor/skills/ds-byovm`) instead of 5; no sub-skill symlinks
- **Installed structure** is now:
  ```text
  .claude/skills/ds-byovm/
    SKILL.md
    references/
      model-acquire.md
      engine-build.md
      pipeline-run.md
      report-generation.md
  scripts/ds-byovm/   (19 scripts, unchanged)
  ```
- **Tests**: updated install dry-run assertions for single-skill structure;
  sub-skill name assertions removed; `assertNotIn` for sub-skill names added

## [1.1.0] — 2026-04-24

### Changed
- **Skill-only architecture**: removed `agents/deepstream-sdk/ds-byovm.md`;
  top-level `SKILL.md` now serves both Claude Code and Cursor (Cursor does not
  support agents — skill is the correct primitive for cross-tool compatibility)
- **Sub-skills renamed** with `nv-`/`ds-` prefix for namespace clarity:
  - `hf-model-acquire` → `nv-model-acquire`
  - `trt-engine-build` → `nv-engine-build`
  - `ds-integration` → `ds-run-pipeline`
  - `benchmark-report` → `nv-byovm-report`
- **SKILL.md enhanced** (version 1.0.1 → 1.1.0): merged pre-flight checks,
  mandatory model folder structure, engine naming convention, run budget table,
  pipeline timing pattern, and report output convention from the removed agent doc
- **install.sh updated**: removed agent installation block, updated symlink
  targets to new sub-skill names; invocation hints say "skill" not "agent"
- **README.md updated**: skill-only usage section for Claude Code + Cursor;
  sub-skill standalone invocation documented; all agent references removed
- **evals.json**: all prompts and assertion text updated from "agent" to "skill"
- **Tests expanded**: dry-run test now asserts all 5 skill names, Cursor skills
  presence, and absence of `.claude/agents/` directory
- **Shebang consistency**: all shell scripts use `#!/usr/bin/env bash` for
  portability across container images and macOS environments
- **Local `.gitignore`**: added skill-level `.gitignore` for portability when
  placed in repos that do not inherit team-mind-hub's root `.gitignore`

## [1.0.1] — 2026-04-23

### Fixed
- `ds-integration` Step 6g KITTI dump produced zero detection files.
  `gie-kitti-output-dir` is a `deepstream-app` `[application]` key — it is
  not read by `nvinfer`, so appending it to the nvinfer config and running a
  `gst-launch-1.0 ... nvinfer ...` pipeline silently wrote no files.
  Step 6g now invokes `scripts/ds-byovm/deepstream/ds-kitti-dump.sh`, which
  wraps `deepstream-app` with the correct `[application]` section.

### Changed
- `hf-model-acquire` Step 2b now uses a **single shared** `build/.venv_optimum`
  for SafeTensors → ONNX export across all models, matching what
  `scripts/ds-byovm/model/safetensors-to-onnx.sh` already does. The previous
  prose created a fresh `build/.venv_$MODEL_NAME` per model, which re-installed
  `optimum`/`transformers`/`torch` every run (~minutes + GBs wasted). New
  models that need extra packages (`timm` for DETR, `onnxsim`, etc.) should
  `pip install` into the shared venv. `cleanup.sh` still removes any legacy
  per-model venvs for backward compatibility, and explicitly preserves the
  shared `build/.venv_optimum`.

## [1.0.0] — 2026-04-22

### Added
- Initial release of the DeepStream Bring Your Own Vision Model (BYOVM) skill
- End-to-end pipeline: HuggingFace / NGC model → ONNX → TensorRT engine → DeepStream → benchmark report
- Four orchestrated sub-skills under `skills/`:
  - `hf-model-acquire` — model download and format routing (ONNX vs SafeTensors)
  - `trt-engine-build` — dynamic TRT engine build + `trtexec` benchmarks
  - `ds-integration` — custom `nvinfer` parser, single-stream + multi-stream DS runs
  - `benchmark-report` — 5-chart Markdown → HTML → PDF report
- Runtime scripts under `scripts/`:
  - `model/` — HF/NGC list + download helpers, ONNX inspection, SafeTensors → ONNX export, scoped cleanup
  - `engine/` — `trtexec` benchmark helper
  - `deepstream/` — single-stream, sweep, KITTI dump, frame extraction helpers
  - `report/` — chart generation, Mermaid → PNG, Markdown → HTML → PDF
- Installer (`install.sh`) with validated `--target` and dry-run mode
- Declared `permissions:` block in `SKILL.md` frontmatter: tool allowlist, MCP
  scope, network egress allowlist (`huggingface.co`, `api.ngc.nvidia.com`,
  `api-inference.huggingface.co`), and filesystem read/write scoping
- Test suite (`tests/test_hardened_scripts.py`) covering input validation for
  all hardened shell scripts and the PDF image-embedding helper (24 tests)

### Security
- All shell scripts validate inputs against `^[A-Za-z0-9._/-]+$` or tighter
  before touching filesystem or network
- `curl` invocations pinned to HTTPS + TLSv1.2 with bounded timeouts; optional
  `$HF_TOKEN` honored for gated HuggingFace repos
- `install.sh` rejects `--target` values that are empty, `/`, contain `..`,
  or don't exist; destructive `rm -rf` is scoped to paths under `$TARGET`
- `scripts/report/md-to-html-pdf.py` base64-inlines images before rendering;
  `wkhtmltopdf` runs without `--enable-local-file-access`
- `scripts/report/render-mermaid-for-pdf.py` refuses user-supplied Puppeteer
  configs; always uses the vetted `mermaid-puppeteer.json` shipped with the skill

### Known limitations
- Nested sub-skills under `skills/` surface a low-severity schema warning
  from some scanners; kept in place because the agent file preloads them by
  name and the installer symlinks them into `.claude/skills/` at the target
- Engine build time depends on GPU, ONNX complexity, and requested batch size;
  the skill retries on OOM by halving batch size but gives up after reaching 1
