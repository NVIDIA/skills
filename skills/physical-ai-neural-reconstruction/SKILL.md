---
name: physical-ai-neural-reconstruction
description: "Router for NVIDIA NuRec/NRE: USDZ rendering, NCore conversion, 3DGS, gRPC sensor sim, carline adaptation, PhysicalAI HF datasets. Do NOT use for SimReady or infra setup."
license: Apache-2.0
version: "0.4.0"
tools:
  - Read
  - Shell
compatibility: >-
  Router skill; downstream sibling skills require Linux x86_64, an
  NVIDIA GPU (Ampere+, CUDA 12.8, >= 24 GB VRAM), Docker >= 23.0.1,
  NVIDIA Container Toolkit >= 1.13.5, an NGC API key, a Hugging Face
  token with the relevant gated licenses accepted, Python 3.10+, and
  `huggingface_hub`. Optional: CARLA / Isaac Sim 5.1 / AlpaSim for
  simulator integration over `serve-grpc`.
metadata:
  author: NVIDIA Physical AI
  tags:
    - physical-ai
    - nurec
    - neural-reconstruction
  upstream:
    sibling_skills:
      - name: physical-ai-datasets
        skill_repo: https://github.com/NVIDIA/nurec
        skill_path: skills/physical-ai-datasets/
        upstream: https://huggingface.co/nvidia
      - name: ncore-data-conversion
        skill_repo: https://github.com/NVIDIA/ncore
        skill_path: skills/ncore-data-conversion/
        upstream: https://github.com/NVIDIA/ncore
      - name: nurec
        skill_repo: https://github.com/NVIDIA/nurec
        skill_path: skills/nurec/
        upstream: nvcr.io/nvidia/nre/nre-ga
        tools_container: nvcr.io/nvidia/nre/nre-tools-ga
        release_tag: "26.04"  # NRE release; NGC image tags are 26.04.01 / 26.04 / 26 / latest (not `release_26.04`)
      - name: asset-harvester
        skill_repo: https://github.com/NVIDIA/asset-harvester
        skill_path: skills/asset-harvester/
        upstream: https://github.com/NVIDIA/asset-harvester
        hf_model: https://huggingface.co/nvidia/asset-harvester
      - name: harmonizer
        skill_repo: https://github.com/NVIDIA/harmonizer
        skill_path: skills/harmonizer/
        upstream: https://github.com/NVIDIA/harmonizer
        hf_model: https://huggingface.co/nvidia/Harmonizer
        container: nvcr.io/nvidia/pytorch:25.10-py3
---

# Physical AI Neural Reconstruction (NuRec) Router

## Purpose

This is a **thin router** for NVIDIA Neural Reconstruction (NuRec)
requests. It points at sibling skills in their product repos
(`physical-ai-datasets`, `ncore-data-conversion`, `nurec`,
`asset-harvester`, `harmonizer`). Use this skill to:

- Identify which sibling skill answers a NuRec question.
- Name each sibling's product repo and path so it can be fetched
  with consent if it is not already installed.
- Order multi-step NuRec workflows (data → conversion → train →
  render → cleanup) before opening the sibling recipe.

The canonical recipes (training, rendering, data conversion, dataset
downloads, object harvesting, frame cleanup) live in the upstream
sibling skills. **Never copy or reconstruct their commands here.**

**Do NOT use this skill for:**

- SimReady packaging of CAD or source meshes → use
  `omniverse-cad-to-simready`.
- Generic USD performance tuning unrelated to NuRec → use
  `omniverse-usd-performance-tuning`.
- AKS / OSMO / NIM Operator infrastructure setup → use
  `physical-ai-infrastructure-setup-and-resilient-scaling`.

## When to Use

Read this skill **first** whenever a user mentions any of:

`nurec`, `nurec router`, `nurec index`, `neural reconstruction`,
`neural reconstruction engine`, `NRE`, `3DGUT`, `3DGRT`, `USDZ`,
`NCore V4`, `sensorsim`, `sensor sim`, `novel view synthesis`,
`PhysicalAI-Autonomous-Vehicles-NuRec`, `PhysicalAI-Robotics-NuRec`,
`PhysicalAI-NuRec-PPISP`, `Cosmos-Drive-Dreams`, `asset harvester`,
`nurec fixer`, `DiffusionHarmonizer`, `harmonizer`, `difix`,
`difix3d`, `carline adaptation`, `serve-grpc`, `render-grpc`,
`warm serve-grpc`, `nre thin client`, `batch_render_rgb`,
`nurec teardown`, "where do I start with NuRec", "which NuRec skill
should I use for X?".

Decide which upstream sibling skill answers the question, fetch it
(see [Locate and fetch the upstream skills](#locate-and-fetch-the-upstream-skills)),
then follow that skill's body.

## Prerequisites

The router itself has no runtime prerequisites beyond `git` for
fetching the upstream. Downstream sibling skills need Linux x86_64, an
NVIDIA GPU (Ampere+, CUDA 12.8, >= 24 GB VRAM), Docker plus the NVIDIA
Container Toolkit, an NGC API key, a Hugging Face token with the
relevant gated licenses already accepted, and Python 3.10+.

Full per-skill detail — driver floors, container names, key resolution
order, which Hugging Face assets are gated, and how to verify secrets
without echoing them — is in
[`references/prerequisites.md`](references/prerequisites.md). Prefer
each sibling's `scripts/validate_setup.py` over hand-written checks.

## What is NuRec?

**NuRec** (NVIDIA Omniverse Neural Reconstruction) turns camera,
LiDAR, radar, or stereo recordings — typically from a self-driving car
or a robot — into a 3D scene that can be re-rendered from any
viewpoint. A typical project runs in three stages: get the input
(convert a recording with `ncore-data-conversion`, or download a
ready-made dataset with `physical-ai-datasets`), train the
reconstruction (`nurec`, which emits a USDZ), then render new views
(`nurec`). Projects that only want to *use* a scene NVIDIA already
published skip the training stage.

Background on the vocabulary — NRE vs NuRec, USDZ, NCore V4, 3DGUT /
3DGRT — is in
[`references/what-is-nurec.md`](references/what-is-nurec.md).

## Pick a skill

Match the user's goal in the left column and open the named upstream
skill on the right. Arrows mean "do these in order".

| I want to… | Upstream skill |
|------------|----------------|
| Find or download a NuRec dataset NVIDIA has published | `physical-ai-datasets` |
| Convert my own camera / LiDAR / radar / depth / stereo recording into NCore V4 | `ncore-data-conversion` |
| Write a new converter for an unsupported sensor setup (drone, RGB-D, ROS 2 bag, COLMAP, ScanNet++) | `ncore-data-conversion` |
| Train a 3D reconstruction from an NCore clip | `ncore-data-conversion` → `nurec` |
| Generate the extra inputs NRE needs (segmentation masks, depth, ego mask, DINOv2, LiDAR-seg visibility) | `nurec` (uses the `nre-tools-ga` container) |
| Render a USDZ along the original camera positions | `nurec` |
| Render at full resolution / highest quality | `nurec` (see "Quality presets") |
| Render along a shifted trajectory (e.g. car moved 3 m left) | `nurec` |
| Adapt an existing USDZ to an augmented target-vehicle rig (carline adaptation) | `nurec` (`export-custom-rig-trajectory` → `render`) → `harmonizer` |
| Render through a server so CARLA / Isaac Sim / AlpaSim / a custom simulator can ask for frames | `nurec` (`serve-grpc`) |
| Render the same USDZ many times back-to-back from Python with minimal per-call latency | `nurec` (warm `serve-grpc` + thin Python client / `batch_render_rgb`) |
| Render LiDAR sweeps (point clouds) from a USDZ | `nurec` (`render-grpc --lidar`) |
| Skip training and just render a NuRec scene NVIDIA already built | `physical-ai-datasets` → `nurec` |
| Skip training and use a pre-built indoor robotics scene | `physical-ai-datasets` → `nurec` (then Isaac Sim 5.1) |
| Extract individual 3D objects (cars, pedestrians) from a driving clip | `asset-harvester` |
| Add, remove, or replace cars / pedestrians in a NuRec scene | `asset-harvester` → `nurec` |
| Clean up or harmonize rendered frames (ghosting, floaters, flicker, lighting/shadows) | `harmonizer`, **or** `--enable-difix` inside `nre` for inline rendering |
| Export the scene as a PLY, mesh, depth maps, ego mask, etc. | `nurec` |
| Upgrade an old USDZ so newer NRE versions load it faster | `nurec` (`upgrade-artifact`) |
| Open a USDZ or PLY in a browser viewer | `nurec` (`viewer` / `ply_viewer`) |
| Measure rendering quality (PSNR, SSIM, LPIPS) against ground truth | `nurec` (`eval-rendering-metrics`) |
| Benchmark different reconstruction methods on the same scenes | `physical-ai-datasets` (`PhysicalAI-NuRec-PPISP`) → `nurec` |
| Train on multiple GPUs or on SLURM | `nurec` |

## Common workflows

Seven end-to-end workflows are documented in
[`references/workflows.md`](references/workflows.md), lettered A–G
by this router:

- **A.** Make a NuRec scene from your own recording.
- **B.** Use a NuRec scene NVIDIA has already trained.
- **C.** Use NuRec for indoor robot simulation.
- **D.** Add, remove, or replace 3D objects in a scene.
- **E.** Clean up rendered frames.
- **F.** Benchmark reconstruction quality.
- **G.** Connect NuRec to a simulator.

Open that file when the user's task spans more than one sibling skill.

## Sibling skills (upstream)

Refer to a sibling by its **name** — that is the portable identifier.
The skill-source column is the product repo and path of that skill.

| Name | Skill source | What it does |
|------|--------------|--------------|
| `physical-ai-datasets` | [`NVIDIA/nurec`](https://github.com/NVIDIA/nurec) → `skills/physical-ai-datasets/` | Catalog and download recipes for every NVIDIA Physical AI dataset on Hugging Face (driving, robotics, manipulation, NuRec scenes, benchmarks). |
| `ncore-data-conversion` | [`NVIDIA/ncore`](https://github.com/NVIDIA/ncore) → `skills/ncore-data-conversion/` | Converts any sensor recording to NCore V4 (the format NRE needs). Also covers writing a new converter. |
| `nurec` | [`NVIDIA/nurec`](https://github.com/NVIDIA/nurec) → `skills/nurec/` | The Neural Reconstruction Engine itself (`nvcr.io/nvidia/nre/nre-ga`, `nvcr.io/nvidia/nre/nre-tools-ga`, NRE 26.04 — image tags `26.04.01` / `26.04` / `latest`). Trains, performs carline adaptation, renders (locally, via warm `serve-grpc` + thin Python client / `batch_render_rgb`, or to an external simulator), exports meshes / point clouds / depth, edits actors, evaluates quality. |
| `asset-harvester` | [`NVIDIA/asset-harvester`](https://github.com/NVIDIA/asset-harvester) → `skills/asset-harvester/` | Open-source Apache-2.0 pipeline (SparseViewDiT + TokenGS) that extracts individual 3D objects from sparse views in a driving clip and saves them as `.ply` Gaussian splats, optionally emitting `metadata.yaml` for the NuRec handoff. |
| `harmonizer` | [`NVIDIA/harmonizer`](https://github.com/NVIDIA/harmonizer) → `skills/harmonizer/` | Standalone NVIDIA **DiffusionHarmonizer** workflow — public successor to the older Fixer / Difix3D+ recipes — that cleans rendered frames, harmonizes inserted actors, evaluates PSNR/LPIPS, and optionally fine-tunes the model. |

For naming overlaps (NRE vs Fixer, ncore-data-conversion vs nurec, AV-NuRec vs
Cosmos-Drive-Dreams, NuRec vs SimReady) see
[`references/mix-ups.md`](references/mix-ups.md).

## Locate and fetch the upstream skills

Select a sibling by its current registered `name` and the
repo/path in the sibling table. Full lookup, consent, and
stale-copy rules are in
[`references/upstream-fetch.md`](references/upstream-fetch.md).

For `ncore-data-conversion`, `nurec`, and `harmonizer`, try the
runtime-local lookup first:

1. `.agents/skills/<name>/SKILL.md` (Cursor, Codex, NemoClaw)
2. `.claude/skills/<name>/SKILL.md` (Claude Code)
3. `.cursor/skills/<name>/SKILL.md` (project-scoped)
4. `~/.cursor/skills/<name>/SKILL.md` (personal skills)

A matching installed definition can be read locally. If it is a
symlink, follow it and use that definition's own companion files.

For `physical-ai-datasets` and `asset-harvester`, generic same-name
discovery cannot distinguish the old bundled copy. Use their
canonical product-repo source instead. A canonical definition
already acquired for this task can be reused.

Do not read sibling skills from a `nurec-skills` checkout: those
copies are retired and may still exist. Use each skill's product-repo
home. In particular, `physical-ai-datasets` and `asset-harvester`
kept their names, so a generic same-name local hit may be the old
copy. Renamed skills may resolve locally under their current names;
never fall back to the retired `ncore`, `nre` or `nurec-fixer`
skill folders.

If the selected current definition is unavailable, follow
[Ask before cloning](references/upstream-fetch.md#ask-before-cloning)
before obtaining it from the mapped product repo. Missing a renamed
local skill never falls back to an old folder or clone. If retrieval
is declined, report the missing skill and stop that handoff.

Read the selected definition before following its workflow. Open its
own references on demand. Companion files (`references/`, `scripts/`,
`assets/`) ship inside **the sibling's own skill directory**. This
catalog router owns its picker and workflow letters; there is no
upstream index to fetch.

## Hard Rules

- Router only — do not duplicate upstream NuRec recipes here. Read
  the upstream sibling skill body before running any mutating command.
- Refer to sibling skills by their `name:` (e.g. `nurec`), not by repo
  path. Folder layouts can change; the name is portable.
- **Never `git clone` a product repo without explicit user consent.**
  Exhaust the local lookup order first (and the same-name exception
  for `physical-ai-datasets` and `asset-harvester`), show the exact
  command, and clone only into a path the user agreed to — never
  silently into `/tmp`. Do not scan broad developer workspaces such
  as `~/Codes` or reuse unrelated old clones. Never fall back to a
  `nurec-skills` checkout.
- Use the GA container channel: `nvcr.io/nvidia/nre/nre-ga` and
  `nvcr.io/nvidia/nre/nre-tools-ga`. The un-suffixed
  `nvcr.io/nvidia/nre/nre` / `nre-tools` names are the legacy
  channel — still valid for cached version pins, but not what a new
  workflow should pull.
- Resolve the NGC key as `${NGC_CLI_API_KEY:-${NGC_API_KEY:-}}` and
  log in with `docker login nvcr.io --username '$oauthtoken'
  --password-stdin`. Never echo a key.
- `physical-ai-datasets` covers gated Hugging Face datasets. Do not
  bypass dataset license terms; the user must accept the
  `PhysicalAI-*` gated licenses on Hugging Face and provide a token
  before downloading.
- Asset Harvester runs **before** packaging into a USDZ. Do not call
  `nre`'s `export-external-assets` on hand-rolled `.ply` files unless
  the user explicitly asks to skip Asset Harvester. The owning skill
  for that command is `nurec`.
- For artifact cleanup, prefer the built-in `--enable-difix` path in
  `nre`. Route to the standalone `harmonizer` only when the user
  needs the public code/model card, paired evaluation, fine-tuning,
  or fixes on previously rendered frames.
- Do not invent NRE / NCore / DiffusionHarmonizer commands from
  memory. Re-read the upstream sibling skill — versions move fast
  (NRE 26.04 — pull `nvcr.io/nvidia/nre/nre-ga:26.04.01` or `:26.04`; the release name
  `release_26.04` is not a valid image tag).
- This router does not deploy infrastructure. Route AKS / OSMO /
  NIM Operator setup to
  `physical-ai-infrastructure-setup-and-resilient-scaling`.

## Limitations

- **Router only.** This skill never executes mutating NuRec commands.
  All training, rendering, conversion, and harmonization happens in
  upstream sibling skills.
- **Upstream-pinned.** Sibling recipes live in their product repos:
  `https://github.com/NVIDIA/nurec`, `https://github.com/NVIDIA/ncore`,
  `https://github.com/NVIDIA/harmonizer`, and
  `https://github.com/NVIDIA/asset-harvester`. A renamed skill already
  installed locally may be used without a network refresh; if a fetch
  is needed, follow
  [Ask before cloning](references/upstream-fetch.md#ask-before-cloning).
- **Hand-curated catalogue.** A newly-added upstream sibling is not
  discoverable here until someone edits the tables (see
  [`references/maintenance.md`](references/maintenance.md)).
- **Gated content.** `nvidia/PhysicalAI-*`, `nvidia/Harmonizer`, and
  `nvidia/Cosmos-Predict2-0.6B-Text2Image` require the user to accept
  license terms on Hugging Face first. For `asset-harvester` only its
  optional DINOv3, Llama Guard and SAM 3D Body models are gated.
  The router cannot bypass this.
- **Heavy footprint.** A complete NuRec workflow can leave 150 GB+
  on disk. See [`references/teardown.md`](references/teardown.md).
- **NVIDIA-only stack.** Requires Linux x86_64 plus an NVIDIA GPU and
  the NVIDIA Container Toolkit. aarch64 / AMD / Intel / Apple Silicon
  are not supported.
- **No Omniverse / Isaac Sim integration steps.** Handing a USDZ to
  Isaac Sim 5.1 (workflow C) is documented in the Isaac Sim docs, not
  in the NuRec skill family.
- **Not a SimReady pipeline.** NuRec produces a renderable USDZ from
  a recording; SimReady packaging of CAD or source meshes is a
  different pipeline (see `omniverse-cad-to-simready`).

## Troubleshooting

Routing-level symptoms — a missing upstream clone, gated-asset `403`s,
NGC login failures, `manifest unknown` on an NRE image, stale cached
skill names — are tabulated in the troubleshooting companion file that
ships alongside this skill.
Symptoms specific to a sibling's own commands belong to that sibling's
skill.

## Cross-skill teardown

A complete NuRec workflow can leave **150 GB+** on disk between
container images, model weights, code clones, conda envs, and output
directories. Each sibling skill has its own dedicated `Teardown`
section — read them in the order documented in
[`references/teardown.md`](references/teardown.md) when the user no
longer needs the workflow. Do **not** revoke `NGC_API_KEY` /
`HF_TOKEN` as part of teardown unless they were leaked.

## Keeping this router up to date

Procedure for adding new sibling skills, renames, or upstream URL
changes lives in [`references/maintenance.md`](references/maintenance.md).
This catalog router is authoritative **for the routing taxonomy and
workflow ordering**. Sibling recipes live in their product repos; it
is not authoritative for `asset-harvester`, which is maintained in
<https://github.com/NVIDIA/asset-harvester>.
