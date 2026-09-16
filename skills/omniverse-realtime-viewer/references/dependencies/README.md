# Dependencies

## Triggers

Use this skill for install, setup, dependency verification, package cache,
ovrtx install, OVStage/ovstage install, OVPhysX/ovphysx install, ovstorage install or integration, ovstream
install, ovui install, NVIDIA runtime acquisition, supplemental dependency
documentation, generated local viewer UI, OpenUSD/pxr setup, Warp, NumPy,
React/Vite, WebRTC client packages, Electron SHM packages, Windows setup
prerequisites, or environment troubleshooting for Omniverse Realtime Viewer apps.

This skill is the source of truth for NVIDIA runtime dependency acquisition.
Other skills should point back here instead of repeating package URLs, release
URLs, registry paths, wheel names, artifact locations, or
ovrtx/ovstage/ovphysx/ovui/ovstream repository URLs. This includes the CAE/CFD
visualization libraries (`warp-simdata`, `cae-openusd-plugins`) — their single
acquisition source is the **CAE/CFD Visualization Libraries** section below.

## How To Use

Start here before writing viewer code. Choose the references that match the selected delivery path and load only those details.

| Need | Read |
|---|---|
| NVIDIA runtime dependency source of truth: `ovrtx`, `ovstage`, `ovphysx`, `ovstorage`, `ovui`, `ovstream`, and the `ov-web-rtc` browser client | `nvidia-runtime.md` |
| Baseline setup, cache paths, package matrix, global requirements | `quick-setup.md` |
| `ovrtx` install, renderer plugin paths, GPU validation | `ovrtx.md` |
| `ovstream`, native streaming libraries, WebRTC server setup | `ovstream.md` |
| React/Vite client and WebRTC browser package setup | `frontend.md` |
| Electron + shared-memory local transport dependencies | `electron-shm.md` |
| Local `ovui`, `usd-core`/`pxr` worker isolation, Warp, NumPy | `local-openusd-gpu.md` |
| Environment variables, verification commands, failure index | `environment-validation.md` |

## Path Selection

- For browser streaming, read `nvidia-runtime.md`, `quick-setup.md`, `ovrtx.md`, `ovstream.md`, `frontend.md`, and `environment-validation.md`.
- For lightweight local `ovui` apps, read `nvidia-runtime.md`, `quick-setup.md`, `ovrtx.md`, `local-openusd-gpu.md`, and `environment-validation.md`.
- For Electron + SHM apps, read `nvidia-runtime.md`, `quick-setup.md`, `ovrtx.md`, `electron-shm.md`, `frontend.md`, and `environment-validation.md`.
- For Tauri/Rust or C++ native apps, read `nvidia-runtime.md`, `quick-setup.md`, `ovrtx.md`, and the delivery skill's own build requirements.
- For Windows-native work, also read `windows-native-setup` after the dependency reference that matches the selected path.

## Critical Rules

- Do not guess install commands or package sources. Use `nvidia-runtime.md` for NVIDIA runtime acquisition.
- Do not hard-code ovrtx, ovstage, ovphysx, ovstorage, ovui, or ovstream GitHub repository URLs in downstream
  skills. Use `nvidia-runtime.md` so dependency locations can be
  updated in one place.
- Keep `ovrtx`, `ovstage`, `ovphysx`, `ovui`, `ovui-data-adapters`, and local UI companion packages on compatible revisions.
- Set `OVRTX_SKIP_USD_CHECK=1` before importing or constructing `ovrtx` components where the selected reference requires it.
- Keep `usd-core`/`pxr` work in a query subprocess unless the exact process import path is verified for the selected wheels.
- Keep OVPhysX USD population in a bounded worker unless `PhysX.attach_ovstage(stage, read_ordinal=...)` is verified against the exact installed OVStage/OVPhysX ABI.
- Let the parent viewer runtime own OVRTX, OVStage stage lifetime, ordinals, and publication state; workers exchange DTOs, not live stage handles.
- Do not add browser 3D renderer dependencies as a fallback for missing GPU or `ovrtx` packages.
- For generated browser-streamed viewers, dependency setup is part of completion. Attempt server runtime installation and verification before declaring the app ready unless the user explicitly opts out or the platform is unsupported.
- Treat vendored packages and local caches as setup aids, not redistribution approval.

See also: `ovstage-runtime`, `ovstage-ovrtx-integration`,
`physics-simulation`, `ovrtx-rendering`, `ovui-local-viewer-recipe`,
`local-viewer`, `streaming-server`, `streaming-client`,
`electron-shm-viewer`, `stage-hierarchy`, `windows-native-setup`, and
`cae-cfd-visualization` (which acquires the `warp-simdata` and
`cae-openusd-plugins` libraries through the section below).

## Dependency Freshness And Local Checkouts

Treat installed packages, vendored files, caches, and local repository clones as
implementation inputs, not authoritative current guidance. For an existing app,
its manifest or lockfile is authoritative for pinned versions. For current API
and setup guidance, the upstream repository and package metadata are authoritative.

A local checkout may substitute for upstream guidance only after checking that:

- its `origin` matches the supplemental repository pointer;
- its working tree is clean, or its relevant local changes are understood;
- its current commit and commit date are recorded; and
- its `AGENTS.md`, `skills/`, README, examples, and release notes are present.

Do not refresh a dependency checkout on every prompt. Refresh it, or consult
upstream directly, when the user asks for latest/current/supported behavior, the
task concerns installation or compatibility, a documented API fails against the
installed package, the checkout is stale or dirty, the installed version differs
from the app manifest, or the dependency is pre-release or fast-moving. Use a
30-day age check as a warning threshold, not as an automatic upgrade rule.

If refresh is unavailable, report the commit or package version used and treat
local guidance as provisional. Never infer that an installed package or local
clone is current solely because it is present on disk.

## ovstage Supplemental Source

`ovstage` follows the same supplemental-repository rule as the other NVIDIA
runtime dependencies. Before implementing stage population, runtime data access,
ordinals, write floors, or attached rendering, read the current `ovstage`
repository documentation, `AGENTS.md`, `skills/`, examples, and release notes
through the pointer in `nvidia-runtime.md`. The local ovstage references provide
viewer-specific composition; upstream is authoritative for the current API.

## OVPhysX Supplemental Source

`ovphysx` acquisition and supplemental repository pointers also live in
`nvidia-runtime.md`. Before implementing physics simulation, tensor binding, or
attached-stage behavior, read the OVPhysX package/repo docs plus
`physics-simulation`. Use the bounded worker handoff unless the exact installed
OVStage/OVPhysX ABI has a verified `PhysX.attach_ovstage(stage, read_ordinal=...)` bridge.

## ovstorage Supplemental Source

Before adding cloud-asset discovery, transfer, caching, or resolver behavior,
read the current upstream `AGENTS.md` and relevant `skills/` guidance through
the `NVIDIA-Omniverse/ovstorage` pointer in `nvidia-runtime.md`. Upstream owns
the current storage API; use this package for viewer-specific routing only.

## CAE/CFD Visualization Libraries

This is the **single acquisition source** for the two domain libraries the
`cae-cfd-visualization` reference family depends on. Every doc in that family
points here by role instead of repeating package locations inline. Refer to them
by role, not by URL, so the prose survives a rename or a repository move:

- **the `warp-simdata` GPU operator library** — NVIDIA Warp kernels that build an
  in-memory `Dataset` from renderable arrays and run the visualization operators
  (`element_faces` / `iso_surface` / `slice` / `streamlines` / points /
  voxelization). Imports Warp lazily; requires CUDA. This is the durable path the
  CAE family builds on.
- **the `cae-openusd-plugins` USD reader/schema library** — a read-only
  `SdfFileFormat` plugin plus OmniSci (`Cae*`/`OmniSci*`) schemas that expose
  CGNS / OpenFOAM / VTK / EnSight / FLASH files as composed USD prims with
  data-on-demand field arrays. Optional: it is frequently absent from a working
  venv, so the CAE family always keeps a library-direct reader fallback (see
  `cae-cfd-visualization/cae-data-ingestion.md`).

> **These libraries are public on GitHub.** This section is the single source of
> truth for their acquisition — the family docs point here rather than repeating URLs.

| Library (role) | Acquisition | Guidance |
|---|---|---|
| `warp-simdata` (GPU operator library) | [NVIDIA/warp-simdata](https://github.com/NVIDIA/warp-simdata) | Before setup, read its current `AGENTS.md`, `skills/`, README, and installation guidance. Follow its supported acquisition and compatibility instructions rather than copying commands into this package. |
| `cae-openusd-plugins` (USD reader/schema library) | [NVIDIA-Omniverse/cae-openusd-plugins](https://github.com/NVIDIA-Omniverse/cae-openusd-plugins) | Before setup, read its current `AGENTS.md`, `skills/`, README, build/install, plugin-registration, and compatibility guidance. It is optional; if its schemas are unavailable, use the library-direct readers in `cae-cfd-visualization/cae-data-ingestion.md`. |

Rules:

- Do not duplicate these acquisition URLs or version pins in the
  `cae-cfd-visualization` family docs. Point back to this section.
- This section is the single route to current upstream setup guidance. Do not
  duplicate acquisition commands, version pins, build flags, or plugin paths in
  this package.
- Let each upstream repository define its supported CUDA/Warp or USD/runtime
  compatibility set; validate the selected combination before building an app.
