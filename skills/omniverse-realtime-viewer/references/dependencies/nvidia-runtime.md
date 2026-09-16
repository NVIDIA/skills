# NVIDIA Runtime Dependency Source Of Truth

This file is the single source of truth for NVIDIA runtime dependency
acquisition in this skill package. Downstream skills may name the runtime they use
and document API behavior, but they should not repeat package URLs, release
URLs, workflow artifact links, registry paths, or fallback install locations.

## Primary NVIDIA Dependencies

| Dependency | Acquisition path | Used by | Guidance |
|---|---|---|---|
| `ovrtx` | NVIDIA Python package index | Local and streaming RTX USD rendering | Resolve the latest available package from this location. |
| `ovstage` | PyPI package | Runtime stage substrate shared by renderer, physics, sensors, and app code | Resolve with the selected `ovrtx` package and verify attached-stage flow. |
| `ovphysx` | NVIDIA Python package index | Physics simulation workers and attached-stage experiments | Resolve with the selected `ovstage`/`ovrtx` package set and verify the bridge ABI. |
| `ovstorage` | [NVIDIA-Omniverse/ovstorage](https://github.com/NVIDIA-Omniverse/ovstorage) | Cloud asset discovery, transfer, and cache integration | Start with the upstream `AGENTS.md` and `skills/`; select the current compatible package or project integration from that guidance. |
| `ovui` | PyPI package | Native local UI and server-side/headless overlay UI | Resolve the latest available package from this location. |
| `ovstream` | PyPI package | WebRTC and SHM streaming server/runtime | Resolve the latest available package from this location. |
| `ov-web-rtc client` (`@nvidia/ov-web-rtc`) | NVIDIA npm package | Browser-side WebRTC client for standalone `ovstream` Direct connections | Use the package guidance below. |

## Version Selection Rule

For new generated viewer apps, first inspect each selected dependency repository's current `AGENTS.md`, `skills/` (when present), README and API documentation, examples, release notes, and package metadata. Then install or integrate the latest available compatible `ovrtx`, `ovstage`, `ovphysx` when physics is selected, `ovstorage` when cloud assets are selected, `ovui`, and `ovstream` from the acquisition locations in this file. Do not copy a resolved version number into downstream skills, templates, or setup recipes. Apply the dependency freshness policy in `dependencies/README.md`; do not treat a repository clone, installed package, or cache as current solely because it is present on disk.

If the host project already has a manifest or lockfile with an explicit runtime
pin, respect that pin unless the user asks to update it. If compatibility
requires a pin, keep it in the project manifest with a short reason rather than
in this dependency source of truth.

For generated apps that select `ovui`, resolve the Python interpreter from the
current `ovui` supplemental repository before creating the virtual environment.
Read the current `NVIDIA-Omniverse/ovui` `README.md`, `AGENTS.md`, relevant
`skills/omniverse-ui-*` references, and available package metadata. Do not copy
a fixed interpreter version into downstream setup recipes. `No matching
distribution found for ovui` usually means the interpreter, OS, architecture, or
selected companion package set does not match the currently available `ovui`
package files. Fix the interpreter or selected package set before changing local
UI code.

## Supplemental Dependency Documentation

These links centralize dependency documentation and examples. Use them for
dependency-specific API behavior that is not covered by the selected viewer
skills.

| Dependency | Current documentation pointer | Use for |
|---|---|---|
| `ovrtx` | <https://github.com/nvidia-omniverse/ovrtx> | Renderer API behavior, Python/C API notes, stage composition, render-var/AOV behavior, picking/selection behavior, and release notes. |
| `ovstage` | <https://github.com/NVIDIA-Omniverse/ovstage> | Runtime stage lifetime, population, queries, ordinals, DLPack, path dictionaries, write floors, and release notes. |
| `ovphysx` | <https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx> | Physics tensor APIs, stage attach behavior, worker examples, binding shapes, and release notes. |
| `ovstorage` | <https://github.com/NVIDIA-Omniverse/ovstorage> | Upstream `AGENTS.md`, the `skills/` catalog, storage client setup, resolver and asset-management patterns, examples, and release notes. |
| `ovui` | <https://github.com/NVIDIA-Omniverse/ovui> | Current Python/setup guidance, widget behavior, `ovwidgets`, `omni.ui`, headless overlay behavior, and native UI conventions. |
| `ovstream` | <https://github.com/NVIDIA-Omniverse/ovstream> | Library-specific `skills/`, sample servers, WebRTC lifecycle, SHM/client behavior, native input, examples, and package release notes. |

Use this table only as supplemental documentation when the selected references do
not contain enough detail for dependency-specific API behavior.

For `ovstorage`, begin with its upstream `AGENTS.md`, then read the relevant
upstream `skills/` guidance before designing cloud-asset access, caching,
resolver behavior, or transfers. This viewer package routes storage-related
work; the upstream repository owns the current storage API and integration
contract.

For `ovstream`, always check the supplemental repository when the task needs
library-specific behavior, newer transport examples, native input details, or
implementation patterns beyond this viewer skill package. That repository owns
additional `skills/` and samples for the streaming library itself.

## Package-Index Dependencies

### ovrtx

Use the NVIDIA Python package index for `ovrtx`.

Current supplemental repository pointer:
<https://github.com/nvidia-omniverse/ovrtx>

```bash
python3 -m pip install --upgrade ovrtx --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple
```

If a project provides `server/requirements.txt`, prefer that project manifest
over an ad hoc direct install. Preserve existing pins in that manifest unless the
user asks to update them.

For ovrtx API behavior, renderer configuration, render vars, picking, selection,
stage composition, or release-specific behavior not covered in this skill package,
use the supplemental documentation pointer above.

### ov-web-rtc client

Use the released `@nvidia/ov-web-rtc` package for the browser client that
connects to standalone `ovstream` WebRTC servers in Direct mode. This skill
package targets `ovstream` plus `ovrtx` viewer services. Those services may be
containerized and launched by OKAS, Kubernetes, or another GPU session
orchestrator. For standalone `ovstream` Direct deployments, do not use connection profiles or fields intended for a different streaming product; after orchestration resolves an endpoint, the frontend uses Direct mode against the exposed `ovstream` signaling host and port. For NVCF self-hosted deployment, read `cloud-deployment/nvcf-self-hosted.md` and follow the upstream workflow before selecting a browser connection model.

Use the current released package. Do not copy resolved client version numbers
into skills, templates, or setup recipes:

```text
registry=https://registry.npmjs.org/
@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/
```

```bash
npm install @nvidia/ov-web-rtc
```

For the `ovstream`-compatible Direct connection shape, use the current
`ovstream` WebRTC browser client example as the reference pattern:
<https://github.com/NVIDIA-Omniverse/ovstream/tree/main/examples/webrtc_client>

Use `@nvidia/ov-web-rtc` for new browser clients. Do not document alternate
browser streaming package names, legacy package names, or connection profiles intended for a different streaming product for generated Omniverse Realtime Viewer apps.

## Centralized Dependencies

### GitHub Asset Retrieval

Use the package URLs and release selectors listed below. If direct browser,
`curl`, or GitHub API access cannot retrieve a listed release or artifact, check
whether GitHub CLI is authenticated and use `gh` for access:

```bash
gh auth status
```

For release assets, use `gh release view` and `gh release download`. For
Actions artifacts, list artifacts through the API and download the named
artifact:

```bash
gh api repos/OWNER/REPO/actions/runs/RUN_ID/artifacts \
  --jq '.artifacts[] | [.name, .expired, .archive_download_url] | @tsv'

gh run download RUN_ID \
  -R OWNER/REPO \
  -n ARTIFACT_NAME \
  -D vendor/ARTIFACT_NAME
```

If `gh auth status` is not authenticated or the token cannot access the listed
repository, report the dependency retrieval failure. Do not use alternate wheel
or tarball locations.


### ovstream

Keep the current `ovstream` package source here rather than in streaming
skills, templates, or setup recipes.

Current supplemental repository pointer:
<https://github.com/NVIDIA-Omniverse/ovstream>

Current Python package:
<https://pypi.org/project/ovstream/>

NVIDIA Python package index mirror:
<https://pypi.nvidia.com/ovstream/>

Install the latest available wheel into the app virtual environment:

```bash
python3 -m pip install --upgrade ovstream
```

The current Python wheels bundle the native ovstream library, StreamSDK,
GStreamer, the bundled `gstnvenc` plugin, CUDA runtime pieces, and
`ovstream_utils`; no separate runtime zip is needed for normal Python apps.

If an environment must route NVIDIA packages through NVIDIA's package index,
use the mirror with PyPI as the fallback:

```bash
python3 -m pip install --upgrade ovstream \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple
```

Use the C/CMake platform zips from the same release only for native C/C++
integrations. Set `OVSTREAM_LIB_PATH` only when running from an extracted
runtime artifact layout, or when explicitly debugging native library discovery.

Rules:

- Use the PyPI package and install instructions from this section for Python
  apps.
- Install the latest available `ovstream` version unless the project manifest
  already pins a compatible version.
- Do not repeat wheel filenames, direct wheel URLs, or alternate package
  locations in app-specific setup notes.
- Do not point app-specific setup notes at unrelated local cache paths.
- Runtime guidance may still document API usage such as `ovstream.Server`,
  callback ordering, `OVSTREAM_LIB_PATH`, and video frame submission.
- For ovstream API or SHM behavior not covered in this skill package, downstream
  skills should ask agents to inspect the current supplemental repository
  pointer's `skills/`, samples, and release notes.

### ovui

Keep the current `ovui` package source here rather than in local-viewer,
overlay, or Windows setup skills.

Current supplemental repository pointer:
<https://github.com/NVIDIA-Omniverse/ovui>

Current Python package:
<https://pypi.org/project/ovui/>

NVIDIA Python package index mirror:
<https://pypi.nvidia.com/ovui/>

Install the latest available wheel into the app virtual environment:

```bash
python3 -m pip install --upgrade ovui
```

Use the wheel matching the selected Python version, OS, and architecture.
Select that Python version by checking the current `ovui` supplemental repo docs
and package metadata instead of hard-coding a version in generated setup notes.

If an environment must route NVIDIA packages through NVIDIA's package index,
use the mirror with PyPI as the fallback:

```bash
python3 -m pip install --upgrade ovui \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple
```

Rules:

- Use the PyPI package and install instructions from this section for Python
  apps.
- Install the latest available `ovui` version unless the project manifest already
  pins a compatible version.
- Keep `ovui`, `ovui-data-adapters`, `ovwidgets`, and related local UI
  companion packages on one compatible package set.
- Keep direct wheel URLs, wheel filenames, and alternate install commands out of
  app-specific setup notes.
- Runtime guidance may still document API usage such as `omni.ui`,
  `omni.ui_scene`, headless overlay contracts, `PYTHONPATH`, and display setup.
- For ovui widget, `ovwidgets`, editor shell, or headless overlay behavior not
  covered in this skill package, use the supplemental documentation pointer above.

### ovstage

Keep the current `ovstage` package source here rather than in renderer, scene,
or interaction references.

Current supplemental repository pointer:
<https://github.com/NVIDIA-Omniverse/ovstage>

Current Python package:
<https://pypi.org/project/ovstage/>

Install the latest available package into the application environment:

```bash
python3 -m pip install --upgrade ovstage
```

For ovstage-specific behavior beyond this viewer skill package, agents must
inspect the current supplemental repository before inventing an implementation.
Read its `AGENTS.md`, `README.md`, `OVERVIEW.md`, `CHANGELOG.md`, relevant
`skills/`, public API headers or Python bindings, examples, and release notes.
Use those sources to confirm population, query, DLPack, path-dictionary,
operation-lifetime, and synchronization details for the selected package.

`ovstage` is pre-release. Resolve it together with the latest `ovrtx` package,
validate the attached-stage flow against upstream examples, and keep an
application-specific pin in that application manifest rather than copying a
fixed version into downstream skills.

When a viewer uses both `ovstage` and `ovrtx`, the parent viewer runtime owns the
stage lifetime, ordinal clock, write floor, renderer attachment, and render-loop
publication state. UI, transport, USD-query workers, and physics workers should
cross that boundary with structured DTOs such as paths, matrices, settings, and
diagnostics rather than borrowed stage handles, path tokens, mapped buffers, or
DLPack objects.

Rules:

- Do not duplicate ovstage package URLs or version pins in feature references.
- Do not assume old renderer-owned USD load/query/write APIs remain current;
  consult the supplemental repo when code needs exact API calls.
- When cloning the supplemental repo for a task, inspect its `skills/` directory
  and examples before implementing a runtime workflow.

### ovphysx

Keep the current `ovphysx` package source here rather than in physics,
interaction, or delivery references.

Current supplemental repository pointer:
<https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx>

Install the latest available package into the physics worker or verified
attached-stage test environment:

```bash
python3 -m pip install --upgrade ovphysx \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple
```

For OVPhysX tensor APIs, binding shapes, simulation stepping, and release-specific
behavior, inspect the supplemental repository pointer above. For viewer-side
integration, read `references/physics-simulation`.

Default runtime boundary:

- The parent viewer process owns `ovrtx.Renderer`, live `ovstage` stage state,
  ordinals, and render publication.
- Use `PhysX.attach_ovstage(stage, read_ordinal=...)` only after a probe verifies the exact installed
  OVStage/OVPhysX ABI and bridge symbols.
- If `attach_ovstage` is missing, reports missing OVStage bridge symbols, or has
  not been verified, run OVPhysX in a bounded child process only when its worker
  API is verified against the installed package. Otherwise report the compatibility
  block; do not guess an OVPhysX population fallback.
- Return pose samples, binding diagnostics, and worker status as structured JSON
  and write accepted transforms through the parent viewer's OVStage/session path
  at monotonic ordinals.

Rules:

- Do not duplicate ovphysx package URLs or version pins in feature references.
- Do not treat an unverified OVPhysX population path as equivalent to a verified
  `attach_ovstage(stage, read_ordinal=...)` bridge; it can create competing USD
  population in the process that owns OVRTX/OVStage and crash native code.
