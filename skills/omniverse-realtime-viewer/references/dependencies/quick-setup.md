# Dependency Quick Setup

## Prepare The Chosen Delivery Path

1. Inspect the existing app's manifest and environment, or resolve the new app's
   package set with `nvidia-runtime.md`. Choose Python/native toolchains from the
   selected libraries' supported platforms before creating an environment.
2. Check that the host GPU/driver supports OVRTX and, for encoded streaming, the
   chosen encoder. Local native windows also need a working display session.
3. Install the app's declared dependencies in an isolated environment. Keep
   renderer, UI, and native transport artifacts from the chosen compatible set;
   give workers separate environments when their native dependencies differ.
4. Build and run the smallest frame path for the selected delivery recipe.
   Follow `environment-validation.md` if initialization fails, then add UI and
   feature dependencies as needed.

| Delivery | Libraries beyond OVRTX/OVStage |
|---|---|
| Browser streaming | ovstream producer and compatible browser client; React/UI toolchain |
| Python native | ovui and its selected companion packages |
| Electron or Tauri SHM | ovstream producer/native client, selected shell toolchain |
| In-process Tauri or C++ | Matching native OVRTX SDK and selected UI/presentation libraries |

Use `frontend.md`, `electron-shm.md`, or `local-openusd-gpu.md` for the relevant
shell. Generate shared UI within the app using `viewer-backend-interface`; it is
not an additional runtime package.

For each selected upstream smoke example, include its documented extras, schema
providers, and sample assets in the same environment before running it unchanged.
An example can require more than the library's base install. Use the interpreter
the generated app will launch, and keep worker checks in their own environments.

## Runtime Boundaries

The viewer runtime owns OVStage and OVRTX. Use `ovstage-ovrtx-integration` for
initialization, schema registration, and frame publication. USD-query workers
return copied metadata; optional simulation follows `physics-simulation` or
`newton-physics` for its chosen in-process or worker boundary.

Configure cache directories only when deployment permissions or isolation
require it, using the selected libraries' supported settings. Preserve reusable
shader/kernel caches across normal app restarts to avoid repeated warmup. Keep
caches, environments, build outputs, and credentials out of version control.

## Setup Evidence

Record the resolved packages, native build identity, GPU/driver, and successful
frame path in the app's validation results. A package import verifies installation;
a displayed or decoded frame verifies the basic delivery integration.
