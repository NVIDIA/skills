# C++ Native Project And Build

The C++ delivery path is already selected. Select the smallest suitable native UI toolkit, keep one responsive window,
and preserve the runtime/UI ownership boundary below. ImGui/GLFW and the file
layout below are examples, not a required stack or project template. If delivery is still undecided,
use `references/routing.md` before returning here.

## Architecture Overview

Use separate UI and runtime ownership so a blocking SDK call does not stop
window events or controls:

```text
UI thread: native events/controls -> bounded intent queue
           latest owned RGBA frame -> toolkit pixel presentation
Runtime owner: commands -> OVStage writes at N -> wait -> seal N
               -> OVRTX update from N -> wait -> native selection/pick state
               -> step with N -> fetch -> map/copy/unmap -> destroy results
               -> latest owned frame and copied state snapshot
Preparation worker: file/composition/metadata work -> generation-tagged result
```

Only the runtime owner accesses the live OVStage and renderer. Keep the UI
handoff lock short; never hold it during SDK calls. Bound the command queue,
coalesce superseded motion, and retain only the latest complete frame. Preserve
button-release and close intent when coalescing. Tag scene-dependent commands,
metadata, picks, and frames with a generation and discard stale work.

Prepare replacement scenes away from the UI and active render loop. Activate a
completed candidate on the owner thread between frames, as described in
`renderer-session.md`. When implementing requested tree/properties, use native
OVStage queries and the property-specific decoding contract in
`references/stage-attribute-reads/README.md`; queries describe data, they do not decode
its values. Optional OpenUSD composition must follow the matched dependency's
ABI or process-isolation boundary.

## Project Skeleton

Split by ownership, for example:

```text
cpp-viewer/
  CMakeLists.txt
  src/
    main.cpp             # selected toolkit, pixel presentation and UI intent
    runtime.cpp          # OVStage, OVRTX and owned frame publication
    runtime.h            # bounded commands and copied snapshots
    camera.cpp
    camera.h
    session_layer.h
    ovrtx_helpers.h
```

The first checkpoint is one real OVRTX frame presented in the selected native window,
with responsive events and normal close. Establish that before generic property
reflection, broad metadata discovery or extra menus. Then add requested camera
controls, picking, tree/details and switching one increment at a time. Use scene
bounds/metadata as needed for correct framing, rather than building a general
inspector before proving the pixel loop.

## Build Setup

Use the current paired OVRTX/OVStage CMake helpers from the upstream native
minimal example, selected through `references/dependencies/README.md`. Keep one matching
loader model, stage both package runtimes, and rebuild when changing the pair.
Use the selected native shell's normal build setup. Do not manually search an
SDK layout or combine static and shared loaders.

`build-details.md` contains the complete illustrative GLFW/ImGui CMake setup
and small C API string/result/wait helpers when needed. It is an implementation
aid, not a required file layout. Keep the owner-thread boundary above even in a
compact project. Continue with `renderer-session.md` and `presentation-loop.md`
to establish the first actual displayed frame.
