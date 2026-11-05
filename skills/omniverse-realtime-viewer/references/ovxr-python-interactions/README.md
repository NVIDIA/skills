# OVXR Python interactions

No reference-application source is required. `NavigationController`, message
names, state machines, and telemetry records in these references are application
patterns to implement, not library imports.

- Start with [architecture](architecture.md) and
  [feature patterns](feature-patterns.md).
- For movement, yaw, and user scale, use the public managed-anchor API described
  in [navigation](../xr-spatial-input/navigation.md).
- For source-safe visual composition and controller/hand lifetimes, use the
  sibling `xr-spatial-input/README.md` reference.
- Build the runtime from
  [the library-only recipe](../ovxr-library-runtime/library-only.md).
- Prove the browser path with the sibling `ovxr-composition-ui-validation/README.md` reference.

Keep one owner thread for renderer, stage, OVXR context, and spatial commands.
Callbacks enqueue bounded immutable intent. Validate finite values, source
identity, epoch, monotonic sequence/time, tracking/focus, and stage generation
before accepting mutation. Commit freshness only after semantic validation so a
malformed future packet cannot poison the current stream.

Use actual installed enums and public wrappers. Do not copy old ABI structs,
assume a version string establishes live native input support, or use camera-update
APIs as an undocumented arbitrary-object write primitive. One accepted command
produces canonical state and a revision; rendered proof additionally requires a
submitted frame, the relevant consumed state, and changing decoded pixels.

Cancel captures and reset input ordering/filters on reconnect or tracking loss.
Keep bounded GPU buffers, scene extraction, mesh construction, and resource
allocation outside the steady-state frame loop. Arm diagnostics for a finite
capture instead of reading back the stage every frame.
