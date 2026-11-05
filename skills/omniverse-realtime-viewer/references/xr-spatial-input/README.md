# OVXR spatial input

Preserve one OVXR-owned XR frame loop. These references are sufficient design
contracts for an application built from libraries; sample class names are not
SDK dependencies.

- Movement, rotation, scale, teleport, and pose conversion:
  [navigation](navigation.md).
- Controller source selection, grip/aim, profiles, models and tools:
  [controllers](controllers.md).
- Hand streams, gesture state, loss and recovery:
  [hands](hands.md).
- Concrete pinch/grab metrics and deterministic precedence:
  [hand recognizers](hand-recognizers.md), [gesture arbitration](gesture-arbitration.md).
- Arc queries, walkability, release freshness and facing-plus-position commit:
  [teleport acceptance](teleport-acceptance.md).
- Decoupled wrist swing/twist and accelerated One Euro panel movement:
  [gesture filtering](gesture-filtering.md).
- Immutable source USD and generated visuals:
  [stage composition](stage-composition.md).
- Spatial proof and bounded diagnostics:
  [feature validation](feature-validation.md).

Use `Context.transform_openxr_pose` when available to match the active anchor,
scene units and up-axis used by the eye cameras. Check pose validity before
conversion. Never apply stage units or navigation twice.

Read [library input and stage-publication](../dependencies/ovxr.md) before SDK integration.

Use upstream openxr-input-and-extensions through
[the library route](../dependencies/ovxr.md) for native extension/handle startup,
action attachment and resource release. The viewer owns snapshot normalization,
source arbitration and tool capture; [frame/input scheduling](../ovxr-library-runtime/frame-input-order.md)
places that work within the selected SDK schedule.

Keep physical identity, reported interaction profile, measured grip/aim, and
visual model calibration separate. Reconnect resets input epochs, captures,
filters, cached handles, and visual publication; preserve the long-lived renderer.

For simulation coupling and extended capture semantics, see [advanced tools](advanced-tools.md). Its eye-derived alignment applies only to explicitly supported legacy packages without managed pose conversion.

For an explicitly controlled browser adapter, use [WebXR sampling](webxr-client.md).
For physical grasps, remote holds and cable pulls, use
[physics interaction](physics-interaction.md) and
[simulation lifecycle](simulation-lifecycle.md).
