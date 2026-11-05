<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# A usable, polished and responsive XR viewer

Use this reference for a user-facing XR editor, spatial panels, toolbelt or
visual-quality work. A minimal stereo probe need not implement an editor.
Start with the [assembly map](../ovxr-library-runtime/viewer-assembly.md) for
construction; this reference supplies the experience and acceptance contract.
These are application design choices, not widgets supplied by OVXR.

## Compose the experience before adding controls

For an editor, a useful starting layout is one hierarchy/properties surface,
a compact toolbelt, and a collapsible settings surface. Keep the primary scene
and active tool target unobstructed. Do not open every debug control by default.
Reuse the user's visual language where supplied; otherwise define a small
shared theme for background, elevated surface, text, muted text, accent,
hover, selected, pending and error. Share these semantic tokens across raster
panels and analytic chrome, not necessarily their rendering implementation.

Use consistent padding, corner radii, icon strokes, type sizes and row rhythm.
Keep secondary metadata subordinate to labels. Use opaque or sufficiently
backed text surfaces against both bright passthrough and dark scenes; global
translucency is not a substitute for contrast. Distinguish hover, selected
target and active manipulation with shape/outline or a label as well as color.
Selection need not repaint a whole panel if a coherent analytic highlight can
represent it. Keep visible text and hit regions tied to the displayed revision.

Place each surface once in its declared LOCAL or world anchor; do not silently
turn a world panel into a head-following HUD. Give movable windows a permanent
rounded grab bar below the content, with a gap and separate hit ID. Keep enlarged
hit bounds clear of adjacent controls. Offer an explicit restore/recenter action
for lost panels, canceling active capture before relocation. Never recenter
automatically in the middle of a drag or teleport.

One optional starting geometry is a 0.96 x 0.48 m panel, a bar 70% of its width,
0.034 m high and 0.018 m below it. These are tunable layout examples, not headset
ergonomic limits. Choose placement, scale and text size for the actual working
distance; evaluate seated/standing use and the requested clients separately.
Use [spatial interaction](spatial-surfaces-and-interaction.md) for anchor math,
hit testing and capture; do not reproduce that math in a theme component.

## Make gestures discoverable without another gesture menu

When using the [editor gesture mapping](../xr-spatial-input/gesture-arbitration.md),
expose a short contextual hint or help surface with the actual bindings:

- Right pinch shows the selection ray; a fresh left pinch selects its UI/scene
  hit. With no selection ray/capture, left pinch holds teleport preview.
- During teleport, right pinch cancels. Both hands must open before a new left
  pinch can start another teleport. No cancel edge also selects an object.
- Select a window's bar, then close an eligible hand into a fist to move it.
  Opening releases immediately; selecting a scene prim clears the panel target.
- Wrist swing/translation aims teleport; twist relative to pinch start changes
  facing. Show that facing on the landing marker, using the same accepted yaw
  that the eventual commit consumes.

Do not require the palm-up menu when it conflicts with the headset shell.
Controller hints come from the active action profile, not hand-gesture labels
or an assumed Quest/PSVR mapping. Hide unsupported actions or show an actionable
unavailable reason. Brief tracking/query uncertainty can retain a bounded
preview, but must visibly distinguish it from a valid landing and must not
authorize movement. See [landing acceptance](../xr-spatial-input/teleport-acceptance.md).

Make pinch strength/release, hold grace, twist gain, smoothing and reach gain
tunable with units and validation. The [filtering recipe](../xr-spatial-input/gesture-filtering.md)
provides no-jump, reversible reach gain and timestamped One Euro math. An
optional panel preset uses 1:1 motion through 5 cm, smoothly reaching 3x at
20 cm; it is displacement gain, not velocity or inertia. Do not require larger
physical movements to conceal a wrong coordinate conversion.

## Sharpness is a per-surface budget

Keep logical layout, physical size and raster size independent. Use actual
fonts and vector icons; package their licensed assets and test missing-font
behavior. For a front-facing width W at distance d, angular width is
`2*atan(W/(2*d))`; use that plus projected eye coverage to choose raster density,
then verify in the headset. A desktop PNG alone cannot establish legibility.
Bound raster dimensions and resolution changes; add hysteresis if adapting to
distance so small hand movements cannot trigger repeated reallocations.

Diagnose text, toolbar silhouettes, gizmo edges and texture minification
separately using [Skia/Yoga](skia-yoga.md) and [depth tools](depth-tools.md).
Increasing the stage-panel raster does not smooth a separately drawn toolbelt.
Keep padded analytic edge coverage, premultiplication and depth behavior
consistent. Test transparent margins over passthrough and scene intersections
at oblique angles. Do not use full-eye blur, disable foveation, or increase all
eye resolutions as an unmeasured UI repair.

Budget actual bytes: a 2048 x 1024 RGBA8 image is 8 MiB before staging/retired
copies. Bound total active surfaces, queued revisions and GPU retirement, not
only one texture. Pose-only motion does not regenerate or upload the raster.
Use the [bounded worker](async-raster-pipeline.md) so hover and settings changes
do not stall the XR owner. Optional decorative effects must fit the same budget;
do not add per-frame large-canvas repainting for an animated highlight.

## Accept the complete experience

Use one representative scene plus purpose-built alpha/depth/material fixtures.
User-provided content is not a required asset bundled with the skill. For each
requested feature, record three separate outcomes: functional, visual, and
performance. None implies the other two.

| Story | Functional and visual proof | Performance observation |
| --- | --- | --- |
| Open/connect | Honest warmup/waiting status, first stereo pixels and head motion | Separate shader warmup from connected steady state |
| Read and inspect | Readable stage labels, coherent property selection, toolbar icons/edges sharp at working distance and oblique angles | Bounded hierarchy work; no unchanged-raster upload |
| Select A then B | One action replaces A; outline, tree, properties and gizmo agree | Bounded active query and no repeated stage traversal |
| Select bar and fist-move | Distinct selected bar, no jump, useful reach, stable motion; selecting a prim clears the bar | Pose-only updates, no text repaint; measure drag-active tails |
| Aim/turn/cancel/commit teleport | Translation and swing remain live while twist changes facing; uncertain targets cannot commit; both-open rearm works | Preview and final async query deadlines; no same-frame GPU wait |
| Cross scene/tool/UI edges | Correct scene occlusion; gizmos over scene, covered UI above gizmos; no dark alpha fringe | Count depth-policy draws/passes and worst-case screen coverage |
| Enable passthrough/settings | Dark opaque geometry stays opaque; background alpha, glass/cutout/matte fixtures behave as configured; controls acknowledge applied state | Matched per-option comparisons, not all options toggled together |
| Mirror, loss and reconnect | Mirror reports stale/unavailable honestly; capture cancels; same authoritative selection/policy recovers | Measure mirror demand separately; no second XR render owner |

Choose a target frame rate and acceptable frame-time/latency budget for the
client, GPU and scene; do not promise a universal FPS. Measure median and tail
owner-frame intervals with idle, hover, drag and teleport, plus compositor and
mirror lanes as in [performance](performance-and-telemetry.md). Preserve visual
configuration in comparisons. If quality and budget conflict, report the
measured tradeoff rather than silently reducing image quality or hiding work.

Finish with [spatial validation](../ovxr-composition-ui-validation/README.md).
Inspect actual captures, label mirror/emulator/headset evidence correctly and
record unavailable measurements. A passing API call, unit test or FPS average
does not prove a beautiful, comfortable or responsive headset experience.
