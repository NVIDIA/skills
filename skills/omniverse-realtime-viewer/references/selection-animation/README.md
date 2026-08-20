# Selection Animation

## Triggers

Use this skill for selection animation, prim animation, hover animation, float
selected object, animate selected, `omni:xform` animation, OVStage transform
samples, `map_attribute`, `AttributeMapping`, or transform not updating.

Use this when selected prims need optional renderer-visible motion feedback.
For new OVStage-based viewers, animation writes runtime transform samples
through the OVStage data plane and the attached ovrtx renderer consumes the
committed publication. Do not implement selection animation by direct
`Renderer.write_attribute()`, `Renderer.bind_attribute()`, or
`Renderer.map_attribute()` calls unless you are maintaining a legacy pre-OVStage
viewer.

## Key Rules

- Runtime attribute: `omni:xform`, not `xformOp:transform` or
  `xformOp:translate`.
- Semantic: matrix transform / `XFORM_MAT4x4` as exposed by the pinned OVStage
  runtime.
- Matrix: float64, row-major, translation in row 3.
- Compose every animated matrix from a saved base transform plus the current
  animation offset. Do not accumulate tiny per-frame deltas.
- Publish each animation update at a monotonic OVStage ordinal before the
  renderer step that should show it.
- Keep native selection outlines, material effects, and segmentation overlays in
  their own managers. Selection animation is motion feedback, not the baseline
  selected-object signal.
- Motion parameters are application choices. Derive direction, magnitude,
  duration, and easing from the product brief, stage units, asset scale, and
  active coordinate system.

## State Machine

`IDLE -> RISING -> HOVERING -> FALLING -> IDLE`

Use a small state machine so selection and deselection are reversible. The
specific motion can be lift, pulse, nudge, scale, or another non-destructive
runtime transform chosen by the app. The example below uses a configurable
translation offset; replace its values for the target product.

```python
import math
import numpy as np

ANIMATION = {
    "direction": np.array([0.0, 1.0, 0.0], dtype=np.float64),  # app-defined
    "distance": 0.05,      # stage units; choose from asset scale/bounds
    "oscillation": 0.0,    # optional additional stage-unit offset
    "frequency_hz": 1.5,
    "rise_seconds": 0.25,
    "fall_seconds": 0.25,
}

def ease_out_quint(t): return 1.0 - (1.0 - t) ** 5
def ease_in_out_sine(t): return -(math.cos(math.pi * t) - 1) / 2
```

## Base Transform Capture

Capture base transforms from current runtime state, not from a newly created
renderer attribute. If a prim has already moved because of a transform tool or
physics handoff, the animation base must include that latest runtime pose. Use a
USD/pxr fallback only for authored metadata or when the runtime has not produced
a current value yet.

```python
import numpy as np

def load_animation_bases(runtime, paths: list[str]) -> dict[str, np.ndarray]:
    bases = {}
    for path, matrix in runtime.read_world_transforms(paths).items():
        matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        if np.isfinite(matrix).all():
            bases[path] = matrix
    return bases
```

Adapter method names are application-owned. Map them to the pinned OVStage API
from `ovstage-data-plane`; the important contract is that the value is copied
out of runtime state before the animator stores it.

## OVStage Write Pattern

Write animation samples through the runtime owner. A compact adapter keeps the
animator independent of whether the application uses Python, C++, streaming,
local ovui, or an Electron/Tauri presentation shell.

```python
import numpy as np

class RuntimeTransformWriter:
    def __init__(self, runtime):
        self.runtime = runtime

    def write_world_xforms(self, xforms_by_path: dict[str, np.ndarray]) -> int | None:
        if not xforms_by_path:
            return None

        paths = list(xforms_by_path)
        values = np.stack([xforms_by_path[path] for path in paths]).astype(np.float64)
        ordinal = self.runtime.next_ordinal()
        op = self.runtime.write_attribute(
            ordinal=ordinal,
            prim_paths=paths,
            attribute_name="omni:xform",
            values=values,
            semantic="XFORM_MAT4x4",
            prim_mode="create_new",
        )
        op.wait()
        self.runtime.advance_write_floor(ordinal).wait()
        return ordinal
```

Each frame, compose the saved base transform with the app-defined transform
offset and publish it through that adapter:

```python
def compose_offset_xforms(
    base_xforms: dict[str, np.ndarray],
    offsets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out = {}
    for path, base in base_xforms.items():
        offset = offsets.get(path)
        offset_mat = np.eye(4, dtype=np.float64)
        if offset is not None:
            offset_mat[3, 0:3] = offset
        out[path] = base @ offset_mat
    return out

def write_offsets(writer, base_xforms, offsets) -> int | None:
    return writer.write_world_xforms(compose_offset_xforms(base_xforms, offsets))
```

The renderer owner then consumes the committed ordinal in the normal loop. If
the pinned integration exposes an explicit `update_from_stage`, call it there;
otherwise the attached renderer step must still run only after the write floor
has advanced.

```python
now = time.monotonic()
dt = max(1.0 / 300.0, min(0.1, now - last_step))
last_step = now

ordinal = animator.update(dt)
if ordinal is not None and hasattr(renderer, "update_from_stage"):
    renderer.update_from_stage(ordinal)
products = renderer.step(render_products={RENDER_PRODUCT_PATH}, delta_time=dt)
```

Updating after `step()` renders the previous transform and makes animation lag
or appear stuck.

## Tensor And Async Notes

Use the OVStage runtime's DLPack and async contracts when writing many animated
prims or GPU-produced transforms. Source CPU/GPU buffers must remain alive until
the runtime write completes, and the render loop must wait or fence before
stepping a frame that depends on the write. Do not reintroduce direct OVRTX
`AttributeBinding` handles as an optimization path in new OVStage viewers.

For sparse selection sets, keep per-path animation state, but still batch the
current frame's changed matrices into one OVStage publication when possible.

## Selection Lifecycle

```python
def select_prim(path: str):
    if animator is not None:
        animator.deselect_all()
        animator.select(path)

def clear_selection():
    if animator is not None:
        animator.deselect_all()
```

Native picking, tree selection, marquee selection, or scripted selection can
all call the same animation methods after they resolve selected prim paths.
Native selection outlines and material/effect pick feedback should be updated
by their own managers in parallel.

On rapid select/deselect, record the current offset so falling reverses smoothly
from the current transform state. On scene switch, discard bases and animation
state for the previous stage generation.

## Transform Tools And Physics

Selection animation and transform manipulator drags must not write transforms
at the same time. Freeze the animator on drag start, use the current visible
runtime transform as the drag base, then update the animator base from the
final OVStage transform before resuming.

When OVPhysX is driving pose samples for a selected prim, either suspend
selection animation for that prim or layer only a clearly requested visual
offset on top of the latest physics pose. The final matrix still publishes
through one OVStage transform writer for that frame.

## Focused Proof

For an app that includes selection animation, capture a concise proof with:

- selected path and active animation mode,
- base transform source,
- OVStage write ordinal and renderer-consumed ordinal for at least one frame,
- measured nonzero transform delta during the animation,
- measured restore to the saved base transform after deselect.

Do not add broad browser or artifact generation just because animation exists;
attach the proof to the app's existing validation report or test output.

## Generated Module Checklist - prim_animation.py

- [ ] `PrimAnimator.__init__(runtime_writer, prim_paths, base_transforms)`
- [ ] `PrimAnimator.select(path: str) -> None`
- [ ] `PrimAnimator.deselect(path: str) -> None`
- [ ] `PrimAnimator.deselect_all() -> None`
- [ ] `PrimAnimator.update(dt: float) -> int | None`
- [ ] `PrimAnimator.current_offset(path: str) -> np.ndarray`
- [ ] `PrimAnimator.freeze(path: str) -> None` and
      `PrimAnimator.resume(path: str) -> None` when transform tools can edit
      animated prims.
- [ ] Base transforms come from current OVStage runtime state, with pxr used
      only as an authored-data fallback.
- [ ] Runtime writes publish `omni:xform` matrices with transform semantics at
      monotonic OVStage ordinals.
- [ ] CPU/GPU buffers used by async runtime writes stay alive until the write
      completes.
- [ ] Falling state restores the saved base transform when complete.
- [ ] No direct OVRTX transform write, binding, or mapping path is used in new
      OVStage viewer code.

See also: `object-selection`, `selection-feedback`, `transform-manipulator`,
`prim-pick-effects`, `physics-simulation`, `stage-management`,
`ovstage-data-plane`, `ovstage-ovrtx-integration`.
