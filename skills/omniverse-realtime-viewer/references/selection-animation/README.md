# Selection Animation

Use this reference for optional selected-object motion: a lift, pulse, or other
reversible transform. Native selection outlines remain the basic selection
signal. Animation contributes local transforms to the application's runtime
owner; it does not own a renderer loop or OVStage publication clock.

## Capture the base transform

Read the current local transform before animation begins. Use the live runtime
value after a gizmo or physics edit; use authored USD only before a runtime
value exists. Copy the matrix so later state updates cannot modify the baseline.

The adapter below is application-owned; use `prim-transform-safety` for authored
transform reads and `ovstage-data-plane` for runtime reads.

```python
import numpy as np

def load_animation_bases(runtime, paths):
    bases = {}
    for path, matrix in runtime.read_local_transforms(paths).items():
        matrix = np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)
        if np.isfinite(matrix).all():
            bases[path] = matrix
    return bases
```

`omni:xform` is local to the parent. Matrices are row-major float64 with
translation in row 3. For world-space motion, convert the offset through the
current effective parent on each update; a parent can move while the child is
selected. Restore the saved local base, so ending animation preserves ancestor
edits. Use the shared ancestry adapter in `prim-transform-safety` for nested and
reset-stack prims.

## Animate from the baseline

Use a reversible state machine such as
`IDLE -> RISING -> HOVERING -> FALLING -> IDLE`. Derive direction and magnitude
from the product brief, stage units, and asset scale. Use elapsed time for easing,
never accumulate transforms from the previous animated frame.

```python
import math
import numpy as np

def ease_out_quint(t):
    return 1.0 - (1.0 - min(1.0, max(0.0, t))) ** 5

def ease_in_out_sine(t):
    return -(math.cos(math.pi * min(1.0, max(0.0, t))) - 1) / 2

def compose_offset_xforms(base_xforms, offsets):
    """Offsets are translations expressed in each prim's parent coordinates."""
    updates = {}
    for path, base in base_xforms.items():
        offset = np.eye(4, dtype=np.float64)
        offset[3, :3] = offsets.get(path, np.zeros(3))
        updates[path] = base @ offset
    return updates
```

`animator.update(dt)` returns changed `{prim_path: local_matrix}` entries. On the
last falling frame it returns the saved base matrix before becoming idle. On
rapid reselection, start from the current offset so motion reverses smoothly.
Once idle, omit unchanged prims from subsequent batches.

## Join the owner's publication

The owner drains input, updates feature state, and composes one final transform
per path. Camera changes, gizmo edits, animation, and completed physics samples
join the same tick. A paused or pending simulation must not suppress camera or
selection-animation updates.

Use the concrete `write_local_transforms` implementation from
[`prim-transform-safety`](../prim-transform-safety/README.md#batch-writes-and-publication).
It builds a 16-lane OVStage tensor in query order and uses
`AttributeSemantic.MATRIX` and `PrimMode.UPSERT`. It waits for its write but does
not allocate an ordinal, advance the floor, or render.

```python
# Inside the runtime owner's existing frame transaction:
# query and ordered_paths cover the final changed transform set.
# local_updates already composes camera, animation, gizmo, and physics ownership.
write_local_transforms(stage, query, ordered_paths, local_updates, ordinal)
# Finish any other attribute writes for this tick before committing it.
stage.advance_write_floor(ordinal).wait()
committed_ordinal = ordinal
products = renderer.step(
    render_products={render_product_path}, delta_time=dt, ordinal=committed_ordinal,
)
```

Allocate `ordinal` once in the owner, only when the tick has changes. Otherwise
render the latest committed ordinal. Release frame results through the normal
presentation lifecycle. The animator only produces matrices; it must not call
`next_ordinal()`, `advance_write_floor()`, or `renderer.step()` itself.

## Coordinate interactions

Keep selection events independent of the presentation shell. Native picks,
outliner selection, and scripted selection all call the same animator methods.

Freeze animation on gizmo drag start and use the visible transform as the drag
base. When dragging ends, refresh the animation baseline from the committed edit
before resuming. When physics owns a body, suspend its selection motion or compose
an explicitly requested visual offset on the current physics pose. Do not let two
feature writers overwrite the same path independently.

On scene replacement, discard animation state and baselines from the old stage
generation. Never restore those values into a new stage with matching path names.

Verify selection, rapid deselection/reselection, and restoration under a rotated
and translated parent. Also verify camera movement and animation continue while
physics is paused or awaiting a step. See `native-picking-selection`,
`prim-transform-safety`, and `realtime-sim-integration` for the connected paths.
