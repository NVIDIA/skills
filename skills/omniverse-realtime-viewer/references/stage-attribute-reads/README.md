# Stage Attribute Reads

## Triggers

Use this skill for requests mentioning `read_attribute`, `read_array_attribute`,
attribute reads, USD attribute values, DLPack, GPU destinations,
`read_attribute_async`, array attributes, inspector values, or current runtime
values.

Use this when inspector panels, query services, animation, transform tools, or
pick effects need current OVStage runtime attribute values. OVRTX renders the
stage and owns renderer outputs/picks; it is not the default scene data service
for new OVStage viewers.

For pinned OVStage/OVRTX attribute API details and DLPack behavior, read
`references/dependencies` for acquisition guidance and supplemental dependency
documentation.

## Choose The Read API

Use `stage-queries` first when you do not know whether an attribute exists or
whether it is scalar or array.

| Runtime read shape | Use |
|---|---|
| Scalar/fixed-shape read | One value per prim: transforms, numeric controls, tokens, shader inputs. |
| Array read | Variable-length arrays such as `points`, `normals`, or `faceVertexCounts`. |
| Async scalar read | Avoid blocking input/message callbacks or inspector panels. |
| Async array read | Large mesh arrays or background property previews. |

The exact function names are app/runtime-adapter choices. They should map to the
pinned OVStage read APIs, validate the active stage generation, and return
copied DTO values before data crosses UI, transport, reload, or process
boundaries.

## Scalar Reads

Scalar/fixed-shape reads return one value per prim. Convert runtime values to a
JSON-safe copy before sending them over a data channel:

```python
import numpy as np

def read_json_scalar(runtime, attr_name: str, paths: list[str]):
    values = runtime.read_attribute(attr_name, paths)
    return np.asarray(values).copy().tolist()

paths = ["/World/Cube"]
xforms = np.asarray(runtime.read_attribute("omni:xform", paths)).reshape(len(paths), 4, 4)
```

Use current OVStage values for transform bases, inspector fields, and
pick-effect state. Use pxr only when the UI asks for authored composition
details that are not represented as runtime attributes.

## Array Reads

Array reads return one array-like value per prim, and lengths may differ:

```python
arrays = runtime.read_array_attribute("points", ["/World/MeshA", "/World/MeshB"])
for path, values in arrays.items():
    points = np.asarray(values)
    preview = points[:1000].copy().tolist()
```

Use arrays for geometry payloads only when the UI truly needs them. For most
inspectors, report counts, dtype, shape, and a capped preview.

## GPU Destinations And Async Flow

If the pinned OVStage runtime exposes DLPack destinations or async reads, keep
the source/destination tensors alive until the operation completes and
synchronize before another consumer reads the data. UI and WebRTC messages must
receive copied values, never borrowed tensors or mapped views.

```python
op = runtime.read_attribute_async("omni:xform", paths)
pending = op.wait(timeout_ns=5_000_000_000)
if pending is None:
    return None

values = pending.fetch(timeout_ns=100_000_000)
if values is None:
    return None

xforms = np.asarray(values).copy()
```

Do not access the value until both `wait()` and `fetch()` have succeeded.

## Inspector Pattern

```python
def inspect_attrs(runtime, path: str, attr_names: list[str]) -> dict:
    descriptors = runtime.query_prims(
        attribute_filter_mode="specific",
        attribute_names=attr_names,
    ).get(path, {})

    values = {}
    for name, desc in descriptors.items():
        if desc.is_array:
            arr = runtime.read_array_attribute(name, [path])[path]
            values[name] = np.asarray(arr)[:1000].copy().tolist()
        else:
            value = runtime.read_attribute(name, [path])
            values[name] = np.asarray(value).copy().tolist()
    return values
```

Keep pxr fallback for variant sets, relationship targets, material bindings, and
USD metadata until native runtime APIs expose those fields directly as
user-readable values.

## Gotchas

- OVStage reads return current runtime values; they do not replace USD
  composition services such as variant-set editing.
- Scalar reads are for one fixed-shape value per prim.
- Array reads are for variable-length values and should return a path-keyed
  result, not a blindly stacked tensor.
- Keep DLPack-backed or mapped views scoped. Take `.copy()` before storing
  values beyond the operation lifetime or sending them to another thread.
- Query descriptors first when a missing attribute would otherwise become an
  exception path in UI code.
- Direct OVRTX read APIs are for renderer-owned diagnostics or legacy
  pre-OVStage samples, not the default inspector path in new viewers.

See also: `stage-queries`, `stage-hierarchy`, `prim-info-display`,
`ovstage-data-plane`, `ovstage-ovrtx-integration`.
