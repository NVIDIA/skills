# Stage Queries

## Triggers

Use this skill for query prims, stage query, find prims, prim type filter, has
attribute, `AttributeFilterMode`, `FilterKind`, `prim_list_handle`, selectable
sets, effect target discovery, or write planning.

Use this when an Omniverse Realtime Viewer needs to discover prim paths, build a
stage tree, find prims by type, find prims with specific attributes, or inspect
attribute schemas before reading or writing.

For new OVStage-based viewers, query the application-owned runtime stage.
OVRTX owns rendering, outputs, native pick queues, and selection visualization;
it is not the source of scene truth for hierarchy, inspector data, transform
handoff, pick effects, or physics pose writes.

## Core API Shape

Keep a narrow runtime query adapter. The exact enum names and method names
should map to the pinned OVStage API, but the generated app should expose a
path-keyed result that can be copied into DTOs for UI and transport.

```python
result = runtime.query_prims(
    attribute_filter_mode="none",
)
paths = sorted(result.keys())
```

Async query paths should stay inside the runtime owner and return copied
results tagged with the active stage generation and observed ordinal:

```python
op = runtime.query_prims_async(
    require_any=[("prim_type", "Mesh"), ("prim_type", "Camera")],
    attribute_filter_mode="specific",
    attribute_names=["omni:xform", "visibility"],
)
pending = op.wait(timeout_ns=5_000_000_000)
if pending is not None:
    result = pending.fetch(timeout_ns=100_000_000)
```

Discard query results whose stage generation no longer matches the active
runtime generation.

## Filter Construction

Each filter is a `(kind, name)` pair:

| Kind | Meaning | Example |
|---|---|---|
| `prim_type` | Match USD type name | `"Mesh"`, `"Xform"`, `"Camera"`, `"SphereLight"` |
| `has_attribute` | Match attribute presence | `"points"`, `"omni:xform"`, `"visibility"`, `"inputs:Fader"` |

Filter lists combine as:

- `require_all`: AND. The prim must match every filter in this list.
- `require_any`: OR. The prim must match at least one filter in this list.
- `exclude`: NOT. The prim must match none of these filters.

```python
meshes_or_lights_with_visibility = runtime.query_prims(
    require_all=[
        ("has_attribute", "visibility"),
    ],
    require_any=[
        ("prim_type", "Mesh"),
        ("prim_type", "SphereLight"),
        ("prim_type", "DistantLight"),
    ],
    exclude=[
        ("prim_type", "Scope"),
        ("has_attribute", "omni:hidden"),
    ],
    attribute_filter_mode="specific",
    attribute_names=["visibility", "purpose", "omni:xform"],
)
```

Omitted lists impose no constraint. An empty query matches every prim.

## Attribute Reporting

Attribute filter mode controls descriptor payload size:

| Mode | Use |
|---|---|
| `none` | Fast path discovery and prim counts. Per-prim descriptor dicts are empty. |
| `specific` | Inspector allowlists and read/write planning. Only requested attributes are reported. |
| `all` | Debugging and rich schema browsing. Avoid for routine data-channel payloads. |

Descriptors should expose enough schema to choose scalar reads, array reads, or
runtime writes:

- `name`
- `dtype`
- `is_array`
- `semantic`

```python
query = runtime.query_prims(
    require_all=[("prim_type", "Mesh")],
    attribute_filter_mode="specific",
    attribute_names=["points", "faceVertexCounts", "omni:xform"],
)

for path, attrs in query.items():
    if "points" in attrs and attrs["points"].is_array:
        points = runtime.read_array_attribute("points", [path])[path]
```

Use the same query descriptors before OVStage writes for pick effects,
visibility commands, and transform tools so generated apps do not invent
attributes that are absent from the active stage.

## Tree Construction

Runtime queries return flat paths. Build hierarchy by splitting paths:

```python
def parent_path(path: str) -> str:
    if path == "/" or path.count("/") <= 1:
        return "/"
    return path.rsplit("/", 1)[0]

def child_index(paths: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for path in paths:
        if path != "/":
            index.setdefault(parent_path(path), []).append(path)
    for children in index.values():
        children.sort()
    return index
```

Use lazy UI expansion: query once after stage load, cache a parent-to-children
index for the active generation, and only send children for expanded rows.

## Query Handles And Follow-Up Work

Lower-level runtimes may expose grouped query results or prim-list handles that
can be passed into dependent reads/writes without converting back through path
strings. Keep those handles inside the runtime owner. UI, WebRTC, worker
processes, and feature managers should receive copied path strings or DTOs, not
borrowed handles.

Python adapters can usually pass grouped path lists into runtime
`read_attribute()`, `read_array_attribute()`, or `write_attribute()` calls.
C/C++ integrations can preserve runtime query handles only until all dependent
operations are enqueued, then release the query result according to the pinned
runtime contract.

## Gotchas

- Query filters match type names and attribute names, not path substrings.
- A specific attribute filter with no attribute names should report no
  descriptors.
- Query result descriptors describe schema; they do not read values. Use
  `stage-attribute-reads` for values.
- Relationship-like values may surface as path IDs or token IDs; use pxr
  fallback for readable relationship target inspection until native relationship
  traversal is complete.
- Keep stage load/reset, query integration, and dependent writes serialized
  through the runtime owner.
- Do not route stage queries through direct OVRTX query APIs in new OVStage
  viewers except for renderer diagnostics or legacy samples.

See also: `stage-attribute-reads`, `stage-hierarchy`, `prim-info-display`,
`prim-pick-effects`, `ovstage-data-plane`, `ovstage-ovrtx-integration`.
