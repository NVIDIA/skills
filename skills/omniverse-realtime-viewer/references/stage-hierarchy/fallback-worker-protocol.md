<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Stage Hierarchy Fallback Worker Protocol

## Fallback Subprocess Protocol

Create an app-local worker (suggested name `server/pxr_worker.py`) and a
parent-side client (suggested class name `PxrWorkerClient`) when variants, rich
metadata, or relationship targets need isolated OpenUSD queries. These are
modules to implement, not files included in this package. The USD query and
serialization building blocks are in [README.md](README.md).

Requests and responses are one UTF-8 JSON object per line on stdin/stdout;
logs go to stderr only. Use line buffering (`bufsize=1`,
`reconfigure(line_buffering=True)`).

```json
{"cmd":"load","path":"/path/to/scene.usd"}
{"cmd":"get_children","path":"/World","filters":["USDGeom"]}
{"cmd":"get_properties","path":"/World/Cube"}
{"cmd":"get_variants","path":"/World/Chair"}
{"cmd":"set_variant","path":"/World/Chair","variant_set":"color","variant_selection":"blue"}
{"cmd":"get_pickable_bboxes","paths":["/World/Cube"]}
{"cmd":"get_material_map"}
{"cmd":"get_world_transforms","paths":["/World/Cube"]}
{"cmd":"get_root_prim_path"}
{"cmd":"get_prim_count"}
{"cmd":"shutdown"}
```

Success: `{"ok":true,...data}`. Error: `{"ok":false,"error":"..."}`. The worker is stateful and single-stage; `load` stores `_stage`, and later commands query that loaded stage.

Keep worker and client on this same `cmd`/`ok` protocol. Browser message
envelopes are a separate boundary; translate them in the app-message adapter.

## Worker Command Standard

Use these command names and response shapes for generated `server/pxr_worker.py`
and `PxrWorkerClient`. Do not emit `get_base_transforms`; the current command
is `get_world_transforms`.

| Command | Request fields | Success response |
|---|---|---|
| `load` | `path` | `{"ok": true}` |
| `get_bbox` | none | `{"ok": true, "empty": false, "center": [...], "size": [...], "max_dim": 123.0}` |
| `get_children` | `path`, optional `filters` | `{"ok": true, "children": [{"name": "...", "path": "...", "children": true, "type": "geom"}]}` |
| `get_root_prim_path` | none | `{"ok": true, "path": "/World"}` |
| `get_prim_count` | none | `{"ok": true, "count": 1234}` |
| `get_properties` | `path` | `{"ok": true, "properties": {"typeName": "Mesh", "visibility": "inherited", "material:binding": "/World/Looks/Mat"}}` |
| `get_variants` | `path` | `{"ok": true, "variants": {"color": {"options": ["red"], "selection": "red"}}}` |
| `set_variant` | `path`, `variant_set`, `variant_selection` | `{"ok": true}` |
| `get_pickable_bboxes` | optional `paths` | `{"ok": true, "bboxes": {"/World/Cube": {"min": [0,0,0], "max": [1,1,1]}}}` |
| `get_material_map` | none | `{"ok": true, "material_map": {"/World/Cube": "/World/Looks/Mat"}}` |
| `get_world_transforms` | optional `paths` | `{"ok": true, "transforms": {"/World/Cube": [[...],[...],[...],[...]]}}` |

`get_pickable_bboxes` returns a dictionary keyed by prim path, not a list:

```json
{
  "ok": true,
  "bboxes": {
    "/World/Mesh": {
      "min": [-1.0, 0.0, -1.0],
      "max": [1.0, 2.0, 1.0]
    }
  }
}
```

The property payload is intentionally a simple object keyed by display/property names. Include `typeName` first, serialize authored attributes, serialize relationship targets as strings or string arrays, and add `material:binding` when `UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()` resolves.

`get_children` with `filters=["USDGeom"]` should match the expected selectable
behavior: it includes prims that are themselves geometry, usually
`UsdGeom.Mesh`, not arbitrary containers that only contain geometry descendants.

## Build The Worker And Client

Implement only the commands the app needs. The complete table above describes
compatible optional coverage, not a requirement to create another hierarchy or
picking implementation. Keep runtime transforms and selection in the viewer
owner; worker results describe the worker's composed USD snapshot.

1. Start the worker with an explicitly configured worker-environment Python
   executable, the generated script's absolute path, UTF-8 text pipes, and
   unbuffered output (`-u`). Resolve that environment through
   [dependencies](../dependencies/README.md). Do not inherit renderer native
   library search paths into an isolated USD worker without verifying them.
2. The worker imports `pxr`, stores one active `Usd.Stage`, and dispatches
   requests through a `cmd`-to-handler dictionary. Its stdin loop catches JSON,
   validation, and handler errors and emits exactly one `ok: false` response;
   successful handlers emit `ok: true` with the table's fields. Flush every
   response. On EOF exit; on `shutdown` acknowledge then exit. Unknown commands
   return an error rather than crashing the process.
3. For `load`, open a candidate stage and check it before replacing the active
   stage. Use an anonymous session layer for temporary variant edits so the
   source USD is never saved or modified. Reject other commands until a stage
   has loaded. `get_children`, properties, variants, bounds, root detection,
   and JSON serialization use the building blocks in [README.md](README.md).
4. For optional `get_world_transforms`, use
   `UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)`
   and serialize four matrix rows. For `get_pickable_bboxes`, compute each
   requested geometry prim's aligned world bound with `UsdGeom.BBoxCache`;
   omit empty bounds. For `get_material_map`, resolve
   `UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()` and return
   concrete material paths. Do not invent an `EffectLayer` suffix unless
   that prim actually exists for the app's material-effect workflow.
5. In the client, allow one outstanding request per worker using a lock or a
   dedicated request queue. This protocol has no request ID: write one line,
   flush, read exactly one response. Drain stderr separately to a bounded log
   or file so a full pipe cannot deadlock the worker. Keep blocking requests
   off UI and ovstream callback threads.
6. Bound each request with a timeout. Use a background stdout reader and a
   queue rather than an unbounded `readline()` on the caller thread. EOF,
   malformed JSON, broken pipes, and timeout invalidate that client session.
   Terminate and join it before restarting: a late response must never be
   consumed as the next request's result. Reload the current scene into a new
   worker before serving queries again.
7. Carry the viewer's scene generation alongside each queued request and
   result. Discard stale results after load/reset. Changing a variant in this
   worker changes only its snapshot: return the accepted variant choice to
   the runtime owner, which rebuilds/publishes the viewer session as described
   in [stage management](../stage-management/README.md). Invalidate metadata,
   bounds, and material caches after recomposition.
8. On shutdown, stop accepting requests, send `shutdown` when the channel is
   healthy, wait with a deadline, then terminate/kill and reap if necessary.
   Close pipes and join reader threads. Stop and restart must be safe after
   partial startup as well as normal use.

Validate with a small authored stage: query before load, successful load,
missing prim, invalid JSON/command, worker crash, timeout, scene switch during
a query, and shutdown. Variant changes must appear in the rendered viewer
after owner publication and leave the original USD unchanged.
