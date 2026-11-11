# ovui Full Editor

## Triggers

Use this reference for the complete USD Viewer editor: docking, RTX viewport,
stage browser, property inspector, transform tools, undo/redo, themes, content
browser, and layer stack.

Use `ovui_widgets.app.application.Application` for the complete editor. For a
minimal single-window shell, read `local-viewer`. For custom controls and panels,
add `viewer-control-patterns` and `viewer-data-view-patterns`.

## Start With The Selected Provider

Resolve the UI, widgets, and adapters through `references/dependencies` before
imports. Read the matching upstream `AGENTS.md` and application/widget skills;
they own exact bootstrap, provider, and adapter contracts. Discover the current
composition recipe and source map from that upstream router before writing the
app. Verify the imports and calls below against the resolved package's exports,
examples, and signatures; adapt them when upstream changes.

Launch the stock editor with an optional scene:

```python
from ovui_widgets.app.application import Application

Application().run(usd_path="path/to/stage.usd")
```

The equivalent module launch is `python -m ovui_widgets.app path/to/stage.usd`;
the console entry point is `ovui-widgets`.

Choose the OpenUSD or native OVStage provider explicitly from its supported
runtime contract. With both adapters installed, the stock application defaults
to OpenUSD; `OVUI_DATA_ADAPTER_PROVIDER=ovstage` selects the native provider.
The OpenUSD editor's verified in-process USD setup is distinct from the default
isolated query workers used by lightweight viewers. Do not copy custom USD
build paths or worker package pins into this editor's environment.

## Compose Through Widgets And Adapters

Widgets live in `ovui_widgets`; adapter interfaces live in
`ovui_data_adapters.common`, and concrete OpenUSD adapters are exported from
`ovui_data_adapters.openusd`. Confirm these namespaces in the current upstream
source map before composing widgets.

| Surface | Current import |
|---|---|
| Application | `ovui_widgets.app.application.Application` |
| RTX viewport | `ovui_widgets.viewport.viewport_widget.ViewportWidget` |
| Stage browser | `ovui_widgets.stage.widget.stage_widget.StageWidget`, `ovui_widgets.stage.window.StageWindow` |
| Property inspector | `ovui_widgets.property.window.PropertyWindow` |
| Content browser | `ovui_widgets.content.ContentBrowserWindow` |
| Layer panel | `ovui_widgets.layers.LayerWindow` |
| OpenUSD rendering and scene state | `ovui_data_adapters.openusd.OvRtxRendererAdapter`, `UsdStageAdapter`, `UsdPropertyAdapter`, `UsdTransformAdapter`, `UsdLayerStackAdapter` |

Prefer the stock Application for complete editor lifecycle. Its provider session
opens the scene and supplies adapters; its file-open flow prepares or reuses the
renderer before activating the replacement, clears undo state, and wires the
panels. Do not replace this with a second stage owner or a fresh renderer on
every file open.

For individual widgets, follow the upstream composition recipe:

1. Initialize the shell and register the shared settings, scheduler, recent-file,
   selection, and undo services before building panels.
2. Use one `SelectionBus` for viewport, tree, and inspector. Set its instance
   before constructing `PropertyWindow`.
3. Open the scene through the selected provider's supported lifecycle. For
   OpenUSD embedding, use the upstream renderer construction/import order.
4. Supply the viewport's renderer, bus, services, and stage adapter provider;
   attach transform/stage adapters and the undo manager. Wire the property
   adapter factory and stage adapter into the inspector.
5. Let the viewport render through its frame-clock contract. Retire scene-bound
   callbacks and services at replacement or shutdown.

The upstream source map supplies constructors and required calls; a
bare `ViewportWidget(renderer=...)` does not constitute complete editor wiring.
Keep lazy frame builders limited to widget creation; callback state must exist
before selection, attach, or adapter callbacks run.

## Interaction And Validation

Route pointer gestures through `viewer-input-routing` and give transform tools
first refusal. Keep undoable mutations behind the provider's adapters and command
stack. Use `prim-transform-safety` for custom live transforms.

Validate the actual desktop or supported headless delivery path, using upstream
inspector guidance for real mouse/keyboard interaction. Capture:

- a nonblank OVRTX frame, viewport resize, and camera navigation;
- selection synchronized across tree, viewport, and inspector, including outline
  changes while the camera stays stationary;
- create, undo, redo, and a subsequent edit of the created prim;
- Delete and Backspace in a numeric property field without deleting the prim;
- scene replacement, cleared stale selection, reconnect when used, and shutdown.

Let the provider adapter own selection-outline activation and renderer update
requirements. Implement selection through its current public API and keep the
render loop advancing so changes appear even while the camera is stationary.
Use the upstream keyboard-focus contract so text editing consumes Delete and
Backspace before scene-level commands. Confirm scene replacement and shutdown
release the provider's renderer, image resources, and scene-bound callbacks.

If the renderer fails, surface the error and preserve usable non-render panels;
a mock renderer or blank viewport does not validate the viewer.
