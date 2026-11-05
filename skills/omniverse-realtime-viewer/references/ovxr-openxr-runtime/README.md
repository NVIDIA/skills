# OVXR runtime integration for a viewer

Read [the library route](../dependencies/ovxr.md): project-setup-python or project-setup-c, application-flow, context-creation and stage-publication. This existing entry point remains compatible; raw descriptor, loader and ABI details live upstream.

Start new apps with [the XR viewer recipe](../xr-viewer-recipe/README.md). Keep one renderer/stage owner and one OVXR frame owner. Isolate XR dependencies from shared GA build targets. Transport handlers enqueue intent instead of operating SDK objects.

Add requested [input](../xr-spatial-input/README.md), [interactions](../ovxr-python-interactions/README.md), [UI](../ovxr-composition-ui/README.md), [passthrough](../xr-session-modes-and-passthrough/README.md) and [mirror](../xr-desktop-mirror/README.md). Canonical navigation/assets/settings survive eligible context refresh; borrowed handles and captures are generation-scoped. Prove first submitted client pixels before tools, then validate the integrated story.
