# Driving Remix with the direct tools

Recipes for using the Remix MCP tools. Follow the prerequisites and save/readback requirements in [SKILL.md](../SKILL.md).

## `add_asset_reference`

Add an asset reference (USD/MDL) onto an existing prim.

Attach an asset reference to "<prim_path>".

1. `remix_append_prim_reference_file_path` prim_path="<prim_path>", asset_file_path="<asset_path>". (Skip prim-existence checks — the append surfaces errors.)
2. `remix_get_prim_reference_file_paths` prim_path="<prim_path>"; confirm the new reference is present.

## `replace_model_asset`

Swap the currently-selected 3D model in the viewport with an ingested asset.

Swap the selected model with an ingested asset.

1. `remix_get_available_ingested_assets` asset_type="models".
2. `remix_get_prim_paths` selection=true, prim_types=["models"].
3. Resolve the requested asset from step 1's list and the intended model from step 2. Use an exact path when provided; otherwise require an unambiguous match. If either is missing, report that and ask the user to ingest or select it. If multiple candidates remain, show them and ask the user to choose instead of taking the first entry.
4. `remix_replace_prim_reference_file_path` prim_path=<resolved model path>, asset_file_path=<resolved asset path>.
