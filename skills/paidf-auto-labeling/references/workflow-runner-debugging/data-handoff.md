# Data Handoff Contract

PAIDF auto-labeling containers exchange state through the DAFT scene directory
for each `DataEntry`.

## Input Record

Each JSONL line should include:

```json
{"id": "clip-001", "media_path": "/experiment/source/clip-001.mp4", "data_path": "/experiment/data/clip-001"}
```

- `media_path` starts as the caller-provided source media, but `prepare_input`
  may rewrite it to a staged effective input such as `sidecars/active.*`.
  Consumers should treat `data_entry.media_path` as the effective input because
  the runtime intentionally uses `SceneContext.from_input(data_entry.media_path)`.
- `data_path` is the shared scene directory used by every stage container.
- The runner mounts local media parents read-only and scene directories
  read-write. Services communicate by writing scene artifacts and
  `sidecars/pipeline_state.json`.

## Scene Layout

```text
<scene>/
  raw/
  contextual/
  task/
  sidecars/
    raw.<ext>
    active.<ext>
    pipeline_state.json
    logs/
```

`sidecars/raw.<ext>` preserves the original media staged by the core pipeline
contract. `sidecars/active.<ext>` is the current media handoff for services that
use the core linear pipeline. Standalone Docker stages also record explicit
artifact paths in `pipeline_state.json`.

## Stage Expectations

| Stage type | Reads | Writes | Active media |
|---|---|---|---|
| Super-resolution | source or active media | enhanced media artifact and `enhanced_media` state | updates only after verified success; auto-gated skips keep active media unchanged |
| Detection/tracking | enhanced media if present, else original media | `contextual/objects.json`, `contextual/instances.json`, optional overlays | unchanged |
| Captioning | tracking overlay, enhanced media, or original media based on `input_source` | captioning sidecars and contextual captions | unchanged |
| VQA | active/original media plus question bank and prior evidence | task QA evidence | unchanged |
| Reasoning | accumulated scene evidence | DAFT `task/`/`contextual/` files | unchanged |
| Training export | accumulated scene outputs across entries | aggregated dataset artifacts | unchanged |

Detection/tracking consumes SR output when `enhanced_media.success` points to an
existing file, but it does not replace active media. Captioning `input_source=auto`
chooses a tracking annotated/re-ID overlay first, then enhanced media, then
original media. For ID-grounded captions, request a tracking overlay and pass
`--input-source tracking` to captioning.

For mixed-resolution video, use SR `resolution_policy=auto`. The expected
handoff is:

- High-resolution input: `enhanced_media.success=false`, `output_path=null`, no
  `sr_output.*`, and `sidecars/active.*` remains byte-identical to `raw.*`.
- Low-resolution input: `enhanced_media.success=true`, `output_path` points to
  `sidecars/sr_output.*`, and `sidecars/active.*` is promoted to that SR output.
- Downstream stages should read the active media or the explicit
  `enhanced_media.output_path`; they should not infer SR success from the
  original input resolution alone.

## Validation

Before starting the next container, verify:

- `sidecars/pipeline_state.json` exists when a stage updates shared state.
- Required task outputs exist and are nonempty.
- Enhanced/active media paths point to existing files before downstream stages
  rely on them.
- Tracking overlays exist when captioning or VQA prompts reference visible IDs.
- Stage logs are under `data_path/sidecars/logs` or the experiment `logs/`
  directory.
