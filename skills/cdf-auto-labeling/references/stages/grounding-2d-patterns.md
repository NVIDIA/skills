# Grounding 2D patterns

## Caption precedence

1. `--caption` / config caption (applies to all entries)
2. `<data_path>/sidecars/input.json` → `{"caption": "..."}` (manual / GT override)
3. Captioning artifacts (`sidecars/captioning/image_captions.json`,
   `contextual/image_captions.json`, …)

When composing `captioning` then `grounding_2d`, omit `sidecars/input.json` so
the dense caption feeds grounding.

## Expression filtering (product default)

Prefer, in order:

1. VLM Step-0 `groundable: true|false` (whole countable objects)
2. SAM3 text-encoder length / CJK limits
3. Instance score / bbox area / max instances per expression

Optional `--expression-filter-policy-path` (JSON with `deny_phrases` /
`deny_nouns`) is an advanced escape hatch only. Do not ship domain keyword packs
as the default cookbook path.

## Images and orchestration

- Slim: `grounding-2d-service` (CI / VLM wiring)
- Live: `grounding-2d-sam3-service` from `services/grounding_2d_service/docker/Dockerfile.gpu`
- Weights mounted read-only; VLM is an external endpoint (scales separately)
- No sibling `FROM detection-and-tracking-sam3-service`, no product `PYTHONPATH`

## Validation checks

- Dry-run shows `grounding-2d-sam3-service` (or slim) without `/workspace`
  PYTHONPATH binds for product configs.
- After a run: `sidecars/grounding_2d/grounding_2d.json` has `expressions` / `instances`;
  `sidecars/grounding_2d/step0_expressions.json` lists VLM phrases.
- Failures: missing caption, unreachable VLM, missing `/models/sam3`, CUDA not
  visible — report the concrete log line; do not invent boxes.
