# Event Video Generation payload contract

Use `scripts/payload.py` as the installed-skill validator. It mirrors the fields consumed by
`EventVideoGenerationDagPayloadConfig`
(`airflow/dags/workflows/event_video_generation_dag/models/payload.py`) without importing the
Airflow repository.

## Top level

| Field | Rule |
|---|---|
| `input_path` | Storage path (`s3://`, `http://`, `https://`) to one seed image or a directory of seed images |
| `max_images` | Integer; default `10`. Positive limits images taken from a sorted directory listing, zero/negative processes all matching images. No effect when `input_path` names one file |
| `output_directory` | Storage directory; the DAG appends `/<run_id>/` per run |
| `external_services` | Service mode switch; defaults to `true` |
| `service_lifecycle` | `vlm_service`, `llm_service`, `image2video_service`; enabled only in internal mode |
| `cosmos` | Image-to-video augmentation configuration |
| `enable_performance_reporting` | Optional bool, default `false`; writes a YAML + HTML dashboard under `reports/` |

Every field has a default in the server model — the shipped defaults are **non-working
placeholders** such as `s3://<your-input-bucket>/<your-workflow>/input`. **Always render an
explicit payload for user runs — never omit `conf.payload` and never copy endpoint URLs or bucket
paths from the repository's checked-in dev payloads or docs examples.**

`cosmos.output_directory` must equal the top-level value and `cosmos.external_services` must
match the top-level value. The model auto-fills both when omitted and raises if they conflict.

Unlike the Image Attribute Augmentation DAG, there is no separate `event_and_person_attribute_search`
top-level payload section: the auto-labeling task group reads the same validated payload dict
directly, so nothing extra needs to be supplied for captioning, visual QA, or person attribute
search configuration.

The `service_lifecycle` flags are derived from `external_services`: the model forces all three
`enabled` values to `not external_services`, logging a warning if the payload disagreed.

## Service modes

External mode requires all of:

- `cosmos.vlm_service_url`
- `cosmos.llm_service_url`
- `cosmos.image2video_service_url`

The renderer maps all three URLs into `cosmos`. URLs must be HTTP(S). Model names are optional
non-empty strings. Credentials belong in the endpoint/server environment, never payload templates.

These URL fields carry non-working placeholder defaults in the server model (for example
`https://<your-vlm-endpoint>/v1`), so the "required" check only fails when a value is explicitly
absent or blank at the API boundary — validation at the model level still accepts the placeholder
string. **Do not rely on the defaults.** Always ask the user for their endpoint URLs and pass them
explicitly in the payload.

Internal mode enables the three lifecycle entries; `vlm_service_url` / `llm_service_url` /
`image2video_service_url` are not required in that case (the DAG resolves deployed endpoints at
runtime through service-startup XCom). Each replica count must be at least one, and each replica
claims GPU capacity — one GPU per VLM/LLM replica, **two** GPUs per image2video replica (see
[setup-and-preflight.md](setup-and-preflight.md)).

## Augmentation controls

- `cosmos.num_augmentation` is singular and must be at least one.
- `max_images` counts input images (or is ignored entirely for a single-file `input_path`), not
  generated videos.
- Estimated generated videos are `processed images * num_augmentation`.
- `cosmos.base_config_path` is an optional MSC-readable URI to a custom Cosmos YAML base
  configuration. Leave it unset unless you intend to override the bundled
  `cosmos_config.yaml` (captioning prompts, augmentation parameters, evaluators).

## Variable distributions — exactly two required keys

`cosmos.variable_distribution.variables` must contain **exactly** these two keys — no more, no
fewer:

- `anomaly_type`: anomaly label -> non-negative sampling weight.
- `env_type`: environment label -> non-negative sampling weight.

This differs from the Image Attribute Augmentation DAG, which requires *all six* of its clothing
attribute keys (or none). Here the model enforces an *exact set* of exactly two keys — adding a
third key (or omitting either of the two) fails Pydantic validation with:

```
variable_distribution.variables must contain exactly 'anomaly_type' and 'env_type'
```

There is no `conditional_variables` support and no lookup-distribution nesting for this DAG —
each of the two keys is a flat weighted map, unlike IAA's `shoe_color depends_on shoe_type`
pattern. Weights must be finite and non-negative, and each distribution's total must be positive;
they do not need to sum to one because the runtime treats them as relative weights. Sampling is
deterministic for a given payload — the DAG uses a fixed seed (42) when constructing the
per-augmentation configs, so the same payload always yields the same anomaly/environment mix.

Use this wrapper shape:

```json
{
  "variables": {
    "anomaly_type": {
      "person_falling": 0.5,
      "person_running": 0.3,
      "person_fighting": 0.2
    },
    "env_type": {
      "warehouse": 1.0
    }
  }
}
```

If `variable_distribution` is omitted entirely, the renderer emits the deterministic default:
`{"anomaly_type": {"person_falling": 1.0}, "env_type": {"warehouse": 1.0}}`, matching
`default_event_video_generation_variable_distribution()` in
`airflow/dags/workflows/event_video_generation_dag/models/payload.py`. The renderer embeds the
distribution verbatim under `cosmos.variable_distribution` (the field the DAG reads), so it always
travels *inside* the payload — there is no separate upload step.

### Creating a custom distribution file

`--variable-distribution` takes a JSON **file path**, not an inline value. To bias toward a mix of
anomaly types in a retail environment:

```bash
cat > /tmp/my-distribution.json <<'EOF'
{
  "variables": {
    "anomaly_type": {
      "stealing_or_shoplifting": 0.4,
      "person_falling": 0.2,
      "person_running": 0.2,
      "person_fighting": 0.2
    },
    "env_type": { "retail": 1.0 }
  }
}
EOF

python scripts/payload.py render \
  --input-path s3://... --output-directory s3://... \
  --service-mode external \
  --variable-distribution /tmp/my-distribution.json \
  --output /tmp/evg-payload.json
```

`assets/variable-distribution.json` is the canonical editable example — copy and modify it rather
than editing it in place.

Vocabulary is defined by the prompt-writing LLM in
`airflow/dags/workflows/event_video_generation_dag/configs/cosmos_config.yaml`
(`captioning.llm.system_prompt`), which contains explicit generation instructions per
`anomaly_type` value:

| `anomaly_type` value | Behavior generated |
|---|---|
| `person_falling` | Natural stumble/fall to a side-seated or hip-and-hand recovery pose |
| `person_climbing` | Unsafe climb onto a shelf, counter, railing, barrier, fixture, or pallet |
| `person_running` | Rapid athletic jog across the open area |
| `smoking_or_vaping` | Object raised to mouth, visible vapor/smoke exhale |
| `person_fighting` | Brief confrontation reading as pushing or grappling |
| `fire_or_smoke` | Visible small flame or smoke plume near a fixture/machine/shelf |
| `stealing_or_shoplifting` | Concealment or grab-and-go theft sequence |

`env_type` has no fixed enum in code — it is interpolated directly into the LLM prompt as a
"smart-space environment" label. The checked-in example payloads and docs use `warehouse` and
`retail`; `school` is called out in the prompt with special handling (it is rewritten to
"education-facility hallway" / "campus hallway" language and restricted to adult subjects). Use a
short, concrete environment noun the prompt writer can plausibly stage a video in.

A value outside these categories still fills the LLM prompt but was not validated against the
attribute-verification evaluator's expectations, so caption/verification quality is not
guaranteed.

## Validation

Validate an existing payload and optionally write its normalized form:

```bash
python scripts/payload.py validate \
  --payload /tmp/evg-payload.json --output /tmp/evg-normalized.json
```

The server's Pydantic model ignores unknown fields at the `EventVideoGenerationCosmosTaskConfig`
level (`extra="ignore"`) but rejects unknown keys inside `variable_distribution.variables`
(`extra="forbid"` plus the exact two-key check). The bundled renderer deliberately outputs only
known fields so misspellings do not silently become part of generated payloads.
