# Image Attribute Augmentation payload contract

Use `scripts/payload.py` as the installed-skill validator. It mirrors the fields consumed by
`ImageAttributeAugmentationDagPayloadConfig`
(`airflow/dags/workflows/image_attribute_augmentation_dag/models/payload.py`) and the nested shared
task models without importing the Airflow repository.

## Top level

| Field | Rule |
|---|---|
| `input_path` | Storage directory (`s3://`, `http://`, `https://`) whose immediate subdirectories are person IDs |
| `max_imgs` | Integer or `null`; positive limits person IDs, zero/negative processes all IDs |
| `output_directory` | Storage directory; the DAG appends `/<run_id>/` per run |
| `external_services` | Service mode switch; defaults to `true` |
| `service_lifecycle` | `vlm_service`, `llm_service`, `image_edit_service`; enabled only in internal mode |
| `cosmos` | Image-edit augmentation configuration |
| `event_and_person_attribute_search` | Attribute search / captioning configuration |
| `enable_performance_reporting` | Optional bool, default `false`; writes a YAML + HTML dashboard under `reports/` |

Every field has a default in the server model, so a run triggered with no `conf.payload` uses the
defaults. **Always render an explicit payload for user runs — never omit `conf.payload` and never
copy endpoint URLs or bucket paths from the repository's checked-in dev or CI payloads.**

`cosmos.output_directory` and `event_and_person_attribute_search.output_directory` must equal the
top-level value, and their `external_services` flags must match the top-level value. The model
auto-fills both when omitted and raises if they conflict.

The `service_lifecycle` flags are derived from `external_services`: the model forces all three
`enabled` values to `not external_services`, logging a warning if the payload disagreed.

## Service modes

External mode requires all of:

- `cosmos.vlm_service_url`
- `cosmos.llm_service_url`
- `cosmos.image_edit_service_url`
- `event_and_person_attribute_search.llm_service_url`

`event_and_person_attribute_search.vlm_service_url` is optional. The renderer maps the same VLM/LLM
URLs into both stages. URLs must be HTTP(S). Model names are optional non-empty strings. Credentials
belong in the endpoint/server environment, never payload templates.

These URL fields have non-empty placeholder defaults in the server model, so the "required" check
only fails when a value is explicitly `null`. **Do not rely on the defaults.** Always ask the user
for their endpoint URLs and pass them explicitly in the payload.

Internal mode emits `null` endpoint URLs and enables the three lifecycle entries. The DAG resolves
deployed endpoints at runtime through service-startup XCom. Each replica count must be at least one,
and each replica claims one GPU.

## Attribute search section

`event_and_person_attribute_search` accepts `config_file`, `mode` (default `image_pas`),
`person_input_dir` (legacy fallback), `vlm_service_url`, `vlm_model`, `llm_service_url`, `llm_model`,
`external_services`, and `output_directory`. The DAG generates and uploads `config_file` per mapped
task, so leave it unset. There are no `tracker` or `threshold` fields.

## Augmentation controls

- `cosmos.num_augmentation` is singular and must be at least one.
- `max_imgs` counts person-ID folders, not individual images.
- Estimated augmented scenes are `processed person IDs * num_augmentation`.
- `cosmos.base_config_path` is an optional `s3://`, `http://`, or `https://` override pointing at a
  custom base `cosmos_config.yaml` to use instead of the DAG's built-in default (`--base-config-path`
  in `scripts/payload.py render`). Leave it unset to use the checked-in default.

## Variable distributions

Use this wrapper shape:

```json
{
  "variables": {
    "shoe_type": {"boots": 0.5, "sneakers": 0.5},
    "colors": {
      "black": {"charcoal black": 0.7, "jet black": 0.3},
      "blue": {"navy blue": 1.0}
    }
  },
  "conditional_variables": {
    "shoe_color": {
      "depends_on": "shoe_type",
      "distributions": {
        "boots": {"black": 1.0},
        "sneakers": {"black": 0.5, "white": 0.5}
      }
    }
  }
}
```

Weights must be finite, non-negative, and have a positive total. They need not sum to one because
the runtime treats them as relative weights. A lookup variable maps lookup keys to weighted
distributions. Every possible parent value must have a conditional distribution.

If the distribution is omitted or `{}`, the renderer emits the deterministic defaults from
`default_variable_distribution()` in `airflow/dags/shared/models/payload.py` (black hoodie / blue
jeans / white sneakers). See `assets/variable-distribution.json` for a safe editable example.

### Creating a custom distribution file

`--variable-distribution` takes a JSON **file path**, not an inline value. When a run requires
specific attributes, write a temporary file first, then pass it to the renderer:

```bash
# All six clothing attributes are REQUIRED — bias one by changing its weights,
# e.g. force a yellow top outer color while keeping the other five at defaults.
cat > /tmp/my-distribution.json <<'EOF'
{
  "variables": {
    "top_outer_color": {"yellow": 1.0},
    "top_outer_type":  {"hoodie": 1.0},
    "bottom_type":     {"jeans": 1.0},
    "bottom_color":    {"blue": 1.0},
    "shoe_type":       {"sneakers": 1.0},
    "shoe_color":      {"white": 1.0}
  }
}
EOF

python scripts/payload.py render \
  --input-path s3://... --output-directory s3://... \
  --service-mode external \
  --variable-distribution /tmp/my-distribution.json \
  --output /tmp/iaa-payload.json
```

`assets/variable-distribution.json` is the canonical editable example — copy and modify it rather
than editing it in place. See the warning below before dropping any attribute.

**Biasing one attribute — you must still supply all six.** ⚠️ A *partial* `variable_distribution`
fails the run. The augmentation captioner initializes against the template
`"{top_outer_color} {top_outer_type}, {bottom_color} {bottom_type}, {shoe_color} {shoe_type}"` and
raises `Failed to initialize captioner: Template has placeholders {...} not found in variables` if
**any** of the six clothing attributes is missing from both `variables` and `conditional_variables`
combined — an attribute supplied only as a `conditional_variables` entry (keyed on another
variable via `depends_on`) still counts, which is exactly how the bundled
`assets/variable-distribution.json` example supplies `shoe_color`. (The task pod still gets deleted,
so the KubernetesPodOperator reports `success`, but no `output.jpg` is written and the downstream
`validate_outputs` task fails — the error is easy to misread as a storage problem.)

To bias one attribute, include **all six** and change only the target's weights. To make the bottoms
yellow, keep the defaults and override `bottom_color`:

```json
{
  "variables": {
    "top_outer_color": {"black": 1.0},
    "top_outer_type":  {"hoodie": 1.0},
    "bottom_type":     {"jeans": 1.0},
    "bottom_color":    {"yellow": 1.0},
    "shoe_type":       {"sneakers": 1.0},
    "shoe_color":      {"white": 1.0}
  }
}
```

Omitting `variable_distribution` **entirely** is fine — the renderer then emits all six defaults;
the failure only happens when you pass a *partial* set. The renderer embeds the distribution
verbatim under `cosmos.variable_distribution` (the field the DAG reads), so it always travels
*inside* the payload — there is no separate upload step.

Attribute values must come from the vocabulary in
`airflow/dags/workflows/image_attribute_augmentation_dag/configs/cosmos_config.yaml`
(`captioning.llm.verification_options`) so the augmentation prompt and attribute verification stay
consistent:

| Attribute | Accepted values |
|---|---|
| `top_outer_color`, `bottom_color` | beige, black, blue, brown, camouflage, green, grey, orange, pink, purple, red, white, yellow |
| `top_outer_type` | camisole, knee-length coat, hoodie, cropped jacket, robe, sweater, vest |
| `bottom_type` | dress, leggings, jeans, shorts, skirt |
| `shoe_type` | barefoot, boots, flip-flops, high heels, sandals, sneakers |
| `shoe_color` | none, beige, black, blue, brown, green, grey, orange, pink, purple, red, white, yellow |

There is no literal `pants`; use `jeans` for generic trousers. A value outside the vocabulary still
fills the image-edit prompt but may fail caption verification.

## Validation

Validate an existing payload and optionally write its normalized form:

```bash
python scripts/payload.py validate \
  --payload /tmp/iaa-payload.json --output /tmp/iaa-normalized.json
```

The server's Pydantic model ignores unknown fields. The bundled renderer deliberately outputs only
known fields so misspellings do not silently become part of generated payloads.
