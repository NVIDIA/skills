# SMT payload contract

Build payloads for `SmtDefectImageGenerationPayload`; `build_work_items` in the
trusted checkout compiles them into simulation and AnomalyGen jobs.

## Collections and modes

| Field | Contract |
|---|---|
| `api_version` | Exactly `"1.0"` |
| `output_directory` | Required, nonempty writable S3 base for a run |
| `pipeline_mode` | `full` (model default), `simulation`, `sim2real`, `defect`; set it explicitly |
| `input_scene` | Required outside defect mode; S3 scene or scene ZIP readable by Simulation |
| `golden_generation` | Required outside defect mode |
| `defect_generation` | Required for full and defect modes |
| `external_services` | Image-edit placement only: `false` deploys it in the cluster, `true` calls `cosmos.image_edit_service_url`. It does not affect AnomalyGen's VLM |
| `cosmos.image_edit_service_url` | External image-edit URL; required when `external_services` is `true` in `sim2real` or `full` |
| `cosmos.image_edit_model` | Optional request model name; does not change the checkpoint served internally |
| `service_lifecycle.image_edit_service.replicas` | Internal image-edit replicas, one GPU each; default 1. The starter payloads set 8 |
| `enable_performance_reporting` | Optional; default false |

The model accepts other `cosmos` and `service_lifecycle` fields, but this workflow
ignores them and the checker rejects them.

Each collection has optional `defaults` and a nonempty `configurations` array.
Every configuration needs a unique ID within its collection matching
`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`; the ID also names its output folder. All
other configuration entries are component override dictionaries. Unknown
top-level fields are rejected.

Merging is `checked-in component template → collection defaults → configuration`.
Dictionaries merge recursively; lists and scalars replace. `per_defect_counts` is
a special case: its entire dictionary is replaced, including by `null`.

With G golden configurations and D defect configurations, full mode builds G
simulation jobs and G × D AnomalyGen jobs. Each golden configuration is a batch,
not one output image. After rendering, one CPU augmentation task runs per
selected image. Simulation and sim2real create no AnomalyGen jobs; defect mode
creates D jobs.

`cosmos.external_services` and `cosmos.output_directory`, if provided, must match
the top level; the model fills them when omitted and enables or disables the
image-edit service itself. Only `sim2real` and `full` run image-edit, so
`external_services` has no effect in `simulation` or `defect` mode; leave it
`false` there.

## Component overrides

| Collection | Override | Passed to | Set by the workflow; do not override |
|---|---|---|---|
| `golden_generation` | `image_generation` | CDF Simulation JobConfig; template `configs/smt/image_generation_config.yaml` | `scene.usd` (from `input_scene`), `output_dir` |
| `golden_generation` | `simulation_output` | Simulation-to-augmentation handoff, below | — |
| `golden_generation` | `augmentation` | CDF Augmentation `PipelineConfig`; template `configs/smt/cosmos_config.yaml` | `data`; the `image_edit` endpoint's `url` and `model` |
| `defect_generation` | `anomaly_input` | Component-type to class mapping for full mode, below | — |
| `defect_generation` | `anomaly_generation` | CDF AnomalyGen `run_pipeline` config; template `configs/smt/anomaly_gen_config.yaml` | `input`, `output`, `recipe`, `checkpoint`, `staging.publish` |

The checker rejects any other override name and the workflow-managed keys above.
Settings inside an override belong to the component, so look them up in the
pinned component documentation linked from SKILL.md. CDF Simulation and
AnomalyGen reject unknown keys when their task starts, so a misspelled key fails
that task rather than the submission.

## Golden configuration

`simulation_output.pass_name` defaults to `component_focus`.
`simulation_output.image` selects a single capture. Give its path relative to the
golden's `simulation/` output, which begins with the pass name: for example
`component_focus/<path listed for that capture in the pass's dataset_index.json>`.
Absolute paths and traversal are rejected. Omit it to augment every RGB capture
in the pass. Simulation-only runs ignore this handoff.

Example **payload-only** Artix golden run; replace the illustrative locations
before execution and adapt the preset for any other board:

```json
{
  "api_version": "1.0",
  "pipeline_mode": "sim2real",
  "input_scene": "s3://example-bucket/scenes/artix-dc-scm.zip",
  "output_directory": "s3://example-bucket/dig-output",
  "external_services": false,
  "service_lifecycle": {"image_edit_service": {"replicas": 1}},
  "golden_generation": {
    "defaults": {
      "simulation_output": {"pass_name": "component_focus"},
      "image_generation": {
        "capture": {"passes": {"component_focus": {
          "limits": {"max_images": 2},
          "roi_selection": [{"component_types": ["ic"], "count": 2, "padding": 0.5}]
        }}}
      }
    },
    "configurations": [{"id": "golden-ic"}]
  }
}
```

The selected pass limit does not disable the template's other capture passes;
review them when budgeting rendering time. For simulation-only, change the mode
to `simulation`; for full mode, add the defect collection below.

## Selecting components

The component passes decide which components become crops. Give
`component_focus` and its mask counterpart `component_focus_roi` the same
`roi_selection` and `limits`, so that the masks match the crops. Requests name
components in three ways:

| The request names | Selector |
|---|---|
| A kind, such as "the ICs" or "the passives" | `component_types` with the board's type names, for example `["ic"]` or `["capacitor"]` |
| A part or package, such as `0603_H100` | `component_types`, which matches type, family, package or footprint metadata; or an exact `component_filter` field: `component_type`, `package_type` or `footprint_id` |
| Specific parts, such as C2 and C14 | `refdes` |

A part name in a request may not be the name the board's metadata uses. If you
cannot confirm a match from the board's metadata, ask for the reference
designators or the metadata value; a selector that matches nothing fails the
simulation task.

- `component_types` selects one match unless you set `count`.
- `refdes` selects every listed part; `component_filter` and `defects` select
  every match.
- `limits.max_images` caps the crops. The template sets 24; `-1` removes the cap.

For example, "good images of 0603_H100, max 3" is this `image_generation`
override:

```json
{"capture": {"passes": {
  "component_focus": {"limits": {"max_images": 3},
    "roi_selection": [{"component_types": ["0603_H100"], "count": 3, "padding": 0.5}]},
  "component_focus_roi": {"limits": {"max_images": 3},
    "roi_selection": [{"component_types": ["0603_H100"], "count": 3, "padding": 0.5}]}
}}}
```

For "N good images" with no named part, keep the template's selection or use
`{"all_components": true, "padding": 0.5}` with `max_images: N`. For every
instance of a part ("full board coverage"), use a `component_filter` selector
with `max_images: -1`. Golden images are component crops; whole-board renders
stay in the simulation output of the template's full-board passes.

## Adapting to another board

The `image_generation` template is written for the Artix sample board. For
another board, supply `input_scene` and override the board-specific template
settings, usually in `golden_generation.defaults.image_generation`:

- `scene.root`, the board's root prim.
- `operations.solder_joints.options.components`, a reference-designator
  allowlist, and `solder_material_prim_path`.
- `annotations.component_classes` and `annotations.geometry.components`, which
  repeat that allowlist.
- The passes under `capture.passes`: `roi_selection` and `limits` for the
  component passes, and `scan_grid` for the full-board passes with their
  `postprocess.stitch` outputs.

Lists replace the template's lists, so supply complete values. Field meanings are
in the CDF Simulation versioned YAML JobConfig reference. Simulation records
each capture's `component_type` from the board's metadata; in full mode, map every
type the selection produces in `anomaly_input.class_by_component_type`.

## Pose defects

Shift, tombstone and sideflip defects come from Simulation, not AnomalyGen. Run
`sim2real`, and in the golden configuration's `image_generation` override add the
pose operation, operation labels, and crops of the components the operation
moved:

```json
{"image_generation": {
  "operations": {"tombstone": {"ratio": 1.0, "angle": [70, 90],
    "component_filter": {"component_type": ["0603_H100"]}}},
  "annotations": {"operation_labels": "defect"},
  "capture": {"passes": {
    "component_focus": {"limits": {"max_images": -1},
      "roi_selection": [{"defects": ["tombstone"], "padding": 0.5}]},
    "component_focus_roi": {"limits": {"max_images": -1},
      "roi_selection": [{"defects": ["tombstone"], "padding": 0.5}],
      "outputs": {"semantic_types": ["class", "defect"]}}
  }}
}}
```

- `shift` takes `translate` in millimeters and `rotate_z` in degrees; `tombstone`
  and `sideflip` take an `angle` range in degrees.
- Each operation takes a `count` or a `ratio`. "Full board coverage" is
  `ratio: 1` over the selected components; without a filter, the operation draws
  from every eligible component on the board.
- Requests rarely state magnitudes. Propose the JobConfig reference's example
  values (shift 0.2 mm and 15°, tombstone 70–90°, sideflip 60–80°) and confirm
  them with the user.
- Use one golden configuration per defect type; its ID, such as `tombstone`,
  names the output folder.
- The workflow writes the crops under `goldens/<id>/augmentation/artifact/`,
  because sim2real treats every output as a golden image. The matching defect
  masks are in `goldens/<id>/simulation/component_focus_roi/`.
- Simulation's `missing` operation removes components and cannot be combined
  with pose operations. A "missing" texture defect is AnomalyGen's
  `<class>+missing` instead.

Pose operations run before the template's solder generation, so solder forms on
the moved geometry.

## Augmentation settings

Image-edit uses the SMT manifest's model `nvidia/cdf-sim2real-smt-solderlight`.
The template gives every image the fixed prompt in `captioning.llm.text` (the
Augmentation text captioner), so no LLM or VLM endpoint is involved. That model
was fine-tuned on this prompt only, so a different prompt can degrade its
output; when a request changes the prompt for the default model, say so before
applying it. To change the prompt for a golden configuration, override that
text; any placeholders in it need matching `captioning.llm.variables`.
Generation parameters use a doubled `augmentation` nesting, because the outer
key names the workflow component and the inner key is a section of the
component's schema:

```json
{"augmentation": {
  "captioning": {"llm": {"text": "Render this PCB component crop as a real inspection-line photograph."}},
  "augmentation": {"parameters": {
    "extra_body": {"seed": 2}
  }}
}}
```

These settings apply to every image in the golden batch. `extra_body` merges with
the template, so the model's released settings (`num_inference_steps` 30,
`true_cfg_scale` 4.0 and its `negative_prompt`) stay unless a request changes
them. `endpoints` is a list,
so an override replaces it and must keep the `image_edit` entry; the workflow
fills that entry's `url` and `model` from the service settings. Its
`api_key_env`, `BUILD_NVIDIA_API_KEY`, carries the deployment's NGC API key.

## Full-mode defects

Add this collection to a full payload, replacing the illustrative asset locations
and the AMG endpoint and model with values supported by the user's deployment:

```json
{
  "defect_generation": {
    "defaults": {
      "anomaly_input": {"class_by_component_type": {
        "ic": "IC", "capacitor": "passive_component"
      }},
      "anomaly_generation": {
        "staging": {"fetch": {
          "input": ["s3://example-bucket/assets/UC1/"],
          "recipe": "s3://example-bucket/recipes/pcb.yaml",
          "checkpoint": "s3://example-bucket/models/pcb.pt"
        }},
        "amg_gen_model": "user-configured-vlm",
        "amg_judge_model": "user-configured-vlm",
        "amg_endpoints": [{"model": "user-configured-vlm", "base_url": "https://vlm.example.com/v1"}]
      }
    },
    "configurations": [
      {"id": "ic-bridge", "anomaly_generation": {"per_defect_counts": {"IC+bridge": 3}}},
      {"id": "passive-missing", "anomaly_generation": {"per_defect_counts": {"passive_component+missing": 1}}}
    ]
  }
}
```

Every defect job receives the complete golden batch, so map **every** selected
component type exactly, even when that job requests defects for only one class.
Classes must be nonempty folder names without `/`, `\`, `.` or `..` path
segments. Full mode packages the goldens into `<class>/clean_image/` and appends
that package to `staging.fetch.input`.

Set asset locations only under `staging.fetch`: the `input` list, `recipe` and
`checkpoint`. AnomalyGen overlays the `input` sources in order, so any clean
images they contain join the generation pool; in full mode, stage shared
reference assets only, not another clean-image dataset. The public
`nvidia/Cosmos-AnomalyGen-PCB-Dataset` holds the UC1 reference assets plus five
sample boards in its `clean_image/` folders, with their `cad_mask/` folders.
Stage it without those folders in both modes: in full mode, a defect generated
on a sample board fails `validate_anomalygen_outputs`, and in defect mode the
sample boards take defects meant for the user's images.

`per_defect_counts` keys are `<class>+<defect>` pairs supported by the staged
assets and by the recipe the checkpoint was trained with; values are requested
counts. `num_sdg` (default 1) supplies a total only when counts are null. Counts
pass through unchanged: three requested defects do not mean three outputs per
golden image, nor three distinct inputs. Preserve the requested counts and report
that limitation instead of multiplying them by batch size.

Map a named component's type to the class the recipe uses for it; the starter
payloads use `IC` and `passive_component`. When a request gives no count,
propose one defect per expected crop and confirm it. For "full board coverage",
select every instance of the component (see
[Selecting components](#selecting-components)) and request at least one defect
per instance. AnomalyGen can still reuse some crops and skip others, so report
the coverage it achieved: the number of distinct `input_images` in
`export_manifest.json`. For several defect types on one component, use one defect
configuration per type so that each type gets its own output folder.

AMG proposes and judges defect masks with a VLM. `amg_endpoints` is a workflow
setting rather than an AnomalyGen key: each entry names a `model` and its
OpenAI-compatible `base_url`, including the version path (AnomalyGen appends
`/chat/completions`). `amg_gen_model` and `amg_judge_model` must each equal an
entry's `model`. The workflow writes the entries to the task's model registry and
adds `VLM_API_KEY` to each. The `cdf-anomalygen` skill's agentic mask generation
reference covers endpoint compatibility and model choice.

## Defect-only input

Set `pipeline_mode: "defect"`, retain `defect_generation`, and omit `input_scene`
and `golden_generation`. A component-type mapping is unnecessary. Set
`anomaly_generation.staging.fetch.input` to a list containing both the shared
reference assets and the prepared clean-image dataset:

```json
["s3://example-bucket/assets/UC1/", "s3://example-bucket/prepared-components/"]
```

The prepared dataset contains `IC/clean_image/` or other recipe class folders
directly at its root. A flat image directory or a previous golden
`augmentation/artifact/` folder is not in this layout: organize those images into
class folders before uploading. A list override replaces the complete template
list, so keep the required reference assets.

For a single attached image, upload it to `<class>/clean_image/` under a prefix in
the user's storage and stage that prefix, where `<class>` is the recipe's class
for the component (the starter payloads use `IC` and `passive_component`). If a
request names a component but supplies no image, ask for clean images. Metal,
glass and other non-PCB surfaces need an AnomalyGen recipe for that material; run
AnomalyGen directly for those.

## Credentials and checker limits

AMG settings belong under `defect_generation.*.anomaly_generation`, not
`cosmos.vlm_service_url`. Provide the VLM credential as `VLM_API_KEY` in the
deployment's secrets and apply it with `make setup` and
`make install sdg-controller`, as described in the product's
`docs/getting-started.md`; when it is unset, the task falls back to the NGC API
key. `make install` first deletes every workload in the controller namespace,
including other users' runs, so confirm with the user that nothing is running
before suggesting it. Payload endpoint entries carry a model and base URL,
never a key. Instead of `amg_endpoints`, a registry file that holds its own keys
can be staged with `staging.fetch.amg_model_registry` and
`amg_model_registry: /work/endpoints.yaml`; confirm that it exists without
printing its contents.

The checker rejects placeholder or credential-bearing locations, missing explicit
staging, unsupported override names, workflow-managed keys and placeholder
image-edit URLs in active external mode. It does not check recipe-supported
defect vocabulary, keys inside component overrides, actual component coverage, or
remote object existence; check those before launching.
