# RT-CV Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys | Foundation | Model family |
|---|---|---|---|
| Detection and tracking (module) | `rtvi-cv` | none — Foundation-independent | chosen via `DS_MODEL_FAMILY` |
| Alerts perception | `perception-alerts` | `alerts` | GDINO |
| Search detection and tracking | `perception-2d-fusion` | `search` | RT-DETR |
| Warehouse 2D perception | `perception-2d` | `warehouse` | RT-DETR (warehouse) |
| Warehouse 3D perception | `perception-3d`, `ds-configurator-3d` | `warehouse` | Sparse4D |

`rtvi-cv` is the profile-neutral module. It is a real `COMPOSE_PROFILES` key that
joins the graph on its own, so a build can request detection without adopting the
`alerts` or `search` Foundation and without inheriting that Foundation's model
family.

Which one to use follows the minimize-additions rule in `../composition.md`:

- Deriving from a Foundation that already ships RT-CV in the requested model
  family — keep that Foundation's clone. It is the smaller delta.
- Composing from modules, or the requested family is not the one the Foundation
  ships — use `rtvi-cv` and set `DS_MODEL_FAMILY`.

RT-DETR and GDINO are not interchangeable — each requires its own configs,
mounts, and class-label taxonomy. With `rtvi-cv` the family is selected rather
than inherited, but the taxonomy consequence is the same. The detector model
family and its emitted class-label taxonomy are authoritatively defined in
`skills/deployment/vss-deploy-detection-tracking-2d/references/integrate-vss-detection-tracking-2d.md`;
the mapping above is the composition surface, not a second source of truth.

## Required peers

- When deriving from a Foundation, use the service key that Foundation defines
  (`perception-alerts`, `perception-2d-fusion`). When composing from modules, use
  `rtvi-cv`. The shared `perception` service in `compose.yaml` remains an
  `extends` template that never joins the project, and is still not a profile key.
- `rtvi-cv` declares every `depends_on` as `required: false`, so it can be
  selected on its own and pulls nothing in implicitly — its peers must be listed
  explicitly, including the Kafka set below.
- `rtvi-cv` needs no blueprint config tree. With neither config mount set,
  ds-start seeds `configs/` from the in-image `reference-configs/<set>` matching
  `DS_MODEL_FAMILY`. Those files carry `<TOKEN>` placeholders for deployment
  model paths: supply each as `RTVI_CV_REF_<TOKEN>` or startup stops and lists
  the ones still missing.
- Warehouse perception additionally requires the blueprint configurator
  ([`configurator.md`](configurator.md)) and the VIOS peers
  ([`vios.md`](vios.md)); `ds-configurator-3d` is the 3D-only DeepStream config
  adaptor.
- Output topic differs by warehouse mode: `2d` publishes tracked detections on
  `mdx-raw`; `3d` (Sparse4D) publishes BEV frames directly to `mdx-bev` and
  leaves `mdx-raw` empty.
- Kafka-backed pipelines require `kafka`, `kafka-topic-init-container`, and
  `broker-health-check`.
- Search RT-CV requires checked-in model/config mounts; model download is
  handled by ds-start phase 0 when `DS_MODEL_DOWNLOAD=auto`.
- Alerts CV mode normally feeds Behavior Analytics; search mode feeds Search
  analytics. Do not add both consumers unless explicitly requested.
- This is a singleton owner: one detector instance per build. When multiple
  pipelines or consumers need detection in one build, they share that single
  detector — resolve to one service key and one model family, not two.
- Selecting or changing the detector/model family is done through the env knobs
  below (`MODEL_TYPE`, `MODEL_NAME_2D`, `DS_MODEL_FAMILY`, `VISION_ENCODER_*`).
  The detector ONNX (and the Search SigLIP vision encoder) is downloaded by the
  RT-CV container at first boot (ds-start phase 0 when `DS_MODEL_DOWNLOAD=auto`)
  from its mounted `models-download.json` into `${VSS_DATA_DIR}/models/`; no
  host-side staging is required. This changes **no service definition**, so it
  needs no `.yml` patch: do not patch `perception-2d-fusion` or `rtvi-cv` for
  a model or detector swap. Selecting and configuring `rtvi-cv` is likewise an env
  delta only — its mounts are interpolated knobs, so it never requires a patch.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VSS_RT_CV_IMAGE`, `VSS_RT_CV_TAG` | Select the RT-CV image. |
| `RT_CV_DEVICE_ID`, `RTVI_CV_PORT`, `RTVI_CV_HOST_PORT` | Select GPU and ports. |
| `MODEL_TYPE`, `MODEL_NAME_2D`, `DS_MODEL_FAMILY` | Select the detector/model family supported by mounted configs. This also fixes the **class-label taxonomy** — the exact class names and their casing emitted on `mdx-raw`. Different model families emit different label sets and casing, so Foundations that ship different families are not interchangeable here. |
| `VISION_ENCODER_MODEL`, `VISION_ENCODER_VERSION` | Select the Search vision encoder NGC package. |
| `NUM_SENSORS`, `STREAM_TYPE`, `DS_MESSAGE_RATE` | Configure input count and event transport. |
| `DS_TRACKER_REID`, `DS_SHOW_SENSOR_ID` | Toggle supported tracking metadata. |
| `HARDWARE_PROFILE`, `PERCEPTION_DOCKERFILE_PREFIX` | Select hardware-specific behavior exposed by the Foundation. |
| `RTVI_CV_MOUNTED_CONFIG_DIR`, `RTVI_CV_DIRECT_CONFIG_DIR` | (`rtvi-cv` only) Supply a config tree. `MOUNTED_` is staged into `configs/` once at startup; `DIRECT_` binds onto `configs/` in place, for configs a configurator rewrites while the service runs. Setting either disables reference-config seeding. |
| `RTVI_CV_MODELS_DIR`, `RTVI_CV_MODELS_MANIFEST` | (`rtvi-cv` only) Host model tree mounted at `/opt/storage/`, and the `models-download.json` that ds-start phase 0 reads. |
| `RTVI_CV_REF_<TOKEN>` | (`rtvi-cv` only) Value for a `<TOKEN>` placeholder in a seeded reference config, e.g. `RTVI_CV_REF_PATH_TO_ONNX_MODEL`. Tokens appearing only on commented-out lines are ignored. |
| `RTVI_CV_ENV_FILE` | (`rtvi-cv` only) Extra env file for keys the service definition does not declare (`SPARSE4D_*`, `VISION_ENCODER_*`, `NUM_SENSORS`, `MODEL_NAME_2D`, `GST_PLUGIN_PATH`). Compose gives `environment:` precedence over `env_file:`, so it cannot override a key the definition already lists. |
| `RTVI_CV_REFERENCE_CONFIG_SET`, `RTVI_CV_REFERENCE_CONFIGS` | (`rtvi-cv` only) Override the `DS_MODEL_FAMILY` -> reference-set mapping, and control seeding: `auto` (default), `never`, or `require` (fail when no set matches). |
| `RTVI_CV_ENGINES_DIR`, `RTVI_CV_EXTRA_MOUNT_1..3` | (`rtvi-cv` only) TensorRT engine cache at `/opt/engines/`, and free-form `src:dst[:opts]` mounts for anything else a build needs. |

Downstream consumers that filter on class labels (Behavior Analytics, for
instance) key on this detector's emitted taxonomy. In a combined build that
converges on a single detector, align those consumer configs to the resolved
model family's label set and casing, not to whatever a source profile's config
happened to ship.

## Placement and sizing

RT-CV has a fixed footprint determined primarily by its model family and stream
count. Prefer a dedicated device; share only when the measured combined budget
fits. See `../sizing.md` for placement resolution and starting stream counts.

## Sources

- `deploy/docker/services/rtvi/rtvi-cv/rtvi-cv.yaml`
- `deploy/docker/services/rtvi/rtvi-cv/compose.yaml`
- `deploy/docker/developer-profiles/dev-profile-alerts/compose.yml`
- `deploy/docker/developer-profiles/dev-profile-search/video-analytics-2d-app/compose.yml`
- `deploy/docker/services/rtvi/rtvi-cv/ds-start.sh`
- `skills/deployment/vss-deploy-detection-tracking-2d/references/environment.md`
- `skills/deployment/vss-deploy-detection-tracking-2d/references/integrate-vss-detection-tracking-2d.md`
