# Warehouse Industry Profile

Warehouse is the only supported **industry** Foundation. This file owns warehouse
end to end: what the Foundation is made of, what constrains it, how a build is
resolved, and how the result is reached and verified. The rest of the skill's
machinery applies unchanged by reference — see Build and resolve below.

`overrides.env` defines further service lists; the `COMPOSE_PROFILES_WH_*` lists
in the table below are supported. The others are out of scope for this skill —
do not compose or deploy them.

**Warehouse is variant selection, not composition.** Pick the one
`COMPOSE_PROFILES_WH_*` list that `MODE` + `BP_PROFILE` + size identify, expand
it verbatim, and deploy it. Do not add, remove or prune service keys, and do not
treat another Foundation as a starting point: each list is a validated
combination. To change the shape of a deployment, select a different variant.

This file is self-contained for warehouse **facts** — it states them once, and
SKILL.md points here rather than restating them. It carries no intake questions
and no step sequence: variant selection is SKILL.md's Q2w, and the lifecycle is
SKILL.md's Steps. It carries the env layers, build artifacts and resolve
pipeline below, and shares the rest of the skill's machinery by reference:
[`../prerequisites.md`](../prerequisites.md),
[`../credentials.md`](../credentials.md), [`../ngc.md`](../ngc.md),
[`../data-directory.md`](../data-directory.md) (**including its
[Warehouse app data](../data-directory.md#warehouse-app-data--check-never-create)
section — a blocking gate this profile cannot deploy without**),
[`../readiness.md`](../readiness.md) and [`../teardown.md`](../teardown.md) all
apply unchanged.

[`../deployment.md`](../deployment.md) applies **from its Review-and-deploy
section onward** — those steps operate on the standalone `resolved.yml` and are
Foundation-agnostic. Skip its Resolve section: it resolves a developer profile
against `dev-profile-<F>/`, which warehouse has no counterpart to. Use the
Resolve block below instead.

## Capabilities and routing cues

- Multi-camera warehouse perception — RT-DETR (2D) or Sparse4D (3D) — with
  behavior analytics over ROI, tripwire, and proximity events.
- Choose for warehouse, loading-dock, forklift/pallet, or depth-aware
  multi-camera requests. Do **not** choose it for generic detection or search —
  `search` is the developer Foundation for those.
- `MODE=2d` with `BP_PROFILE=bp_wh` is the only variant with an agent, UI, and
  RTVI VLM. Every other variant is headless perception plus analytics.

## Profile Service Set

Authoritative source:
`deploy/docker/industry-profiles/warehouse-operations/overrides.env`. Select one
list by variant; expand it verbatim into `COMPOSE_PROFILES` and record its name
in `FOUNDATION_VARIANT`. The lists in scope are:

| `MODE` | `BP_PROFILE` | Extended list | Minimal list |
|---|---|---|---|
| `2d` | `bp_wh` | `COMPOSE_PROFILES_WH_2D` | — |
| `2d` | `bp_wh_kafka` | `…_WH_KAFKA_2D` | `…_WH_KAFKA_2D_MINIMAL` |
| `2d` | `bp_wh_redis` | `…_WH_REDIS_2D` | `…_WH_REDIS_2D_MINIMAL` |
| `3d` | `bp_wh_kafka` | `…_WH_KAFKA_3D` | `…_WH_KAFKA_3D_MINIMAL` |
| `3d` | `bp_wh_redis` | `…_WH_REDIS_3D` | `…_WH_REDIS_3D_MINIMAL` |
| `mv3dt` | `bp_wh_kafka` | `…_WH_KAFKA_MV3DT` | `…_WH_KAFKA_MV3DT_MINIMAL` |
| `mv3dt` | `bp_wh_redis` | `…_WH_REDIS_MV3DT` | `…_WH_REDIS_MV3DT_MINIMAL` |
| `auto-calibration` | `bp_wh_auto_calib` | `COMPOSE_PROFILES_WH_AUTO_CALIB` | — |

`mv3dt` is multi-view 3D tracking with BEV fusion — the same analytics shape as
`3d`, fed by a multi-camera tracker rather than single-view Sparse4D.

**`auto-calibration` is a mode, not a size or a broker choice.** It pairs only
with `BP_PROFILE=bp_wh_auto_calib`. It writes a calibration rather than reading
one, so the calibration-presence rule below does not apply, and there is no
`warehouse-auto-calibration-app` tree for a mode-embedded path to point at. It
runs no perception and no broker, so `STREAM_TYPE` does not apply either. It
ships without the SDRC chain — `init-dirs`, `render-config`,
`wdm-env-from-config`, `wait-for-redis`, `sdr-controller` — so its
infrastructure floor is the common set only.

**Auto-calibration build requirements.** In `override.env`, restore the
direct-VST defaults — `overrides.env` sets the SDRC trio for `2d`/`3d`/`mv3dt`,
and a later env layer can only override a variable, not un-set it:

```text
VST_USE_SDRC=false
VST_NGINX_MODE=vst
STREAM_PROCESSOR_MODULE_ENDPOINT=http://vss-vios-streamprocessing:30001
```

Create the project store before `up` — it is not in the repo, and the container
writes to it as uid 1000:

```bash
mkdir -p "$VSS_APPS_DIR/services/auto-calibration/projects"
chmod 0777 "$VSS_APPS_DIR/services/auto-calibration/projects"
```

**VGGT is optional — nothing to do.** `VGGT available: False` in the logs is
expected, not a failure: the weight is gated on HuggingFace, so the service may falls back to base AMC.

Extended adds ELK, `vss-video-analytics-api`, `vss-haproxy-ingress`,
`import-calibration-output-container-<mode>`, and monitoring (`dcgm-exporter`,
`prometheus`, `grafana`, `node-exporter`, `cadvisor`). Minimal lists carry none
of these.

> Size is selected *only* by which list `COMPOSE_PROFILES` points at.

## Capability owners present

`<mode>` is `2d`, `3d` or `mv3dt`; the suffix is on the compose *service* name
only ([`../services/vios.md`](../services/vios.md)).

| Owner | Service profile keys |
|---|---|
| RT-CV | `perception-2d` / `perception-3d`; 3D additionally requires `ds-configurator-3d`. **`mv3dt` does not follow the `perception-<mode>` pattern**: it is two services, `vss-rtvi-cv-mv3dt` (per-camera tracking) and `vss-rtvi-cv-bev-fusion` (multi-view merge) |
| Behavior Analytics | `vss-behavior-analytics-<mode>` |
| Configurator | `bp-configurator-<mode>` |
| ELK | `kafka`, `kafka-topic-init-container`, `redis`, `broker-health-check`, `elasticsearch`, `elasticsearch-init-container`, `kibana`, `logstash`, `kibana-init-container-<mode>` |
| MQTT | `mosquitto` — **`mv3dt` lists only**; no 2d/3d list carries it (`MQTT_HOST_PORT` 1883) |
| VIOS | `nvstreamer-<mode>`, `sensor-ms-<mode>`, `streamprocessing-ms-<mode>`, `centralizedb`, `vst-ingress`, `sdr-controller`, `turnserver`, `turnserver-init`, `init-dirs`, `render-config`, `wdm-env-from-config`, `wait-for-redis`, `sensor-bp-wait-bp-configurator` |
| Video Analytics API | `vss-video-analytics-api`, `import-calibration-output-container-<mode>` |
| Ingress | `vss-haproxy-ingress` |
| Monitoring | `dcgm-exporter`, `prometheus`, `grafana`, `node-exporter`, `cadvisor` — observational, so nothing else requires them. `GRAFANA_HOST_PORT` defaults to `35000` → container `3000`, with no HAProxy route. `node-exporter` and `cadvisor` set no `container_name` and appear as `<project>-node-exporter-1` / `-cadvisor-1` |
| Agent / RT-VLM / LLM NIM | `bp_wh` only: `vss-agent`, `vss-ui`, `vss-va-mcp`, `phoenix`, `alert-bridge`, `rtvi-vlm`, `llm_${LLM_MODE}_${LLM_NAME_SLUG}` |

`redis` is in **every** warehouse list — it backs `sdr-controller` regardless of
broker choice, and is additionally the CV broker when `STREAM_TYPE=redis`.

`vios-apt-cache-init` resolves into every warehouse build without appearing in
any `COMPOSE_PROFILES_WH_*` list: it carries no `profiles:` gate and is a
`depends_on` of `streamprocessing-ms-*`. Expect it as a one-shot `Exited (0)`;
its absence from the service list is not a defect.

## Profile-specific environment knobs

| Knob | Purpose |
|---|---|
| `MODE`, `BP_PROFILE`, `STREAM_TYPE` | Select the variant. These three pick the `COMPOSE_PROFILES_WH_*` list; they are not free-form. |
| `SAMPLE_VIDEO_DATASET`, `NUM_STREAMS` | Must match each other and the variant — see Hard constraints. |
| `DATASET_TYPE` | `MODE=3d` **only** — inert for `2d`, `mv3dt` and `auto-calibration`. `real` or `synthetic`; selects the whole Sparse4D bundle: ONNX and engine (`v3.0` vs `v2.3`), the `_ov_kmeans900_v*_r50.npy` anchor, `labels-<type>.txt`, and the NGC TAO Sparse4D artifact the configurator writes into `warehouse-3d-app/models-download.json`. Must track `SAMPLE_VIDEO_DATASET` — see Hard constraints. Effective default is `synthetic`, from `overrides.env`; the `${DATASET_TYPE:-synthetic}` fallback in `warehouse-3d-app.yml` only fires if no env layer sets it. |
| `HARDWARE_PROFILE` | Selects perception tuning in `blueprint-configurator/blueprint_config.yml` and LLM NIM sizing, including the per-mode stream ceiling in [`../sizing.md`](../sizing.md). Not validated by Compose; the configurator only uppercases it, so an unrecognized value (including a spacing or hyphenation variant such as `IGX THOR`) silently matches no tuning section. |
| `VSS_APPS_DIR`, `VSS_DATA_DIR` | Ship as `/path/to/…` sentinels — always set both. See the closure table under Build and resolve. |
| `RT_CV_DEVICE_ID` (0), `RT_VLM_DEVICE_ID` (1), `LLM_DEVICE_ID` (2) | GPU layout. |
| `LLM_MODE`, `LLM_NAME`, `LLM_NAME_SLUG`, `LLM_BASE_URL` | `bp_wh` + `MODE=2d` only; `none` everywhere else. For `remote`, `LLM_BASE_URL` is the endpoint root **without** a trailing `/v1` — the agent config appends it. |
| `VLM_MODE`, `VLM_NAME_SLUG` | Keep both `none`. Warehouse uses the integrated RTVI VLM, never the standalone VLM NIM path, and remote VLM is not wired end to end on the Docker path — see below. |
| `VSS_RT_CV_TAG` | Must be `sbsa`-tagged when `HARDWARE_PROFILE=GB300` or `DGX-SPARK`. |
| `BP_CONFIGURATOR_ENV_FILE` | Point at the build's generated `configurator.env`. Without it the configurator reads the checked-in `overrides.env` and bakes the `<HOST_IP>` sentinel — see [`../services/configurator.md`](../services/configurator.md). |
| `NVSTREAMER_CONFIG_DIR`, `TURN_PUBLIC_HOST` | Easily-missed closure members. `TURN_PUBLIC_HOST` derives from `HOST_IP` only transitively, through `EXTERNAL_IP` and `VSS_PUBLIC_HOST`. |

## Hard constraints

Each of these fails at bring-up or silently at runtime, not at `docker compose
config` — `scripts/validate_warehouse_env.py` checks them before deploy.

| Constraint | Symptom if violated |
|---|---|
| `MODE` must be `2d`, `3d`, `mv3dt` or `auto-calibration`, and `BP_PROFILE` one of `bp_wh`, `bp_wh_kafka`, `bp_wh_redis`, `bp_wh_auto_calib` | routes to a service list this skill does not support |
| `BP_PROFILE=bp_wh` is 2D-only | unsupported combination |
| `BP_PROFILE=bp_wh_auto_calib` pairs only with `MODE=auto-calibration`, and vice versa | routes to a list that is not the one selected |
| `BP_PROFILE=bp_wh` is rejected on `IGX-THOR` and `DGX-SPARK` | configurator refuses |
| `HARDWARE_PROFILE=GB300` or `DGX-SPARK` requires an `sbsa` `VSS_RT_CV_TAG` | configurator refuses |
| `LLM_MODE=local` requires `services/nim/<LLM_NAME_SLUG>/hw-<HARDWARE_PROFILE>.env` | compose dies with a bare "no such file" |
| `NUM_STREAMS` must equal the dataset's camera count: `nv-warehouse-4cams` 4, `warehouse-loading-dock-3cams-synthetic` 3, `warehouse-4cams-20mx20m-synthetic` 4. **Dataset and mode are independent** — every shipped dataset now carries calibration for `2d`, `3d` and `mv3dt`, so any dataset pairs with any analytics mode | short stream count with every container healthy |
| Under `MODE=3d`, `DATASET_TYPE` must track the footage family: `nv-warehouse-4cams` → `real`; both `*-synthetic` → `synthetic`; a custom dataset → whatever the footage actually is. It must also be non-empty — the label mount interpolates it into a **host path** | wrong Sparse4D weights and detection thresholds (`real` 0.85/0.75 vs `synthetic` 0.5/0.3), every container healthy, no error anywhere. Empty resolves to `labels-.txt`, so Docker creates a directory where a file is expected |
| `STREAM_TYPE=redis` iff `BP_PROFILE=bp_wh_redis` | no metadata reaches the broker |
| A custom `SAMPLE_VIDEO_DATASET` has no checked-in `calibration.json` | Docker creates a directory where a file is expected; perception emits nothing |
| `MODE=3d` or `mv3dt` on a `…_MINIMAL` list has no Elasticsearch | `mdx-bev` never persisted; BEV output unverifiable |
| The checkout was cloned under a restrictive `umask` (e.g. `027`), leaving repo config non-world-readable | containers run as a different user and cannot read the config they bind-mount — see below |

### Repo file permissions

Compose bind-mounts config out of the checkout into containers that run as a
different user, so those files must be world-readable and their parent
directories world-traversable. Clone with a `umask` of `022` or looser; `027`
yields `0640` files and `0750` directories that containers cannot read.

`umask` applies to the shell that runs `git clone`, so reading it later says
nothing about an existing checkout — the modes on the tree are what matter.

### Remote VLM is not supported (Docker path)

Warehouse uses the integrated RTVI VLM, which runs locally only, so `VLM_MODE`
and `VLM_NAME_SLUG` stay `none` — `scripts/validate_warehouse_env.py` enforces
it. The supported remote configuration is **remote LLM with local RT-VLM**
(`LLM_MODE=remote`, `LLM_BASE_URL` without a trailing `/v1`).

Setting `VLM_MODE=remote` looks plausible and is not wired: the Docker path never
switches `RTVI_VLM_MODEL_TO_USE` (`cosmos-reason3`) or `VLM_MODEL_TYPE` (`rtvi`),
so the deployment starts cleanly and keeps routing through the local RT-VLM
against a model path of `none`.

### Calibration is already in the repo

The shipped sample datasets **need no calibration run**. Each carries a
checked-in `calibration.json` that Compose bind-mounts by path:

```text
warehouse-<mode>-app/calibration/sample-data/${SAMPLE_VIDEO_DATASET}/calibration.json
```

All three shipped datasets carry one **for each of `2d`, `3d` and `mv3dt`**, which
is why dataset and mode are chosen independently. 3D mounts it three ways —
behavior analytics reads `/resources/calibration.json`, `ds-configurator-3d` and
perception read `/opt/data/ds-configurator/calibration.json`. Nothing is staged
under `$VSS_DATA_DIR`.

Only a **custom** dataset needs a calibration run — produced by
`vss-generate-video-calibration`, or by a `MODE=auto-calibration` build — dropped
at the path above under its dataset name.
`scripts/validate_warehouse_env.py` fails the build when it is missing.

A custom dataset needs a **second** thing under `MODE=3d`: an explicit
`DATASET_TYPE`. The three shipped datasets determine it (`nv-warehouse-4cams` is
`real`, both `*-synthetic` are `synthetic`), but custom footage carries no
provenance the skill can infer, so the value has to be asked for rather than
derived — SKILL.md's Q2w-datatype. The validator can only check that some valid
value is set; it cannot tell whether it matches the footage.

`MODE=auto-calibration` is the exception to this whole section: it is the run that
*produces* a calibration, so it has no `warehouse-auto-calibration-app` tree and the
presence check is skipped for it. Pick the dataset you intend to calibrate and match
`NUM_STREAMS` to it, exactly as for an analytics mode.

`import-calibration-output-container-<mode>` (extended lists only) imports
calibration into the analytics store; it does not produce calibration.

## Build and resolve

### Env layers

Warehouse resolves from the industry-profile directory, not a `dev-profile-*`
one. Four ordered layers, later overriding earlier:

```text
deploy/docker/containers.env
deploy/docker/industry-profiles/warehouse-operations/.env
deploy/docker/industry-profiles/warehouse-operations/overrides.env
_builds/<name>/override.env
```

### Build artifacts

```text
_builds/<name>/
├── override.env
├── compose.yml
├── configurator.env       # generated, never hand-edited
└── resolved.yml
```

`override.env` carries `FOUNDATION=warehouse`, `FOUNDATION_VARIANT=<the
COMPOSE_PROFILES_WH_* name>`, the expanded literal `COMPOSE_PROFILES` (never a
`${...}` reference to the baseline), the variant knobs above, and the
dependent-value closure below. `FOUNDATION_VARIANT` must be the variant `MODE` +
`BP_PROFILE` select; `scripts/validate_warehouse_env.py` enforces that pairing.

`compose.yml` appends the shared TURN relay overlay that every warehouse
deployment needs. It is an in-tree shared file, not a build-local patch, so it
belongs in the include path list:

```yaml
include:
  - path:
      - ../../deploy/docker/compose.yml
      - ../../deploy/docker/services/infra/compose-no-turn-tcp-relay.yml
```

`configurator.env` exists because `bp-configurator-<mode>` does **not** read its
environment through Compose interpolation. It declares

```yaml
env_file:
  - ${BP_CONFIGURATOR_BASE_ENV_FILE:-$VSS_APPS_DIR/.../.env}
  - ${BP_CONFIGURATOR_ENV_FILE:-$VSS_APPS_DIR/.../overrides.env}
```

so with those knobs unset it loads the **checked-in** files directly, bypassing
`--env-file` layering entirely — `override.env` cannot reach it, and the pristine
`HOST_IP='<HOST_IP>'` sentinel is baked into the container that renders every
stream and hardware config. Generate the file and point
`BP_CONFIGURATOR_ENV_FILE` at it from `override.env`. Regenerate it whenever
`override.env` changes.

### Dependent-value closure

Compose expands each env file as it is read, so a value derived in an earlier
layer is **not** recomputed when a later layer changes its input. Materialize the
full closure in `override.env`, and follow references **transitively** — `HOST_IP`
reaches `TURN_PUBLIC_HOST` only through two hops, so a single-level scan for
`${HOST_IP}` misses it and the build bakes the sentinel into
`streamprocessing-ms-<mode>`.

| Change | Also re-materialize |
|---|---|
| `HOST_IP` | `EXTERNAL_IP` → `VSS_PUBLIC_HOST` → `TURN_EXTERNAL_IP`, `TURN_PUBLIC_HOST` |
| `MODE` | `SDR_CONTROLLER_CONFIG_PATH` — embeds the mode |
| `VSS_APPS_DIR` | `SDR_CONTROLLER_CONFIG_PATH`, `SENSOR_FILE_PATH`, `NVSTREAMER_CONFIG_DIR`, `VLM_AS_VERIFIER_CONFIG_FILE`, `VLM_AS_VERIFIER_CONFIG_FILE_REALTIME`, `VLM_AS_VERIFIER_ALERT_TYPE_CONFIG_FILE` |

`overrides.env` ships `VSS_APPS_DIR` and `VSS_DATA_DIR` as `/path/to/...`
placeholders — always set both. A missed closure member surfaces as a
`<HOST_IP>` or `/path/to/deploy/docker` sentinel in `validate_resolved_yml.py`.

### Resolve

Run from the repository root.

```bash
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"
FOUNDATION_DIR="$REPO/deploy/docker/industry-profiles/warehouse-operations"
SCRIPTS="$REPO/skills/vss-build-vision-ai/scripts"

# Establish the helper-script runner FIRST: every script below goes through
# "${VSS_SKILL_PY[@]}". Calling `uv run` directly would strand a host that
# passed the prerequisite check on the python3 fallback.
if command -v uv >/dev/null 2>&1; then
  VSS_SKILL_PY=(uv run)
else
  VSS_SKILL_PY=(python3)
fi

# BEFORE `config`: materialize the configurator's env_file.
"${VSS_SKILL_PY[@]}" "$SCRIPTS/render_warehouse_configurator_env.py" \
  "$BUILD_DIR" --repo-root "$REPO"

# containers.env derives every managed image tag from VSS_CONTAINER_TAG and is
# expanded before override.env, so only an exported value reaches those tags.
VSS_CONTAINER_TAG="${VSS_CONTAINER_TAG:-$(
  sed -n 's/^VSS_CONTAINER_TAG=//p' "$BUILD_DIR/override.env"
)}"
tag_args=()
if [ -n "$VSS_CONTAINER_TAG" ]; then
  export VSS_CONTAINER_TAG
  tag_args=(--expect-container-tag "$VSS_CONTAINER_TAG")
fi

docker compose \
  --env-file "$REPO/deploy/docker/containers.env" \
  --env-file "$FOUNDATION_DIR/.env" \
  --env-file "$FOUNDATION_DIR/overrides.env" \
  --env-file "$BUILD_DIR/override.env" \
  -f "$BUILD_DIR/compose.yml" \
  config --no-consistency > "$BUILD_DIR/resolved.yml"

"${VSS_SKILL_PY[@]}" "$SCRIPTS/normalize_resolved_yml.py" "$BUILD_DIR/resolved.yml"
"${VSS_SKILL_PY[@]}" "$SCRIPTS/validate_resolved_yml.py" \
  "$BUILD_DIR/resolved.yml" --repo-root "$REPO" "${tag_args[@]}"
"${VSS_SKILL_PY[@]}" "$SCRIPTS/validate_warehouse_env.py" \
  "$BUILD_DIR" --repo-root "$REPO"
```

Only stdout may reach `resolved.yml` — never merge stderr into it. Deploy the
resolved file standalone: `config` bakes the env layers in, so pass no
`--env-file` to `up`, `ps` or `down`.

`validate_warehouse_env.py` enforces what `resolved.yml` structurally cannot:
`MODE`, `BP_PROFILE`, `HARDWARE_PROFILE` and `SAMPLE_VIDEO_DATASET` appear in no
service `environment:` block. Its rules fail at bring-up or silently at runtime,
never at `config` time.

## Access points

Prefer the HAProxy ingress when the selected list includes it — one
browser-reachable origin that rewrites paths to internal services. The
`…_MINIMAL` lists omit HAProxy, so use direct ports there. Routes are defined in
`deploy/docker/services/infra/haproxy/haproxy.cfg.template`.

### Via the ingress — `http://<EXTERNAL_IP>:${HAPROXY_HOST_PORT:-7777}`

| Path | Backend | Available when |
|---|---|---|
| `/` (catch-all) | `vss-ui` | `bp_wh` only; other ingress-enabled variants have no UI backend, so `/` returns 503 |
| `/vst`, `/vst/...` | `vst-ingress` | any ingress-enabled variant — VST is proxied, so this is the browser path to its UI |
| `/storage/...` | `vst-ingress` (rewritten to `/vst/storage/...`) | any ingress-enabled variant |
| `/kibana/...` | `kibana` | `bp_wh`, or extended Kafka/Redis |
| `/elasticsearch/...` | `elasticsearch`, path-stripped; `GET/HEAD/POST/OPTIONS` only, cluster-admin and bulk-mutating paths denied | same as `/kibana` |
| `/video-analytics-api/...` | `vss-video-analytics-api`, path-stripped | same as `/kibana` |
| `/rtvi-cv/...` | `vss-rtvi-cv`, path-stripped | `bp_wh`, or extended Kafka/Redis |
| `/rtvi-vlm/...` | `rtvi-vlm`, path-stripped | `bp_wh` only |
| `/alert-bridge/...`, `/phoenix/...`, `/va-mcp` | `alert-bridge`, `phoenix`, `vss-va-mcp` | `bp_wh` only |
| `/api`, `/chat`, `/static`, `/websocket` | `vss-agent` (`/api/chat` matches `vss-ui` first) | `bp_wh` only |
| `/behavior-analytics/...` | — | **never works.** The route exists and the container runs, but it publishes no HTTP listener, so the backend never passes its check and every request 503s. Read behaviors from `mdx-behavior` or the `mdx-behavior-*` indices |
| `/perception-sdr/...`, `/rtvi-embed/...` | — | **never** — neither container is deployed by any warehouse list |

### Direct ports

| Service | URL | Available when |
|---|---|---|
| NvStreamer UI | `<HOST_IP>:31000` (`NVSTREAMER_HTTP_HOST_PORT`) | all variants; no ingress route |
| VST UI | `<HOST_IP>:30888/vst/` (`VST_INGRESS_HOST_PORT`) | all variants; prefer `/vst/` via ingress |
| SDR controller | `<HOST_IP>:10000` (`SDRC_PROXY_HOST_PORT`); also publishes 5003, 8011, 9902 | all except `auto-calibration`, which deploys none |
| Elasticsearch | `<HOST_IP>:9200` (`ELASTICSEARCH_HOST_PORT`) | `bp_wh`, or extended Kafka/Redis |
| Kibana | `<HOST_IP>:5601/kibana` (`KIBANA_HOST_PORT`) | same — served under `/kibana` either way |
| Video Analytics API | `<HOST_IP>:8081` (`VIDEO_ANALYTICS_API_HOST_PORT`) | same |
| Grafana | `<HOST_IP>:35000` (`GRAFANA_HOST_PORT`) | `bp_wh`, or extended Kafka/Redis; no ingress route |
| VSS Agent, Phoenix | `<HOST_IP>:8000`, `<HOST_IP>:6006` | `bp_wh` only; prefer `/api` and `/phoenix` |
| Auto-calibration API | `<HOST_IP>:8010/docs` (`VSS_AUTO_CALIBRATION_HOST_PORT`) | `auto-calibration` only; no ingress route |
| Auto-calibration UI | `<HOST_IP>:5000` (`VSS_AUTO_CALIBRATION_UI_HOST_PORT`) | `auto-calibration` only; no ingress route |

Nothing listens on `8001` — there is no VST MCP container.

> **A wrong `Host` header looks like "every path 404s".** HAProxy first denies
> any request whose `Host` is not in its `known_host` ACL — `VSS_PUBLIC_HOST`,
> `EXTERNAL_IP`, `HOST_IP`, `localhost`, `127.0.0.1`, each with and without
> `:HAPROXY_PORT` — with a 404, then routes matching traffic through the
> identical ACL. `EXTERNAL_IP` defaults to `${HOST_IP}`; set it to the
> browser-reachable name. On Brev the ingress, agent and UI additionally need
> `https`/`wss` on the secure-link domain ([`../brev.md`](../brev.md)).

## Working-tree side effects

A warehouse deploy **modifies checked-in files in the repo tree**. Several config
JSONs are bind-mounted read-write (only `models-download.json` is `:ro`), and
`bp-configurator-<mode>` rewrites several in place on first boot — it logs
`Created backup: …` and `Successfully wrote JSON file`, leaving both a
`*.backup_<timestamp>.json` and a reformatted original. After a deploy, `git status`
shows several config files nobody edited.

`models-download.json` being `:ro` describes only its **container** mount — it does
not mean the file is never modified. On `MODE=3d` the configurator rewrites
`warehouse-3d-app/models-download.json` on the host, pointing the model and anchor
entries at the Sparse4D bundle `DATASET_TYPE` selects, and leaves a backup beside
it. Expect that file in `git status` after any 3D deploy.

## Stock readiness checks

Container-state gating is the shared Gate 0 in [`../readiness.md`](../readiness.md);
warehouse's one-shot init containers are expected `Exited (0)` and pass it
unchanged. Warehouse additionally needs a **liveness** check — every container
can be `Up` while zero streams are processed:

```bash
docker logs --since 60s vss-rtvi-cv 2>&1 | grep -aE "stream_name" | tail -8
docker logs --since 60s vss-rtvi-cv 2>&1 | grep -a "Active sources" | tail -1
```

Expect one `stream_name` line per source at roughly source framerate, and an
active-source count equal to `NUM_STREAMS`. Do **not** `grep -i fps` —
DeepStream's only line containing that string is a valueless header, so it
reports success regardless.

On `mv3dt` the perception container is `vss-rtvi-cv-mv3dt`, and `mdx-bev` comes
from a second service, `vss-rtvi-cv-bev-fusion` — check it is `healthy` too.
**`mv3dt` publishes `mdx-raw` as well as `mdx-bev`** (per-camera tracks and the
fused result), unlike `3d`. First output can take several minutes; a
`vss-vios-ingress` restart count in the low teens while `sdr-controller` comes up
is normal, provided it settles to `healthy`.

HTTP probes, when the selected list ships them:

```bash
curl -sf "http://${HOST_IP}:${HAPROXY_HOST_PORT:-7777}/vst/"
curl -sf "http://${HOST_IP}:9200/_cluster/health"          # extended, or bp_wh
curl -sf "http://${HOST_IP}:8081/livez"                    # liveness only; extended, or bp_wh
curl -sf "http://${HOST_IP}:5601/kibana/api/status"        # extended, or bp_wh
curl -sf "http://${HOST_IP}:8000/health"                   # bp_wh only
```

> Endpoint quirks that read as a dead service are in
> [`../services/elk.md`](../services/elk.md); routes that never answer are in
> Access points above.

### `MODE=auto-calibration` readiness

This mode deploys no `vss-rtvi-cv` and processes no streams, so the liveness
check above does not apply. Use instead:

```bash
curl -sf -o /dev/null -w '%{http_code}\n' "http://${HOST_IP}:8010/docs"   # API  -> 200
curl -sf -o /dev/null -w '%{http_code}\n' "http://${HOST_IP}:5000/"       # UI   -> 200
docker inspect -f '{{.RestartCount}} {{.State.Health.Status}}' vss-auto-calibration
docker logs --since 60s vss-auto-calibration 2>&1 | grep -c "Failed to persist project state"
```

Expect `200`, `200`, `0 healthy`, and a count of **0** — the last probe is what
confirms the `projects/` mount is writable.

`/docs` is the endpoint that answers; `/openapi.json`, `/v1/health` and `/health`
return **404** on this image.

## Sources

- `deploy/docker/industry-profiles/warehouse-operations/.env`
- `deploy/docker/industry-profiles/warehouse-operations/overrides.env`
- `deploy/docker/industry-profiles/warehouse-operations/compose.yml`
- `deploy/docker/industry-profiles/warehouse-operations/warehouse-{2d,3d}-app/`
- `deploy/docker/industry-profiles/warehouse-operations/blueprint-configurator/blueprint_config.yml`
- `deploy/docker/services/infra/compose-no-turn-tcp-relay.yml`
- `deploy/docker/services/infra/haproxy/haproxy.cfg.template`
