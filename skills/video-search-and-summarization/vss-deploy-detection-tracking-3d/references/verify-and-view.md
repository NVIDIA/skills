# Verify and view the deployed MV3DT stack

Parent: [`../SKILL.md`](../SKILL.md). Run **after** [`deploy-rtvi-cv-3d-stack.md`](deploy-rtvi-cv-3d-stack.md) returns. Goal: confirm perception + fusion are running, BEV is flowing through the broker, and the user has a working browser viewing path (VST video wall).

Overlay viz uses the existing VST video wall — there's no separate visualization skill for MV3DT. Whether bounding boxes actually render depends on which profile was deployed (see [`../SKILL.md` Q0](../SKILL.md#q0--profile-size-overlays-or-not)).

## Step 1 — Container health

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}' \
  | grep -E 'mv3dt|mosquitto|kafka|redis|vios|centralizedb|configurator|broker-health-check|behavior|elasticsearch|logstash|kibana|video-analytics|auto-calib'
```

Expected — all the following must show `Up` (or `Up (healthy)` where a health check applies):

### Always-deployed (both profiles)

| Container | Expected state |
|---|---|
| `vss-rtvi-cv-mv3dt` | `Up` (no compose health check — see Step 2 for FPS sanity) |
| `vss-rtvi-cv-bev-fusion` | `Up (healthy)` — health check is `/tmp/fusion_ready` sentinel |
| `mosquitto` | `Up (healthy)` |
| `kafka` *or* `redis` (per `STREAM_TYPE`) | `Up` |
| `broker-health-check` | `Exited (0)` — one-shot, then completes |
| `vss-vios-sensor` (= `sensor-ms-mv3dt`) | `Up (healthy)` |
| `centralizedb` | `Up (healthy)` |
| `vss-configurator-mv3dt` | `Up (healthy)` |
| `vss-vios-nvstreamer-mv3dt` | `Up` (sample/videos only — absent when feeding external RTSP) |
| `vss-auto-calibration` / `-ui` | `Up` (gated under `bp_wh_*_mv3dt`) |
| `vss-behavior-analytics-mv3dt` | `Up` (always — NOT gated by MINIMAL_PROFILE) |

### Extra under extended (`MINIMAL_PROFILE=""`)

| Container | Expected state |
|---|---|
| `elasticsearch` | `Up (healthy)` |
| `elasticsearch-init-container` | `Exited (0)` — one-shot |
| `logstash` | `Up` |
| `kibana` | `Up (healthy)` |
| `vss-kibana-init-mv3dt` | `Exited (0)` — one-shot |
| `vss-video-analytics-api-mv3dt` | `Up (healthy)` |
| `vss-import-calibration-output-mv3dt` | `Exited (0)` — one-shot |

If anything stays `(starting)` or `(unhealthy)` past ~15 min, jump to [`troubleshooting.md`](troubleshooting.md).

## Step 2 — Perception FPS

```bash
docker logs --tail 200 vss-rtvi-cv-mv3dt 2>&1 | grep -iE 'fps|engine|error' | tail -20
```

Expected on a healthy bring-up (first run, in order):

1. `Build engine successfully` lines (BodyPose3DNet TRT engine compile — 3–8 min).
2. `FPS = …` lines per camera once streams flow.

For ongoing monitoring:

```bash
docker logs -f vss-rtvi-cv-mv3dt 2>&1 | grep -i fps
```

Target FPS depends on `HARDWARE_PROFILE` — see the per-GPU `max_streams_supported` table in `SKILL.md` Prerequisites §3 (anchored at `blueprint_config.yml`). Roughly: ~30 FPS / camera on datacenter-class GPUs running at or below their cap; lower on edge platforms or when running at the cap. Confirm against the canonical table in `blueprint_config.yml` for your GPU before reporting "low FPS" — you may simply be at expected throughput.

**Stream count check.** If perception logs report fewer FPS lines than `NUM_STREAMS`, the per-GPU cap has been applied (see [`configure-cameras.md`](configure-cameras.md) Step 2). Compare:

```bash
ls "${VSS_DATA_DIR}/videos/${SAMPLE_VIDEO_DATASET}/"*.mp4 | wc -l
docker logs vss-rtvi-cv-mv3dt 2>&1 | grep -c 'Source.*added'
```

If the second number is less than the first, the `keep_count` op trimmed videos at deploy time.

## Step 3 — BEV Fusion ready

The fusion container marks itself ready by creating `/tmp/fusion_ready` and the compose health check probes that file. **Don't try to `docker exec ... test -f /tmp/fusion_ready` — the image strips out `test`/`ls` from PATH.** Use the compose-evaluated health status instead:

```bash
docker inspect --format '{{.State.Health.Status}}' vss-rtvi-cv-bev-fusion
# Expected: healthy
```

If `unhealthy` or `starting` past 5 min, the sentinel never appeared. Diagnose:

```bash
docker logs --tail 100 vss-rtvi-cv-bev-fusion 2>&1 | tail -30
```

Common causes: broker topic `mdx-raw` not yet produced (perception hasn't emitted), or `MAX_EXPECTED_SENSORS` differs from actual stream count (see [`configure-cameras.md`](configure-cameras.md)).

## Step 4 — Broker offsets growing

Confirm metadata is flowing end-to-end by watching the two topics MV3DT uses:

- `mdx-raw` — per-camera detections (perception → fusion)
- `mdx-bev` — fused BEV frames (fusion → downstream)

### Kafka path

The shipped image is `confluentinc/cp-kafka:8.2.0`, which exposes `kafka-get-offsets`. The older `kafka-run-class kafka.tools.GetOffsetShell` does **not** exist in this image — `ClassNotFoundException`. Use:

```bash
# Latest offsets — repeat after 30s, numbers must grow on both topics
docker exec kafka kafka-get-offsets --bootstrap-server localhost:9092 --topic mdx-raw
docker exec kafka kafka-get-offsets --bootstrap-server localhost:9092 --topic mdx-bev

# Output is `<topic>:<partition>:<offset>` — sum partitions for total messages.

# Optional: peek at one fused message
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 --topic mdx-bev \
  --from-beginning --max-messages 1
```

### Redis path

```bash
# Stream length — repeat, must grow
docker exec redis redis-cli XLEN mdx-raw
docker exec redis redis-cli XLEN mdx-bev

# Optional: peek at one message
docker exec redis redis-cli XRANGE mdx-bev - + COUNT 1
```

If `mdx-bev` is empty but `mdx-raw` is growing: fusion isn't producing output — check [`troubleshooting.md`](troubleshooting.md).

## Step 5 — VST video wall

```
http://<HOST_IP>:30888/vst
```

Use `HOST_IP` from the `.env` (or whatever the user can actually reach from a browser — see "Browser reachability" below for cloud VMs / corp VPN).

### Bounding-box overlays (extended profile only)

Overlays render only when Elasticsearch is populated with the metadata index — i.e. **`MINIMAL_PROFILE=""` (extended)**. Under minimal mode, ELK + `vss-video-analytics-api-mv3dt` + `vss-import-calibration-output-mv3dt` are not deployed, and VST shows raw video without overlays. This matches `vss-deploy-profile/references/warehouse.md` lines 37 / 211.

If you're on minimal and the user wants overlays: tear down ([`teardown.md`](teardown.md)), set `MINIMAL_PROFILE=""`, redeploy ([`deploy-rtvi-cv-3d-stack.md`](deploy-rtvi-cv-3d-stack.md)).

In the VST UI, enable overlays via the player's options menu — by default the 3D bounding box overlay is off; toggle it on per stream.

### Tune BEV `group` / `region` for better overlays

If the BEV top-view floor map looks **stretched or squished**, or overlays sit off to one side, the `group`/`region` values in `calibration.json` (and/or the `Top.png` aspect) need refining. For API-only AMC runs these were set to schema-valid **placeholders** by [`calibration-workflow.md` § 4a](calibration-workflow.md) — enough to boot the stack, but not geometrically accurate. This is expected; tune them now that everything is deployed.

Surface the current values to the user first:

```bash
CAL_DIR="${VSS_APPS_DIR}/industry-profiles/warehouse-operations/warehouse-mv3dt-app/calibration/sample-data/${SAMPLE_VIDEO_DATASET}"
jq '.sensors[0] | {group, region, place}' "${CAL_DIR}/calibration.json"
```

Then point the user at the canonical customization docs to set them properly:

- **Accurate `group.origin` / `group.dimensions`** are derived from camera **FOV coverage** (union of per-camera ground-projected frustums), not from the image size. The VSS Configurator normally computes these automatically; to (re)generate manually, run `spatial-ai-data-utils`'s `tools/camera_grouping/calculate_origin.py` against `calibration.json` (`--overwrite`, optionally `--map_file <Top.png> --visualize`).
- **`group_id` / `region` labels** per camera are defined in the Sensor Info File (`camera_info.json`, with `SENSOR_INFO_SOURCE=file`).
- Field meanings and the camera-grouping tools are documented in the NVIDIA **VSS Warehouse 3D-Vision-AI Profile → Customization** guide: `https://docs.nvidia.com/vss/latest/warehouse-docs/3D-profile.html#customization`.

After editing `calibration.json`, re-import it (re-run the one-shot `import-calibration-output-container-mv3dt` compose service) and restart `vss-vios-streamprocessing` so VST reloads it, then hard-refresh the VST tab (`Ctrl+Shift+R`).

> **Floor-map aspect.** VST renders `Top.png` into a fixed-aspect (≈16:9) panel. A plan-view image whose aspect is far from 16:9 (e.g. a tall/portrait layout) will appear stretched **regardless of `region` values** — pad/letterbox `Top.png` to ~16:9 (origin-preserving, so world↔pixel mapping is unchanged) if needed.

### Browser reachability

The VST UI loads over TCP/30888, but video playback uses **WebRTC**. The browser must reach:

1. **TCP/30888** — UI itself.
2. **Outbound STUN** — VST's `vst_config.json` defaults `stunurl_list` to `stun.l.google.com:19302`. Corp / VPN networks often block this.
3. **Inbound UDP** on a wide port range — VST's `webrtc_port_range` defaults to random UDP (`{min: 0, max: 0}`). Corp / cloud / on-prem firewalls that don't pass arbitrary UDP will make WebRTC fail at ICE negotiation. This is the most common reason "VST UI loads but playback fails" on hosts that are otherwise healthy.

**Symptom of WebRTC failure:** UI loads fine, but clicking play on a sensor shows `Playback Error: Error 22: Failed to create Video Source` — even when the data pipeline is healthy (`mdx-raw` / `mdx-bev` offsets growing, `vss-vios-streamprocessing` is recording chunks).

**Sensor-status caveat.** Even when WebRTC is failing, `GET /vst/api/v1/sensor/list` may report `state: "offline"` and `url: null` on each sensor. That field reflects browser-reachability, not pipeline health. If `streamprocessing` is actively writing files to disk under `${VSS_DATA_DIR}/data_log/`, the data pipeline is fine — the issue is the browser→host transport.

**Workarounds.**

1. **Run the browser on the host.** VNC, X-forward, or RDP into the deploy host — bypasses the WebRTC firewall entirely.
2. **Bypass VST UI; use RTSP directly.** VST publishes the per-sensor stream at `rtsp://<HOST_IP>:30554/live/<sensorId>`. Open with `ffplay`, `vlc`, or `mpv` if TCP/30554 is reachable. No overlays, but lets you see the raw stream.
3. **Bypass UI entirely; consume `mdx-bev`.** The data is on the broker — write a downstream consumer in your language of choice.
4. **Self-host TURN.** Heavyweight: stand up a TURN server on TCP/443 (reachable through corp HTTPS) and point VST at it. Out of scope for this skill; needs VST config edits.

If the user is on a host without these restrictions (LAN, public IP with permissive firewall), Step 5 just works.

## Step 6 — Other diagnostic endpoints

| Surface | URL | Notes |
|---|---|---|
| NvStreamer UI | `http://<HOST_IP>:31000` | Configure / inspect the RTSP server (sample / videos mode only) |
| Auto-Calibration UI | `http://<HOST_IP>:5000` | Available because AMC is gated under `bp_wh_*_mv3dt`; use to re-calibrate without tearing down |
| VST sensor list (API) | `http://<HOST_IP>:30888/vst/api/v1/sensor/list` | `jq` it to confirm `NUM_STREAMS` sensors are registered |
| VST MCP | `http://<HOST_IP>:8001` | Read-only diagnostics |
| Kibana (extended only) | `http://<HOST_IP>:5601` | Dashboards for `mdx-bev` and friends |

No HAProxy ingress under MV3DT — `vss-haproxy-ingress` is gated under `bp_wh` (agents profile) which is not valid for `MODE=mv3dt`. Access services on direct ports.

## When something is wrong

See [`troubleshooting.md`](troubleshooting.md).
