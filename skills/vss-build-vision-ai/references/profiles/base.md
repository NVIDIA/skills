# Base Developer Profile

## Capabilities and routing cues

- General VSS agent, UI, video ingest/storage, LLM, and integrated RT-VLM.
- Choose for "deploy VSS", general video understanding, or the smallest
  developer foundation.
- Prefer as the Foundation when the request needs VIOS + inference but no
  ELK-backed alerts, search, or long-video summarization.
- RTX PRO 4500 Blackwell (`RTXPRO4500BW`) is not a valid hardware profile here;
  route that SKU to Alerts with a remote LLM.
- The Agent and UI layer is **optional in a delta** — see
  [Capability owners present](#capability-owners-present) before assuming a
  request for "Q&A" requires it.

## Profile Service Set

Authoritative source:
`deploy/docker/developer-profiles/dev-profile-base/overrides.env`.

```text
phoenix,redis,vss-haproxy-ingress,vss-ui,vss-agent,centralizedb,vst-ingress,sensor-ms,streamprocessing-ms,llm_${LLM_MODE}_${LLM_NAME_SLUG},rtvi-vlm
```

## Capability owners present

| Owner | Service profile keys |
|---|---|
| Agent | `vss-agent`, `vss-ui`, `phoenix` |
| Ingress | `vss-haproxy-ingress` |
| VIOS | `centralizedb`, `vst-ingress`, `sensor-ms`, `streamprocessing-ms` |
| LLM NIM | `llm_${LLM_MODE}_${LLM_NAME_SLUG}` |
| RT-VLM | `rtvi-vlm` |

`redis` is a shared peer used by this profile graph (see `services/elk.md` for
when it is retained).

`vss-haproxy-ingress` is the optional single-origin front door in an explicit
headless capability-pruning build: retain it only when the Agent/UI tier is
present or the request explicitly asks to expose surfaces through one browse
origin; otherwise prune it (headless clients reach each backend on its own
port). This does not apply to a Q3-only harness delta, which keeps ingress and
every unrelated Foundation key. See `services/ingress.md`. When it is ambiguous
whether a browse origin is wanted, ask rather than silently retaining it.

## Profile-specific environment knobs

| Knob | Purpose |
|---|---|
| `HARDWARE_PROFILE` | Select current hardware defaults. |
| `LLM_MODE`, `LLM_NAME`, `LLM_NAME_SLUG`, `LLM_DEVICE_ID`, `LLM_BASE_URL` | Select and place the LLM. |
| `VLM_MODE`, `VLM_NAME`, `RT_VLM_DEVICE_ID`, `VLM_BASE_URL` | Select and place integrated or remote VLM inference. |
| `RTVI_VLM_MESSAGE_BUS` | Generated-output bus; current Compose defaults to `kafka`. Set it explicitly and pair it with legacy `RTVI_VLM_KAFKA_ENABLED=true`/`false` while VSS Compose still forwards that compatibility field. Current RT-VLM uses `MESSAGE_BUS`, not `KAFKA_ENABLED`. |
| `STREAM_TYPE` | Already `kafka`; do not repeat it in a delta unless changing broker type. |
| `VSS_APPS_DIR`, `VSS_DATA_DIR`, `HOST_IP`, `EXTERNAL_IP` | Resolve repository, data, and network paths. |
| `HAPROXY_HOST_PORT`, `VSS_PUBLIC_*`, `VSS_*_HOST_PORT` | Change public ingress or host port bindings. |

## Stock readiness checks

```bash
curl -sf "http://${HOST_IP}:8000/health"                        # vss-agent
curl -sf "http://${HOST_IP}:8018/v1/health/ready"               # rtvi-vlm
curl -sf "http://${HOST_IP}:${LLM_PORT:-30081}/v1/health/ready" # LLM NIM
curl -sf "http://${HOST_IP}:3000/"                              # vss-ui
```

In an explicit headless delta that drops the whole Agent/UI tier, skip the
`:8000`, `:3000`, and LLM NIM probes because those services are absent by
design. In a Q3-only harness delta, `vss-agent` leaves with its two private
peers - the LLM NIM and `phoenix` - so skip `:8000` and the LLM NIM probe, and
keep the UI and RT-VLM probes because those Foundation services remain. (An
`lvs` build keeps the NIM for `lvs-server`; probe it there.)

For remote LLM/VLM mode, probe the selected remote `/v1/models` endpoint
instead of the absent local service.

## Sources

- `deploy/docker/developer-profiles/dev-profile-base/.env`
- `deploy/docker/developer-profiles/dev-profile-base/overrides.env`
- `deploy/docker/developer-profiles/dev-profile-base/compose.yml`
- `deploy/docker/services/agent/compose.yml`
- `deploy/docker/services/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml`
- `deploy/docker/services/vios/compose.yml`
