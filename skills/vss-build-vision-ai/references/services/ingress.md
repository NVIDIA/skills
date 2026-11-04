# Ingress Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys |
|---|---|
| Single-origin HTTP ingress for browse and host-CLI operate surfaces | `vss-haproxy-ingress` |

## Access role

A bridge-network reverse proxy that fronts HTTP surfaces on one origin (default
port `7777`) behind a host-header allowlist. It is infrastructure, not a
capability producer — it only routes to owners the build already deploys. Two
independent uses: (a) the interactive tier's public front door (UI + agent);
(b) a single origin for a headless build, fronting its **browse** and host-CLI
**operate** surfaces (detailed below). It is reached only when the request asks
to expose surfaces through one origin; otherwise it is pruned. NvStreamer is
never fronted here (see below).

## Required peers

- Routes only to services already in the build; a route whose backend is not
  deployed simply returns `503` (every backend uses `init-addr none`), so the
  proxy starts regardless of which surfaces are present.
- The interactive front-door use requires the Agent owner (`vss-agent` / UI).
  The headless use requires none of that tier — only the browse/operate owners it
  routes to (ELK/Kibana, VIOS, Alerts, Elasticsearch, RT-Embed, RT-CV).
- Requires the host-identity env below, or the host-header allowlist returns
  `404` for every request.

## Headless single-origin ingress

This section applies only when the build includes `vss-haproxy-ingress`. A
headless build that exposes no single origin prunes the proxy (see Access role)
and produces no curated patch — do not reach for it when no service-definition
change to the ingress is needed.

The shipped `haproxy.cfg.template` is authored for the full stack: its catch-all
plus the `/api/chat`, `/chat`, `/static`, `/websocket`, `/phoenix`, and `/va-mcp`
routes target the interactive tier that a headless build prunes. Two headless
modes, not interchangeable — **default to the curated patch** when the build
exposes a chosen set of surfaces through one origin (a "single ingress" request,
or any named-surface list). It is a build-generation artifact, so a validate-only
pass is no reason to skip it. Use as-is only when the caller explicitly accepts
advertised dead routes and a `503` `/` landing.

**Curate by consumer class, not by profile.** Retain, for the backends the build
deploys, the routes each consumer of the origin needs:

| Consumer of the origin | Routes to retain (for deployed backends) |
|---|---|
| Human **browse** | `/kibana`, `/vst`, `/storage`, `/video-analytics-api`, and (combined only) `/alert-bridge` |
| Host-CLI **operate** (`vss configure`) | `/vst`, `/elasticsearch` (read-only guard, verbatim), `/rtvi-embed`, `/rtvi-cv`, `/rtvi-vlm` (only where RT-VLM is driven from the host; see note below), and `/api` when an agent ships |

The operate set is the read-path subset of what `vss configure` probes to resolve
a deployment through one origin (`vss_cli/config.py:INGRESS_SERVICES`). A queryable
headless build **must** carry it — post-#1469 `vss search run` takes no endpoints, so a build missing these routes is unqueryable (no ingress-less read path). But RT-Embed or Elasticsearch in the build does **not**
make it queryable — an ingestion/indexing-only build that requests no read surface
prunes the proxy regardless. For `/rtvi-vlm` the canonical
`haproxy.cfg.template` is the **single authored source** — it already
carries the `/rtvi-vlm` ACL and the `bk_rtvi_vlm_strip` prefix-rewrite backend, so a
build that needs the route copies it verbatim into any curated config and authors
**no new backend rule**. Front RT-VLM at `/rtvi-vlm` whenever the host drives it:
for result critique and visual follow-up Q&A, because `vss configure` records
`rt_vlm` from this probe alone and the CLI's critic is fail-open — no route, no
critique, retrieval unaffected; and for a tagging leg driven by hand rather than
by the webhook receiver. This is a conscious tradeoff: it re-exposes RT-VLM's
SSE-generation and stream/file-mutation endpoints through HAProxy, so the
origin's host-allowlist is the only boundary — never front it on an
unauthenticated origin. Only a build where nothing outside the Compose network
reaches RT-VLM — its tagging leg driven by VIOS webhooks, no host-side critique,
no CLI operate surface asked for — keeps it loopback-only.

- **Curated (patch).** Write the trimmed config to `patches/haproxy.cfg`, beside
  the `patches/vss-haproxy-ingress.yml` service-definition patch that overrides the
  config volume, keeping the browse + operate routes above for the build's deployed
  backends and replacing the catch-all. Bind the payload by an absolute
  `${BUILD_DIR}/patches/haproxy.cfg` source (per `composition.md`): the patch is
  pulled in through the ordered `path:` list, so a relative `./haproxy.cfg` would
  resolve against the root Compose file's directory (`deploy/docker/`), not the
  patch's.
- **As-is (explicit shortcut only).** Activate `vss-haproxy-ingress` and set the
  host-identity env. Interactive routes 503 harmlessly; the browse and operate
  routes work, but dead routes are advertised and `/` 503s.

Discipline for the trimmed config:

- **Prune, do not author.** Derive it by deleting the backends and routes for
  pruned services from the shipped template and swapping the catch-all — never
  write one from scratch, so it stays a faithful subset and is re-derivable when
  the template moves. Copy every `replace-path` block verbatim.
- **Prune only on a structural test, never on a liveness `503`.** Drop a route
  only when its backend **never binds an HTTP port** — not when it merely `503`s
  now. Behavior-Analytics (the `vss-search-analytics-2d-fusion` worker) is
  Kafka-only, so it always `503`s: drop its `/behavior-analytics` route (reached
  via Kibana/ES instead). Every browse/operate backend above binds HTTP, so none
  qualify. A required operate route that transiently `503`s (backend not yet ready)
  is **not** a drop reason: `vss configure` records it as *absent*, not
  present-but-broken, and a re-probe after readiness resolves it — keep required
  routes exposed regardless.
- **Redirect only the bare origin; 404 every other unmatched path.** Bounce
  only `/` to `/kibana/` (the headless browse landing) and `http-request deny
  deny_status 404` the rest. A blanket `redirect … if h_main !p_routed` would
  302 unrouted probe paths (`/api`, `/rtvi-vlm`) to Kibana's 200, so `vss
  configure` records absent services as present. HAProxy runs **every
  `http-request` rule before any `use_backend`**, so the 404 must exclude both
  the landing (`!p_root`) and the kept routes (`!p_routed`, in sync with the
  `use_backend` routes) or it preempts real routing.
- **`/kibana` stays no-strip.** Kibana runs with `server.basePath: "/kibana"` +
  `server.rewriteBasePath: true` (`kibana.yml`), so the proxy must not strip the
  prefix — unlike `alert-bridge` / `video-analytics-api`, which do strip.
- **Lint before use:** `haproxy -c -f <cfg>` must pass; add it to the build's
  validate step so a bad config fails at build time, not at container start.
- **NvStreamer is not routed here.** It is reached directly on its published
  port (default `31000`) in every profile; do not add a subpath route for it
  (its UI has no base-path setting, so a subpath mount breaks its assets).

### Reference trimmed config (headless: browse + operate)

Pruned from `haproxy.cfg.template`; mount it via the patch. Keep the omitted
verbatim blocks exactly as the template has them.

```haproxy
global
    log stdout format raw local0 info
    maxconn 40000

defaults
    mode http
    log global
    option httplog
    option forwardfor
    timeout connect 10s
    timeout client 120s
    timeout server 120s
    timeout tunnel 3600s

resolvers docker
    nameserver dns 127.0.0.11:53
    accepted_payload_size 8192
    hold valid 10s

# --- Browse backends (rewrites copied verbatim); operate backends added below ---

backend bk_vst_ingress
    server s1 "${VST_INGRESS_SERVICE_HOST}:${VST_PORT}" check resolvers docker init-addr none

backend bk_vst_storage_compat
    http-request replace-path ^/storage/(.*) /vst/storage/\1
    http-request replace-path ^/storage$ /vst/storage
    server s1 "${VST_INGRESS_SERVICE_HOST}:${VST_PORT}" check resolvers docker init-addr none

backend bk_vst_prefixed_compat
    http-request replace-path ^/[^/]+:[0-9]+/vst/(.*) /vst/\1
    http-request replace-path ^/[^/]+:[0-9]+/vst$ /vst
    server s1 "${VST_INGRESS_SERVICE_HOST}:${VST_PORT}" check resolvers docker init-addr none

backend bk_kibana
    server s1 "${KIBANA_SERVICE_HOST}:${KIBANA_PORT}" check resolvers docker init-addr none

backend bk_video_analytics_api_strip
    http-request replace-path ^/video-analytics-api/(.*) /\1
    http-request replace-path ^/video-analytics-api$ /
    server s1 "${VIDEO_ANALYTICS_API_SERVICE_HOST}:${VIDEO_ANALYTICS_API_PORT}" check resolvers docker init-addr none

# Combined (alerts) builds only — drop this backend if no alerts capability ships:
backend bk_alert_bridge_strip
    http-request replace-path ^/alert-bridge/(.*) /\1
    http-request replace-path ^/alert-bridge$ /
    server s1 "${ALERT_BRIDGE_SERVICE_HOST}:${ALERT_BRIDGE_PORT}" check resolvers docker init-addr none

# Operate route-set (`vss configure`) — add these for the backends the build
# deploys; drop any whose backend is not shipped. Copied verbatim from the template.
backend bk_elasticsearch_strip
    http-request replace-path ^/elasticsearch/(.*) /\1
    http-request replace-path ^/elasticsearch$ /
    server s1 "${ELASTICSEARCH_SERVICE_HOST}:${ELASTICSEARCH_SERVICE_PORT}" check resolvers docker init-addr none

backend bk_rtvi_embed_strip
    http-request replace-path ^/rtvi-embed/(.*) /\1
    http-request replace-path ^/rtvi-embed$ /
    server s1 "${RTVI_EMBED_SERVICE_HOST}:${RTVI_EMBED_SERVICE_PORT}" check resolvers docker init-addr none

backend bk_rtvi_cv_strip
    http-request replace-path ^/rtvi-cv/(.*) /\1
    http-request replace-path ^/rtvi-cv$ /
    server s1 "${RTVI_CV_SERVICE_HOST}:${RTVI_CV_SERVICE_PORT}" check resolvers docker init-addr none

backend bk_rtvi_vlm_strip
    http-request replace-path ^/rtvi-vlm/(.*) /\1
    http-request replace-path ^/rtvi-vlm$ /
    server s1 "${RTVI_VLM_SERVICE_HOST}:${RTVI_VLM_SERVICE_PORT}" check resolvers docker init-addr none

frontend fe_http
    bind "${HAPROXY_BIND_ADDR}:${HAPROXY_PORT}"

    # known_host allowlist + `http-request deny deny_status 404 if !known_host`
    # and the full h_main ACL block: COPY BOTH VERBATIM from haproxy.cfg.template.

    # storage preflight + route (copy the ACLs + HEAD/OPTIONS returns verbatim):
    acl p_storage path /storage
    acl p_storage path_beg /storage/
    # ... (HEAD/OPTIONS return blocks copied verbatim from the template) ...
    use_backend bk_vst_storage_compat if h_main p_storage

    acl p_video_analytics path /video-analytics-api
    acl p_video_analytics path_beg /video-analytics-api/
    use_backend bk_video_analytics_api_strip if h_main p_video_analytics

    # Combined (alerts) builds only:
    acl p_alert_bridge path /alert-bridge
    acl p_alert_bridge path_beg /alert-bridge/
    use_backend bk_alert_bridge_strip if h_main p_alert_bridge

    acl p_kibana path /kibana
    acl p_kibana path_beg /kibana/
    use_backend bk_kibana if h_main p_kibana

    acl p_vst path /vst
    acl p_vst path_beg /vst/
    acl p_vst_prefixed path_reg ^/[^/]+:[0-9]+/vst(/|$)
    use_backend bk_vst_prefixed_compat if h_main p_vst_prefixed
    use_backend bk_vst_ingress if h_main p_vst

    # --- Operate route-set (`vss configure`) — keep for the deployed backends ---
    # Elasticsearch is read-only at the edge: COPY the guard block below (method-
    # deny + admin/mutating-deny) VERBATIM from haproxy.cfg.template — it is
    # security-bearing, like the storage-preflight and h_main blocks.
    acl p_elasticsearch path /elasticsearch
    acl p_elasticsearch path_beg /elasticsearch/
    acl es_read_method method GET HEAD POST OPTIONS
    acl es_admin_path path_reg ^/elasticsearch/+_(cluster|nodes|snapshot|security|settings|shutdown|license)(/|$)
    acl es_mutating_op path_reg ^/elasticsearch/.*/(_delete_by_query|_update_by_query|_update|_bulk|_forcemerge|_close|_open)(/|$)
    http-request deny status 405 if h_main p_elasticsearch !es_read_method
    http-request deny status 403 if h_main p_elasticsearch es_admin_path
    http-request deny status 403 if h_main p_elasticsearch es_mutating_op
    use_backend bk_elasticsearch_strip if h_main p_elasticsearch

    acl p_rtvi_embed path /rtvi-embed
    acl p_rtvi_embed path_beg /rtvi-embed/
    use_backend bk_rtvi_embed_strip if h_main p_rtvi_embed

    acl p_rtvi_cv path /rtvi-cv
    acl p_rtvi_cv path_beg /rtvi-cv/
    use_backend bk_rtvi_cv_strip if h_main p_rtvi_cv

    acl p_rtvi_vlm path /rtvi-vlm
    acl p_rtvi_vlm path_beg /rtvi-vlm/
    use_backend bk_rtvi_vlm_strip if h_main p_rtvi_vlm

    # Landing + catch-all: only the bare origin bounces to Kibana (no UI in
    # headless); every other unmatched path 404s, so `vss configure` probes for
    # unrouted services (agent, rt-vlm) record absent instead of following a
    # unrouted services (agent, and rt-vlm on non-tagging builds) record absent instead of following a
    # use_backend, so p_routed must exclude the kept routes from the 404 (drop
    # any whose backend the build does not deploy).
    acl p_root path /
    acl p_routed path_beg /kibana /vst /storage /video-analytics-api /alert-bridge /elasticsearch /rtvi-embed /rtvi-cv /rtvi-vlm
    acl p_routed path_reg ^/[^/]+:[0-9]+/vst(/|$)
    http-request redirect location /kibana/ code 302 if h_main p_root
    http-request deny deny_status 404 if h_main !p_root !p_routed
```

## Configuration knobs

| Environment variable | Use |
|---|---|
| `HAPROXY_HOST_PORT`, `HAPROXY_PORT`, `HAPROXY_BIND_ADDR` | Publish and bind the proxy origin. |
| `VSS_PUBLIC_HOST`, `VSS_PUBLIC_PORT`, `EXTERNAL_IP`, `HOST_IP` | Host-header allowlist — required, or every request 404s. |
| `KIBANA_SERVICE_HOST`, `KIBANA_PORT`, `VST_INGRESS_SERVICE_HOST`, `VST_PORT`, `BEHAVIOR_ANALYTICS_SERVICE_HOST`, `VIDEO_ANALYTICS_API_SERVICE_HOST`, `ALERT_BRIDGE_SERVICE_HOST` (+ ports) | Per-backend browse targets; Docker-DNS defaults suit the shipped service keys. |
| `ELASTICSEARCH_SERVICE_HOST`/`_PORT`, `RTVI_EMBED_SERVICE_HOST`/`_PORT`, `RTVI_CV_SERVICE_HOST`/`_PORT`, `RTVI_VLM_SERVICE_HOST`/`_PORT` | Per-backend operate targets (`vss configure` route-set); add for the backends the build deploys. |

## Sources

- `deploy/docker/services/infra/haproxy/compose.yml`
- `deploy/docker/services/infra/haproxy/haproxy.cfg.template`
- `deploy/docker/services/infra/elk/kibana/configs/kibana.yml`
