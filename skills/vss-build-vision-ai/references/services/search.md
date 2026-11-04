# Search Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys |
|---|---|
| Search analytics and indexing flow | `vss-search-analytics-2d-fusion` |

## Required peers

- Requires RT-CV, RT-Embed, Elasticsearch, Kafka, Logstash, and the Search
  profile's VIOS/NvStreamer path.
- Agent search requires `COSMOS_EMBED_ENDPOINT`, `ELASTIC_SEARCH_ENDPOINT`, and
  `ELASTIC_SEARCH_INDEX`.
- RT-VLM serves result critique and visual follow-up Q&A, chosen **per request**
  and never at deploy time: the agent's `use_critic` (default on) and every
  `vss search run`. There is no build-time flag.
- `rtvi-vlm` is **opt in**. Retrieval never needs it, so closure drops it unless
  the request asks to verify results or to ask questions about clips. Both
  consumers call it directly, so include it whenever either is requested, with
  or without a `vss-agent` tier. When it is absent, say so: verdicts come back
  `unverified` and `vss-ask-video` stops rather than deploying one.
- Presence alone is not enough. The CLI verifies only when the recorded
  deployment has both `rt_vlm` and `vst`, so a build that ships `rtvi-vlm` must
  also route `/rtvi-vlm` through the ingress. Prune that route and
  `vss configure` records the service absent while the container runs.
- Deploying and routing it for either use makes it **no** webhook receiver:
  continuous tagging is a separate capability, requested separately
  ([`rt-vlm.md`](rt-vlm.md), [`vios.md`](vios.md)).
- `vss-video-analytics-api` is a **separate** service key from
  `vss-search-analytics-2d-fusion`. The analytics API (`:9901`) provides the
  REST query/browse surface over ES indices; the search-analytics service is the
  Behavior-Analytics container that produces `mdx-embed-filtered` and
  `mdx-behavior`. Both are in the Search Foundation's service set; do not
  confuse or substitute one for the other. Like the ingress, the analytics API is
  an exposed read surface — include it only when the request asks to expose a
  query/browse/REST surface, and prune it otherwise even though the Foundation
  ships it. It is a shared singleton: include `vss-video-analytics-api` when
  required, and never introduce a second API key or container.
- When this container also serves another capability on one shared instance (a
  combined build), it is the shared **Behavior-Analytics** instance — converge its
  mounted JSON config per [`behavior-analytics.md`](behavior-analytics.md); its
  operating mode (`numWorkersFor*` gates) is not env-expressible.

## Write-path topic flow

Surface this topic-level flow in the architecture preview (SKILL.md step 6
requires principal data flows and topics); it is authoritatively defined in
`skills/deployment/vss-setup-behavior-analytics/references/integrate-behavior-analytics-service.md`.

- Detection: `perception-2d-fusion -> mdx-raw`.
- Embeddings: `rtvi-embed -> mdx-embed -> vss-search-analytics-2d-fusion ->
  mdx-embed-filtered`. `vss-search-analytics-2d-fusion` is the **sole** producer of
  `mdx-embed-filtered`.
- Behaviors: `vss-search-analytics-2d-fusion -> mdx-behavior`.
- Indexing: Logstash consumes these output topics into Elasticsearch index
  patterns `mdx-raw-*`, `mdx-behavior-*`, and `mdx-embed-filtered-*`.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VSS_BEHAVIOR_ANALYTICS_IMAGE`, `VSS_BEHAVIOR_ANALYTICS_TAG` | Select the Search analytics image inherited from Behavior Analytics. |
| `STREAM_TYPE` | Select the checked-in Kafka or Redis analytics config. |
| `COSMOS_EMBED_ENDPOINT` | Point the Agent at RT-Embed. |
| `ELASTIC_SEARCH_ENDPOINT`, `ELASTIC_SEARCH_INDEX` | Point the Agent at indexed search data. |

## Sources

- `deploy/docker/developer-profiles/dev-profile-search/.env`
- `deploy/docker/developer-profiles/dev-profile-search/overrides.env`
- `deploy/docker/developer-profiles/dev-profile-search/video-analytics-2d-app/compose.yml`
- `deploy/docker/services/analytics/behavior-analytics/compose.yml`
- `skills/vss-build-vision-ai/references/deployment_resolution.md`
