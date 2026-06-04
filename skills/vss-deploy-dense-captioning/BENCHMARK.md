# Skills Eval Benchmark

Generated: 2026-05-29 06:23:00 UTC
Specs: 2

---

## Skill Eval - `skills/vss-deploy-dense-captioning/evals/standalone_api.json`

Branch: `fix/nvbug-6228187-rtvi-vlm-skill` - 2 platforms - spec `standalone_api`
Result source: B200 standalone validation, RTX PRO manual validation, latest
standalone skill retest summary, and PR static checks. Formal skills-eval trace
artifacts were not rerun after the documentation-only follow-up fixes.

| Platform | Mode | Query | Result | Reward | Duration | Turns | Prompt tok | Cached tok | Trace |
|---|---|---|---|---|---|---|---|---|---|
| B200 | standalone | Deploy standalone RT-VLM from the copied compose, validate health/API/OpenAPI/chat/completions, register and clean up the RTSP sample stream, and leave the service running for verifier probes. | PASS - core flow passed | n/a | n/a | n/a | n/a | n/a | manual |
| RTX PRO 6000 Blackwell | standalone + Kafka | Deploy standalone RT-VLM from the copied compose on GPU 7, validate health/API/chat/completions, precheck RTSP, generate captions, publish Kafka caption and incident records, and clean up the temporary stream. | PASS - RTSP caption and Kafka path passed | n/a | n/a | n/a | n/a | n/a | manual |
| Unspecified retest host | standalone + Kafka + files | Retest standalone RT-VLM deployment, model startup, REST APIs, file captioning, RTSP captioning, and Kafka caption/incident publishing. | PASS - 37 pass, 4 documentation gaps fixed in follow-up | n/a | n/a | n/a | n/a | n/a | manual |

Observed coverage:
- Standalone compose copy flow worked.
- The copied compose required removing the dangling `rtvi-vlm.depends_on` block; the normalized copy passed `docker compose config --quiet`.
- RT-VLM deployed successfully, downloaded and served Cosmos Reason 2, and became healthy.
- API validation passed for `/v1/health/ready`, `/v1/models`, `/openapi.json`, `/v1/assets/stats`, text-only `/v1/chat/completions`, and expected HTTP 400 for text-only legacy `/v1/completions`.
- RTSP stream registration, dense caption generation, stop-captioning, and stream cleanup worked.
- File upload, metadata/list/content, file captioning, and file delete worked in the latest standalone retest.
- RTX PRO validation used image `nvcr.io/nvidia/vss-core/vss-rt-vlm:3.2.0`, cached Cosmos Reason 2 FP8 model `nim_nvidia_cosmos-reason2-8b_0303-fp8-dynamic-kv8`, and RTSP precheck discovered `video,852,480`.
- RTX PRO Kafka validation confirmed live env topics `mdx-vlm`, `mdx-vlm-incidents`, and `vision-llm-errors`; offsets moved to `mdx-vlm:0:46`, `mdx-vlm-incidents:0:21`, and `vision-llm-errors:0:0`.
- RTX PRO validation cleaned up the temporary `rt-vlm-eval-rtxpro-local` stream and restored the pre-existing `bench-kafka` container after using a temporary broker with `host.docker.internal:9092` advertised.
- Full repo infra Kafka compose is not required for standalone validation and may fail with a minimal RT-VLM env because it includes full-profile SDRC compose fragments.

Expected verifier checks after this PR:
- Use host port `8018` consistently in the skill/evals/deploy reference.
- Derive the RT-VLM image tag from compose; current documented fallback is `3.2.0`.
- Treat `/openapi.json` as endpoint source of truth; do not call `/v1/license` unless exposed.
- Require an RTSP probe to discover video stream/caps; an exit code of `0` with unknown media type is not sufficient without a cross-check.
- Handle agent-safe secret handoff for `NGC_CLI_API_KEY` and call out manual `sudo chown` ownership when required.
- Use live API-shaped `media_info={"type":"offset","start_offset":0,"end_offset":10}` for file slicing, not the old `*_offset_ms` fields.
- Treat `/v1/metrics` as unauthenticated on current 26.05 standalone builds.
- Treat singular CV-style `/v1/stream/*` as compatibility-only; use plural `/v1/streams/*` IDs for dense-caption validation.
- Provide a documented Kafka fallback image and non-`ss` port checks for restricted agent environments.

---

## Skill Eval - `skills/vss-deploy-dense-captioning/evals/alerts_profile_api.json`

Branch: `fix/nvbug-6228187-rtvi-vlm-skill` - 1 platform - spec `alerts_profile_api`
Result source: B200 RT-VLM validation with Kafka enabled after broker/listener
correction, plus PR static checks. Formal skills-eval trace artifacts were not
rerun after the documentation-only follow-up fixes.

| Platform | Mode | Query | Result | Reward | Duration | Turns | Prompt tok | Cached tok | Trace |
|---|---|---|---|---|---|---|---|---|---|
| B200 | alerts real-time / RT-VLM direct API | Validate RT-VLM readiness/API/OpenAPI/chat/completions against an alerts-style deployment and show Kafka incident-consumer guidance. | PASS - core API and Kafka path passed after documented broker setup | n/a | n/a | n/a | n/a | n/a | manual |

Observed coverage:
- Kafka validation worked after starting a broker with an advertised listener reachable at `${HOST_IP}:9092` and restarting RT-VLM.
- The standalone validation used the live RT-VLM env topic values; caption and incident topics received messages.
- Source-backed alerts/profile topic defaults are `mdx-vlm` and `mdx-vlm-incidents`; a bare copied compose without env overrides falls back to `vision-llm-messages` and `vision-llm-events-incidents`.
- Caption and incident message keys matched for the same request/chunk.

Expected verifier checks after this PR:
- Read `KAFKA_TOPIC`, `KAFKA_INCIDENT_TOPIC`, and `ERROR_MESSAGE_TOPIC` from the live `vss-rtvi-vlm` container or deployment source before consuming Kafka records.
- For the VSS alerts/profile source, use `mdx-vlm-incidents` as the incident topic; use the `vision-llm-events-incidents` fallback only for a bare copied compose with no topic override.
- Start Kafka before RT-VLM when Kafka is enabled, or restart/recreate `rtvi-vlm` after Kafka comes up or the advertised listener changes.
- Use a self-contained standalone Kafka broker for RT-VLM-only validation; use full repo infra Kafka only when the full profile env/config is present and `docker compose config --quiet` passes.
- If Kafka is already running, confirm with the user whether to reuse it or launch/replace a broker before stopping anything.
- Use deterministic positive alert prompts first when validating Kafka wiring, then switch back to scene-analysis prompts.
- Preserve model cache volumes; avoid `docker compose down -v` unless intentionally forcing a large model re-download.

All core RT-VLM deployment, API, caption generation, Kafka caption publishing, and
Kafka incident publishing checks passed during manual validation on B200 and RTX
PRO after the documented prerequisites were satisfied. The remaining changes in
this PR are skill/documentation reliability fixes intended to make the same
outcome repeatable for automated validation.

---
