# Evaluation Report

Evaluation of the `vss-deploy-video-embedding` skill before publication through NVSkills-Eval.

This benchmark documents validation of the standalone RT-Embed (`rtvi-embed`) bring-up path. The goal is to record whether the skill is safe, correct, and useful for agents before broader workflow use.

## Evaluation Summary

- Skill: `vss-deploy-video-embedding`
- Evaluation date: 2026-05-29
- Eval spec: `skills/vss-deploy-video-embedding/evals/standalone_deploy.json`
- Platform (manual): RTX Pro (3× RTX PRO 6000)
- Overall verdict: **PASS** (manual validation)
- Tier 3 live agent evaluation (L40S Skills Eval): **pending** — add results here after a clean run passes

## Agents Used

- Manual validation: operator following `SKILL.md` on RTX Pro
- Tier 3 Skills Eval agent: not yet recorded in this report

## Eval Spec Coverage

The `standalone_deploy.json` spec checks:

1. Standalone deploy from `deploy/docker/services/rtvi/rtvi-embed/rtvi-embed-docker-compose.yml` with profile `bp_developer_search_2d` (no full VSS profile)
2. Standalone env: `RTVI_EMBED_PORT=8017`, writable `VSS_DATA_DIR` with `data_log/vst/clip_storage`, Kafka/Redis disabled for standalone
3. First-boot warmup (`start_period: 1200s`) before declaring ready
4. Live probes: `/v1/ready`, `/v1/models` (`cosmos-embed1-448p`), container `vss-rtvi-embed`
5. No fabricated credentials

## Manual Validation Results (RTX Pro)

All checks below passed on port **8017** using the standalone flow documented in `SKILL.md`.

| Area | Result | Evidence |
|---|---|---|
| Deploy | ✅ PASS | `docker compose` + `bp_developer_search_2d`; standalone env (`VSS_DATA_DIR`, `RTVI_EMBED_KAFKA_ENABLED=false`, `ENABLE_REDIS_ERROR_MESSAGES=false`) |
| Verify | ✅ PASS | `/v1/ready` 200; `/v1/models` reports `cosmos-embed1-448p` |
| Text embeddings | ✅ PASS | `POST /v1/generate_text_embeddings` → non-empty `.data[0].embeddings` |
| Video file embeddings | ✅ PASS | `POST /v1/files` + `POST /v1/generate_video_embeddings` → `chunk_responses` with embeddings |
| Live RTSP SSE | ✅ PASS | `POST /v1/streams/add` + streaming embed → `data:` events with embeddings |
| Metrics | ✅ PASS | `/v1/metrics`, `/v1/assets/stats` |
| Credentials | ✅ PASS | Used host env keys; no fabricated tokens in operator output |

## Tier 3 Skills Eval (L40S)

**Not included in this report.** After a passing Skills Eval run on `standalone_deploy.json`, replace this section with the sanitized benchmark table (pattern: [PR #802](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/pull/802)).

To trigger a new run: open a PR that touches `skills/vss-deploy-video-embedding/`, then ask a PR vetter to comment:

```text
/ok to test <head-commit-sha>
```

Request a dedicated L40S pool member (no concurrent trial on the same Brev instance) for a clean score.

## Publication Recommendation

The standalone deploy path and API operations documented in `SKILL.md` are **validated manually** and suitable to proceed toward publication. Refresh this file when a **passing** Tier 3 Skills Eval run completes or when the eval spec, skill behavior, or target platform materially changes.
