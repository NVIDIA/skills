# Agent-Facing Failure Modes

Use this file first when a VSS deploy, runtime probe, `/generate` request, or skill handoff fails. It consolidates cross-profile symptoms.

After identifying the failure class here, continue in the matching profile reference:

- `base.md` for base profile agent/VLM/VIOS failures.
- `lvs-profile.md` for long-video summarization and `vss-lvs` / `vss-rtvi-vlm` failures.
- `search.md` for Cosmos Embed1, Elasticsearch, and search-profile failures.
- `alerts.md` for alerts profile failures.
- `warehouse-debug.md` for warehouse profile stream, perception, and analytics failures.

## Quick Triage

Run these checks before changing configuration:

```bash
docker compose -f "$REPO/deploy/docker/resolved.yml" ps
grep -n '\${' "$REPO/deploy/docker/resolved.yml" | head -20
docker logs vss-agent --tail 200
```

If `resolved.yml` does not exist, return to `SKILL.md` Step 3 and run the compose dry-run before deploying.

## Failure Mode Table

| Symptom | Grep / check | Likely cause | Corrective action |
|---|---|---|---|
| `resolved.yml` contains `${...}` | `grep -n '\${' "$REPO/deploy/docker/resolved.yml"` | Compose did not see required env values such as `BP_PROFILE`, `MODE`, `HARDWARE_PROFILE`, `LLM_MODE`, or `VLM_MODE`. This can cause every profile's services to deploy. | Fix the missing values in the profile `generated.env`, regenerate `resolved.yml`, re-run the grep check, then deploy. |
| `docker compose up` says no `resolved.yml` | `test -f "$REPO/deploy/docker/resolved.yml"` | The dry-run step was skipped. | Run `docker compose --env-file "$ENV_GEN" config > "$REPO/deploy/docker/resolved.yml"` first. |
| NIM container is up but `/generate` or model calls time out | `docker logs <nim-container> --tail 200` and `curl -sf http://<host>:<port>/v1/models` | NIM cold start or model still loading. | Keep polling `/v1/models` or the service health endpoint before retrying the agent request. Do not restart a loading NIM unless logs show a hard failure. |
| `CUDA out of memory` | Search `docker logs <container> 2>&1` for `out of memory`. | LLM, VLM, RT-VLM, or embedding service is too large for the selected GPU placement. | Follow the profile sizing reference. Typical fixes are lowering `NIM_KVCACHE_PERCENT`, lowering `RTVI_VLLM_GPU_MEMORY_UTILIZATION`, lowering max model length / max sequences, reducing streams, or switching one side to remote mode with user approval. |
| Container exits with code `137` or `OOMKilled` | Search `docker inspect <container>` for `OOMKilled`. | Host RAM or GPU memory pressure. | Check `free -h` and `nvidia-smi`. Reduce workload/model memory, free memory, or pick a larger host/profile placement. |
| `authentication required`, `401`, or image pull fails from `nvcr.io` | Search `docker compose logs 2>&1` for `authentication required`, `unauthorized`, `401`, or `nvcr.io`. | Missing, invalid, or expired `NGC_CLI_API_KEY`. | Re-export `NGC_CLI_API_KEY`, run the NGC checks in `ngc.md`, then retry login/pull or redeploy. |
| DGX Spark standalone LLM NIM exits or never reaches ready | `docker logs nemotron-dgx-spark --tail 200` and `curl -sf http://localhost:30081/v1/health/ready` | Missing NGC credentials, image pull/access failure, or too much KV cache/context for unified memory. Logs can include `No available memory for the cache blocks`. | Follow `edge.md`: verify `NGC_API_KEY`, restart the standalone NIM with lower `NIM_MAX_NUM_SEQS`, or lower `NIM_KVCACHE_PERCENT` / `NIM_GPU_MEM_FRACTION` by `0.05`. Do not use `NIM_MAX_MODEL_LEN` with the DGX Spark variant. |
| Remote LLM/VLM returns HTTP `401` | Search `docker logs vss-agent --tail 200` for `401`, `unauthorized`, or `authentication`. | Missing or invalid remote endpoint API key, usually `NVIDIA_API_KEY`. | Verify the endpoint key and model name, update `generated.env`, regenerate `resolved.yml`, and redeploy affected services. |
| Remote LLM/VLM returns HTTP `5xx` | Search `docker logs vss-agent --tail 200` for `5xx`, `InternalServerError`, `BadGateway`, or `ServiceUnavailable`. | Remote endpoint unavailable, overloaded, wrong model, or transient provider failure. | Confirm endpoint URL and model name. Retry after the endpoint is healthy, or switch backend placement with user approval. |
| LVS remote VLM hangs or OOMs | Check `VLM_MODE`, `RTVI_VLM_MODEL_PATH`, and `RTVI_VLM_ENDPOINT` in `$ENV_GEN`. | `VLM_MODE=remote` was set but `RTVI_VLM_MODEL_PATH` still points to local weights, so RT-VLM tries to load and proxy. | Set `RTVI_VLM_MODEL_PATH=none`, ensure `RTVI_VLM_ENDPOINT=<endpoint>/v1`, regenerate `resolved.yml`, and redeploy `vss-rtvi-vlm` / `vss-lvs`. |
| Thor Edge 4B fails to pull weights | `curl -sf -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2` | Missing, invalid, or unauthorized `HF_TOKEN` for gated Hugging Face weights. | Set a valid `HF_TOKEN` with model access, rerun the `edge.md` verification, and restart the standalone vLLM container. |
| Thor Edge 4B agent produces planning text instead of tool calls | Search `docker logs vss-agent --tail 200` for `[USER]` or missing tool calls. | `config_edge.yml` prompt is missing explicit tool-call routing rules for Edge 4B. | Use the Thor Edge 4B prompt guidance in `edge.md`, then redeploy/restart the agent. |
| WebSocket query returns `error_message` | `docker logs vss-agent --tail 200` | LLM or VLM backend is not healthy or not reachable from the agent container. | Check model service `/v1/models`, verify `LLM_BASE_URL` / `VLM_BASE_URL` in `resolved.yml`, then restart/redeploy the affected service. |
| Empty report or empty video answer | `docker logs vss-agent --tail 200` | VLM unreachable, bad VST URL, missing video ingest, or backend still cold. | Verify VST upload/listing, VLM `/v1/models`, and agent env URLs. Retry after health checks pass. |
| `unknown or invalid runtime name: nvidia` | Search `docker info 2>/dev/null` for `runtimes`. | NVIDIA Container Toolkit is not installed or Docker was not restarted. | Follow `prerequisites.md`, restart Docker, and rerun the pre-flight check. |
| GPU not detected | `nvidia-smi` and `docker run --rm --gpus all ubuntu:22.04 nvidia-smi` | Driver, kernel module, or Docker GPU runtime issue. | Load modules with `sudo modprobe nvidia && sudo modprobe nvidia_uvm`, then follow `prerequisites.md` if Docker still cannot see GPUs. |
| `cosmos-reason2-8b` crashes or is restarted in shared GPU mode | `docker logs nvidia-cosmos-reason2-8b --tail 200` | Known CR2/NIM restart limitation in shared GPU mode. Restarting the CR2 container alone may not recover service for now. | Redeploy the full affected VSS stack (workaround until Cosmos Reason 3 is released). |


## Rule of Thumb

- If the failure appears before `docker compose up`, check env generation and `resolved.yml`.
- If containers start but API calls fail, check service health and `vss-agent` logs.
- If model services fail, check GPU memory, model names, endpoint URLs, and credentials.
- If a corrective action changes env values, regenerate `resolved.yml` before redeploying.
