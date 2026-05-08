# LVS Debugging — Long-form Reference

`deploy-lvs-service.md` §15 has the short table. This reference expands each row.

## "Nothing happens" on `docker compose up`

**Symptom**: `docker compose up -d` succeeds but `docker ps` shows no container;
`docker compose ps` reports zero services.

**Root cause**: The LVS blueprint compose puts `lvs-server` under
`profiles: ["bp_developer_lvs_2d"]`. Without `--profile bp_developer_lvs_2d`,
compose treats the service as profile-gated and skips it.

**Fix**:
```bash
docker compose \
  -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d \
  up -d
```

Add the flag to every `config`, `pull`, `up`, `down`, `logs`, and `ps`
invocation. A shell function is convenient:
```bash
dcl() {
  docker compose \
    -f "${MDX_SAMPLE_APPS_DIR}/lvs/compose.yml" \
    --profile bp_developer_lvs_2d \
    "$@"
}
dcl up -d && dcl logs -f lvs-server
```

## `bind source path does not exist: /lvs/.env`

**Symptom**: compose fails with a bind-mount error referencing a literal
path that starts with `/lvs/...`.

**Root cause**: `MDX_SAMPLE_APPS_DIR` is unset, so compose interpolates the
variable into an empty string and the `env_file` / volume `source` paths
collapse to `/lvs/.env` etc.

**Fix**:
```bash
export MDX_SAMPLE_APPS_DIR=~/met-blueprints/deployments
ls "$MDX_SAMPLE_APPS_DIR/lvs/.env" "$MDX_SAMPLE_APPS_DIR/lvs/configs/config.yaml"
```

You can also set it in your shell rc / systemd unit so it persists across
terminals.

## `docker: Error response from daemon: unauthorized`

**Symptom**: `docker compose pull` or first `up` fails with an NGC auth error
on the `nvcr.io/nvstaging/vss-core/vss-video-summarization` image.

**Root cause options**:

1. No `docker login nvcr.io` performed.
2. Your NGC API key doesn't have access to the `nvstaging` org — this tag is
   a staging / release-candidate build, often gated to internal users.

**Fix 1 (auth)**:
```bash
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
```

**Fix 2 (pin a public tag instead)**:
```bash
echo 'CONTAINER_IMAGE=nvcr.io/nvidia/vss-core/vss-long-video-summarization:3.1.0' \
  >> "$MDX_SAMPLE_APPS_DIR/lvs/.env"
```

The `3.1.0` tag has different behavior from the `3.2.0-rc1` build — some
env vars (e.g. `USE_RTVI_VLM`) may not be wired up. Consult the public docs.

## Container exits immediately (`Exited (1)`)

**Diagnose**:
```bash
docker logs --tail 200 vss-lvs
```

Look for the specific line that preceded the exit:

| Log line | Meaning | Fix |
|---|---|---|
| `Error: No GPUs were found` | `nvidia-smi` returns no devices inside container | Install / reinstall NVIDIA Container Toolkit; restart docker daemon; verify `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` works |
| `Error: annoy is not installed in the image` | Wrong image (some baseline without `annoy`) | Verify `CONTAINER_IMAGE` matches an LVS release |
| `Please set BACKEND_PORT env variable` | `BACKEND_PORT` unset | Set in `.env` (default `38111`) |
| FastAPI startup error re: database connection | ES/Milvus unreachable | `curl $ES_HOST:$ES_PORT` from host; fix network or point at a live backend |
| pydantic / OpenAI SDK error re: `api_key` | `LVS_LLM_API_KEY` resolves to empty | Set `NVIDIA_API_KEY` or `OPENAI_API_KEY` in `.env` |

## OOM (`Exited (137)` or `OOMKilled: true`)

```bash
docker inspect vss-lvs | grep -A2 OOMKilled
nvidia-smi
```

**Fixes** (in escalating order):

1. Lower `VLM_BATCH_SIZE` in `.env` — start with `1` or `2` on ≤48 GB cards.
2. Lower `VLLM_GPU_MEMORY_UTILIZATION` (e.g. `0.5`).
3. Lower `VLM_INPUT_WIDTH` / `VLM_INPUT_HEIGHT`.
4. Increase `VLM_DEFAULT_NUM_FRAMES_PER_CHUNK` to reduce parallel calls.
5. Pin to a larger GPU: `GPU_DEVICES=<idx>`.

Remember: this LVS container does NOT host the VLM — the OOM might be on the
**upstream VLM NIM's** GPU, not this one. Check `nvidia-smi` on whichever
host runs the NIM.

## Port already in use

**Symptom**: `bind: address already in use` for 38111 / 38112 / 38113.

**Why port-remapping won't help here**: the compose uses `network_mode:
host`, which ignores any `ports:` mapping. You must change the app's own
listening port.

**Fix**:
```bash
# Pick new free ports
cat >> "$MDX_SAMPLE_APPS_DIR/lvs/.env" <<EOF
BACKEND_PORT=48111
LVS_MCP_PORT=48112
FRONTEND_PORT=48113
EOF
docker compose -f "$MDX_SAMPLE_APPS_DIR/lvs/compose.yml" \
  --profile bp_developer_lvs_2d up -d --force-recreate
```

Then probe the new port: `curl http://localhost:48111/v1/ready`.

## Healthcheck failing past the 120 s grace period

The healthcheck is `curl -f http://localhost:$BACKEND_PORT/v1/ready`. If it
still fails after `start_period: 120s`:

1. Is the process up? `docker exec vss-lvs ps aux | head`
2. Is it listening on the expected port?
   `docker exec vss-lvs ss -tnlp | grep 381`
3. Does `curl -v http://localhost:38111/v1/ready` work from the **host**?
4. Check for upstream failure — the `/v1/ready` endpoint returns not-ready
   if a declared LLM/VLM is unreachable:
   ```bash
   docker logs vss-lvs | grep -Ei "llm|vlm|endpoint|unreachable|connect"
   ```

Common upstream causes:
- LLM NIM down: `curl $LLM_BASE_URL/v1/models`
- VLM NIM down: `curl $VLM_BASE_URL/v1/models`
- Firewall between LVS host and NIM host
- `HOST_IP` misconfigured (e.g. set to a docker-bridge IP not reachable from
  inside host-networked LVS)

## `could not select device driver "nvidia" with capabilities: [[gpu]]`

Missing NVIDIA Container Toolkit.
```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 503 "Another video is being processed"

By design: VIA processes one video at a time. Options:

- Poll `/v1/files` + `/v1/summarize` and serialize from the client.
- Run a second LVS instance on another host (not another container on the
  same host — `container_name: vss-lvs` is hardcoded).

## Incomplete JSON from VLM

Docs' Known Issue. The VLM sometimes fails to satisfy the CA-RAG schema's
`events[]` requirements.

Tunables in `$MDX_SAMPLE_APPS_DIR/lvs/configs/config.yaml`:
- `functions.summarization_using_llm.events` — simpler list → fewer failures
- `functions.summarization_using_llm.prompts.caption` — more explicit prompt
- Increase `VLM_DEFAULT_NUM_FRAMES_PER_CHUNK`
- Retry failed chunks at the client level

## MCP server unreachable

```bash
# Is the env flag on?
docker exec vss-lvs env | grep -E 'LVS_ENABLE_MCP|LVS_MCP_PORT'

# Is the port listening?
curl -I http://localhost:38112/sse

# If not, check logs for mcp startup
docker logs vss-lvs | grep -i mcp
```

Make sure any host firewall allows 38112.

## Stale logs from a previous run polluting diagnostics

The entrypoint clears `$VIA_LOG_DIR/*` on startup if `VIA_LOG_DIR` is set and
exists. If you're not bind-mounting logs and want to preserve them across
recreates, set `VIA_LOG_DIR=/var/log/via` and mount that path.

## Fast-path: "is LVS up and serving"?

```bash
docker ps --filter name=vss-lvs --format '{{.Status}}' && \
  curl -fsSL http://localhost:${BACKEND_PORT:-38111}/v1/ready && \
  echo "LVS is ready"
```
