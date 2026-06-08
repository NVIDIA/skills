# Deploy troubleshooting

### Step 3b — Verify resolved.yml has no unexpanded ${...} tokens

**Skipping this is the #1 cause of "I deployed `search` but it brought
up `base` + `lvs` + `search` services."** The `.env` line near 90 is
literal `COMPOSE_PROFILES=${BP_PROFILE}_${MODE},...` — docker compose
expands it at `config` time using the same env file. If any upstream
var (`BP_PROFILE`, `MODE`, `HARDWARE_PROFILE`, `LLM_MODE`,
`VLM_MODE`) is missing from the env, the rendered profile list
collapses to the empty string, and compose then includes **every**
service from **every** profile.

```bash
if grep -q '\${' "$REPO/deploy/docker/resolved.yml"; then
  echo "FAIL: resolved.yml has unexpanded variables:"
  grep -n '\${' "$REPO/deploy/docker/resolved.yml" | head -5
  exit 1
fi
```

If this check fails, re-apply the Step 2 env overrides directly to
the `.env` file at the path above, regenerate `resolved.yml` (Step 3),
and re-run this check before continuing.

## NIM endpoint probes
<a id="nim-probes"></a>

Cross-profile LLM/VLM reachability checks for the "Debugging a Deployment"
quick-checks in [`../SKILL.md`](../SKILL.md#debugging-a-deployment). Extract the
selected modes/URLs from `generated.env`, then skip `localhost:3008x` when the
matching `*_MODE=remote` (a connection refused there is expected) and probe the
selected `*_BASE_URL/v1/models` via `scripts/probe_remote_models.sh` instead:

```bash
if [ -n "${ENV_GEN:-}" ] && [ -f "$ENV_GEN" ]; then
  # Use `sub(/^[^=]*=/,""); print` (the whole value after the first '='), NOT
  # `print $2`, so a value containing '=' — e.g. a base URL with a query
  # string like `?api-version=...` — is not truncated at the first '='.
  LLM_MODE="${LLM_MODE:-$(awk -F= '$1=="LLM_MODE"{sub(/^[^=]*=/,""); print}' "$ENV_GEN" | tail -1)}"
  VLM_MODE="${VLM_MODE:-$(awk -F= '$1=="VLM_MODE"{sub(/^[^=]*=/,""); print}' "$ENV_GEN" | tail -1)}"
  LLM_BASE_URL="${LLM_BASE_URL:-$(awk -F= '$1=="LLM_BASE_URL"{sub(/^[^=]*=/,""); print}' "$ENV_GEN" | tail -1)}"
  VLM_BASE_URL="${VLM_BASE_URL:-$(awk -F= '$1=="VLM_BASE_URL"{sub(/^[^=]*=/,""); print}' "$ENV_GEN" | tail -1)}"
  LLM_NAME="${LLM_NAME:-$(awk -F= '$1=="LLM_NAME"{sub(/^[^=]*=/,""); print}' "$ENV_GEN" | tail -1)}"
  VLM_NAME="${VLM_NAME:-$(awk -F= '$1=="VLM_NAME"{sub(/^[^=]*=/,""); print}' "$ENV_GEN" | tail -1)}"
fi

# VLM NIM responding (base/lvs profiles)
if [ "${VLM_MODE:-}" = "remote" ]; then
  echo "VLM_MODE=remote — skip localhost:30082; probing ${VLM_BASE_URL:-<remote-vlm-base-url>}/v1/models"
  REMOTE_API_KEY="${NVIDIA_API_KEY:-}" \
    "$REPO/skills/vss-deploy-profile/scripts/probe_remote_models.sh" "$VLM_BASE_URL" "${VLM_NAME:-}"
else
  curl -sf http://localhost:30082/v1/models | python3 -m json.tool
fi

# LLM NIM responding
if [ "${LLM_MODE:-}" = "remote" ]; then
  echo "LLM_MODE=remote — skip localhost:30081; probing ${LLM_BASE_URL:-<remote-llm-base-url>}/v1/models"
  REMOTE_API_KEY="${NVIDIA_API_KEY:-}" \
    "$REPO/skills/vss-deploy-profile/scripts/probe_remote_models.sh" "$LLM_BASE_URL" "${LLM_NAME:-}"
else
  curl -sf http://localhost:30081/v1/models | python3 -m json.tool
fi
```
