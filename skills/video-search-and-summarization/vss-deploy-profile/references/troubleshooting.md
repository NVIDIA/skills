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
