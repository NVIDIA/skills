# Deploy — env_overrides reference

### Step 2 — Build env_overrides

Build a dictionary of env var overrides based on user intent. Only include vars that differ from the profile's `.env` defaults.

**Always set (they have placeholder defaults in the template):**

| Var | Value |
|---|---|
| `HARDWARE_PROFILE` | Detected or user-specified |
| `VSS_APPS_DIR` | `<repo>/deployments` |
| `VSS_DATA_DIR` | `<repo>/data` (or user-specified) |
| `HOST_IP` | Detected host IP |
| `NGC_CLI_API_KEY` | From environment or user |

**Common overrides by user intent:**

| User intent | Env overrides |
|---|---|
| Remote LLM | `LLM_MODE=remote`, `LLM_NAME_SLUG=none`, `LLM_BASE_URL=<host>` (no `/v1`), `LLM_NAME=<model>`, `NVIDIA_API_KEY=<key>` |
| Remote VLM | `VLM_MODE=remote`, `VLM_NAME_SLUG=none`, `VLM_BASE_URL=<host>` (no `/v1`), `VLM_NAME=<model>`, `NVIDIA_API_KEY=<key>` |
| **Remote LLM AND remote VLM** (aka `remote-all`) | **BOTH of the above** — you must set `LLM_MODE=remote`, `VLM_MODE=remote`, `LLM_NAME_SLUG=none`, `VLM_NAME_SLUG=none`, `LLM_BASE_URL`, `VLM_BASE_URL`, `LLM_NAME`, `VLM_NAME`. The presence of a remote VLM endpoint does not imply `VLM_MODE=remote` — you have to write it explicitly. |
| NVIDIA API for remote inference | `LLM_BASE_URL=https://integrate.api.nvidia.com` |
| Dedicated GPUs | `LLM_MODE=local`, `VLM_MODE=local`, `LLM_DEVICE_ID=0`, `VLM_DEVICE_ID=1` |
| Different LLM model | `LLM_NAME=<name>`, `LLM_NAME_SLUG=<slug>` |
| Different VLM model | `VLM_NAME=<name>`, `VLM_NAME_SLUG=<slug>` |

**Extracting remote endpoints from user intent.**

If the user says "remote LLM" or mentions an LLM endpoint URL, you MUST do
all of the following before `docker compose up`:

1. Identify the endpoint URL and model name. If the user gave them in
   their prompt (e.g. *"deploy with remote LLM at
   `http://launchpad:11571` serving `nvidia/nvidia-nemotron-nano-9b-v2`"*),
   use those values directly. Strip any trailing `/v1` (see callout below).
2. If the user said "remote" without providing a URL or model, **stop and
   ask the user** for:
   - The LLM endpoint URL (without `/v1`)
   - The LLM model name served there
   - (same pair for VLM if they also said "remote VLM")
   - An `NVIDIA_API_KEY` if the endpoint requires one
3. Write `LLM_MODE=remote` + `LLM_NAME_SLUG=none` + `LLM_BASE_URL=<url>` +
   `LLM_NAME=<model>` into
   `deploy/docker/developer-profiles/dev-profile-<profile>/generated.env`
   (the skill's per-deploy working copy — see ``SKILL.md`` (see `../SKILL.md`)
   Step 1c). Do the same set for VLM if the user said remote VLM. Use
   `sed -i "s|^KEY=.*|KEY=VALUE|"` — the source `.env` template ships
   with placeholder rows for these keys, which `cp` to `generated.env`
   so the same `sed` patterns work.
4. After writing, `grep -E '^(HARDWARE_PROFILE|LLM_MODE|VLM_MODE|LLM_NAME_SLUG|VLM_NAME_SLUG|LLM_BASE_URL|VLM_BASE_URL)=' <env-file>`
   and verify every line shows the value you intended. A silent miss on
   `LLM_MODE` / `VLM_MODE` is the #1 cause of deployments coming up with
   wrong compose profiles.

Never leave `LLM_MODE` or `VLM_MODE` unset when the user said "remote".
The `.env` template's placeholder value is `LLM_MODE=local` — failing to
overwrite it means you silently deployed in local mode with remote URLs
dangling, which no `COMPOSE_PROFILES` path correctly supports.

> **Important — `/v1` suffix on base URLs**
>
> `LLM_BASE_URL` and `VLM_BASE_URL` must **not** include a trailing `/v1`.
> The agent's `config.yml` appends `/v1` automatically (`base_url: ${LLM_BASE_URL}/v1`),
> so including it yourself produces `/v1/v1/chat/completions` and requests will fail
> with connection / 404 errors.
>
> If a user or endpoint documentation gives you a URL ending in `/v1`, strip it
> before writing to `.env`. Examples:
> - User says: "LLM is at `http://10.0.0.5:31081/v1`" → write `LLM_BASE_URL=http://10.0.0.5:31081`
> - User says: "Use `https://integrate.api.nvidia.com/v1`" → write `LLM_BASE_URL=https://integrate.api.nvidia.com`

See the profile reference doc for full env override recipes.

**Do NOT set `COMPOSE_PROFILES` directly** — it is computed from `BP_PROFILE`, `MODE`, `HARDWARE_PROFILE`, `LLM_MODE`, `LLM_NAME_SLUG`, `VLM_MODE`, `VLM_NAME_SLUG`.
