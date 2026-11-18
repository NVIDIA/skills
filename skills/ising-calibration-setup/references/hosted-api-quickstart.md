# Hosted Model API quickstart (build.nvidia.com)

## Get an API key

1. Sign in at `https://build.nvidia.com` with an NVIDIA account.
2. Go to `https://build.nvidia.com/settings/api-keys` (or: profile menu ->
   Settings -> API keys) and click **Generate API Key**. You will be asked to
   name the key and set an expiration.
3. The key starts with `nvapi-` and is shown only once. Save it in a password
   manager, and load it into your shell without it appearing in your command
   history or terminal scrollback:

   ```bash
   read -rs NVIDIA_API_KEY
   export NVIDIA_API_KEY
   ```

   Paste the key at the prompt (input will not echo), then press Enter.
   `export VAR=nvapi-...` typed as a literal command instead would leave the
   raw key in `~/.bash_history` / `~/.zsh_history` -- avoid that form.

Do not commit this key, paste it into chat, or write it to a world-readable
file.

### If you will be connecting regularly

The `read -rs` pattern above is fine for a one-off session, but is
inconvenient if you re-run scripts or notebooks against this API often. For
repeated interactive use, pull the key from an OS-level secret store at the
moment it is needed, instead of storing it in a plaintext file:

- Linux: `secret-tool` (GNOME Keyring / freedesktop Secret Service) or the
  `pass` password manager
- macOS: the `security` CLI against Keychain
- Cross-platform: a password manager CLI (1Password `op`, Bitwarden `bw`,
  etc.)

For example: `export NVIDIA_API_KEY=$(secret-tool lookup service nvidia-api key nvapi)`.
A `.env` file loaded per-project (e.g. via `direnv`) is an acceptable middle
ground if `chmod 600` and excluded via `.gitignore` -- do not source it from
a global shell rc file.

If this key is used by an application or pipeline that runs repeatedly
(rather than a person at a terminal), use a real secrets manager (Vault, or
your cloud provider's Secrets Manager) and inject the key as an environment
variable at process start. Never embed it in code, container images, or
anything committed to a repo.

## Endpoint

```
POST https://integrate.api.nvidia.com/v1/chat/completions
```

OpenAI-compatible chat completions API. Required header:

```
Authorization: Bearer $NVIDIA_API_KEY
```

## Example request

```bash
curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/ising-calibration-1.5-31b",
    "messages": [
      {"role": "user", "content": "Describe what this calibration plot shows."}
    ],
    "temperature": 0.2,
    "max_tokens": 8192
  }'
```

This returns a standard OpenAI-compatible chat completion response, e.g.:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "nvidia/ising-calibration-1.5-31b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 14, "completion_tokens": 187, "total_tokens": 201}
}
```

## Hello world: describe a real calibration plot

The text-only ping above proves auth works; this optional request goes further
and exercises the multimodal path. It pulls one real sample image from the
public QCalEval benchmark dataset (no local test image needed) and asks the
model to describe it. The image request shape
(`type: image_url`) comes from the
[Hugging Face model card](https://huggingface.co/nvidia/Ising-Calibration-1.5-31B-BF16);
the build.nvidia.com interactive code tab currently ships with the request
body blanked out, so do not rely on it for this shape.

Requires `jq`.

```bash
# 1. Grab a real calibration-plot image URL from the public QCalEval dataset
IMAGE_URL=$(curl -s "https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FQCalEval&config=default&split=test" \
  | jq -r '.rows[0].row.images[0].src')

# 2. Ask the model to describe it
curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg img "$IMAGE_URL" '{
    model: "nvidia/ising-calibration-1.5-31b",
    messages: [{
      role: "user",
      content: [
        {type: "text", text: "Describe what this calibration plot shows."},
        {type: "image_url", image_url: {url: $img}}
      ]
    }],
    temperature: 0.2,
    max_tokens: 512
  }')"
```

This returns a domain-accurate analysis of the plot's calibration result.
The `image_url.url` field takes a plain HTTPS URL directly -- no base64
data-URI encoding needed for a publicly reachable image. The QCalEval
`images[].src` URLs are signed and expire (see the `Expires=` query
parameter), so always fetch a fresh one via the dataset API rather than
hardcoding a URL from a prior run.

If `jq` is not installed, the same two steps work with `python3` -- parse
`rows[0].row.images[0].src` out of the dataset response and substitute it into
the JSON body.

## Terms

Usage is governed by the NVIDIA API Trial Terms of Service (linked from the
model page).

## Note on credentials

This `nvapi-` key is a **different credential** from the NGC personal key
used to pull the self-hosted NIM image (see
`self-hosted-nim-quickstart.md`). They are not interchangeable.
