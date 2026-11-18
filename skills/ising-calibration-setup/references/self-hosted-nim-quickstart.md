# Self-hosted NIM quickstart

- Container: `nvcr.io/nim/nvidia/ising-calibration-1-5-31b:1.5.0`
- Catalog page: `https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/ising-calibration-1-5-31b`

## Get an NGC personal key

This is a **separate credential** from the `nvapi-` key used for the hosted
API (see `hosted-api-quickstart.md`) -- do not reuse one for the other.

1. Sign in at `https://org.ngc.nvidia.com`.
2. Go to `https://org.ngc.nvidia.com/setup/personal-keys` (or: account icon ->
   Setup -> Generate Personal Key).
3. Click **Generate Personal Key**. The dialog has three required fields:
   - **Key Name** -- any label, e.g. `ising-nim`.
   - **Expiration** -- pick the shortest period that covers your use.
   - **Services Included** -- a dropdown of per-service checkboxes defining the
     key's scopes. **NGC Catalog** is the only one needed: it grants download
     access to the public catalog this NIM image is served from. Without it the
     key still looks valid but fails at `docker login` with an auth error. The
     remaining services apply to other NVIDIA products.
4. The key is displayed **once**, when it is created. Copy it now; if you lose
   it you must generate a new one (there is no way to re-display it). Load it
   into your shell without it appearing in your command history or terminal
   scrollback:

   ```bash
   read -rs NGC_API_KEY
   export NGC_API_KEY
   ```

   Paste the key at the prompt (input will not echo), then press Enter.
   `export NGC_API_KEY=...` typed as a literal command instead would leave
   the raw key in `~/.bash_history` / `~/.zsh_history` -- avoid that form.

5. Confirm it landed, without printing it:

   ```bash
   test -n "$NGC_API_KEY" && echo "NGC_API_KEY is set" || echo "NGC_API_KEY is EMPTY"
   ```

   Note that this only lives in the current shell, so a new terminal will not
   have it. That matters for `docker login`, and for `docker run`, which passes
   it into the container with `-e NGC_API_KEY=$NGC_API_KEY`. Once login succeeds,
   `docker pull` uses the stored credential instead.

The `docker login` in the next section is the real test of whether the key and
its scopes are correct. Run it before pulling anything, so a credential problem
surfaces in two seconds rather than partway through a multi-gigabyte pull.

### Supplying the key again later

`docker login` caches the credential locally, so routine
re-pulls do not need `NGC_API_KEY` in your shell again. You need it again when
the key expires, on a new machine, or in CI, where every run starts from a clean
environment.

For those cases read the key from a secret store at the moment it is needed
rather than from a plaintext file -- `secret-tool` or `pass` on Linux, the
`security` CLI on macOS, or a password manager CLI. In automation, use your
platform's secrets manager and inject `NGC_API_KEY` at pull time. Never embed it
in a Dockerfile, image layer, or anything committed to a repo.

## Log Docker in to nvcr.io

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Copy the command as written -- `$oauthtoken` is a fixed literal, not a
placeholder for your username, and the single quotes around it are required.

Expect `Login Succeeded`. If you get an auth error instead, the problem is the
key, not Docker: confirm `NGC_API_KEY` is set in *this* shell and that the key
includes the **NGC Catalog** service scope. Fix it here before pulling.

A `docker login` that succeeds but a `docker pull` that then 401s or reports
the image as not found usually means the opposite problem: the credential is
fine, but the image is not visible to your account -- typically an access
agreement that still needs accepting on the catalog page.

## Pull and run

```bash
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-~/.cache/nim}
mkdir -p "$LOCAL_NIM_CACHE"

docker pull nvcr.io/nim/nvidia/ising-calibration-1-5-31b:1.5.0

docker run --gpus all -it --rm \
  -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  nvcr.io/nim/nvidia/ising-calibration-1-5-31b:1.5.0
```

There are **two** large downloads, and they are easy to confuse:

1. `docker pull` fetches the container image (~46GB on disk once unpacked).
   It authenticates with the credential `docker login` stored.
2. On startup the container picks a configuration suited to your GPU and
   downloads the matching model weights. For this image they come from the
   public Hugging Face repositories `nvidia/Ising-Calibration-1.5-31B-BF16` and
   `nvidia/Ising-Calibration-1.5-31B-NVFP4`, pinned to exact revisions in the
   image's model manifest.

`-e NGC_API_KEY=$NGC_API_KEY` passes your key into the container; this is the
form NVIDIA's NIM documentation uses. The shell expands `$NGC_API_KEY` when you
run the command, so the key must be loaded (with `read -rs`) in the same shell
first. Because the expanded value lands in that process's command line, do not
paste the key literally in its place.

The `-v` flag is what makes that second download reusable. It reads as
`host_path:container_path`:

```
-v "$LOCAL_NIM_CACHE:/opt/nim/.cache"
    |                 |
    |                 +- where the container writes weights. Fixed by the
    |                    image (its NIM_CACHE_PATH); do not change it.
    +- a directory on your machine, ~/.cache/nim by default. Change this
       freely to put the cache on a bigger disk.
```

With that mapping the weights land on your machine and survive the container
being deleted. Point the container side anywhere else -- `/root/.cache/nim`, for
instance -- and the mount silently does nothing: the container still writes to
`/opt/nim/.cache` on its own filesystem, and `--rm` discards it, so every run
re-downloads the full weights.

To confirm the mapping works, watch the host directory while the container
starts. If `du -sh "$LOCAL_NIM_CACHE"` stays near-empty while the log prints
`Downloaded filename: ...`, the container side of the `-v` is wrong.

## GPU requirements

Requires an NVIDIA Ada Lovelace, Hopper, or Blackwell architecture GPU.

| GPU family           | Memory / count              |
|----------------------|-----------------------------|
| L40S                 | 48GB, 2 GPUs (BF16 / NVFP4) |
| H200 SXM             | 141GB, 1-2 GPUs             |
| GH200 / B200 / GB200 | 1-2 GPUs                    |
| GB300+               | single GPU                  |

This table lists the configurations NVIDIA has published as tested. A GPU of a
supported architecture with comparable memory may work without appearing here.
The container picks a configuration for whatever GPU it finds at startup; there
is nothing to select yourself.

## Calling the running container

Once up, the container serves an OpenAI-compatible HTTP API on port 8000 for
chat completions and multimodal (image) calibration-plot analysis.

### Knowing when it is ready

The first run downloads model weights before serving; later runs load them from
the cache mount. Either way startup takes a while before the server accepts
requests. Wait for this line:

```
The server is up and ready to serve!
```

followed by `Application startup complete.` and
`Uvicorn running on http://0.0.0.0:8000`. The container also prints example
cURL requests for its own endpoints at this point. Requests sent before this
will fail with a connection error -- that is expected, not a misconfiguration.

### Health check

```bash
curl -s http://localhost:8000/v1/health/ready
# {"object":"health.response","message":"Service is ready","status":"ok"}
```

Returns `200` once the model is loaded; use it rather than inferring readiness
from log output. `/v1/health/live` is also available.

### Endpoints

The container describes its own API, so read the list from the container rather
than from documentation that can drift:

```bash
curl -s http://localhost:8000/openapi.json | jq -r '.paths | keys[]'
```

Open `http://localhost:8000/docs` in a browser for the same thing as browsable
Swagger UI, with request and response schemas.

The container also prints its endpoints, with example cURL requests, to the log
at startup.

### Model ID

The container serves the model as `nvidia/NVIDIA-Ising-Calibration-1.5-31B`,
which differs in case and prefix from the hosted API's
`nvidia/ising-calibration-1.5-31b`. The two are not interchangeable, so read
the ID from the container rather than copying it from a hosted example:

```bash
curl -s http://localhost:8000/v1/models
```

The `model` field is optional against a NIM serving a single model -- omit it
and the response reports the resolved ID.

### Hello world: describe a real calibration plot

A text ping confirms the server is up; this optional request goes further and
exercises the multimodal path. It needs no local test image -- it pulls a real
sample plot from the public QCalEval benchmark dataset. Requires `jq`:

```bash
# 1. Grab a real calibration-plot image URL from the public QCalEval dataset
IMAGE_URL=$(curl -s "https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FQCalEval&config=default&split=test" \
  | jq -r '.rows[0].row.images[0].src')

# 2. Ask your container to describe it
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg img "$IMAGE_URL" '{
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

The request shape is the same as the hosted API's: a `content` array mixing a
`text` block and an `image_url` block, taking a plain HTTPS URL with no base64
encoding. The QCalEval `images[].src` URLs are signed and expire, so fetch a
fresh one rather than reusing a URL from a previous run.

If `jq` is not installed, the same two steps work with `python3` -- parse
`rows[0].row.images[0].src` out of the dataset response and substitute it into
the JSON body.

A working response finishes with `finish_reason: stop` and reads the plot in
domain terms: whether the calibration succeeded, what the data trends show, and
what to try next.

The first request after startup is slow: warmup work completes on first
inference. Later requests are markedly faster.
