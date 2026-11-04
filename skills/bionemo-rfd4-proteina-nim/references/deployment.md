# Deploy the public RFD4-Proteina NIM

Use the released image and prerequisites from the
[public Deploy tab](https://build.nvidia.com/ipd/rfd4-proteina/deploy) and
[NGC container listing](https://catalog.ngc.nvidia.com/orgs/nim/teams/ipd/containers/rfd4-proteina).
No private source checkout or custom deployment script is required.

## Requirements

Provide Docker with NVIDIA Container Toolkit, a supported NVIDIA GPU and driver,
NGC access, and space for the image and model cache. Check the product page for
supported hardware; this skill does not establish a minimum GPU-memory figure.
Hosted inference does not require a local GPU or Docker.

The public image is `nvcr.io/nim/ipd/rfd4-proteina:latest`. `latest` is mutable:
for reproducibility, select a published version/digest from NGC or record the
resolved image digest after pulling it.

## Launch

Set `NGC_API_KEY` through the user's normal secret mechanism. Keep shell tracing
off and never print the key. Login passes the key on standard input:

```bash
set -e
: "${NGC_API_KEY:?Set your NGC API key in the environment}"
printf '%s' "$NGC_API_KEY" |
  docker login nvcr.io --username '$oauthtoken' --password-stdin

export RFD4_NIM_IMAGE=nvcr.io/nim/ipd/rfd4-proteina:latest
export RFD4_NIM_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/nim/rfd4-proteina"
mkdir -p "$RFD4_NIM_CACHE"
docker pull "$RFD4_NIM_IMAGE"
RFD4_NIM_IMAGE_PINNED="$(docker image inspect --format '{{index .RepoDigests 0}}' "$RFD4_NIM_IMAGE")"
case "$RFD4_NIM_IMAGE_PINNED" in
  nvcr.io/nim/ipd/rfd4-proteina@sha256:*) ;;
  *) echo "Could not resolve the public image digest; stopping" >&2; exit 1 ;;
esac
printf 'Using image: %s\n' "$RFD4_NIM_IMAGE_PINNED"

docker run --detach --name rfd4-proteina \
  --runtime=nvidia --gpus '"device=0"' \
  -p 127.0.0.1:8000:8000 \
  -e NGC_API_KEY \
  -v "$RFD4_NIM_CACHE:/opt/nim/.cache" \
  "$RFD4_NIM_IMAGE_PINNED"
```

This adapts the published command to bind the unauthenticated inference service
to loopback and run the pulled image by its recorded digest, detached. Retain
that digest with the run artifacts; pinning records image identity, not a
security or model-quality assessment. Select a GPU allocated to the task. If the container
name already exists, inspect/reuse the intended deployment or choose another
name; do not remove a container to make the command succeed.

The container needs the runtime `NGC_API_KEY` for model downloads, in addition
to registry login. Ensure the cache is writable by the image's runtime user.
If startup reports a permissions error, inspect the image user and adjust this
dedicated cache's ownership or use an appropriate managed volume; do not change
permissions recursively on unrelated directories.

## Readiness and schema discovery

Readiness must return a JSON `true` body. HTTP 200 alone is insufficient.
After launch, run this bounded check in the same host environment:

```bash
python - <<'PY'
import http.client
import json
import time

deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=5)
    try:
        connection.request("GET", "/v1/health/ready")
        response = connection.getresponse()
        if response.status == 200 and json.loads(response.read()) is True:
            print("RFD4-Proteina is ready")
            break
    except (OSError, ValueError, http.client.HTTPException):
        pass
    finally:
        connection.close()
    time.sleep(5)
else:
    raise SystemExit("Not ready after 15 minutes; inspect container logs")
PY
```

Inspect `docker logs rfd4-proteina` for failures. Discover the running service's
capabilities before using optional task or metric settings:

```bash
curl --noproxy '*' --fail --silent --show-error --max-time 30 \
  http://127.0.0.1:8000/v1/tasks
curl --noproxy '*' --fail --silent --show-error --max-time 30 \
  http://127.0.0.1:8000/openapi.json
```

From the installed skill directory, submit to the local route without a hosted
API key:

```bash
python scripts/design.py --mode local --base-url http://localhost:8000 \
  --request assets/requests/16_ppi_binder.json \
  --output-dir /tmp/rfd4-local-binder
```

Local inference uses `/biology/ipd/rfd4-proteina/generative`, without `/v1`.
The hosted URL uses the `/v1` prefix. A host with a different port mapping can
set `RFD4_NIM_URL` or pass `--base-url`.

This procedure follows the public instructions; it is not a record of a local
deployment executed as part of this skill revision. Warmup, compilation, model
downloads, request length, batching, and search can all affect runtime.
