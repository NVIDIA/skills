# Riva TTS Custom Model Deployment

> **Agent:** Announce each phase before presenting it: **Phase N/4 — Phase Title** (e.g., "**Phase 1/4 — Obtain a .riva File**").
>
> **Source of truth.** This skill describes the 4-phase custom-deployment workflow, which is stable. For per-release detail — per-model `riva-build` syntax, supported NeMo architectures, NGC artifact paths, voice model configurations — **fetch or open the canonical doc page or run `riva-build -h` inside the container.** See [Looking up current information](#looking-up-current-information) below.

## Purpose

Deploy a custom or fine-tuned TTS pipeline as a Riva NIM when pre-built NIMs do not meet voice, language, or pronunciation requirements. Covers the full pipeline: obtain a deployable `.riva` checkpoint, build an RMIR, deploy the model repository, and launch the NIM. If the user has their own fine-tuned `.nemo` TTS checkpoint, use the inline `nemo2riva` method inside `riva-build`. For runtime-only customizations — zero-shot voice cloning, custom pronunciation dictionaries, SSML, per-request `custom_configuration` keys — see [`tts-pipelines.md`](tts-pipelines.md), which does not require a rebuild.

## Looking up current information

| Question type | Fetch this page |
|---|---|
| **Custom deployment workflow, `.nemo` / `.riva` artifact paths, `riva-build` / `riva-deploy` syntax, and model-specific examples** | https://docs.nvidia.com/nim/speech/latest/tts/custom-deployment.html |
| Current Magpie TTS artifacts (`deployable` `.riva` and `trainable` `.nemo` versions) | https://catalog.ngc.nvidia.com/orgs/nvidia/riva/models/speechsynthesis_multilingual_magpietts_ipa/- |
| Which base NIM container image to use for a given TTS model family | https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html |
| GPU / VRAM / driver minimums | https://docs.nvidia.com/nim/speech/latest/get-started/prerequisites.html |
| Live, version-accurate parameter list (run inside the container) | `riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> -h` |
| Runtime SSML, custom dictionaries, and `custom_configuration` keys | https://docs.nvidia.com/nim/speech/latest/tts/customization.html |
| Zero-shot voice cloning | https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html |

**Do not infer from this skill's text:** which base container image to use for a specific TTS model family, the exact `nemo2riva` inline-config block for a given architecture, or what the current `riva-build` defaults are. The custom-deployment page, the model's NGC page, and the image-specific `--help` output are authoritative.

## Workflow

4-phase pipeline: obtain a `.riva` file → build an RMIR with `riva-build` → deploy the model repository with `riva-deploy` → launch the custom NIM.

## Prerequisites

- Complete [`setup.md`](setup.md): NVIDIA Container Toolkit, `NGC_API_KEY` exported (driver minimum: see prerequisites page cited above)
- If no NeMo fine-tuning was performed, use a `deployable_vX.Y` `.riva` artifact from the model's NGC `_finetune` package.
- Use `trainable_vX.Y` / `.nemo` only when the user has fine-tuned with NeMo. Fine-tuned `.nemo` checkpoints are passed directly to `riva-build` via the inline `nemo2riva` `source_path` config. Use the model-specific example on the custom-deployment page when one is documented, and confirm the config against `riva-build ... -h` in the matching image.

## Instructions

Follow the 4-phase pipeline below. Run `riva-build` and `riva-deploy` inside the NIM container (enter with `--entrypoint /bin/bash`). All paths like `/riva_build_deploy/` refer to the mounted directory inside the container.

For runtime customizations that don't require a rebuild (zero-shot voice cloning, custom dictionaries, SSML, per-request `custom_configuration` keys): see [`tts-pipelines.md`](tts-pipelines.md).

## Phase 1 — Obtain a `.riva` or `.nemo` File

Two sources:

**Option A — Download a deployable `.riva` artifact from NGC** (default if you have not fine-tuned):

```bash
ngc registry model download-version \
  nim/nvidia/<model-name>_finetune:<version> \
  --dest /path/to/artifacts/
```

Use `deployable_vX.Y` versions from the model's `_finetune` package. These contain the `.riva` file ready for `riva-build` and are the right source when you only need to change Riva pipeline parameters such as voice configuration, synthesis mode, or pronunciation. `trainable_vX.Y` versions contain `.nemo` assets for NeMo fine-tuning, not direct deployment.

**Option B — Use your own fine-tuned NeMo checkpoint (`.nemo`):**

Do this only when the user has a `.nemo` checkpoint from NeMo fine-tuning. Pass the `.nemo` file directly to `riva-build` via the inline `nemo2riva` block in `source_path`. The inline-config syntax is **per model family**. Use the **Riva Build** section and the model-specific examples on the custom-deployment page:

https://docs.nvidia.com/nim/speech/latest/tts/custom-deployment.html

Confirm the documented example against `riva-build ... -h` inside the matching NIM image. Do not copy an inline block from a different model family.

---

## Phase 2 — Build RMIR with `riva-build`

Run `riva-build` inside the NIM container. This creates the RMIR (Riva Model Intermediate Representation) file.

The base NIM container image must match the TTS model family you're deploying. Fetch the TTS support matrix to find the right base image.

```bash
export CONTAINER_ID=<base-NIM-image-matching-your-TTS-model-family>
export NIM_EXPORT_PATH=~/nim_export
export ARTIFACT_DIR=/path/to/artifacts         # directory containing your .riva file

mkdir -p $NIM_EXPORT_PATH
sudo chown 1000:1000 $NIM_EXPORT_PATH
```

See [setup.md → Cache directory ownership](setup.md#cache-directory-ownership) for the `chown 1000:1000` rationale.

```bash
# Launch interactive shell inside the NIM container
docker run --gpus all -it --rm \
  --ulimit nofile=65536:65536 \
  -v $ARTIFACT_DIR:/riva_build_deploy \
  -v $NIM_EXPORT_PATH:/model_tar \
  --entrypoint="/bin/bash" \
  --name riva-tts-build-deploy \
  nvcr.io/nim/nvidia/$CONTAINER_ID:latest
```

> **`--ulimit nofile=65536:65536`** raises the file-descriptor cap inside the build container. Without it, certain large-model edge cases can cascade into `OSError: Too many open files` during cleanup.

Inside the container, run `riva-build`. The `--config-path` and `--config-name` values are per TTS model family — fetch or open the custom-deployment page and confirm the values with the image-specific `--help` output:

**Starting from a `.riva` artifact:**

```bash
riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> \
  output_path=/riva_build_deploy/custom_model.rmir \
  'source_path=[/riva_build_deploy/model.riva]'

# Force overwrite if .rmir already exists
riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> \
  force=true \
  output_path=/riva_build_deploy/custom_model.rmir \
  'source_path=[/riva_build_deploy/model.riva]'
```

> **Note:** `riva-build` does NOT accept a `-f` CLI flag; pass `force=true` as a Hydra-style config parameter to overwrite an existing RMIR. `riva-deploy` accepts `-f` (Phase 3).

**Starting from a `.nemo` checkpoint (inline `nemo2riva` config):**

```bash
# Inline nemo2riva block — exact form is per model family; use the matching custom-deployment example
riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> \
  output_path=/riva_build_deploy/custom_model.rmir \
  'source_path=[{path: /riva_build_deploy/model.nemo, nemo2riva: {<model-specific-conversion-options>}}]'
```

The inline `nemo2riva` block is **per model family** — use the matching example on the custom-deployment page and verify available parameters with `riva-build ... -h` in the same image.

For the full parameter set and current per-config options, run `riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> -h` inside the container.

For synthesis pipeline configuration options (audio encoding, sample rate, SSML): see [`tts-pipelines.md`](tts-pipelines.md).

---

## Phase 3 — Deploy Model Repository with `riva-deploy`

Still inside the container (or re-enter it), run `riva-deploy` to build the Triton model repository. Use `-f` so repeated builds replace stale generated files:

```bash
riva-deploy -f /riva_build_deploy/custom_model.rmir /data/models
```

**Important:** Always deploy to `/data/models` inside the container. Deploying elsewhere requires manual path fixes in Triton config files.

After deploy completes, create the tar archive:

```bash
cd /data/models
tar -czf /model_tar/custom_model.tar.gz *
```

Exit and remove the container:

```bash
exit
docker stop riva-tts-build-deploy 2>/dev/null; docker rm riva-tts-build-deploy 2>/dev/null
```

Your `custom_model.tar.gz` is now in `$NIM_EXPORT_PATH` on the host.

---

## Phase 4 — Launch the Custom TTS NIM

```bash
docker run -it --rm --name=$CONTAINER_ID \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NGC_API_KEY \
  -e NIM_TAGS_SELECTOR \
  -e NIM_DISABLE_MODEL_DOWNLOAD=true \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -p 9000:9000 \
  -p 50051:50051 \
  -v $NIM_EXPORT_PATH:/opt/nim/export \
  -e NIM_EXPORT_PATH=/opt/nim/export \
  nvcr.io/nim/nvidia/$CONTAINER_ID:latest
```

> **Security note:** Environment variables passed via `-e` to Docker are visible in `docker inspect` output and process listings. For production, use Docker secrets or a secrets manager.

`NIM_DISABLE_MODEL_DOWNLOAD=true` prevents the container from downloading pre-trained models from NGC and uses the custom repository from `NIM_EXPORT_PATH` instead.

## Verify Readiness

```bash
curl -X GET http://localhost:9000/v1/health/ready
# Expected: {"status":"ready"}
```

Confirm TTS models are actually loaded (inline probe, needs only `pip install nvidia-riva-client`):

```bash
python3 - <<'PY'
import sys, riva.client
from riva.client.proto.riva_tts_pb2 import RivaSynthesisConfigRequest
auth = riva.client.Auth(uri="0.0.0.0:50051")
try:
    cfg = riva.client.SpeechSynthesisService(auth).stub.GetRivaSynthesisConfig(
        RivaSynthesisConfigRequest(), metadata=auth.get_auth_metadata())
except Exception as e:
    print(f"UNHEALTHY: {e}"); sys.exit(2)
if not cfg.model_config:
    print("UNHEALTHY: server responded but exposes no TTS models"); sys.exit(2)
print(f"OK: {len(cfg.model_config)} model(s)")
for m in cfg.model_config:
    voices = m.parameters.get("voice_name", "")
    langs  = m.parameters.get("language_code", "")
    print(f"  - {m.model_name}  [{langs}]  voices={voices}")
PY
```

## Run Inference on the Custom Model

```bash
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Hello from my custom TTS voice." \
  --voice <VOICE_NAME> \
  --output output.wav
```

List voices exposed by the custom NIM:

```bash
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --list-voices
```

For runtime feature support on your custom model, fetch the customization page for SSML and custom dictionaries, and the voice-cloning page for zero-shot synthesis. Feature support depends on the underlying model architecture.

---

## Examples

**Build RMIR from a `.riva` artifact (inside NIM container):**

```bash
riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> \
  output_path=/riva_build_deploy/model.rmir \
  'source_path=[/riva_build_deploy/model.riva]'
```

**Build RMIR from a `.nemo` checkpoint (use the matching custom-deployment example):**

```bash
riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> \
  output_path=/riva_build_deploy/model.rmir \
  'source_path=[{path: /riva_build_deploy/model.nemo, nemo2riva: {<model-specific-conversion-options>}}]'
```

**Launch the custom TTS NIM:**

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' \
  -e NGC_API_KEY -e NIM_DISABLE_MODEL_DOWNLOAD=true \
  -v $NIM_EXPORT_PATH:/opt/nim/export \
  -e NIM_EXPORT_PATH=/opt/nim/export \
  nvcr.io/nim/nvidia/$CONTAINER_ID:latest
```

**Lookup flow — agent question "which base container should I use for a fine-tuned Magpie TTS?":**

1. Fetch or open the TTS support matrix
2. Locate the Magpie TTS family entry, copy its `CONTAINER_ID`
3. Use that as the base image in Phase 2

Do not pick a base image from this skill's text alone — the catalog rotates per release.

## Troubleshooting

- **Match container to model architecture** — use the NIM base image that matches your TTS model family. Fetch the support matrix to find the right one.
- **Deploy to `/data/models` only** — other paths break Triton config references without manual edits.
- **`NIM_DISABLE_MODEL_DOWNLOAD=true` is required** — without it, the container ignores the custom model and downloads the default pre-trained model.
- **`force=true` for `riva-build`, `-f` for `riva-deploy`** — `riva-build` rejects `-f` as unrecognized; pass `force=true` as a config parameter. `riva-deploy` accepts `-f`.
- **Phase 3 runs on target GPU** — `riva-deploy` optimizes TensorRT engines for the deployment GPU; run it on the same GPU class you'll use in production.
- **`.nemo` architecture support** — not all NeMo TTS architectures are supported by every NIM image. Check the custom-deployment page for a matching model example and verify the exact inline-config keys with the image-specific `riva-build ... -h` output.
- **Voice names from custom NIM** — the voices exposed by a custom NIM depend on the trained model checkpoint. Always run `--list-voices` to discover the actual voice names rather than copying from documentation.

## Limitations

- x86_64 architecture only — `riva-build` runs inside the NIM container
- NVIDIA AI Enterprise license required for self-hosting
- `.nemo` → RMIR conversion happens inside `riva-build` via the inline `nemo2riva` block; supported architectures and inline-config keys are version-locked per release — verify against the current custom-deployment page and the matching image's `riva-build ... -h` output before converting
- Runtime-only customizations (zero-shot voice cloning, custom pronunciation dictionaries, SSML, `custom_configuration` keys) do not require a rebuild — see [`tts-pipelines.md`](tts-pipelines.md)
