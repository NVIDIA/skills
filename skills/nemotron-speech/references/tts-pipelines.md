# Riva TTS Synthesis Pipeline Configuration

> **Note:** Build-time options run `riva-build` **inside the NIM container** — enter with `--entrypoint /bin/bash` (see [`tts-custom.md`](tts-custom.md)). Most TTS customizations are **runtime-only** and do not require a rebuild.
>
> **Source of truth.** This skill describes synthesis pipeline concepts and parameter shapes, which are stable. For per-release detail — full parameter list, per-model `custom_configuration` keys, supported SSML tags, default sample rates, pronunciation dictionary format — **fetch or open the canonical doc page and answer from that, not from this skill's text.** See [Looking up current information](#looking-up-current-information) below.

## Purpose

Configure TTS synthesis pipeline options: audio encoding, sample rate, offline vs streaming mode, SSML, per-request `custom_configuration` keys (e.g., emotional styles), zero-shot voice cloning, and custom pronunciation dictionaries. Most options are runtime-only (no rebuild needed). Build-time options (embedding a custom pronunciation dictionary at deploy time) require re-running `riva-build` and `riva-deploy`.

## Looking up current information

| Question type | Fetch this page |
|---|---|
| **Request-time customization** — SSML, custom dictionaries, and `custom_configuration` keys | https://docs.nvidia.com/nim/speech/latest/tts/customization.html |
| **Voices and emotional styles** | https://docs.nvidia.com/nim/speech/latest/tts/voices.html |
| **Zero-shot voice cloning** — supported models, prompt requirements, quality options | https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html |
| **Custom deployment and build-time synthesis options** | https://docs.nvidia.com/nim/speech/latest/tts/custom-deployment.html |
| **gRPC proto contract** — `SynthesizeSpeechRequest`, `ZeroShotData`, `custom_configuration` map | https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/protos.html |
| **HTTP REST API** — offline and streaming synthesis request fields and response formats | https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/http-tts.html |
| **Realtime WebSocket API** — OpenAI-realtime-compatible TTS sessions, event schemas | https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/realtime-tts.html |
| Current model catalog, voice lists, supported languages, VRAM minimums | https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html |
| Latency / throughput benchmarks per model and GPU | https://docs.nvidia.com/nim/speech/latest/reference/performances/tts/performance.html |
| **Get all build-time parameters from inside the container** | `riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> -h` |

**Do not infer from this skill's text:** which `custom_configuration` keys a specific model supports, which SSML tags are available, what the current default sample rate is, or whether zero-shot voice cloning is supported for a given model. Use the topic-specific pages above and the live model configuration.

## Prerequisites

- For **runtime-only** options (streaming mode, SSML, `custom_configuration`, zero-shot, custom dictionaries per-request): a running TTS NIM (see [`tts.md`](tts.md)) and `pip install nvidia-riva-client`
- For **build-time** options (embedding a pronunciation dictionary into the model repo): complete [`tts-custom.md`](tts-custom.md) first — NIM container available, `NGC_API_KEY` exported

## Instructions

**Determine first whether a rebuild is needed.** Most TTS pipeline tuning is runtime-only and takes effect per-request. Build-time options (embedding a pronunciation dictionary into the model repo) require Phase 2–3 from [`tts-custom.md`](tts-custom.md).

For runtime tuning: set the relevant parameters in your `SynthesizeSpeechRequest` or `talk.py` flags. Fetch the customization page for the authoritative list of per-model `custom_configuration` keys and supported SSML tags.

---

## Offline vs Streaming Synthesis

Choose based on latency requirements and response size:

| Mode | RPC | When to use |
|---|---|---|
| Offline (`Synthesize`) | Unary | Short to medium text; full audio needed before playback; gRPC response ≤ 4 MB |
| Streaming (`SynthesizeOnline`) | Server-streaming | Long text; low time-to-first-audio; bypasses 4 MB gRPC limit |

**The gRPC 4 MB limit is the hard boundary:** a single `Synthesize` (offline) response cannot exceed 4 MB. At 16-bit PCM 22050 Hz mono, 4 MB ≈ 90 seconds of audio. For longer synthesis, use `SynthesizeOnline` (`--stream` in `talk.py`) or HTTP streaming.

**Time-to-first-audio:** streaming begins returning audio chunks as soon as the first TTS chunk is ready, before the full utterance is synthesized. Use streaming when playback latency matters more than simplicity.

**gRPC:**

```bash
# Offline
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Your text here." \
  --voice <VOICE_NAME> \
  --output output.wav

# Streaming
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Your text here." \
  --voice <VOICE_NAME> \
  --stream \
  --output output.wav
```

---

## Audio Encoding

Two supported encodings:

| Encoding | Format | Best for |
|---|---|---|
| `LINEAR_PCM` | Uncompressed 16-bit PCM | Maximum compatibility, lossless, larger files |
| `OGG_OPUS` | Opus-compressed audio in Ogg container | Lower bandwidth, streaming, browser playback |

`LINEAR_PCM` is the default. Use `OGG_OPUS` when bandwidth matters or when the downstream player supports it. Verify per-model encoding support on the customization page — not all models expose Opus.

**gRPC (inline quick path):**

```python
import os, wave, riva.client

auth = riva.client.Auth(uri="0.0.0.0:50051")
tts = riva.client.SpeechSynthesisService(auth)

resp = tts.synthesize(
    text="Your text here.",
    voice_name="<voice-name-from-list-voices>",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.OGGOPUS,    # or .LINEAR_PCM
    sample_rate_hz=22050,
)
with open("output.ogg", "wb") as f:
    f.write(resp.audio)
```

**HTTP:**

```bash
# Offline — returns WAV (LINEAR_PCM)
curl -sS http://localhost:9000/v1/audio/synthesize --fail-with-body \
  -F language=en-US \
  -F text="Your text here." \
  -F voice=<VOICE_NAME> \
  --output output.wav

# Streaming — returns raw LPCM (NOT WAV); wrap with sox
curl -sS http://localhost:9000/v1/audio/synthesize_online --fail-with-body \
  -F language=en-US \
  -F text="Your text here." \
  -F voice=<VOICE_NAME> \
  -F sample_rate_hz=22050 \
  --output output.raw
sox -b 16 -e signed -c 1 -r 22050 output.raw output.wav
```

> **HTTP streaming returns raw LPCM, not WAV.** The response has no WAV header. Pass the same sample rate you requested to `sox` (`-r <rate>`). A mismatched rate produces pitched-wrong or sped-up output.

---

## Sample Rate

The model synthesizes at its native sample rate. Requesting a different rate causes server-side resampling, which may reduce quality.

- Default is model-specific — fetch from the support matrix or inspect the `GetRivaSynthesisConfig` probe output (see [`tts.md`](tts.md) Step 2)
- Common values: 22050 Hz (legacy models), 44100 Hz (higher-quality models)
- Pass `--sample-rate-hz` to `talk.py` or `sample_rate_hz` in the gRPC request to request a specific rate

Do not hardcode a sample rate without checking the model's native rate on the support matrix — requesting lower than native discards quality; higher than native wastes bandwidth with no benefit.

---

## SSML (Speech Synthesis Markup Language)

SSML allows per-word or per-phrase control over prosody, emphasis, breaks, and phoneme pronunciation. Support is **per-model** — not all Riva TTS models accept SSML input.

**Before using SSML:**
1. Fetch the customization page to confirm SSML support for your model
2. Verify which SSML tags the model accepts (supported tags vary by model version)

**General usage pattern (wrap text in `<speak>` root):**

```python
import wave, riva.client

auth = riva.client.Auth(uri="0.0.0.0:50051")
tts = riva.client.SpeechSynthesisService(auth)

ssml_text = """<speak>
  Welcome to <emphasis level="strong">NVIDIA</emphasis> TTS.
  <break time="500ms"/>
  Synthesis begins now.
</speak>"""

resp = tts.synthesize(
    text=ssml_text,
    voice_name="<voice-name-from-list-voices>",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=44100,
)
with wave.open("output.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
    w.writeframes(resp.audio)
```

Do not assume specific SSML tag support from this skill's text — fetch the customization page and answer from the fetched content.

---

## Custom Configuration Keys (`custom_configuration`)

`custom_configuration` is a per-request string-to-string map for model-specific synthesis parameters. Common uses: emotional styles, speaking rate adjustments, exaggeration factors.

**These keys are per-model and per-release** — always fetch the customization page to find which keys a given model supports and what their valid ranges are.

**gRPC (inline quick path):**

```python
import wave, riva.client
from riva.client.proto import riva_tts_pb2

auth = riva.client.Auth(uri="0.0.0.0:50051")
tts = riva.client.SpeechSynthesisService(auth)

req = riva_tts_pb2.SynthesizeSpeechRequest(
    text="Hello, this is expressive synthesis.",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=44100,
    voice_name="<voice-name-from-list-voices>",
    custom_configuration={
        "exaggeration_factor": "1.5",   # example key — verify on the customization page
    },
)
resp = tts.stub.Synthesize(req, metadata=auth.get_auth_metadata())
with wave.open("output.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
    w.writeframes(resp.audio)
```

**`talk.py` (comma-separated `key:value` pairs):**

```bash
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Hello, this is expressive synthesis." \
  --voice <VOICE_NAME> \
  --custom-configuration "exaggeration_factor:1.5" \
  --output output.wav
```

Do not infer key names or valid ranges from this skill's text — fetch the customization page.

---

## Zero-Shot Voice Cloning

Zero-shot synthesis lets you synthesize in a new voice from a short audio prompt (reference audio + transcript), without fine-tuning. Support is **per-model** — not all deployed NIMs expose this capability.

> **Agent:** Run the pre-flight check below **before** providing zero-shot synthesis code or proceeding with any zero-shot workflow. Do not skip this step even if the user says "I already deployed the right model." A model that does not support zero-shot returns no error during deployment — the failure only surfaces at inference time as an `UNIMPLEMENTED` gRPC error or silent fallback. The pre-flight check catches this before the user wastes time preparing audio prompts.

### Zero-Shot Pre-flight Check

Run this probe to confirm that the deployed NIM actually supports zero-shot synthesis:

```python
#!/usr/bin/env python3
import sys, riva.client
from riva.client.proto.riva_tts_pb2 import (
    RivaSynthesisConfigRequest, SynthesizeSpeechRequest, ZeroShotData,
)
from riva.client.proto.riva_audio_pb2 import AudioEncoding

SERVER = "0.0.0.0:50051"   # adjust if cloud or non-default port

auth = riva.client.Auth(uri=SERVER)
tts  = riva.client.SpeechSynthesisService(auth)

# Step 1 — confirm NIM is healthy and a model is loaded
try:
    cfg = tts.stub.GetRivaSynthesisConfig(
        RivaSynthesisConfigRequest(), metadata=auth.get_auth_metadata()
    )
except Exception as e:
    print(f"BLOCKED: NIM not reachable — {e}"); sys.exit(1)

if not cfg.model_config:
    print("BLOCKED: NIM responded but no TTS models are loaded."); sys.exit(1)

for m in cfg.model_config:
    print(f"Running model: {m.model_name}")

# Step 2 — probe zero-shot capability with a minimal dummy request
# The probe intentionally uses invalid audio data.
# UNIMPLEMENTED / "not supported" → model does not support zero-shot → BLOCKED
# INVALID_ARGUMENT / any other error → model accepts ZeroShotData but rejected bad input → OK
try:
    probe = SynthesizeSpeechRequest(
        text="probe",
        language_code="en-US",
        encoding=AudioEncoding.LINEAR_PCM,
        sample_rate_hz=22050,
        zero_shot_data=ZeroShotData(
            audio_prompt=b"\x00" * 100,   # intentionally invalid; triggers INVALID_ARGUMENT if supported
            sample_rate_hz=16000,
            encoding=AudioEncoding.LINEAR_PCM,
            quality=1,
            transcript="probe",
        ),
    )
    tts.stub.Synthesize(probe, metadata=auth.get_auth_metadata())
    # Unexpectedly succeeded — zero-shot is supported
    print("OK: zero-shot is supported by the running model.")
except Exception as e:
    err = str(e).lower()
    if "unimplemented" in err or "not supported" in err or "zero_shot" in err or "zero-shot" in err:
        print(f"BLOCKED: the deployed model does not support zero-shot synthesis.\n  {e}")
        print("\nNext steps:")
        print("  1. Check the TTS support matrix for models that advertise zero-shot capability.")
        print("  2. Redeploy using a zero-shot-capable model (e.g., a Magpie variant — verify on support matrix).")
        print("  3. Re-run this check after redeployment before proceeding.")
        sys.exit(1)
    # Any other error (e.g., INVALID_ARGUMENT) means the server accepted ZeroShotData
    # but rejected our dummy input — zero-shot IS available on this model
    print(f"OK: zero-shot is supported (probe rejected dummy input as expected: {type(e).__name__}).")
```

**If the check exits with code 1:** stop here. Do not proceed with zero-shot synthesis. Redeploy with a zero-shot-capable model (fetch the TTS support matrix for the current list) and re-run the check.

**If the check prints OK:** proceed with the synthesis steps below.

---

**Requirements (verify on the voice-cloning page):**
- A 5–30 second clean audio clip of the target voice (WAV, mono, 16-bit PCM recommended)
- The verbatim transcript of that audio clip
- A model that passed the pre-flight check above

**`quality` parameter:** integer 1–40 (default 20). Higher values improve voice similarity at the cost of synthesis latency. Fetch the voice-cloning page for per-model guidance on valid range and recommended starting point.

**Inline quick path (gRPC):**

```python
import wave, riva.client
from riva.client.proto import riva_tts_pb2

# Read the reference audio prompt
with open("reference_audio.wav", "rb") as f:
    audio_prompt = f.read()

# Get sample rate from WAV header
with wave.open("reference_audio.wav") as w:
    prompt_sr = w.getframerate()

auth = riva.client.Auth(uri="0.0.0.0:50051")
tts = riva.client.SpeechSynthesisService(auth)

req = riva_tts_pb2.SynthesizeSpeechRequest(
    text="Text to synthesize in the cloned voice.",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=44100,
    zero_shot_data=riva_tts_pb2.ZeroShotData(
        audio_prompt=audio_prompt,
        sample_rate_hz=prompt_sr,
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        quality=20,
        transcript="The exact verbatim transcript of the reference audio.",
    ),
)
resp = tts.stub.Synthesize(req, metadata=auth.get_auth_metadata())
with wave.open("output.wav", "wb") as out:
    out.setnchannels(1); out.setsampwidth(2); out.setframerate(44100)
    out.writeframes(resp.audio)
```

> **Audio prompt quality matters.** A clean, quiet recording with no background noise produces significantly better voice similarity. Stereo files must be downmixed to mono before use: `ffmpeg -i input.wav -ac 1 -ar 16000 prompt.wav`. Resampling artifacts or clipping in the prompt noticeably degrade similarity.

---

## Custom Pronunciation Dictionaries

Override how specific words or phrases are pronounced using grapheme-to-phoneme (G2P) mappings. Support and format are **per-model** — verify on the customization page.

**Runtime per-request (no rebuild required):**

Pass the dictionary as a string in the `custom_dictionary` field. The phoneme format (CMU ARPA, IPA, etc.) is model-specific — fetch from the customization page.

```python
import wave, riva.client
from riva.client.proto import riva_tts_pb2

auth = riva.client.Auth(uri="0.0.0.0:50051")
tts = riva.client.SpeechSynthesisService(auth)

# Dictionary format is model-specific — fetch from the customization page
custom_dict = """
NVIDIA  en-US  N IH V IH D IY AH
NIM  en-US  N IH M
"""

req = riva_tts_pb2.SynthesizeSpeechRequest(
    text="NVIDIA NIM enables fast inference.",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=44100,
    voice_name="<voice-name-from-list-voices>",
    custom_dictionary=custom_dict,
)
resp = tts.stub.Synthesize(req, metadata=auth.get_auth_metadata())
with wave.open("output.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
    w.writeframes(resp.audio)
```

**Build-time (embedded into model repository):**

Embedding a pronunciation dictionary at build time avoids passing it per-request, but requires re-running `riva-build` and `riva-deploy` (see [`tts-custom.md`](tts-custom.md)). The build-time parameter name and dictionary path format are per-model — fetch the custom-deployment page and confirm with the image-specific `riva-build ... -h` output.

---

## Get All Available Build-Time Parameters

To see all configurable parameters for the version you're running, enter the NIM container and run:

```bash
riva-build --config-path=pkg://servicemaker.configs.tts --config-name=<model-config-name> -h
```

The `--config-name` value is per TTS model family — fetch or open the custom-deployment page and confirm it with the matching image's `--help` output. Defaults shown in this skill are illustrative and may differ per release.

---

## Examples

**Streaming synthesis to WAV:**

```bash
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Streaming synthesis reduces time to first audio." \
  --voice <VOICE_NAME> \
  --stream \
  --sample-rate-hz 44100 \
  --output output.wav
```

**HTTP streaming output converted to WAV:**

```bash
curl -sS http://localhost:9000/v1/audio/synthesize_online --fail-with-body \
  -F language=en-US \
  -F text="Convert raw LPCM to WAV with sox." \
  -F voice=<VOICE_NAME> \
  -F sample_rate_hz=22050 \
  --output output.raw

sox -b 16 -e signed -c 1 -r 22050 output.raw output.wav
```

**Synthesis with `custom_configuration` (verify key names on customization page):**

```bash
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Expressive synthesis with custom parameters." \
  --voice <VOICE_NAME> \
  --custom-configuration "exaggeration_factor:1.5" \
  --output output.wav
```

**Lookup flow — agent question "does Magpie TTS support zero-shot voice cloning?":**

1. Fetch or open the voice-cloning page
2. Confirm that the deployed Magpie model is listed as zero-shot capable
3. Answer from the fetched content

Do not answer feature-support questions from this skill's text alone.

## Troubleshooting

- **gRPC 4 MB limit hit** — switch to `--stream` (`SynthesizeOnline`) or HTTP streaming for long text. At 22050 Hz 16-bit PCM mono, 4 MB ≈ 90 seconds.
- **HTTP streaming output not playable** — the `/v1/audio/synthesize_online` endpoint returns raw LPCM with no WAV header. Always wrap with `sox -b 16 -e signed -c 1 -r <rate> output.raw output.wav` using the exact rate you requested in the API call.
- **`custom_configuration` key rejected or silently ignored** — key names and valid ranges are per-model and per-release. Fetch the customization page to confirm supported keys for your model.
- **SSML tags silently ignored** — not all models support SSML, and models that do support a subset of tags. Fetch the customization page to confirm SSML support before debugging.
- **Zero-shot pre-flight check exits with BLOCKED** — the deployed model does not support zero-shot synthesis. Fetch the TTS support matrix to identify a zero-shot-capable model, redeploy, and re-run the check before attempting synthesis.
- **Zero-shot pre-flight check returns UNIMPLEMENTED** — same as above; `UNIMPLEMENTED` is the gRPC status code returned when the server has no handler for `ZeroShotData`. Redeploy with a supported model.
- **Zero-shot voice similarity poor** — check audio prompt quality: background noise, resampling artifacts, clipping, or stereo-to-mono issues all degrade similarity. Use a clean mono 16-bit PCM prompt.
- **Sample rate mismatch in sox conversion** — always pass to `sox` the exact `sample_rate_hz` you requested in the API call; a mismatch produces pitched-wrong or sped-up output.
- **Voice name not recognized after custom NIM launch** — voice names exposed by a custom NIM depend on the trained checkpoint. Run `--list-voices` against the running NIM to discover the actual names.

## Limitations

- `custom_configuration` keys, SSML tags, and zero-shot support are per-model and change per release — verify request-time options on the customization page and zero-shot behavior on the voice-cloning page
- Zero-shot voice cloning requires a model that supports it; not all TTS NIMs expose this capability
- Build-time pronunciation dictionary embedding requires a full `riva-build` + `riva-deploy` cycle (see [`tts-custom.md`](tts-custom.md))
- HTTP streaming (`/v1/audio/synthesize_online`) returns raw LPCM, not WAV — client-side wrapping with `sox` or equivalent is always required
- gRPC unary responses (`Synthesize`) are capped at 4 MB — approximately 90 seconds at 22050 Hz 16-bit PCM mono
- x86_64 architecture only for self-hosted deployment
