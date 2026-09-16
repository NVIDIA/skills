# Riva TTS Pronunciation Discovery and Dictionary Application

> **Agent:** When walking the user through the discovery workflow, announce each step: **Step N/3 — Step Title** (e.g., "**Step 1/3 — Generate IPA Candidates**").
>
> **Source of truth.** This skill describes a stable workflow for discovering and applying pronunciations. For per-release detail — which models support `custom_dictionary`, the exact IPA phone set for a given model, SSML `<phoneme>` tag support, `--phone_dictionary_file` riva-build parameter name — **fetch or open the canonical doc page and answer from that, not from this skill's text.** See [Looking up current information](#looking-up-current-information) below.

---

## Purpose

Discover and apply the correct pronunciation for a specific word or phrase in Riva TTS. Covers the full workflow: generating IPA candidates using the agent's linguistic knowledge, synthesizing audio variants so the user can compare and choose, and then applying the chosen pronunciation in all three delivery formats.

**Use this reference when:** the user does not know the phoneme sequence for their word and needs to find it through listening and iteration.

**Use [`tts-pipelines.md`](tts-pipelines.md) instead when:** the user already knows the phoneme string and just needs help with the wire format or application method.

## Looking up current information

| Question type | Fetch this page |
|---|---|
| IPA phone set for current Riva TTS models | https://docs.nvidia.com/nim/speech/latest/tts/phoneme-support.html |
| `<phoneme>` tag support and runtime custom-dictionary format | https://docs.nvidia.com/nim/speech/latest/tts/customization.html |
| Build-time dictionary configuration and `riva-build` workflow | https://docs.nvidia.com/nim/speech/latest/tts/custom-deployment.html |
| gRPC proto contract — `SynthesizeSpeechRequest.custom_dictionary` field | https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/protos.html |
| `--custom-dictionary` CLI flag format for `talk.py` | https://docs.nvidia.com/nim/speech/latest/tts/customization.html |

**Do not infer from this skill's text:** which IPA phone symbols a specific model supports, whether SSML `<phoneme>` is supported for the deployed model, or whether `custom_dictionary` is accepted. Use the phoneme-support page for the IPA inventory and the customization page for request-time feature support.

---

## Prerequisites

- A running TTS NIM (see [`tts.md`](tts.md))
- `pip install nvidia-riva-client`
- A voice name discovered via `--list-voices` (see [`tts.md`](tts.md) Step 3)
- The word or phrase to fix, and a short carrier sentence that contains it

---

## IPA Phoneme Notation

Modern Riva / Nemotron Speech TTS models use **IPA (International Phonetic Alphabet)**, not ARPABET/CMU, unless the model was built with `--phone_set arpabet` at `riva-build` time. The agent generates IPA candidates using its own linguistic knowledge — no external lookup tool is required for common English words.

**Worked example — "NVIDIA":**

| Label | IPA | Character difference |
|---|---|---|
| A | `ɛnˈvɪdiə` | Initial vowel `ɛ` (as in "bed"), primary stress on second syllable |
| B | `ɪnˈvɪdiə` | Initial vowel `ɪ` (as in "it"), same stress |
| C | `ɛnˈvidɪə` | Variant vowel in third syllable |

Two key conventions:
- `ˈ` marks primary stress on the following syllable; `ˌ` marks secondary stress. Omitting stress marks lets the model infer, but including them produces more consistent output.
- `ʌ` and `ɚ` are not natively in modern models' phone sets — they are auto-converted to `ə` and `ɝ` respectively. Use `ə` and `ɝ` directly.

For the authoritative phone inventory, fetch the phoneme-support page from the routing table above.

---

## Step 1 — Generate IPA Candidates

The agent produces 2–3 distinct IPA strings for the target word, labeled A, B, C. Each variant should differ in a meaningful and audible way — typically stress placement, initial or final vowel choice, or syllable reduction. Present them as a short table so the user can read the differences before listening.

> **Agent instruction:** Generate variants using your linguistic knowledge. Do not use phone symbols that are absent from the model's phone set. Fetch the phoneme-support page if uncertain whether a specific symbol is supported.

---

## Step 2 — Synthesize Audio Variants

Run this script to produce one WAV file per candidate. The user listens to each and picks the one that sounds correct.

**Prerequisites:** set `SERVER` and `VOICE` environment variables (or edit inline), then run:

```bash
export SERVER=0.0.0.0:50051              # or grpc.nvcf.nvidia.com:443 for cloud
export VOICE="<voice-name-from-list-voices>"
```

```python
#!/usr/bin/env python3
import os, wave, riva.client
from riva.client.proto import riva_tts_pb2

SERVER   = os.getenv("SERVER", "0.0.0.0:50051")
VOICE    = os.getenv("VOICE", "")
is_cloud = "nvcf" in SERVER

WORD     = "NVIDIA"                              # word to fix
SENTENCE = f"Welcome to {WORD} neural inference."  # short carrier sentence

# Agent fills in these IPA candidates for the word above
CANDIDATES = {
    "A": "ɛnˈvɪdiə",
    "B": "ɪnˈvɪdiə",
    "C": "ɛnˈvidɪə",
}

md = None
if is_cloud:
    fid = os.environ["FUNCTION_ID"]
    md  = [["function-id", fid],
           ["authorization", f"Bearer {os.environ['NVIDIA_API_KEY']}"]]

auth = riva.client.Auth(uri=SERVER, use_ssl=is_cloud, metadata_args=md)
tts  = riva.client.SpeechSynthesisService(auth)
SR   = 44100

for label, ipa in CANDIDATES.items():
    # custom_dictionary wire format: "WORD  IPA"  ← double space between word and phonemes
    # Multiple entries: "WORD1  IPA1,WORD2  IPA2"  ← comma-separated, no spaces around comma
    req = riva_tts_pb2.SynthesizeSpeechRequest(
        text=SENTENCE,
        language_code="en-US",
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=SR,
        voice_name=VOICE,
        custom_dictionary=f"{WORD}  {ipa}",
    )
    resp = tts.stub.Synthesize(req, metadata=auth.get_auth_metadata())
    fname = f"variant_{label.lower()}.wav"
    with wave.open(fname, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(resp.audio)
    print(f"Wrote {fname}  (/{ipa}/)")

print("\nListen to variant_a.wav, variant_b.wav, variant_c.wav and tell me which sounds correct.")
```

> **Double-space rule:** The `custom_dictionary` field uses **two spaces** between the word and its phoneme string: `"WORD  IPA"`. A single space silently misparses the entry, and the model falls back to its default G2P without error.

---

## Step 3 — Apply the Chosen Pronunciation

After the user identifies their preferred variant, output all three delivery formats. The user can choose whichever fits their workflow.

### Format 1 — Per-request `custom_dictionary` (runtime, no rebuild)

Pass the dictionary string directly in each synthesis request. No `riva-build` or `riva-deploy` needed.

**Wire format:**

```
Single word:   "NVIDIA  ɛnˈvɪdiə"
Multiple words: "NVIDIA  ɛnˈvɪdiə,NIM  nɪm"
```

Rules: double space between word and phonemes; comma-separated entries with no spaces around the comma; no trailing comma.

**gRPC (inline quick path):**

```python
import os, wave, riva.client
from riva.client.proto import riva_tts_pb2

auth = riva.client.Auth(uri="0.0.0.0:50051")
tts  = riva.client.SpeechSynthesisService(auth)

req = riva_tts_pb2.SynthesizeSpeechRequest(
    text="Welcome to NVIDIA.",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=44100,
    voice_name="<voice-name-from-list-voices>",
    custom_dictionary="NVIDIA  ɛnˈvɪdiə",
)
resp = tts.stub.Synthesize(req, metadata=auth.get_auth_metadata())
with wave.open("output.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
    w.writeframes(resp.audio)
```

**talk.py equivalent** (reads from a file — see Format 3):

```bash
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Welcome to NVIDIA." \
  --voice <VOICE_NAME> \
  --custom-dictionary my_pronunciations.txt \
  --output output.wav
```

---

### Format 2 — SSML `<phoneme>` inline tag

Override pronunciation for a single occurrence of a word directly in the synthesis text, without a dictionary file.

```xml
<speak>
  Welcome to <phoneme alphabet="ipa" ph="ɛnˈvɪdiə">NVIDIA</phoneme> neural inference.
</speak>
```

```python
import wave, riva.client

auth = riva.client.Auth(uri="0.0.0.0:50051")
tts  = riva.client.SpeechSynthesisService(auth)

ssml = '<speak>Welcome to <phoneme alphabet="ipa" ph="ɛnˈvɪdiə">NVIDIA</phoneme>.</speak>'
resp = tts.synthesize(
    text=ssml,
    voice_name="<voice-name-from-list-voices>",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=44100,
)
with wave.open("output.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
    w.writeframes(resp.audio)
```

> **SSML `<phoneme>` support is per-model.** Fetch the customization page to confirm your deployed model accepts SSML and specifically `<phoneme alphabet="ipa">`. A model that does not support SSML will silently ignore the tag and fall back to its default G2P pronunciation — no error is raised.

---

### Format 3 — Session dictionary file (reuse across requests)

Build a persistent file of pronunciation overrides and load it for every request, or share it across team members.

**File format** — one entry per line, double space between word and IPA, UTF-8 encoding:

```
# Pronunciation dictionary — one WORD  IPA entry per line
# Lines starting with # and blank lines are ignored
NVIDIA  ɛnˈvɪdiə
NIM  nɪm
Riva  ˈɹiːvə
```

**Load helper (Python):**

```python
def load_dict_file(path: str) -> str:
    """Read a session dictionary file and return the custom_dictionary wire string."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return ",".join(entries)

# Usage:
custom_dictionary = load_dict_file("my_pronunciations.txt")
req = riva_tts_pb2.SynthesizeSpeechRequest(
    text="...",
    custom_dictionary=custom_dictionary,
    ...
)
```

**talk.py usage** — pass the file path directly (talk.py reads and formats it internally):

```bash
python3 python-clients/scripts/tts/talk.py \
  --server 0.0.0.0:50051 \
  --text "Welcome to NVIDIA NIM." \
  --voice <VOICE_NAME> \
  --custom-dictionary my_pronunciations.txt \
  --output output.wav
```

---

## Build-time Option (Permanent Embedding)

If every request to a self-hosted NIM needs the same pronunciation overrides, bake the dictionary into the RMIR at build time using `--phone_dictionary_file <path>` in `riva-build`. This removes the need to pass `custom_dictionary` per-request but requires a full `riva-build` + `riva-deploy` cycle (see [`tts-custom.md`](tts-custom.md)).

The phone set selected at `riva-build` time must match the phoneme alphabet used in the dictionary file. Fetch the custom-deployment page and confirm the exact parameter name and accepted format with the matching image's `riva-build ... -h` output.

---

## Troubleshooting

- **Pronunciation unchanged despite `custom_dictionary`** — verify the double-space delimiter is present (not single space or tab); verify the model supports `custom_dictionary` for the deployed version by fetching the customization page; confirm the word spelling in the dictionary matches the exact spelling in the synthesis text (case-sensitive on some models).
- **SSML `<phoneme>` tag silently ignored** — the model may not support SSML or may not support the `<phoneme>` tag specifically. Fetch the customization page and confirm SSML support for your model before debugging further.
- **IPA character causes synthesis error or unexpected sound** — the IPA symbol may not be in the model's phone set, or a Unicode lookalike may have been substituted (e.g., Latin `ɛ` U+025B vs a visually similar character from another Unicode block). Copy IPA symbols from a verified source; check the phoneme-support page for the authoritative symbol list.
- **Multiple entries: last entry silently dropped** — check for spaces around the comma separator or a trailing comma. Correct format: `"WORD1  IPA1,WORD2  IPA2"` — no whitespace around commas, no trailing comma.
- **Phone set mismatch at build time** — `--phone_set` in `riva-build` must match the alphabet used in `--phone_dictionary_file`. Mixing IPA strings into an ARPABET-configured model produces garbled or default-G2P output.

## Limitations

- `custom_dictionary` and SSML `<phoneme>` support are per-model and per-release — verify feature support on the customization page and symbols on the phoneme-support page
- IPA candidates are generated from the agent's linguistic knowledge; for high-stakes or unusual terms, cross-validate against a phonetic dictionary (e.g., CMU Pronouncing Dictionary) or a native-speaker reference recording
- `custom_dictionary` applies per-request only; it does not persist across sessions unless included in every request or loaded from a session file
- Build-time dictionary embedding requires `riva-build` + `riva-deploy`; see [`tts-custom.md`](tts-custom.md)
- All phoneme strings must use UTF-8 encoding; ASCII-lookalike substitutions silently produce incorrect phonemes
- x86_64 architecture only for self-hosted deployment
