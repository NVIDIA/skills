# Speech Customization

Use when the agent must recognize or pronounce domain terms such as product names,
people, acronyms, medicines, menu items, or internal vocabulary.

This is runtime customization, not model retraining:

- ASR word boosting biases recognition toward approved words and phrases.
- TTS pronunciation customization maps approved words to supported IPA pronunciations.

Language and locale routing is a separate model-selection concern. Resolve it through
`models/language-routing.md`. This file owns only domain vocabulary.

## Trigger

Offer speech customization when the request names a domain with specialized vocabulary,
or when the user asks for boosting, hotwords, a glossary, pronunciation, or phonemes.
Skip it for a generic assistant unless requested.

After the base model table is approved, ask once:

```text
This use case has domain vocabulary. Should I add ASR word boosting and TTS pronunciation
rules? If yes, send any required terms or I will draft a glossary for approval.
```

## Verify Support First

Read the current official documentation before drafting or wiring:

- [ASR customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html)
- [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)
- [TTS customization](https://docs.nvidia.com/nim/speech/latest/tts/customization.html)
- [TTS support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html)

Confirm the locked model and API support the requested feature. Word-boosting modes,
limits, score ranges, and model support differ by decoder and release. TTS phoneme and
custom-dictionary support also differs by model and language.

If the locked model lacks a required feature, show the affected model row and propose one
supported streaming alternative. Never swap models silently.

### Platform Source

| Routed stack | Customization source |
| --- | --- |
| Cloud, or NIM on a workstation | Speech NIM docs and matrices above |
| vLLM plus NeMo-Speech.cpp | [NeMo-Speech.cpp ASR configuration](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/configuration.md) and [TTS configuration](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/tts/configuration.md) |

The single-GPU stack does not run Speech NIM, so Speech NIM tags and customization
procedures do not apply to it. Read the two configuration documents above before drafting
a glossary for DGX Spark, Jetson Thor, or a low-concurrency workstation.

Its two speech directions are not symmetric, and that asymmetry has to reach the user
before a glossary is approved.

**ASR boosting is supported and depends on the decoder.** Approved phrases and a score
travel per request in the recognition configuration's speech contexts. The cache-aware
streaming Nemotron models this stack uses accept boosting with no extra artifacts.
Flashlight beam decoding on a CTC model needs a language model and a lexicon first, and a
greedy CTC model with no language model ignores boosting entirely, as does an offline-only
model. Score ranges are not comparable across decoders, so read the range for the decoder
actually loaded and never carry a score over from a NIM deployment. Boosting also depends
on the tokenizer embedded in the model file, so an older GGUF may need reconversion before
it takes effect.

**TTS pronunciation customization is not available on this stack.** The synthesis service
takes plain text and exposes no request-time phoneme, IPA, or pronunciation-dictionary
path. Say that plainly rather than drafting IPA that cannot be wired. Two supported
options remain, and both belong in the proposal instead:

- text-normalization grammars, which decide how written numbers, dates, and currency are
  spoken (`platforms/single-gpu.md` §One-time speech model setup)
- the spelling of the term in the LLM instruction, so the model emits a form the voice
  already pronounces correctly (`domain/agent-behavior.md`)

If correct pronunciation of a specific term is a hard requirement, say that the NIM path
or cloud is where that control exists, and let the user decide before build rather than
discovering it at verification.

## Glossary Approval

Draft one table and show every term before writing files:

| Term | Locale | ASR phrase | Boost score | TTS IPA |
| --- | --- | --- | --- | --- |
| user or domain term | approved locale | exact spoken form | value from current ASR docs | supported IPA |

Use the current docs for score ranges and IPA support. Start with the lowest recommended
boost that solves the ambiguity. Higher scores increase false positives.

On the single-GPU stack the TTS column stays empty, because that path has no pronunciation
control. Show the table with that column marked unsupported rather than filling it, and
name the alternative you are proposing for each affected term.

Wait for explicit approval of the displayed terms, scores, and pronunciations. “Use your
suggestions” counts only after the complete table has been shown. Do not create a glossary
or wire customization before this confirmation.

## ASR

Keep streaming ASR. Choose the customization mode from the current ASR docs:

- Per-stream boosting for a request-specific glossary.
- Global boosting only when the approved list or deployment requires server-side
  configuration.

Wire the approved phrases and scores through the selected framework's current NVIDIA ASR
API. Query its documentation MCP for field names. Do not reuse score ranges across CTC,
RNNT, TDT, or Nemotron ASR without checking the current docs.

## TTS

On cloud and the NIM path, use the current TTS docs to choose request-time SSML phonemes
or a custom pronunciation dictionary. Use only phonemes supported by the locked model and
language.

Wire the approved dictionary through the selected framework's current NVIDIA TTS API.
After startup, synthesize every customized term and let the user confirm pronunciation.

The single-GPU stack has no pronunciation control, so there is nothing to wire there. Use
the alternatives in §Platform source and still synthesize every term at verification, so
the user hears what the agent will actually say.

## Omni

Omni has no ASR slot, so ASR word boosting does not apply. Do not add ASR just for domain
terms. TTS pronunciation customization still applies wherever the routed stack supports
it.

Domain terms may also be added to the Omni system instruction for response consistency,
but that is not ASR boosting.

## Generated Project

When customization is approved:

- Write the confirmed terms to `speech_glossary.json` with this schema:

```json
{
  "version": 1,
  "terms": [
    {
      "term": "approved display form",
      "locale": "en-US",
      "asr": {
        "phrase": "approved spoken form",
        "boost": 4.0
      },
      "tts": {
        "mode": "ipa",
        "grapheme": "approved display form",
        "pronunciation": "approved IPA"
      }
    }
  ]
}
```

- Omit `asr` for Omni or when boosting is not approved. Omit `tts` when pronunciation
  customization is not approved or the routed stack does not support it. Do not write
  placeholder or `null` entries.
- Keep model ids, scores, and pronunciation data out of `.env`.
- Load the glossary from agent code and map it to the framework APIs documented today.
- Include a short README note explaining how to edit and revalidate the glossary.

## Verify

Test each term in a natural sentence:

1. ASR returns the intended spelling without causing nearby false positives.
2. TTS pronounces the term acceptably, and where pronunciation is not controllable, the
   user hears it and accepts the result.
3. A normal sentence without glossary terms still works.
4. The full spoken exchange in `operations/run.md` still passes.

Tune one term or score at a time. Use `operations/iterate.md` for later glossary changes.

## Anti-Patterns

- Writing or wiring a glossary before showing it and receiving approval.
- Assuming every ASR model supports word boosting, or that the loaded decoder does.
- Carrying a boost score across decoders or across deployment stacks.
- Applying ASR boosting to Omni.
- Inventing IPA unsupported by the selected TTS model or language.
- Drafting pronunciation rules for a stack that has no pronunciation control, instead of
  saying so before approval.
- Raising boost scores without checking false positives.
