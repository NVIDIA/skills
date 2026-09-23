# Model Catalog

Exact ids for the proposal table. Never write them from memory. Open the file for the
slot you are resolving.

| Slot | Read | Discover first | Confirm on |
| --- | --- | --- | --- |
| Language / locale | `models/language-routing.md` | ASR + TTS language tables | selected framework docs MCP |
| Cascaded LLM | `models/llm.md` | LLM NIM support matrix for NIM, or the locked model page for raw vLLM, plus `/v1/models` | [Lightning](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b?nim=self-hosted) |
| Omni | `models/llm.md` and `frameworks/omni.md` | VLM NIM matrix for NIM, or the locked model page for raw vLLM, plus `/v1/models` | [Nemotron 3 Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) |
| ASR | `models/asr.md` | [Speech docs](https://docs.nvidia.com/nim/speech/latest/) → [ASR matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) | `<slug>` and `<slug>/deploy` |
| TTS | `models/tts.md` | [TTS matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html) | same pattern |

Family browse: [NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron#section-nvidia-nemotron-models).

Nemotron 3 Omni is the product name for the Omni slot, and its slug, repository ids, and NIM
image carry `nano-omni`. The proposal table shows the product name. Generated code and
Compose carry the published ids verbatim
(`models/llm.md` §The Omni name and its ids differ).

## Shared Rules

- Proposal table gets exact catalog ids. Family names alone are wrong. Those ids describe
  the choice, and agent code uses the runtime ids resolved after startup.
- Lock language routing before selecting ASR and TTS.
- Model ids, voice ids, and function ids go in generated code or config. `.env` is secrets only.
- Model-service tools (`list-model-profiles` and `NIM_MODEL_PROFILE`) never apply to ASR
  or TTS. Select the LLM or VLM NIM support matrix from the pipeline.
- The LLM NIM support matrix and its profile assumptions apply to Cascaded LLM only. Omni
  NIM uses the VLM NIM matrix, profiles, utilities, and environment documentation in
  `frameworks/omni.md`.
- Speech profile tags (`NIM_TAGS_SELECTOR`, Speech matrices, `/deploy`) do not apply to
  generated LLM deployments. Pin an LLM with `NIM_MODEL_PROFILE`.
- For self-hosted Speech, the matrix owns profile fit and tags. The `/deploy` page owns
  image URI and launch shape. Stop if they still conflict.
- Shared-GPU placement uses the runtime budget in `preflight.md` and the LLM controls in
  `models/llm.md`, plus the explicit TTS batch profile in `models/tts.md` on the NIM path.
  It stays provisional until the routed stack's measured gate passes
  (`preflight.md` §Shared rules).
- The single-GPU stack bypasses Speech NIM selection entirely and resolves ASR and TTS
  through `platforms/single-gpu.md`. That covers DGX Spark, Jetson Thor, and a
  low-concurrency workstation.
- If the user names a model outside these families, say so and wait. Do not substitute.

## Where They Live

| Identifier | Who | Source |
| --- | --- | --- |
| Catalog slug | Proposal table | `build.nvidia.com/nvidia/<slug>`. Intent only, never agent code |
| Container image | Self-hosted LLM | the self-hosted build page or NGC command that actually pulls |
| `NIM_MODEL_PROFILE` | Self-hosted Cascaded LLM or Omni NIM | `list-model-profiles` from that service's NIM image on the assigned GPU set |
| Quantization variant | Self-hosted LLM on vLLM | the locked Hugging Face card's variant for the probed compute capability |
| Runtime model id | LLM (and listing checks) | `GET /v1/models` · cloud `https://integrate.api.nvidia.com/v1/models` |
| Function id | Cloud ASR / TTS | build.nvidia.com model page. gRPC `grpc.nvcf.nvidia.com:443`. Empty when self-hosted |
| `CONTAINER_ID` + `NIM_TAGS_SELECTOR` | Self-hosted ASR / TTS on Speech NIM | Speech matrix, then `/deploy` |
| GGUF file name | Self-hosted ASR / TTS on the single-GPU stack | that model's Hugging Face page, read at fetch time |
| Voice id | TTS | running service's documented voice-discovery API |

Slug, image, and runtime model id are three separate strings (`models/llm.md` §Resolve
three names). Agent code uses the runtime model id only.
