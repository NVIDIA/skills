# Output Contract

Read after approval, container readiness, exact local profile pinning, and any required
row reconfirmation. The generated project must be runnable and repeatable without
re-reading this skill.

Write in two passes, because some values only exist once the services answer. First write
everything needed to start them: dependencies, secret placeholders, Compose, and cache
paths. Then start the services, resolve the runtime model id and TTS voice, and finalize
the agent file, `scripts/smoke.sh`, and the README with those real values. No placeholder
model, voice, endpoint, or command may survive the second pass.

## Always Write

| File | Purpose |
| --- | --- |
| `bot.py` or `agent.py` | selected framework pipeline |
| `pyproject.toml` | dependencies resolved from current framework docs |
| `.env.example` | secret placeholders only |
| `.gitignore` | excludes `.env`, credential variants, caches, and generated model state while preserving `.env.example` |
| `scripts/smoke.sh` | proves the stack before any client connects |
| `README.md` | setup, start, verify, stop, logs, and troubleshooting |

Also write:

- `compose.yaml` when any model service is self-hosted or the project owns coturn
- `speech_glossary.json` when speech customization is approved
- `signaling_server.py` when custom TURN is required for Pipecat WebRTC
- `nvidia_omni_multimodal_service.py` and `audio_only_smart_turn_strategy.py` from the
  current upstream blueprint when the pipeline is Omni

## Generated Compose

The skill ships no static Compose file. Generate `compose.yaml` from current authoritative
deployment instructions:

| Service | Source |
| --- | --- |
| Workstation NIM LLM, ASR, or TTS | each locked model's build.nvidia.com self-hosted page |
| Workstation NIM Omni | the Omni build page plus the current VLM NIM quickstart, support matrix, profiles, utilities, and environment documentation in `frameworks/omni.md` §Workstation NIM |
| Single-GPU LLM or Omni | the locked model's build.nvidia.com page and the current [vLLM `serve` CLI reference](https://docs.vllm.ai/en/stable/cli/serve/) |
| Single-GPU speech | the NeMo-Speech.cpp container's current NGC page for the run command, and its repository documentation for engine keys |
| Project-owned coturn | current coturn deployment documentation |

Translate each current launch command into one Compose service. Preserve its image,
command, environment, mounts, shared-memory or IPC settings, ports, and GPU requirements
except for the approved profile/tags, host ports, cache paths, and GPU placement this skill
explicitly locks. Use current health checks from the same source.

Memory controls are required Compose entries on any shared GPU, not optional tuning. Which
controls depends on the routed stack:

| Routed stack | Required in Compose |
| --- | --- |
| Workstation NIM Cascaded | the exact pinned LLM NIM profile, the approved context length, exactly one documented LLM NIM runtime memory-control path, and the explicit TTS batch profile |
| Workstation NIM Omni | the exact Compatible VLM NIM profile, the approved context length, only controls documented for that VLM NIM release, and the explicit TTS batch profile |
| Single-GPU vLLM plus NeMo-Speech.cpp | `--gpu-memory-utilization`, `--max-model-len`, and every required speech model path |
| Single-GPU Omni | the above, plus the audio extras installed before `vllm serve` and pinned to the image's vLLM version, plus the modality limits and media path the model requires (`platforms/single-gpu.md` §Omni needs audio extras) |

Do not rely on auto-selected LLM precision, default vLLM memory utilization, or a container
default. The memory-control value comes from the budget in `preflight.md` §Deployment fit
and, on the single-GPU stack, from the measured speech reserve in
`platforms/single-gpu.md` §Memory. Do not reuse a number from another build. If
memory measured at startup differs from the budget, update Compose before starting rather
than proceeding with a stale value. Document the gate that applies in the README, which is
`platforms/deployment.md` §Shared-GPU memory gate on NIM and
`platforms/single-gpu.md` §Start and verify on the single-GPU stack.

Preserve the documented cache or model mount for every service. A speech NIM downloads
models and can build engines on first boot. The single-GPU stack instead mounts a
pre-populated read-only model tree and caches compiled vLLM kernels. Either way that work
must survive container replacement.

| Routed stack and pipeline | Service names |
| --- | --- |
| Workstation NIM cascaded | `llm`, `tts`, `asr` |
| Workstation NIM Omni | `omni`, `tts` |
| Single-GPU cascaded | `llm`, `speech` (ASR and TTS in one service) |
| Single-GPU Omni | `omni`, `speech` (TTS paths only) |

Generate no HTTP health check for the single-GPU `speech` service. It serves gRPC only, so
readiness is the configuration query in `platforms/single-gpu.md` §Readiness is gRPC only.

For a hybrid deployment, generate only the services assigned locally. Cloud slots remain
agent configuration and do not get placeholder Compose services.

For a multi-GPU workstation running NIM, follow `platforms/deployment.md` §Scaling across
GPUs. The single-GPU stack serves one device, so it pins every service to the same GPU. In
Compose, use `device_ids` or `count`, never both. A multi-GPU LLM service must expose the
exact topology required by its compatible profile.

On DGX Spark and Jetson Thor, confirm an `arm64` image exists for every generated service
before writing it.

Compose may read secret values such as `NVIDIA_API_KEY` and `HF_TOKEN` from the user
environment or `.env`. Write only placeholders to `.env.example`; the user supplies real
values after file generation is approved. Keep model ids, endpoints, profiles, tags, ports, and GPU placement
in `compose.yaml` or agent code, not `.env`.

On the single-GPU stack, mount the speech model tree produced by the one-time download as
read-only, from a path outside the generated project. Do not populate or repair it during
`docker compose up`, and do not create it by starting Compose first, because Docker then
creates it as root (`platforms/readiness.md` §Speech model tree).

Validate before handover:

```bash
docker compose config --quiet
```

Then start one service at a time and wait for readiness. Do not rely on Compose startup
order alone.

## README

Write the locked choices and exact commands for this project:

1. framework, pipeline, host class, routed local stack and the concurrency it assumes,
   transport, language route, per-slot local/cloud placement, container image, pinned
   profile or quantization variant, runtime model id, and discovered TTS voice
2. prerequisites and required credentials
3. one-time setup, including the speech model download on the single-GPU stack
4. dependency installation
5. service startup in the required order for the routed stack
6. health and model-id checks, then `scripts/smoke.sh`
7. agent start command and client connection path
8. spoken-exchange acceptance test
9. log commands, stop commands, and links to troubleshooting

Item 3 must name the long first start for the routed stack, which log line shows progress,
and which path keeps the result. On NIM that is a speech model download and engine build.
On the single-GPU stack it is the vLLM kernel compile, plus the one-time speech model
download and its ownership rules.

Item 5 must give the order the routed stack requires, because the two are opposite. NIM
starts the LLM first. The single-GPU stack starts speech first, so its reserve can be
measured before the LLM memory fraction is set.

Do not leave placeholders such as “start the services” or “use the deploy page.” Resolve
those instructions while building and persist the resulting commands in the README.

## Smoke Before Client

The check list below is fixed by this contract. Plan it in the first pass and do not reduce
it later. The second pass only fills in resolved values: take the LLM model string from that
placement's `/v1/models` and the voice from the TTS voice query in `models/tts.md`, then
write both into `bot.py` or `agent.py` in place of anything carried over from the proposal
table.

Generate `scripts/smoke.sh` from the same resolved endpoints. Include only the checks that
apply to the approved pipeline and deployment: cloud slots probe their configured cloud
endpoints instead of local service health. It must exit non-zero on the first failure so the
user never debugs this from a browser:

1. `/v1/models` returns the exact served id the agent uses
2. Cascaded: one **streaming** chat completion with reasoning off, asserting non-empty
   `delta.content` and empty or absent `delta.reasoning_content`. A non-streaming answer
   still passes while a reasoning parser routes every token away from TTS
   (`models/llm.md` §Reasoning parser)
3. Omni: endpoint readiness and one minimal **audio** request, and no LLM or ASR checks. It
   must be an audio request, because a text request passes on a server that cannot accept
   audio at all
4. TTS voice query, then one sentence synthesized in the approved locale using a voice that
   query actually returned
5. Cascaded: ASR configuration from the running service, asserting the loaded model and
   language rather than HTTP readiness alone
6. Cascaded: feed the synthesized sentence back through streaming ASR, which is the only
   check that proves both speech directions before a microphone is involved
7. in-process construction of the same services the agent builds, using the same classes
   and settings, which is what catches a missing API-key argument or a misplaced reasoning
   payload

On the single-GPU stack, checks 4 and 5 use the gRPC configuration queries rather than an
HTTP voice list, and check 6 must also assert that a mid-stream final result arrives, which
is what proves ASR endpointing is enabled
(`platforms/single-gpu.md` §Set explicitly).

Use the agent's own settings, including per-slot cloud endpoints on a hybrid layout. Print
no secrets. Run it after the services are healthy and before the runner starts. A passing
run is still not a working agent, so the spoken exchange in `operations/run.md` remains the
bar.

## Verify the Generated Project

Re-read this section after generation.

- No real secret is written to disk.
- `compose.yaml`, when generated, parses and exposes only the intended GPUs.
- Every local service reaches readiness.
- `scripts/smoke.sh` passes.
- The documented agent command starts successfully.
- A new terminal can follow the README to complete a spoken exchange.
