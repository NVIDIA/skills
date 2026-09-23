# Run

The last step. Start the stack, prove audio flows in both directions, and leave the user
able to do it without you.

Do not declare success because a process started. A voice agent can come up cleanly and
still be deaf or mute, so the finish line is a spoken exchange, not a running port.

## Order

Services first, agent last. An agent pointed at a NIM that is still loading fails in a way
that reads like a model or credential problem, which sends the user chasing the wrong bug.

Cloud runs have no services to start, so they go straight to smoke and then the agent.

## Cloud

The user copies `.env.example` to `.env` and fills in their own key. Never do this for
them, and never write a real key to disk.

Then sync dependencies. Anything missing from `.env` surfaces as an authentication failure
rather than a crash, so read the error before assuming the pipeline is wrong.

## Self-hosted

Start or reuse local model services and wait for readiness exactly as
`platforms/deployment.md` or the selected platform guide defines. Use the generated Compose
services and commands in the project README. Move on only after every local slot is ready.

Expect one long first start, and which service it is depends on the routed stack. A speech
NIM sits at `starting` while it downloads models and builds an engine
(`platforms/deployment.md` §First boot takes much longer). On the single-GPU stack the
speech service starts quickly and vLLM is the slow one, because it compiles kernels
(`platforms/single-gpu.md` §First boot compiles kernels). Read the logs rather than
restarting, and treat a long wait on the wrong service as a symptom instead of expected
behavior.

## Smoke

Run `scripts/smoke.sh` before the agent starts: on self-hosted or hybrid, once every local
service is healthy, and on cloud as soon as dependencies are installed, since there are no
local services to wait on. It proves the locked model answers on the streaming endpoint, the
locked voice speaks and is transcribed back, and the agent's own services construct. Cloud
and hybrid layouts run it against their configured endpoints. A silent connect failure in the browser is far more
expensive to diagnose than this script, so do not skip it to save a minute.

Fix whatever it reports through `operations/troubleshoot.md` and run it again. Only a
passing run earns a client connection.

## Connect

Pipecat's runner serves a client at `/client`. Use `localhost` when the browser and agent
are on the same machine, and the host address when they are not. For LiveKit, follow
`frameworks/livekit.md` §Connect and verify: open the LiveKit project that matches `.env`
`LIVEKIT_URL`, refresh Agents, and select the worker. A worker's local HTTP or health port
is never a client URL.

When the transport row said both, `GET /status` lists what actually registered, and the
client can switch between them. If `/status` names only one transport, the runner was
started with `-t` when it should not have been. See `frameworks/pipecat.md`.

For remote Pipecat browsers, NAT, coturn, or a separate GPU host, follow
`networking/remote-webrtc.md`. For LiveKit, use its current connectivity docs.

## Prove It Works

Speak and verify the selected pipeline:

- Cascaded: transcript appears → LLM reply is generated → TTS speaks it back.
- Omni: user audio produces an Omni reply → TTS speaks it back. A user transcription is
  optional.

The spoken reply must also pass `domain/agent-behavior.md`. Trace the exchange under one
turn id and confirm the stage events in `operations/observability.md`.

The first failed stage identifies the broken slot.
Use `operations/troubleshoot.md` for the symptom-first path.

## Stop

Interrupt the agent, then use the generated README to stop the Compose services you
started. Leaving them running holds the GPU and ports, which can make a later start fail
to bind.

## Hand Over

Leave the user with five things, written down rather than implied.

- How to start, check health and logs, and stop it through the generated README.
- Where the model ids and endpoints live in the project, which is the code, not `.env`.
- Which credentials `.env` needs, and that it is theirs to fill.
- `operations/troubleshoot.md` as the debugging entry point.
- `operations/iterate.md` for future changes.

## Anti-Patterns

- Starting the agent before the services report ready.
- Opening a client before `scripts/smoke.sh` passes.
- Treating a running process as a working agent.
- Writing a real credential into `.env` on the user's behalf.
- Leaving the stack running with no stop instructions.
