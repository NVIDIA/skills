# Pi adapter (guard + recorder + NVIDIA provider)

TypeScript extensions for the [Pi coding agent](https://pi.dev). Pack drivers
load them with `-e`; requires Pi 0.85.1+ (credential-safe bash spawn hooks).

| File | Role |
|---|---|
| `guard.ts` | Blocks known dead ends BEFORE tokens burn: per-session tool-call budget (`PI_KIT_TURN_BUDGET`), near-duplicate-command loop breaker, no-fabricated-results check (a train "ok" commit requires `Execution status: PASS` in the real log), destructive-command safety net, and credential-access guards across bash/read/edit/write. Bash subprocesses receive only allowlisted non-secret environment variables, never provider keys or shell startup hooks. The tool-call budget includes file tools and aborts the turn on exhaustion. |
| `recorder.ts` | Appends every substantive executed command to `$RD/commands.log` (guard-blocked calls excluded). Run dir comes from `PI_KIT_RD`, falling back to the newest `PI_KIT_RUN_PREFIX`* dir under `$PI_KIT_WS/results`. Extend the match list via `PI_KIT_RECORD_PATTERNS` — comma-separated regexes, so a pattern cannot itself contain a comma (e.g. `{1,3}` quantifiers are unrepresentable; use an unbounded form instead). Invalid fragments are logged and skipped, never fatal. |
| `nvidia-provider.ts` | Optional `nim/` provider for the user's own OpenAI-compatible endpoint: thinking disabled via chat-template kwargs with the `:off` model suffix, temperature pinned to 0 for every `nim/` model. It has no default endpoint and no built-in model id: it is registered only when `PI_KIT_NIM_BASE_URL` (alias `NVIDIA_INFERENCE_BASE_URL`) holds the base URL, including `/v1`, and `PI_KIT_NIM_MODELS` names at least one model id; drivers and the smoke test add the id from `MODEL=nim/<id>` automatically. Also configure `PI_KIT_NIM_API_KEY_VAR` (name of the key variable; default `NVIDIA_INFERENCE_API_KEY`). Not needed for Pi built-in providers. |

## Security boundary

These guards are defense in depth, **not an OS sandbox**. Arbitrary shell
programs and access to the Docker daemon can access host files indirectly.
Run packs only on an isolated, trusted worker without unrelated credentials,
with datasets staged and images authenticated/pulled during approved preflight.
Do not run untrusted cards on a personal workstation. Credential-file guards
(including `~/.docker/config.json` and `~/.ngc/config`) resolve symlinks for
read/edit/write and for path-like words in bash commands (after `~`/`$VAR`
expansion); paths built at runtime (`$(...)`, `eval`, globs) are not
resolved, so shell deny patterns catch common access but do not prove
arbitrary shell programs safe.

Provider keys remain available to Pi itself; only its bash subprocess
environment is filtered. New packs needing authenticated subprocesses require
an explicitly designed credential channel, not a blanket environment passthrough.

## Offline regression tests

The regression tests of the guard, the pack drivers, and card mining use
fixtures only and run from a checkout of the repository,
https://github.com/NVIDIA-TAO/tao-skill-bank; this plugin package does not carry
them. The separate `smoke_test.sh` calls a real model API.

## Workflow-specific guards

`guard.ts` also carries three guards discovered on the DEFT AOI reference
hardware (a cuDNN/sm_75 embedding crash, a disk-headroom check before
training, a container CLI quirk). They are harmless elsewhere — each matches a
specific command signature — and they double as the pattern to copy: **one
guard per quirk your first run discovers**, with a reason that says why it is
blocked and what to do instead.

## Env contract

Set by pack drivers; set them yourself only when running sessions by hand:
`PI_KIT_TURN_BUDGET`, `PI_KIT_RD`, `PI_KIT_WS`, `PI_KIT_RUN_PREFIX`,
`PI_KIT_RECORD_PATTERNS`. For hand-run sessions the guard also honors bare
`RD`/`ITER` and the recorder honors bare `WS`; the recorder deliberately
trusts only `PI_KIT_RD` for the run dir (a bare-RD fallback could misfile
records into a previous run's commands.log).
