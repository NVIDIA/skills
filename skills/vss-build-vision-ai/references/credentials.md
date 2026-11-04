# Credential Gate

Run this before writing `_builds/<name>/override.env` or starting any image
pull. Validate credentials early: a bad key should fail in seconds, not after
a cold NIM start.

## Required By Mode

- `NGC_CLI_API_KEY` or `NGC_API_KEY`: required for any local NIM image pull
  (`LLM_MODE` or `VLM_MODE` set to `local` / `local_shared`). These are the
  same underlying NGC personal API key with different consumer conventions:
  the NGC CLI and build override use `NGC_CLI_API_KEY`; NIM / RT-VLM
  containers receive the key as `NGC_API_KEY`.
- `NVIDIA_API_KEY`: required for a remote endpoint on build.nvidia.com
  (`integrate.api.nvidia.com`), and the key a remote NIM reads
  ([`env-overrides.md`](env-overrides.md)). A keyed endpoint elsewhere still
  needs a runtime key persisted, named for its consumer: `OPENAI_API_KEY` for an
  OpenAI-compatible LLM/VLM the agent calls, `RTVI_VLM_API_KEY` for RT-VLM's own
  endpoint. Only a genuinely keyless endpoint needs none. `REMOTE_API_KEY` is
  probe-only and persists nothing.
- `HF_TOKEN`: required only for a gated or private Hugging Face checkpoint
  (`--require hf`); the RT-Embed Cosmos-Embed defaults are public, and no
  in-tree edge path needs it.
- Customer LLM/VLM endpoint URL + model name: required for any selected
  remote endpoint. This includes build.nvidia.com / NVIDIA API catalog
  endpoints because their `/v1/models` response can list many models.

## Discovery

Surface discovered credentials to the user; do not auto-source them without confirmation.

- If either `$NGC_CLI_API_KEY` or `$NGC_API_KEY` is set, normalize both names
  to the same resolved NGC key before probes and before writing
  `_builds/<name>/override.env`.
- If both `$NGC_CLI_API_KEY` and `$NGC_API_KEY` are set and differ, stop and
  ask which NGC personal API key to use. Do not silently choose one.
- If neither NGC env var is set but `~/.ngc/config` exists, extract the
  account metadata and ask: `Use NGC account <org>/<team> for the deploy?`
- If `$HF_TOKEN` is unset but `~/.cache/huggingface/token` exists, ask before exporting it.

## Probes

Run the credential gate, naming the credentials the chosen mode requires with
`--require`. It validates the NGC key when set, reporting it as validated,
rejected by the service, not validated because the service never answered, or
skipped, resolves `NGC_CLI_API_KEY` / `NGC_API_KEY` to one key, and reports a
conflict when both are set and differ. Only a `401`/`403` is a verdict on the
key itself; a timeout, a rate limit, or a `5xx` means the probe got no verdict,
so retry or check the service rather than replacing the key. The probe is
bounded at 5s to connect and 15s in total.

`NVIDIA_API_KEY` and `HF_TOKEN` are the exceptions: both are checked for
presence, never validated. `integrate.api.nvidia.com/v1/models` is a public
catalog, and a gated repository's Hugging Face metadata is public too, so each
answers `200` for a good key, a junk key and none alike. A probe against either
proves nothing, and claiming a validated key on that basis is worse than saying
nothing. The endpoint that will actually be called is probed by
`probe_remote_models.sh` below, and checkpoint access is the artifact probes'
job.

| Chosen mode | Pass |
| --- | --- |
| `LLM_MODE` or `VLM_MODE` of `local` / `local_shared` | `--require ngc` |
| `LLM_BASE_URL` / `VLM_BASE_URL` on build.nvidia.com (`integrate.api.nvidia.com`) | `--require nvidia-api` |
| A **gated or private** Hugging Face checkpoint | `--require hf` |

Key the second row on the endpoint, not on the mode being remote: a keyless
endpoint such as `http://localhost:30081` needs no `--require` at all, and
requiring `NVIDIA_API_KEY` for it would block a build that needs no such key.

`--require nvidia-api` is a gate-time presence check, not the runtime wiring. A
keyed endpoint elsewhere — a self-hosted NIM, a third-party gateway — still
needs its key written into `override.env` as the variable its consumer reads,
named under "Required By Mode" above. `REMOTE_API_KEY` only feeds
`probe_remote_models.sh` and is read by no deployed service, so setting it alone
leaves the build with no usable key and fails at the first inference.

The Cosmos-Embed checkpoints RT-Embed uses by default are public, so a build
that loads only those needs no token and no `--require hf`.

```bash
bash skills/vss-build-vision-ai/scripts/check_credentials.sh --require ngc
```

Branch on the exit code, not on the printed lines. Any non-zero exit blocks:
`0` is every required credential present and validated where probed, `2` is a
gate failure with a `BLOCKER` summary on stderr naming each cause, and `1` is a
usage error, which means the gate never checked a single credential — fix the
invocation and re-run rather than treating it as a pass. Which credentials are
required is the caller's to declare — with no `--require` the gate only fails on
an NGC key-name conflict.

After the NGC key validates, set **both** `NGC_CLI_API_KEY` and `NGC_API_KEY` to
that one resolved key in `_builds/<name>/override.env` — the NGC CLI and VSS env read
`NGC_CLI_API_KEY`; NIM / RT-VLM containers read `NGC_API_KEY`. Do not leave only
one set.

Set the key **before** generating `resolved.yml`. If either key changes after
`resolved.yml` exists, regenerate it (re-run `docker compose … config`): the
deploy reads only the self-contained `resolved.yml` and passes no `--env-file`,
so a key baked as `''` cannot be fixed by a shell export or an env file at
`docker compose up`.

This token probe is not sufficient for local NIM / RT-VLM deployments. It
proves the key authenticates, but it does not prove that the key's org/team can
access the selected `nvcr.io/...` images or `ngc:...` model repositories. After
`resolved.yml` exists, run the artifact probes below before starting Compose.

## Artifact Entitlement Probes

Build the artifact list from the selected deployment:

- `resolved.yml`: every `image:` under `nvcr.io/...` that Compose will pull.
- `_builds/<name>/override.env`: NGC-backed model/resource paths such as
  `RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos3-nano-reasoner:bf16-final`. Skip
  `none`, local paths, remote endpoint URLs, and `git:` paths other than
  Hugging Face.
- `_builds/<name>/override.env`: Hugging Face checkpoints, written as
  `git:https://huggingface.co/<org>/<repo>`. This is the only place
  `HF_TOKEN`'s access is established — the gate checks presence, not validity.
- Profile staging instructions: NGC model/resource downloads such as
  alerts/search perception models.

Probe each artifact with the credential that will fetch it — the normalized NGC
key for the NGC artifacts, `HF_TOKEN` for the Hugging Face ones:

- Container images: after `docker login nvcr.io`, run `docker manifest inspect
  <nvcr.io/...>`, or use `ngc registry image info ...` when it maps cleanly to
  an NGC image path. A `401` or `403` from a gated repository proves the key
  lacks the required org/team entitlement.
- NGC models/resources: run `ngc registry model info ...` or `ngc registry
  resource info ...` for the exact repository and tag. Pass the org and team
  explicitly — an `ngc:` path already carries them, and without them the CLI
  fails `Missing org - If Authenticated, org is also required.` on a perfectly
  entitled key. Split `ngc:<org>/<team>/<name>:<version>` and hand back both:

  ```bash
  # RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos3-nano-reasoner:bf16-final
  ngc registry model info nim/nvidia/cosmos3-nano-reasoner:bf16-final \
    --org nim --team nvidia
  ```

  Treat that error as a malformed probe, not a failed entitlement, and do not
  reach for `ngc config set` ([`ngc.md`](ngc.md)) to fix a one-off check. Do not
  use `docker manifest inspect` for non-OCI models or a raw
  `Authorization: Bearer <key>` REST call; their expected failures are false
  entitlement signals. If the NGC CLI is unavailable, use the selected gated
  container-image probe as the org/team entitlement signal.
- Profile-staged TAO/perception models: run the corresponding NGC model or
  resource probe before downloading them.
- Hugging Face checkpoints: probe the exact selected repository with
  `auth-check`, the endpoint `huggingface_hub`'s `HfApi.auth_check()` calls.

  ```bash
  # RTVI_VLM_MODEL_PATH=git:https://huggingface.co/<org>/<repo>
  curl -s -o /dev/null -w '%{http_code}\n' --connect-timeout 5 --max-time 15 \
    -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/api/models/<org>/<repo>/auth-check
  ```

  `200` is access, `401` a missing or invalid token, `403` a valid token that
  has not been granted the repository. Do not substitute `/api/models/<repo>` or
  its `/tree/main`: Hugging Face keeps a **gated** repository's metadata and
  file list public, so both answer `200` for a good token, a junk token and none
  alike — the same reason the gate cannot validate `HF_TOKEN` on its own. Only a
  private repository refuses those, which is why metadata is not the probe.
  A public repository answers `200` to anyone here too, so run this against the
  **selected** repository and nothing else; RT-Embed's public Cosmos-Embed
  defaults need neither a probe nor a token.

  `200` clears repository access, not the download: a token scoped below what
  the download needs still fails at `hf download`, which stays authoritative.

On `401`, `403`, permission, membership, or missing repository errors, stop and
request access for the credential that probe used: an NGC key entitled to the
NGC artifacts, or a Hugging Face token granted the selected repository. Do not
defer this failure to image pull, model download, or NIM cold start.

## Remote Endpoint Probes

For every selected remote LLM/VLM endpoint, probe the endpoint before writing
it into `_builds/<name>/override.env`. Do this even when the endpoint is on
localhost; it
catches wrong ports, stale tunnels, missing auth, and model-name mismatches
before the deploy flow spends time generating compose or warming containers.

Use the base URL without a trailing `/v1`; the script strips `/v1` and
`/v1/models` if the user supplied them. If the endpoint requires auth, set
`REMOTE_API_KEY` to the key that the agent will use for that endpoint. It scopes
to this probe only — no deployed service reads it, so the runtime key still has
to be persisted separately, named under "Required By Mode" above.

On build.nvidia.com this clears the endpoint and the advertised model, not the
key: `/v1/models` there is a public catalog that answers `200` with no
`Authorization` header, so neither it nor the credential gate can reject a bad
`NVIDIA_API_KEY`. One surfaces at the first authenticated inference request —
read that `401` as a credential failure, not an endpoint fault
([`troubleshooting.md`](troubleshooting.md)). An authenticated probe would not
settle it sooner: a bogus key and a valid key without access to the named model
both answer `403`.

Aggregate endpoints such as `https://integrate.api.nvidia.com` can advertise
many LLM and VLM models. Do not auto-select the first returned model from such
endpoints. If the endpoint lists multiple models and the user has not selected
an exact model id, stop and ask which model to use.

Run the skill script:

```bash
REMOTE_API_KEY="$NVIDIA_API_KEY" \
  skills/vss-build-vision-ai/scripts/probe_remote_models.sh "$LLM_BASE_URL" "$LLM_NAME"

skills/vss-build-vision-ai/scripts/probe_remote_models.sh \
  "http://localhost:30081" "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
```

If `/v1/models` fails or does not advertise the selected model, stop and ask
the user for the correct endpoint/model before writing the build override.

## Decision Rule

Any non-zero exit from the credential gate, a selected NGC or Hugging Face
artifact access failure, or a selected remote endpoint that fails `/v1/models`
is a blocker. Exit `2` names a credential cause; exit `1` means the gate did not
run at all, so never read it as a pass — fix the invocation and re-run. Prompt
the user, re-probe, and do not proceed to env mutation until it resolves — an
unset key that reaches `resolved.yml` is baked in as `''` and no later export
fixes it.

The gate applies the declared requirements itself, so a `skip` for a key the
mode does not use exits `0` and is fine, as is a rejected or unvalidated key
the mode does not need. `not validated` means the service gave no verdict — no
answer within the probe timeout, a rate limit, or a `5xx`: fix the host's
egress or wait for the service, rather than replacing a key that may be good.

An exit `0` is not a statement that `HF_TOKEN` or `NVIDIA_API_KEY` works. The
gate only saw that they were set. The checkpoint stays unproven until its
artifact probe runs; `NVIDIA_API_KEY` stays unproven even after
`probe_remote_models.sh`, which clears the endpoint and the model but not the
key, until the first authenticated inference request. Do not read either
`set — not validated here` line as a pass.
