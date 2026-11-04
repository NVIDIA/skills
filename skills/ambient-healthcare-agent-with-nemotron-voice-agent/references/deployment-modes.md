# NVA Service and Deployment Modes

Use this reference after inspecting the target NVA checkout. Follow the public
NVA deployment and configuration documentation for supported profiles,
hardware checks, credentials, catalog shape, startup commands, and verification.

## Runtime Choice

Treat LLM, ASR, and TTS as one explicit runtime decision. Report the inspected
default for each service, including its key, display name, model or server,
endpoint, and catalog source.

The default mode is `public-endpoints`: use public NVIDIA AI Endpoints from the
generic `services.cloud.yaml` catalog for the LLM, ASR, and TTS. Tell the user
the absolute path to the checkout's `.env` file, created from `.env.example` if
absent, and instruct them to fill in `NVIDIA_API_KEY=` there for this mode. Keep
the credential out of chat and preserve any existing `.env`. Tell the user
that they can instead request either:

- `nva-managed-local`: NVA Compose self-deploys the supported NIM services.
- `existing-nim-endpoints`: NVA points to NIM services the user already runs.

The user may also request a mixed service layout. In that case, use one complete
NVA recipe profile and configure each service slot through the documented NVA
catalog and registry workflow. Never combine multiple recipe profiles to
assemble a hybrid deployment.

If the user does not request an override after seeing this disclosure, record
the decision as `public-endpoints`. Do not silently fall back to public services
after the user selects a local or existing-endpoint mode.

## Public NVIDIA AI Endpoints

Preserve an inspected LLM, ASR, or TTS default when it resolves to the `cloud`
section of `services.cloud.yaml`. If a default is not public, choose a compatible
cloud entry reported by the inspector and use the public NVA configuration
instructions to update `examples.generic-assistant.defaults`.
Prefer the existing key when that key has a cloud entry; otherwise use the first
compatible cloud entry in catalog order and tell the user what changed.

Use the public NVA deployment documentation to select the cloud-only Generic
Cascaded recipe. In current NVA repos this is normally
`generic-assistant`, but inspect the checked-out release rather than assuming it.

After applying the healthcare overlay and passing static checks, start the
customized Generic recipe. Its authenticated app and selected-service health
checks are the credential and endpoint gate. Run the authorized live
expected-conversation validation against the selected public LLM after that gate.

## NVA-Managed Local NIMs

Ask which supported hardware target should host the NIMs and whether the user
wants all services local or a mixed layout. Then follow the public NVA
deployment documentation to:

1. inspect the actual GPU, memory, and compute capability;
2. inspect the checked-out NVA deployment guidance and choose exactly one supported Generic recipe (for example, current releases may expose `server` and `single-gpu` families); do not infer a profile name from an older release;
3. identify NGC/Hugging Face prerequisites and any model-profile settings;
4. use the documented catalog workflow to select matching local LLM, ASR, and TTS keys;
5. after applying the healthcare overlay and passing static checks, satisfy
   the selected prerequisites, start the recipe, and wait for every required
   sidecar to be ready.

Do not call the public inference API merely to validate a key after the user has
opted out of public inference. Validate credentials and image access through the
checks required by the public NVA deployment documentation instead.

The local LLM must be running before live expected-conversation validation.
Because the validator runs on the host, pass the local LLM key plus the
host-reachable base URL documented by the target repo when its catalog contains
a Compose DNS name. Apply any validation-driven prompt fix, rebuild or restart
as directed by the public NVA documentation, and rerun the full suite.

## Existing NIM Endpoints

Collect the non-secret connection details required by the target repo catalogs:

- LLM base URL and model ID;
- ASR server, protocol, and model settings;
- TTS server, protocol, model, voice, and language settings;
- the names of environment variables that hold credentials.

Keep credentials out of chat; request only the names of credential variables.
Use the documented NVA catalog workflow to add or update entries and set the three registry defaults.
Then choose a single documented app recipe that does not start unwanted
sidecars; when all three services are remote, this is normally the cloud-only
Generic Cascaded app recipe.

After applying the overlay and passing static checks, verify reachability and
authentication for the user-provided endpoints. Run expected-conversation
validation against the host-reachable LLM endpoint. Verify ASR and TTS through
the service checks available in the target NVA repo and the deployed app logs.

## Ordering and Handoff

For every runtime mode, first verify checkout compatibility and Docker Compose
access, select one documented recipe, obtain the scenario selection and required
fixture authorization, apply the healthcare overlay, and run static validation.
Only then start the customized Generic app. Do not start the unmodified recipe.
Missing credentials or unavailable endpoints block the later live and service
checks, not the overlay and static checks.

- `public-endpoints`: start the customized cloud recipe, verify the app and
  selected LLM, ASR, and TTS services, then run authorized live histories
  against the selected public LLM.
- `existing-nim-endpoints`: start the customized app recipe without unwanted
  sidecars, verify the configured endpoints, then run authorized live histories
  against the host-reachable LLM endpoint.
- `nva-managed-local`: start the customized recipe and its required LLM and
  speech sidecars together. After their health checks pass, run authorized live
  histories against the documented host-reachable LLM endpoint.
- Mixed service layouts use one complete documented recipe, verify every
  selected service, and run authorized live validation only against the
  configured host-reachable LLM endpoint.

Source changes introduced by this skill require the build behavior documented
for the checked-out NVA release. YAML-only service/default changes use that
release's documented restart or re-apply behavior.

After any validation-driven restart, reverify the app and every selected service
and complete one real microphone-to-ASR-to-LLM/tool-to-TTS round trip. The handoff
must report the chosen runtime mode; selected LLM, ASR, and TTS keys and
endpoints; recipe profile; credential and endpoint checks; static and live
validation results; voice round-trip result; running app and sidecar services;
and UI URL. Report any failed final gate as blocked or pending.
