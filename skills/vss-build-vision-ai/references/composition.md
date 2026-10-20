# Delta Profile Composition

- [Model](#model)
- [Select the foundation](#select-the-foundation)
- [Compute the delta](#compute-the-delta)
- [UI configuration](#ui-configuration)
- [Clarification gate](#clarification-gate)
- [Artifact contract](#artifact-contract)
- [Resolve](#resolve)
- [Validate](#validate)
- [Sources](#sources)

## Model

A Foundation is one reviewed, current developer profile selected as the closest
starting point for a request. A Delta Profile is the smallest environment and
optional Compose service-definition patch applied to exactly one Foundation.
The Foundation remains in place; the delta does not copy its `.env`,
`overrides.env`, Compose files, configs, or skill bundle.

Current Foundations:

- `base`
- `alerts`
- `lvs`
- `search`

Use only developer profiles here. The `warehouse` industry blueprint is **not**
composed as a delta — it selects one of nine validated service lists — and is
documented end to end in [`profiles/warehouse.md`](profiles/warehouse.md),
including its own env layers, artifacts and resolve pipeline. Do not route
warehouse or another industry profile through the rules below.

## Select the foundation

1. Translate the request and any eval specification into required and forbidden
   capabilities.
2. Compare those requirements with all files under `profiles/`.
3. Prefer an exact capability match and use stock mode.
4. Otherwise minimize service-set additions, removals, and definition changes,
   in that order.
5. Do not break a Foundation tie by guessing; an equally small delta between
   two profiles is a clarification-gate blocker (see below).

The selected profile's checked-in `overrides.env` is authoritative for its
Profile Service Set. The copied list in `profiles/` is a routing aid and must be
checked against source before writing a delta.

## Compute the delta

Start with the Foundation's effective `COMPOSE_PROFILES`.

- Add an existing service by adding its exact, self-named profile key.
- Remove a service by omitting its exact key.
- Keep dynamic NIM keys in their existing form:
  `llm_${LLM_MODE}_${LLM_NAME_SLUG}` and
  `vlm_${VLM_MODE}_${VLM_NAME_SLUG}`.
- Never invent an umbrella profile, a `bp_developer_*` name, or a `*-patched`
  name.
- For a genuinely new service, use the user- or source-provided service key as
  its self-profile. Do not derive a separate aggregate profile name.
- Read every selected owner's `Required peers`. Add a peer only when it is not
  already present and the requested capability needs it.
- Put user-configurable values in the env delta. Do not copy default values that
  are unchanged.

**Harness-only guard.** Before running the generic pruning pass below, compare
the requested capability set with the Foundation. If they are identical and
the only change is a Q3 answer, this is a harness-only delta. Preserve the
Foundation list and remove only the agent-owned keys: `vss-agent`, `phoenix`
(its trace sink), the `llm_*` key when no remaining key consumes the LLM
(`lvs-server` does), plus an unrequested `vss-va-mcp` when the existing VA-MCP
rule applies. Set `REQUESTED_PROFILES` to the
explicitly requested profile keys, including an empty value when none were
named. Do not interpret Q3 **no** as headless and do not run forward-closure
pruning: the host-side `vss` CLI is the driver, while UI, ingress, the VLM,
Redis, VIOS, and all other Foundation services stay.
Run this exact check before artifact generation:

```bash
uv run "$REPO/skills/vss-build-vision-ai/scripts/resolve_service_graph.py" \
  --foundation "$FOUNDATION_PROFILES" \
  --final "$FINAL_PROFILES" \
  --requested "${REQUESTED_PROFILES:-}"
```

An explicit request for headless operation or capability removal does not meet
this guard and proceeds through normal pruning.

The Foundation is a starting graph to trim, not a floor to inherit. A delta is
symmetric: after adding requested owners and their peers, prune every Foundation
service that no requested capability needs. Compute the reachable set by forward
closure: start from the requested capabilities, resolve each to its owner, then
follow that owner's `Required peers` transitively. Remove every Foundation
service outside that closure, including a peer whose only consumer was removed
(for example an LLM required solely by an orchestration owner that is not
requested). A service is retained only because a requested capability reaches it,
never because the Foundation happened to ship it.

When more than one requested capability maps to the same owner, converge on a
single instance (one service key, one variant, one config), never two variants
of one owner for the same role (for example, one detector feeding two pipelines).
Where a shared singleton's behavior lives in a mounted config, that config
converges too: decide it against the build's service set rather than inheriting
a Foundation's by default (VIOS's `notification_config.json`, for one).
If that owner's output feeds another service, align the consumer's config to the
variant you selected, not to the one its Foundation shipped. Owner contracts
state which owners are singletons, what output each fixes, and which consumer
keys track it; read them before merging configs.

Service activation alone is never a Compose-definition change.

## UI configuration

For stock and delta builds with `vss-ui`, default
`NEXT_PUBLIC_SIDEBAR_CHAT_CHAT_UPLOAD_FILE_ENABLE=false` unless explicitly
requested otherwise. Write it to `override.env` when it differs from the
inherited value, and verify it in `resolved.yml`.

### UI subtitle

When `vss-ui` is enabled, derive `NEXT_PUBLIC_APP_SUBTITLE` from the final
`COMPOSE_PROFILES`. In order, include `Alerts - CV` or `Alerts - VLM` for
`alert-bridge` according to `MODE`, `LVS` for `lvs-server`, and `Search` for
`vss-search-analytics-2d-fusion`. Join enabled labels with ` + ` inside
`Vision (...)`: real-time VLM alerts plus summarization becomes
`"Vision (Alerts - VLM + LVS)"`. With none of those owners, use `Vision (Base)`
if `rtvi-vlm` remains, otherwise `Vision`. Write the result to `override.env`
only when it differs from the inherited subtitle; honor an explicit user
choice. Check that `resolved.yml` gives `vss-ui` the resulting value.

## Clarification gate

This gate is generic: it settles any resolution blocker the rules cannot,
not just Foundation choice. Resolution is deterministic: parse the request into
required and excluded capabilities, map each to an owner, select the Foundation,
then compute the delta by forward closure, prune, and singleton convergence. Run
that pass in full first. Only when it leaves a blocker the rules cannot settle,
ask exactly one structured clarification. The gate is a last resort that
replaces a reasoning loop; it is never an early exit or a substitute for
resolution the rules can do.

Ask only for a blocker in this closed set:

- **Unmapped capability**: a required capability resolves to no owner.
- **Ambiguous owner**: a capability maps to more than one owner and no routing
  cue in `profiles/` or `services/` disambiguates.
- **Foundation tie**: two Foundations have an equally small delta.
- **Singleton conflict**: two requested capabilities force two incompatible
  variants of one singleton owner (for example two detector families feeding one
  consumer taxonomy).
- **Capability contradiction**: a requested capability overlaps an excluded one.

Anything outside this set is resolved with the rules, not a question. Do not ask
before the pass completes, do not ask about a decision the rules already settle
(the strictly-closest Foundation, peers fixed by `Required peers`, pruning
outcomes), and do not guess past a blocker or improvise a service set.

Bound the interaction so it cannot loop:

- Make one resolution attempt. If blockers remain, stop and ask; never re-run
  the same resolution expecting a different result.
- Batch every open blocker into one clarification. Do not ask serially.
- After the answer, re-resolve once. If the answer creates a new blocker, ask
  once more and say the answer introduced it. Never repeat an answered question.

State the clarification exactly, with no hedging:

1. Lead with what is resolved (the Foundation and the mapped capabilities and
   owners), so the question is bounded.
2. State each unresolved item as one closed question, never an open-ended one.
3. Offer concrete options: the candidate owners, service keys, or Foundations.
4. Mark a recommended option when the rules lean one way, with a one-clause
   reason.
5. Ask only what is unresolved. Do not restate settled decisions or hedge
   ("I think", "maybe", "possibly").

## Artifact contract

Always write:

```text
_builds/<name>/
├── override.env
├── compose.yml
├── resolved.yml
└── patches/         # optional; service definitions this build changes or adds,
                     # and the payload files it mounts
```

Nothing else sits beside the three primary files: every build-local payload
lives in `patches/`, whether a `.yml` patch mounts it or an environment variable
selects it (see the patch rules below).

`<name>` is a filesystem label supplied by the user or a neutral description of
the requested build. It is never a Compose profile. If the user supplies no
`<name>`, derive a neutral one rather than writing elsewhere.

The only writable location is `_builds/<name>/`. Never create or edit files under
`deploy/docker/**` or a `dev-profile-*` Foundation directory, and never write
`resolved.yml` outside `_builds/<name>/`. The env artifact is always named
`override.env`.

`override.env` contains:

1. `FOUNDATION=<base|alerts|lvs|search>`.
2. The full effective `COMPOSE_PROFILES` after additions and removals.
3. Every customized environment value and every Foundation value transitively
   derived from it. Do not repeat unrelated Foundation defaults.

Compose expands each env file as it is read; values expanded in a Foundation
file are not recomputed when a later file changes one of their inputs.
Therefore, materialize the complete dependent-value closure in `override.env`.
For example:

- changing `VSS_APPS_DIR` also requires the effective `VST_CONFIG_PATH`,
  `SDR_CONTROLLER_CONFIG_PATH`, and any selected profile-specific config paths;
- changing `HOST_IP` also requires the effective `EXTERNAL_IP`,
  `VSS_PUBLIC_HOST`, public VIOS/Agent URLs, and selected UI/API endpoints.
- choosing Alerts `MODE=2d_vlm` requires the mode-selected Manage Alerts UI
  flags from [`profiles/alerts.md`](profiles/alerts.md#mode-selected-manage-alerts-editors):
  real-time alerts enabled and CV verification disabled. The Compose defaults
  enable both editors; setting `ALERT_AGENT_ALWAYS_ON=true` alone does not
  change them.

Find the exact closure by following variable references in the selected
Foundation's `.env` and `overrides.env`; do not assume a later primitive
override will update an earlier derived value.

`compose.yml` is the build entrypoint. With no service-definition changes:

```yaml
include:
  - path:
      - ../../deploy/docker/compose.yml
```

When a service definition must change, append its patch after the root Compose
file:

```yaml
include:
  - path:
      - ../../deploy/docker/compose.yml
      - ./patches/<service>.yml
```

The ordered `path` list merges the patch into the included root model before
including it in the build. Use Docker Compose 2.20.3 or newer.

Create a `.yml` patch under `patches/` only when:

- a requested service does not exist in the root Compose graph; or
- an existing service definition must change in a way Compose env interpolation
  cannot express.

A patch may contain only the changed or new `services:` entries under the
canonical service key. The patch **filename** must be
`patches/<canonical-COMPOSE_PROFILES-key>.yml` — the same key the service uses
in `COMPOSE_PROFILES`, not a shortened or generic name. Do not copy unchanged
services, volumes, networks, or profile files. Add multiple patch paths after
the root file when multiple service definitions change.

A build-local payload is referenced by an absolute
`${BUILD_DIR}/patches/<file>` source, with `BUILD_DIR` set to the build's
absolute path in `override.env`. A relative `./` source would resolve against the
root Compose file's directory (`deploy/docker/`), not the patch's — the ordered
`path:` list sets the included model's project directory from its first entry — so
Docker would create a stray root-owned directory there at `up`. A checked-in repo
file a patch mounts is likewise bound by its absolute repo path, never copied into
the build.

Most payloads sit beside the patch that mounts them — a curated `haproxy.cfg`
next to `patches/vss-haproxy-ingress.yml`. Some are instead selected by an
environment variable the root Compose file already interpolates into a mount and
need no patch: a projected `notification_config.json` chosen by
`VST_NOTIFICATION_CONFIG_PATH` ([`services/vios.md`](services/vios.md)). A
payload with no `.yml` beside it is expected, not an error.

`resolved.yml` is the fully interpolated output of `docker compose config`.
Resolution filters the root graph through `COMPOSE_PROFILES`, so only the
effective service set and its dependencies are serialized. Normalization then
removes their now-redundant service profile gates. It is the exact, standalone
Compose model used directly for validation, deployment, readiness, and teardown:
`config` bakes the `name`, `env_file`, and interpolation, so it needs no
`--env-file`. Pass the resolve env layers only to `config`, never to `up`, `ps`,
or `down`, and deploy with `pull --ignore-buildable && up -d --build`
(see [`deployment.md`](deployment.md)).

All three primary files are required in stock and delta mode. `_builds/` is
gitignored because `override.env` and `resolved.yml` can contain credentials.
Keep them local and never commit them.

## Resolve

From the repository root:

```bash
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"
FOUNDATION="$(sed -n 's/^FOUNDATION=//p' "$BUILD_DIR/override.env")"
FOUNDATION_DIR="$REPO/deploy/docker/developer-profiles/dev-profile-$FOUNDATION"

if command -v uv >/dev/null 2>&1; then
  VSS_SKILL_PY=(uv run)
elif python3 - <<'PY' >/dev/null 2>&1
import yaml
PY
then
  VSS_SKILL_PY=(python3)
else
  echo "Install uv or install PyYAML for python3 before normalizing resolved.yml." >&2
  exit 1
fi

env_args=(
  --env-file "$REPO/deploy/docker/containers.env"
  --env-file "$FOUNDATION_DIR/.env"
  --env-file "$FOUNDATION_DIR/overrides.env"
  --env-file "$BUILD_DIR/override.env"
)

# containers.env derives every managed image tag from VSS_CONTAINER_TAG and is
# expanded before override.env, so only an exported value reaches those tags.
VSS_CONTAINER_TAG="${VSS_CONTAINER_TAG:-$(
  sed -n 's/^VSS_CONTAINER_TAG=//p' "$BUILD_DIR/override.env"
)}"
tag_args=()
if [ -n "$VSS_CONTAINER_TAG" ]; then
  export VSS_CONTAINER_TAG
  tag_args=(--expect-container-tag "$VSS_CONTAINER_TAG")
fi

docker compose "${env_args[@]}" \
  -f "$BUILD_DIR/compose.yml" \
  config --no-consistency > "$BUILD_DIR/resolved.yml"

"${VSS_SKILL_PY[@]}" "$REPO/skills/vss-build-vision-ai/scripts/normalize_resolved_yml.py" \
  "$BUILD_DIR/resolved.yml"

"${VSS_SKILL_PY[@]}" "$REPO/skills/vss-build-vision-ai/scripts/validate_resolved_yml.py" \
  "$BUILD_DIR/resolved.yml" --repo-root "$REPO" "${tag_args[@]}"
```

`docker compose config` writes the resolved model to stdout and its warnings and
errors to stderr. Only stdout may reach `resolved.yml`, so write it with the `>`
redirect shown: never merge the streams (`2>&1`, `&>`, or a combined `tee`), and
never reconstruct `resolved.yml` from the command's captured output — an agent
shell interleaves stderr into that capture, so the warnings pollute the YAML even
with no explicit merge. Leave stderr on the terminal: do not silence it
(`2>/dev/null`) or divert it to a build-directory file (`2> resolve.err`) —
stderr is transient diagnostics, never a persisted build artifact. Then act on what it reports: a non-zero exit code
means resolution failed (a required variable, missing file, or invalid
definition) and must be fixed before continuing; on success, the
`variable is not set. Defaulting to a blank string.` lines are informational:
unset optional knobs are expected, but scan them for any value that belonged in
`override.env`'s dependent-value closure and set it if so.

The env layers are ordered from broad defaults to build-specific customization;
later values override earlier values. Regenerate `resolved.yml` whenever
`override.env`, `compose.yml`, a patch, or a Foundation source changes.
Normalization removes only optional dependency references to services omitted
by profile filtering, then removes service profile gates from the already
filtered model. It fails rather than remove a missing required dependency.
If validation reports real unresolved `${...}` Compose interpolation, do not
deploy the raw output. Add only the missing concrete value or derived value to
`override.env`, regenerate `resolved.yml` from the same ordered env layers, and
rerun normalization and validation. Escaped container-shell variables such as
`$${HOST_IP}`, `$${NUM_STREAMS}`, or `$${VAR:-default}` are valid in
`resolved.yml` and must not be counted as unresolved Compose interpolation.

That layer ordering is why the common container image tag is exported rather
than left to `override.env`: `containers.env` has already derived every
`VSS_*_TAG` from `VSS_CONTAINER_TAG` by the time the build layer is read, and
the shell environment outranks every `--env-file`. A build that records the tag
but resolves without the export gets `vss-agent-ui` — the one image whose
Compose default reads the common knob directly — on the requested tag and every
other image on `develop-latest`; `--expect-container-tag` fails on exactly that
mixture. Service-specific `VSS_*_TAG` pins and `VSS_CONTAINER_TAG_SUFFIX` still
supersede or augment the common tag; see the SBSA rules in
[`sizing.md`](sizing.md).

## Validate

```bash
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"

if command -v uv >/dev/null 2>&1; then
  VSS_SKILL_PY=(uv run)
elif python3 - <<'PY' >/dev/null 2>&1
import yaml
PY
then
  VSS_SKILL_PY=(python3)
else
  echo "Install uv or install PyYAML for python3 before validating resolved.yml." >&2
  exit 1
fi

"${VSS_SKILL_PY[@]}" "$REPO/skills/vss-build-vision-ai/scripts/validate_resolved_yml.py" \
  "$BUILD_DIR/resolved.yml" --repo-root "$REPO"

docker compose -f "$BUILD_DIR/resolved.yml" config --quiet
docker compose -f "$BUILD_DIR/resolved.yml" config --services
docker compose -f "$BUILD_DIR/resolved.yml" config --images
```

Then verify:

- `FOUNDATION` names one current developer profile.
- Every `COMPOSE_PROFILES` token exists in the current Compose graph after env
  interpolation.
- The resolved service list is non-empty.
- Added capability owners and their required peers resolve.
- Removed services do not resolve.
- Except in a harness-only delta, every retained service is transitively
  required by at least one requested capability; no orphaned Foundation
  carryover survives ordinary capability pruning. A harness-only delta instead
  preserves every unrelated Foundation key by invariant.
- A shared singleton owner resolves to exactly one variant, and every consumer
  config that keys on that owner's output (class-label taxonomy and casing,
  topic names) matches the resolved variant; no consumer filters on a taxonomy
  the resolved owner does not emit.
- No unrequested service definition is present in a patch.
- Any patch contains only changed or new service entries.
- `resolved.yml` contains no real unresolved `${...}` Compose interpolation and
  every selected service's environment is filled in. Escaped `$${...}` variables
  are container-shell expressions, not Compose interpolation failures.
- Every credential key the selected mode requires (see `credentials.md`
  Required By Mode) resolves to a non-empty literal. A required credential that
  resolved empty is a blocker: set it and re-resolve, never deploy the empty
  value, since a baked `''` cannot be supplied at deploy time. Keys the mode
  does not require (for example `HF_TOKEN` off edge) may be empty.
  `validate_resolved_yml.py` enforces this — it fails when `nvcr.io/`/`ngc:`
  artifacts are present but `NGC_API_KEY`/`NGC_CLI_API_KEY` resolve empty; pass
  `--required-secret KEY` for other mode-required keys.
- With an NGC key the non-empty check is necessary but not sufficient: run the
  `credentials.md` Artifact Entitlement Probes against the exact baked `nvcr.io/`
  images and `ngc:` paths. A `401`/`403`/missing-repo result is a blocker — a
  Validate gate on every build, deploy or not.
- `resolved.yml` contains no stock sentinels such as
  `/path/to/deploy/docker` or `<HOST_IP>`.
- Every checked-in bind source exists and a file target is not backed by a
  directory. This is a validation check only: do not create placeholder files
  or directories under `deploy/docker/` to satisfy it.
- A service governed by a mounted config file has that config reconciled to the
  requested capabilities and ingestion mode — e.g. a mounted analytics JSON's
  live-vs-simulation mode and processor gates must match the request, as
  env-delta resolution cannot express them — per its owner contract, not left at
  a source profile's default.
- The resolved services and knobs satisfy every observable check from the user
  request or eval specification.

## Sources

- `deploy/docker/compose.yml`
- `deploy/docker/containers.env`
- `deploy/docker/developer-profiles/dev-profile-*/.env`
- `deploy/docker/developer-profiles/dev-profile-*/overrides.env`
- `deploy/docker/developer-profiles/dev-profile-*/compose.yml`
- `deploy/docker/services/**/compose.yml`
- `deploy/docker/services/**/compose.yaml`
- `deploy/docker/services/**/docker-compose.yaml`
- `deploy/docker/services/**/docker-compose.yml`
