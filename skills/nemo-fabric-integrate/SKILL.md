---
name: nemo-fabric-integrate
description: Use this skill when integrating NVIDIA NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer's own application, job, or deployment config into an in-memory FabricConfig, choosing the single-invocation convenience API or an explicitly started runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Integrate NVIDIA NeMo Fabric Through The Python SDK

Use this skill when a consumer codebase — an application, service, evaluation
harness, or platform — needs to run agent harnesses through NeMo Fabric's typed
Python SDK. The consumer owns its own configuration object and translates it
into an in-memory `FabricConfig`; NeMo Fabric owns adapter selection, the runtime
lifecycle, and normalized results.

## Integration Boundary

Use the public, in-memory contract. These rules keep a consumer integration
supported and upgrade-safe:

- Import only from the public `nemo_fabric` package. Never import `_native` or
  any adapter-internal module.
- Build configuration as a typed `FabricConfig` in memory and pass it directly to
  NeMo Fabric. Create every deployment or evaluation variant with ordinary Python
  functions and `model_copy(deep=True)`. A platform integration can serialize
  the typed config inside a private transient run specification when it crosses
  a process boundary; that transport is not a public authoring format.
- Let NeMo Fabric own harness control. Do not reimplement start, invoke, or stop
  logic, and do not manage adapter threads, sessions, or processes directly.
- Treat `runtime_id`, `invocation_id`, and `request_id` as opaque correlation
  strings, not parsable or reusable state.

Refer to [config-mapping.md](references/config-mapping.md) for how to translate a
consumer config object into `FabricConfig`, and for the full list of mechanics
that stay hidden behind this boundary.

## Install And Set Up The Environment

The consumer or its execution environment owns installation; NeMo Fabric validates
runtime assumptions but never installs harnesses or credentials at run time.

- NeMo Fabric supports Python 3.11 through 3.14. Use Python 3.11 through 3.13
  for Hermes Agent; the Harbor integration requires Python 3.12 or later.
- Install the runtime with `uv pip install nemo-fabric` (add the `harbor` extra
  for the Harbor integration). Refer to the
  [installation guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx).
- Select the harness adapter through `HarnessConfig.adapter_id`. To install the
  NeMo Fabric runtime, adapter, and supported harness in one environment, use
  `nemo-fabric[claude]`, `nemo-fabric[codex]`,
  or `nemo-fabric[deepagents]`.
- Hermes Agent 0.20 and later is no longer installable from PyPI. Follow the
  [Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation),
  then install the `nemo-fabric[hermes-agent]` package
  into the Python environment that runs Hermes Agent. These packages do not
  install Hermes Agent.
- In a separate adapter environment, install
  `nemo-fabric-adapters-<adapter>[harness]`. This installs the adapter and
  supported harness dependencies without the NeMo Fabric runtime. Use `full`
  instead when that adapter package provides package-installable optional
  integrations.
- Point the runtime to a separate adapter environment with `ADAPTER_PYTHON`.
  Use matching NeMo Fabric release versions for the runtime and adapter package
  unless a different pairing has been explicitly validated.
- If the adapter environment already manages a compatible harness, install the
  bare `nemo-fabric-adapters-<adapter>` distribution. Bare adapter
  distributions contain only adapter-owned runtime dependencies.
- LangChain Deep Agents and Hermes Agent adapter packages provide `relay` and
  include the NeMo Relay Python package in `full`. The Hermes Agent extras do
  not install Hermes Agent. Claude and Codex do not provide `relay`; their
  `harness` and `full` extras install the supported `nemo-relay` CLI alongside
  the harness SDK.
- Provide model credentials through environment variables named by the config
  (`ModelConfig.api_key_env`), never as literals in code.
- Confirm the native extension is importable; SDK calls raise
  `FabricNativeUnavailableError` when it is missing.

## Build The Typed Config From Consumer Config

Map the consumer's application, job, or deployment object into a `FabricConfig`
with the public models and helper methods:

```python
from nemo_fabric import (
    FabricConfig,
    HarnessConfig,
    InstructionConfig,
    InstructionsConfig,
    MetadataConfig,
    ModelConfig,
    RuntimeConfig,
    ToolsConfig,
)


def to_tools_config(job) -> ToolsConfig | None:
    enabled = job.enabled_tools
    blocked = list(job.blocked_tools)
    if enabled is None and not blocked:
        return None
    return ToolsConfig(
        enabled=None if enabled is None else list(enabled),
        blocked=blocked,
    )


def to_fabric_config(job) -> FabricConfig:
    config = FabricConfig(
        metadata=MetadataConfig(name=job.name),
        harness=HarnessConfig(adapter_id=job.adapter_id, resolution="preinstalled"),
        models={
            "default": ModelConfig(
                provider=job.provider,
                model=job.model,
                api_key_env=job.api_key_env,
                base_url=job.base_url,
            )
        },
        instructions=(
            InstructionsConfig(
                system=InstructionConfig(
                    content=job.system_instruction,
                    mode=job.system_instruction_mode,
                ),
            )
            if job.system_instruction is not None
            else None
        ),
        runtime=RuntimeConfig(
            input_schema="chat",
            output_schema="message",
            timeout_seconds=job.timeout_seconds,
            max_turns=job.max_turns,
        ),
        tools=to_tools_config(job),
    )
    config.add_skill_path(job.skill_dir)
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="${GITHUB_MCP_URL}",
        exposure="harness_native",
    )
    return config
```

- Shape capabilities with `ToolsConfig`, `add_tool_definition`, `block_tools`, `add_skill_path`,
  `remove_skill_path`,
  `add_mcp_server`, `remove_mcp_server`, and `enable_relay`.
- Use `add_tool_definition` only when the selected adapter accepts
  `tools.definitions` and publishes a `tool_definition_schema`.
- Use a restricted `allowed_tools` list or non-empty `blocked_tools` on
  `add_mcp_server` only when the selected adapter declares both `mcp` and
  `mcp.tool_filters`. An unfiltered server requires only `mcp`.
  `allowed_tools=None` exposes every discovered tool, while an empty list
  exposes none; blocked tools are removed after applying that allowlist. Tool
  names must be non-blank, and planning rejects a tool that appears in both
  lists.
- Configure MCP authentication only when the selected adapter declares
  `mcp.auth.oauth2` or `mcp.auth.service_account`, matching the authentication
  type.
- Create deployment or evaluation variants with `model_copy(deep=True)` and
  ordinary Python functions; each copy plans and runs independently.
- Pass `base_dir=...` to any `Fabric` call when the config uses relative paths,
  so skills, workspaces, and artifacts anchor to the consumer's own layout.

The repository [`code_review_agent` example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent)
shows this pattern end to end with complete Hermes Agent, Codex, Deep Agents,
environment, MCP, and telemetry variants. Reuse it rather than duplicating config
construction.

## Choose A Lifecycle

Pick the smallest lifecycle the consumer needs:

- **Single invocation** — one input, no retained state after the call.
  `await Fabric().run(config, input=...)` runs the full start, invoke, and stop
  cycle and returns a `RunResult`. Pass
  `request=RunRequest(...)` instead of `input=...` when the invocation needs a
  caller-owned request ID or context (the two are mutually exclusive).
- **Stateful runtime** — ordered turns over one logical harness lifecycle. Start it with
  `start_runtime(...)` and use the returned `Runtime` as an async context
  manager so cleanup runs on exit — shutdown is attempted, not guaranteed
  (`stop()` can raise `FabricRuntimeError`; see Consume Results And Handle
  Errors). A runtime accepts one active invocation at a time; overlapping calls
  raise `FabricStateError`.
- **Native OpenAI stream** — adapter-native OpenAI Chat Completions chunks plus
  a separate terminal normalized result. Check
  `runtime.supports_openai_streaming`, call
  `runtime.invoke_openai_stream(...)`, iterate the returned
  `OpenAIInvokeStream`, and then await `stream.result()`. The selected adapter
  descriptor must declare `capabilities.streaming`. Each yielded mapping has
  `object == "chat.completion.chunk"`; an empty stream is valid. If iteration
  stops early, call `await stream.aclose()` to drain without cancelling the
  target invocation. This path does not require NeMo Relay or
  `streaming=True`.
- **NVIDIA NeMo Relay stream** — live, raw ATOF records plus a terminal normalized
  result. Enable NeMo Relay, pass `streaming=True` to `start_runtime(...)`, call
  `runtime.invoke_stream(...)`, iterate the returned `InvokeStream`, and then
  await `stream.result()`. Iteration ending does not indicate invocation
  success; invocation exceptions raise from `result()`, while harness-reported
  failures remain normalized `RunResult` values. If iteration stops early,
  call `await stream.aclose()` before starting another turn. `aclose()` waits
  for the turn to finish; it does not cancel the harness invocation. The SDK
  intentionally exposes only ATOF records generated by NeMo Relay. This path is
  independent of native OpenAI streaming. The listener
  limits each record to 1 MiB and its queue to 1,024 records or 16 MiB of
  encoded data. It correlates records through the NeMo Fabric request ID for
  in-process harnesses. For gateway harnesses, it uses the NeMo Relay turn-scope
  role and 1-based turn index. It yields only the matched scope tree. Delayed
  prior-turn records therefore do not enter the next stream. If gateway and
  NeMo Fabric turn sequences do not align, the SDK discards the uncorrelated records
  and emits a `RuntimeWarning` after natural stream exhaustion. The listener
  binds to `NEMO_FABRIC_STREAMING_HOST`, which defaults to `127.0.0.1`.
  Override it when the gateway must reach the SDK through another network
  interface, and restrict access to that interface. If async iteration reaches
  its post-turn drain timeout without a NeMo Relay connection, or receives data
  without a matching turn root, the SDK emits one `RuntimeWarning` for that
  failure mode; callers that only await `stream.result()` do not run that
  warning check. The SDK also warns when a NeMo Relay upload terminates before
  completing its chunked request body because yielded records can be incomplete.
  The `streaming=True` flag does not enable NeMo Relay by itself. Without
  `streaming=True`, startup leaves the NeMo Relay configuration unchanged and
  does not inject the SDK-owned ATOF stream sink.

The selected adapter owns the execution topology. The bundled Claude, Codex,
Deep Agents, and Hermes Agent adapters retain their native client, graph/checkpointer,
or agent/database inside one local host for the full runtime. Local `process`
and `python` adapters use this host lifecycle; consumers do not select another
local execution mechanism in `FabricConfig`. Do not replay an invocation after
a runtime failure. Stop the failed runtime and explicitly start a new one
according to the application's retry policy.

The lifecycle fragment below shows the available forms. It assumes the caller
has already set `config = to_fabric_config(job)` and chosen `base`, as described
in the configuration example above:

```python
import asyncio

from nemo_fabric import Fabric


async def main() -> None:
    fabric = Fabric()

    # Single invocation
    result = await fabric.run(config, base_dir=base, input="Review the changes.")

    # Multi-turn
    async with await fabric.start_runtime(config, base_dir=base) as runtime:
        first = await runtime.invoke(input="Inspect the repository")
        second = await runtime.invoke(input="Now review the latest patch")

    # Adapter-native OpenAI Chat Completions chunks
    async with await fabric.start_runtime(config, base_dir=base) as runtime:
        if runtime.supports_openai_streaming:
            stream = runtime.invoke_openai_stream(input="Review the latest patch")
            async for chunk in stream:
                print(chunk)
            openai_streamed_result = await stream.result()

    # NeMo Relay streaming
    streaming_config = config.model_copy(deep=True).enable_relay()
    async with await fabric.start_runtime(
        streaming_config,
        base_dir=base,
        streaming=True,
    ) as runtime:
        stream = runtime.invoke_stream(input="Review the latest patch")
        async for record in stream:
            print(record)
        streamed_result = await stream.result()


asyncio.run(main())
```

NeMo Fabric owns no application scheduling queue, worker pool, retry policy, or
global concurrency policy. Each runtime still permits only one active
invocation; start independent runtimes for parallel work. The NeMo Relay
streaming path uses an internal bounded transport queue and TCP backpressure
only to carry one invocation's ATOF records. Treat `stream.result()` as
authoritative, and reconstruct nested work from ATOF `uuid` and `parent_uuid`
fields rather than stream order.

For native OpenAI streaming, the SDK owns the authenticated loopback HTTP
transport, chunked NDJSON framing, and correlation values. Consumer code
supplies no listener or credentials. The adapter executes exactly one
invocation, and the terminal `RunResult` remains separate from the chunk stream.
Fully consume the stream or call `await stream.aclose()` before starting another
turn. Awaiting `stream.result()` also drains and discards unread native OpenAI
chunks, so consume the iterator first when the application needs every chunk.

## Validate Before Running

Resolve and diagnose before spending work on a runtime, especially in a new
environment or before relying on an optional capability:

```python
fabric = Fabric()
plan = fabric.plan(config, base_dir=base)             # sync: adapter + capabilities
report = await fabric.doctor(config, base_dir=base)   # async: preflight checks

print(plan.adapter.adapter_id, report.status)
```

- Use `plan(...)` to confirm adapter selection and capability routing before
  running. Planning validates `harness.settings` against the exact resolved
  Adapter Descriptor and, when present, `workflow.settings` against the exact
  resolved Adapter Target Descriptor.
- Use `doctor(...)` to check adapter availability, resolution, environment
  context, and declared requirements such as required environment variables. Its
  aggregate `status` is `pass`, `warn`, or `fail`. Invalid, unknown, or
  misspelled adapter settings fail before diagnostics or runtime startup. A
  resolved descriptor without a settings schema accepts only an empty settings
  map.

## Consume Results And Handle Errors

Every invocation that reaches the adapter boundary returns a normalized
`RunResult`, even when the harness invocation itself failed. Inspect the failure
fields before reading output:

```python
result = await fabric.run(config, base_dir=base, input="Review the changes.")

if result.status == "succeeded":
    use_output(result.output, result.artifacts, result.telemetry)
else:
    handle_failure(result.status, result.error, result.events)  # failed, cancelled, ...
```

- Treat `status == "succeeded"` as the only success. Other terminal values
  (`failed`, `cancelled`) are unsuccessful, so branch on `status`, not on
  `error`. Read `status`, `error`, and `events` before processing `output`.
- Capture `artifacts` and `telemetry` references as the returned evidence for
  platforms and evaluations. Store and log `runtime_id`, `invocation_id`, and
  `request_id` separately as opaque strings.
- Catch `FabricError` subclasses for lifecycle failures that prevent a
  normalized result: `FabricConfigError`, `FabricCapabilityError`,
  `FabricRuntimeError`, `FabricStateError`, and `FabricNativeUnavailableError`.
- The consumer owns retries and failure policy; NeMo Fabric does not retry by
  default. `run(...)` and `async with` runtimes attempt cleanup automatically,
  so prefer them over manual `stop()` — but shutdown is not guaranteed: `stop()`,
  including the automatic call when an `async with` block exits, can raise
  `FabricRuntimeError`. On a normal exit that error propagates; after an
  invocation error the cleanup failure is attached to the original exception. Be
  ready to handle a shutdown failure.

Refer to [results-and-errors.md](references/results-and-errors.md) for the full
result-field and error inventory, and
[sdk-api-inventory.md](references/sdk-api-inventory.md) for when to use each
`Fabric` and `Runtime` method.

## Test And Validate The Integration

- Write focused integration tests that build the consumer's `FabricConfig`,
  assert `plan(...)` selects the expected adapter and capabilities, and — where
  a harness and credentials are available — run one invocation and assert the
  `RunResult` status and evidence.
- `plan(...)` is credential-free — use it as the CI gate that validates adapter
  selection and capability routing without a model or secrets. `doctor(...)` also
  runs without calling a model, but it checks declared environment requirements
  (such as required API-key variables) and returns `fail` when they are unset, so
  run it where the environment is provisioned and read its per-check results.
- Run the consumer project's own build and test commands. For a source checkout
  of NeMo Fabric, `just build-all` rebuilds the native extension and
  `just test-python` runs the Python suite.
- Confirm the typed config is passed directly to NeMo Fabric and no non-public
  imports were added.

## Checklist

- [ ] The consumer config object is translated directly into an in-memory `FabricConfig`.
- [ ] Only public `nemo_fabric` symbols are imported; no `_native` or adapter internals.
- [ ] The consumer config is built in memory and passed directly to NeMo Fabric.
- [ ] The right lifecycle is chosen: `run(...)` for a single invocation,
  `start_runtime(...)` with `async with` for multi-turn,
  `invoke_openai_stream(...)` for descriptor-gated OpenAI chunks, or
  `invoke_stream(...)` for raw NeMo Relay ATOF.
- [ ] `plan(...)` and `doctor(...)` validate adapter selection, capabilities, and environment before execution.
- [ ] Installation, adapter dependencies, and credentials are owned by the environment, not consumer code.
- [ ] `RunResult` status, error, and events are inspected before output; artifacts and telemetry are captured.
- [ ] `FabricError` subclasses are handled, including a `FabricRuntimeError` raised by shutdown; cleanup is delegated to `run(...)` or `async with` (attempted, not guaranteed).
- [ ] Correlation IDs are stored and logged as opaque strings.
- [ ] Focused integration tests pass and NeMo Fabric validation (`plan`/`doctor`, tests) succeeds.

## Related Documentation

Link to these canonical sources instead of duplicating them:

- [Python SDK guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx)
- [NeMo Fabric overview](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/about-nemo-fabric/overview.mdx) and
  [installation guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx)
- Generated API reference (public API index; the installed `nemo_fabric` type
  stubs are authoritative for exact signatures, fields, and defaults):
  [client](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md),
  [runtime](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md),
  [native OpenAI streaming](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.openai_streaming.md),
  [Relay streaming](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.streaming.md),
  [models](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.models.md),
  [types](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.types.md),
  [errors](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.errors.md)
- Canonical in-memory config example:
  [examples/code_review_agent](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent)
- Platform and evaluation-harness integration:
  [examples/harbor](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/harbor) and
  [nemo_fabric.integrations.harbor](https://github.com/NVIDIA/NeMo-Fabric/tree/main/sdk/python/nemo-fabric-runtime/src/nemo_fabric/integrations/harbor).
  Harbor constructs a typed config from explicit agent inputs and transports it
  inside a private transient run specification at the task-process boundary.
  Follow the code-review example for consumer integration code; Harbor's
  transport representation is an internal process-boundary contract.
