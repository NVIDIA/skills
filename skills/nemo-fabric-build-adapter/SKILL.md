---
name: nemo-fabric-build-adapter
description: Build, migrate, review, and maintain third-party NVIDIA NeMo Fabric adapters against the public adapter contract. Use when creating adapter or target descriptors, mapping AgentConfig into an agent harness or custom-agent runtime, implementing start/invoke/stop, declaring schemas and capabilities, packaging discovery metadata, or assessing adapter conformance. Do not use for consumer applications that only call the NVIDIA NeMo Fabric SDK.
---

# Build an NVIDIA NeMo Fabric Adapter

Build against the published southbound contract. Keep the adapter thin: let
NeMo Fabric own planning and consumer-facing behavior, and let the adapter own
only target translation and lifecycle state.

## Read the Contract

Read the current
[adapter contract](https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs/adapter-contract)
before changing code. Start with the overview, choose an integration shape,
then follow the numbered descriptor, configuration, execution, results,
registration, and verification stages. Read the custom-agent page when the
target loads application-defined agents or workflows. Read the optional native
OpenAI streaming page only when the adapter claims that capability.

Use the committed
[adapter-contract JSON Schemas](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas/adapter-contract)
or the
schemas installed with the matching NeMo Fabric release for exact wire shapes.
Do not reconstruct a schema from examples or copy field lists into adapter
code.

## Establish the Boundary

Establish the adapter boundary before defining its descriptor:

1. Identify the adapter implementation and its stable `adapter_id`.
2. Choose a harness adapter, a shared framework adapter with registered
   targets, or a dedicated custom-agent adapter.
3. Reuse one shared adapter across custom agents when the framework provides
   stable loading and invocation semantics. Use a dedicated adapter when the
   agent itself is the only unambiguous execution boundary.
4. List the normalized fields the target can actually enforce.
5. Separate adapter-wide `harness.settings`, per-target `workflow.settings`,
   and typed `extensions`.
6. Keep installation, environment preparation, Relay orchestration, caller
   scheduling, and consumer result enrichment outside the adapter.

If the requested behavior cannot be expressed by the current contract, surface
the gap. Do not silently consume an unsupported northbound field or hide it in
an unrelated extension.

## Define the Descriptor First

Create one self-contained `*.fabric-adapter.json` before implementing target
translation:

- Set the current `contract_version`, a globally stable `adapter_id`,
  `adapter_kind`, and runner binding.
- Declare only normalized `config.accepts` fields the implementation enforces.
- When accepting `instructions.system`, declare the exact supported
  `config.system_instruction_modes`. New descriptors must not rely on the
  legacy omitted-value behavior, which means `replace` only.
- Declare `mcp.auth.oauth2` or `mcp.auth.service_account` only when the adapter
  implements the corresponding MCP authentication mode.
- Publish closed `settings_schema`, `model_schema`, `tool_definition_schema`,
  and `extension_schemas` where applicable. Use
  `model_schema` only for static model/provider compatibility and model settings;
  keep credential validity and provider availability in startup validation.
- Declare runtime requirements and telemetry outputs without secret values.
- Leave optional capability flags false unless the installed NeMo Fabric runtime
  exposes and tests that adapter operation. Set `capabilities.streaming` only
  when the adapter implements native OpenAI Chat Completions streaming through
  `invoke_openai_stream`. Relay-backed ATOF streaming is independent and does
  not require this capability.

If the adapter loads registered targets, list their types in `target_types`.
Create one `*.fabric-target.json` per target. The target record owns its
`adapter_id`, type-specific entry point, and workflow settings schema. It uses
the same `contract_version` as the Adapter Descriptor.

Validate descriptor schemas without importing adapter code. Keep all schema
references local to the descriptor document; do not rely on HTTP or file
references.

## Package Discovery Metadata

Install the descriptor in the standard shared-data location. For setuptools:

```toml
[tool.setuptools.data-files]
"share/nemo-fabric/adapters/acme" = ["acme.fabric-adapter.json"]
"share/nemo-fabric/targets/acme" = ["email.fabric-target.json"]
```

Depend on `nemo-fabric-adapter-contract` for typed standard-library dataclasses.
Install its optional `pydantic` extra only for Pydantic interoperability. Add
`nemo-fabric-adapters-common` only if the adapter chooses its lifecycle or
Relay helpers. A bare adapter package should not depend on the NeMo Fabric
runtime.

For a TypeScript adapter, depend on
`nemo-fabric-adapter-contract`. Import descriptor, configuration,
runtime-context, request, and result types from the package root, matching the
Python package's single model namespace. TypeScript types do not validate data
received from a process or network boundary; validate untrusted values against
the JSON Schemas included with the package.

## Map AgentConfig

Accept a validated `AgentConfig` and translate each declared field once at the
adapter boundary:

- Resolve named model roles into target-native model clients or settings.
- Apply normalized instructions and runtime limits only when declared. Validate
  `instructions.system.mode` at the adapter startup boundary as well as during
  planning; `replace` discards the harness default, while `append` preserves it
  and adds the configured content after it.
- Convert MCP servers, tool definitions, tool policy, and skills into native
  target constructs.
- Resolve workflow entry points and construction settings during `start` in
  the task environment.
- Read identity, environment, artifacts, and telemetry from `RuntimeContext`,
  not from workflow settings.

Reject unsupported values with stable, safe error codes. Do not log complete
configs, environment values, headers, credentials, or arbitrary user input.

Use typed extension models and publish their schemas at the exact descriptor
extension point. Never treat `extensions` as an unchecked dictionary escape
hatch.

## Implement the Lifecycle

Implement exactly one `start`, zero or more ordered `invoke` operations, and
one `stop` for each NeMo Fabric runtime.

- Construct and retain target state in `start`.
- Accept `AgentRunRequest` and `RuntimeContext`, then return one
  `AgentRunResult` from `invoke`.
- Make `stop` safe after partial startup and failed invocation.
- Isolate mutable state between independent runtimes.
- If the descriptor declares `capabilities.streaming`, implement
  `async invoke_openai_stream(request, context, emit)`. Execute the target exactly once,
  await `emit(chunk)` only for the `openai.chat_completions.chunk/v1` profile,
  and return one `AgentRunResult`. Each chunk requires
  non-empty `id` and `model`, a nonnegative integer `created`, the exact
  `chat.completion.chunk` discriminator, and structurally valid `choices`. An
  invocation that emits no chunks is valid.
- Do not add an adapter streaming method for Relay-backed
  `Runtime.invoke_stream()`; execute ordinary `invoke` and use the provided
  telemetry context.

For native OpenAI streaming, the SDK owns the authenticated loopback HTTP
transport with chunked NDJSON framing. The common host validates the transport,
removes its credentials from the adapter payload, and supplies the `emit`
callback. Do not persist or log stream credentials, write chunks to stdout, add
SSE framing, or forward other target-native event profiles.

For a Python adapter that opts into the common host:

```python
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle


class TargetRuntime:
    async def start(self, payload):
        config: AgentConfig = payload["config"]
        ...

    async def invoke(
        self,
        request: AgentRunRequest,
        context: RuntimeContext,
    ) -> AgentRunResult:
        native = await self.target.run(request.input)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": native.text},
        )

    async def invoke_openai_stream(self, request, context, emit):
        async for chunk in self.target.stream(request.input):
            await emit(chunk)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": self.target.final_text},
        )

    async def stop(self):
        ...


def main() -> None:
    lifecycle.serve(TargetRuntime, config_loader=AgentConfig.from_mapping)
```

The common host decodes the internal lifecycle envelope before calling the
adapter and encodes its terminal result afterward. Adapter code does not parse
the transport envelope or infer failure from fields inside `output`.
Return `AgentRunStatus.FAILED` with an `AgentRunError` when the target completes
with a failed outcome. Raise an exception when the adapter cannot produce a
normalized terminal result.

For in-process Relay SDK telemetry where the adapter owns the invocation-level
Agent scope, wrap that scope with
`nemo_fabric_adapters.common.utils.relay_request_context(context.request_id)`.
The helper uses a UUID request ID as Relay's propagated root and always returns
`nemo_fabric_request_id` metadata, including for non-UUID request IDs. Do not
apply this pattern to an external Relay gateway or an upstream integration that
creates an isolated scope context unless its boundary accepts a per-turn
propagation context.

## Handle Custom Agents

For a shared framework adapter, select the registered target with
`FabricConfig.workflow.target_id`. Use the selected Adapter Target Descriptor's
`spec.entrypoint.kind` for adapter-scoped resolution semantics and `ref` for
the factory identity. Validate `workflow.settings` against that target's
`spec.settings_schema`.

- Define only entry-point kinds that the shared adapter resolves
  unambiguously. The v1alpha2 contract does not define a global kind catalog.
- Map a declared `factory` intent to the corresponding target-native factory.
  The current NeMo Agent Toolkit reference maps `fabric.agent.react` to its
  ReAct workflow factory.
- Supply factories an adapter-defined build context containing already
  resolved native values; do not require custom agents to parse `FabricConfig`
  or `AgentConfig`.
- Use a dedicated adapter without an artificial workflow entry point when the
  selected adapter already identifies one application-owned agent.

Compare the
[NeMo Agent Toolkit shared adapter](https://github.com/NVIDIA/NeMo-Fabric/tree/main/external/nat)
with the
[dedicated LangGraph example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/langgraph_custom_agent)
before choosing the custom-agent boundary.

## Validate Before Handoff

Complete these checks before handing off an adapter:

1. Install the built wheel in an isolated adapter environment.
2. Confirm discovery below `share/nemo-fabric` and inspect the resolved adapter
   and target descriptors in `Fabric().plan(...)`.
3. Exercise one accepted normalized config and rejection for unsupported
   fields and each declared schema.
4. Run `doctor(...)` with both missing and satisfied requirements.
5. Test start, success, target failure, malformed output, repeated invocation,
   stop, partial-start cleanup, EOF cleanup, and two-runtime isolation.
6. If native OpenAI streaming is claimed, test empty and multi-chunk streams,
   malformed and oversized records, invalid chunks, sequence and identity
   mismatches, a missing end record, early consumer close without cancellation,
   a separate terminal result, one active turn, and exactly one target
   invocation.
7. Test Relay correlation separately if telemetry support is claimed.
8. Report the adapter package version, contract version, required-profile
   result, and every optional capability as supported or unsupported.

Do not claim automated NeMo Fabric conformance until the published conformance
suite exists and the exact adapter release passes it.
