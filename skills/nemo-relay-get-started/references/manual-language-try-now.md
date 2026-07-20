<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Manual Language Try-Now Reference

Use this path only when the application directly owns its tool or LLM call
sites and a maintained framework integration is not the better boundary. This
is the most involved try-now path because it changes application code.

## Select The Language

Detect the existing project language from its manifest and use
`nemo-relay-install` for that package:

- `pyproject.toml` -> Python package
- `package.json` -> Node.js package
- `Cargo.toml` -> Rust crate

Do not introduce a second language or create a source checkout when a published
package fits the target application.

Use the matching quick start:

- [Python quick start](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start/python)
- [Node.js quick start](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start/nodejs)
- [Rust quick start](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start/rust)

## Attach The Smallest Working Boundary

Preview the code change and obtain confirmation. Keep the first example to one
scope, one tool call, one LLM call, and one short-lived event subscriber:

- **Python**: register a subscriber, enter `nemo_relay.scope.scope(...)`, then
  use `nemo_relay.tools.execute(...)` and `nemo_relay.llm.execute(...)`.
- **Node.js**: register a subscriber, use `withScope(...)`, then
  `toolCallExecute(...)` and `llmCallExecute(...)`.
- **Rust**: register a subscriber, push an agent scope, use
  `tool_call_execute(...)` and `llm_call_execute(...)`, then pop the scope.

Prefer managed execution helpers over manual start/end lifecycle calls. Keep
the tool and model callbacks deterministic and local for the first pass; they
need to demonstrate Relay lifecycle behavior, not production provider setup.

Flush asynchronous subscriber delivery before checking output, then deregister
the short-lived subscriber. Do not add middleware, codecs, external exporters,
or plugin configuration until the basic lifecycle is visible.

## Define Success And Continue

Treat this path as successful when:

- The application prints or records scope, tool, and LLM lifecycle events.
- The tool and model callbacks still return their expected values.
- The events show the tool and LLM work under the intended root scope.

Returned callback values without event evidence do not verify Relay
instrumentation. Summarize event names and parentage without dumping complete
payloads.

After this proof, preserve the demonstrated boundary and recommend one
goal-aligned plugin as the primary next step. If the trial used only a
short-lived subscriber, configure plugin-managed Observability first; otherwise
choose Adaptive, NeMo Guardrails, PII Redaction, or Model Pricing based on the
user's outcome. Use `nemo-relay-instrument-calls` only when the demonstrated
boundary does not yet cover the real application workflow.

For all supported languages, see the
[Quick Start](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start).
