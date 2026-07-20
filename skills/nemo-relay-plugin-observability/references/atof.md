<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export Raw ATOF Events

Use ATOF when the user needs the canonical NeMo Relay lifecycle event stream as
JSONL for local debugging, offline inspection, or delivery to a raw-event
collector. ATOF preserves events; it does not project them into trajectories or
trace spans.

## Default Plugin Path

Prefer plugin-managed lifecycle for reusable process configuration:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 2

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "logs"
filename = "events.jsonl"
mode = "append"

[[components.config.atof.sinks]]
type = "stream"
url = "http://localhost:8080/events"
transport = "http_post"
header_env = { authorization = "NEMO_RELAY_ATOF_AUTH_HEADER" }
```

Use `overwrite` for an isolated one-run artifact and `append` for repeated local
runs. Add a `stream` sink when the same events should also be delivered
remotely. Use `header_env` to map stream header names to environment variables,
keeping credentials out of configuration files. Before activation, set each
named variable to the complete header value; validation rejects missing or
blank values.

Use the manual `AtofExporter` API only when the caller needs a custom subscriber
name or explicit registration window. Each manual exporter owns one sink, so
register one exporter per destination when you need fan-out. The lifecycle is:
create, register, run instrumented work, force flush, deregister, then shut
down.

## Verify

Verify the export with the following checks:

- Confirm the output file exists and contains one JSON object per line.
- Confirm the expected root scope plus tool or LLM lifecycle events are present.
- Check UUID and parent UUID relationships instead of relying only on event
  order.
- Confirm sensitive fields are absent before retaining or transmitting output.
- For stream sinks, verify file output separately from remote delivery.

Common failures include an unwritable output directory, an invalid mode, an
empty stream URL, an unsupported stream transport, abrupt process termination,
or interruption before `shutdown()` finishes flushing pending events.

For the complete exporter configuration, refer to
[ATOF observability](https://docs.nvidia.com/nemo/relay/dev/configure-plugins/observability/atof).
