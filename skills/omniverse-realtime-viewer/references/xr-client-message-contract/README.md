<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XR Client Message Contract

## Triggers

Use this reference for: client sends a message but nothing happens, XR message
envelope, prim manipulation from a headset, anchor update, camera transform
message, throttling transform messages, what wire format the XR client should
send, adapting a specified existing XR client protocol.

Read `cloudxr-opaque-data-channel/README.md` first for the transport, and
`streaming-messages/README.md` for the dispatch pattern this mirrors.

## One envelope

```json
{"event_type": "openStageRequest", "payload": {"url": "stage02.usd"}, "request_id": "r1"}
```

| Field | Rule |
| --- | --- |
| `event_type` | Required, a string. Otherwise log and drop. |
| `payload` | Required, an object. Its keys are the handler's arguments. |
| `request_id` | Optional; echo it on the reply when present. |

Replies use the same shape. Malformed messages are logged and dropped, never
raised — one bad message must not take down the frame loop.

This is the viewer application's protocol, not a protocol implemented by OVXR.
Use the same canonical envelope as the other viewer transports in this bundle.
If an existing client speaks a different envelope, document that client's
public/versioned contract and translate at a single adapter boundary. Enable
only the aliases required by that integration, and reply in one canonical form.
Do not infer a legacy format from a client or headset name.

## Payload conventions that bite

- Use camelCase application payload keys consistently; envelope keys remain
  `event_type`, `payload`, and optional `request_id` as shown above.
- Define transforms as 16 finite row-major values using the bundle's row-vector
  convention, with declared coordinate space and units. An existing client with
  another layout needs a conversion fixture from its identified contract; do
  not guess a 12-value layout or scale factor from historical examples.
- Read resolved scene units through upstream scene/anchor guidance and publish
  the applicable conversion to the client. Convert once; never assume every
  stage uses centimetres or hardcode a metres-to-stage factor.

## Server handler shape

Keep the dispatch design from `streaming-messages/README.md`: a dict of event
type to bound method, one `send` sink, `request_id` echoed when present. Do not
reuse an ovstream input handler — `ovstream` has no XR transport, and XR input
arrives through OpenXR, not as `InputEvent`s.

A minimal, useful surface:

| Request | Does |
| --- | --- |
| `getStatsRequest` | fps, frame time, device info, scene info — including `metersPerUnit`, so the client can convert correctly |
| `openStageRequest` | repopulate the OVStage from a URL |
| `setPrimTransformRequest` | write a transform to a prim through the OVStage data plane |
| `anchorUpdateRequest` | move the prim named by `tracking_anchor_path` |

**Validate paths from the network.** `openStageRequest` carries an
attacker-controlled string; confine it to an allowed directory or it is a
file-read primitive.

**Writes use one application publication coordinator.** Co-writing viewers
select OVXR application ordinal mode and supply its owned ordinals for every
stage-writing call. Resolve all lower writes before bind publishes and renders;
begin only locates views. Do not give each writer a floor-plus-one counter.
Read [the library publication contract](../dependencies/ovxr.md).

## Rate discipline

Native headset tracking belongs on the OpenXR path; ordinary application
messages must not replace latency-critical predicted poses. An explicitly
controlled WebXR proof client may send separately labelled observed input
snapshots under the [controlled-client contract](../xr-spatial-input/webxr-client.md).
Those observations do not claim to be OVXR-private predicted eye poses.

For continuous interaction such as a drag or slider, coalesce superseded
updates, bound the send cadence and measure the accepted input-to-render delay.
Select limits for the actual runtime/client and message family rather than
copying an undocumented legacy frequency. Keep ordered begin/end/cancel edges.

Pump only bounded channel work through the runtime owner using the vendor's
connection/readiness contract. A disconnected channel remains a readiness state,
not a reason to spin or flood frame logs.

## Common mistakes

| Mistake | Consequence | Prevention |
| --- | --- | --- |
| Assuming an unspecified client protocol | Messages rejected or misdecoded | Identify the client contract and translate once into the canonical application envelope |
| Hardcoding ×100 for centimetres | Scene off by 100× on metre-scale stages | Use the stage's `metersPerUnit` and publish it |
| Assuming a client's matrix matches yours | Transform garbage or silently rejected | State the shape in the handler error; convert at the edge |
| Replacing native predicted poses with JSON telemetry | Latency and jitter | Keep native tracking on OpenXR; label controlled-client observations separately |
| Unbounded `openStageRequest` paths | Arbitrary file read | Confine to an allowed directory |
| Raising out of a handler | Frame loop dies | Catch, reply with an error, keep rendering |

## Related References

- `cloudxr-opaque-data-channel/README.md` — the transport and channel lifecycle
- `xr-client-connection/README.md` — getting connected at all
- `streaming-messages/README.md` — the dispatch pattern
- `ovstage-runtime/README.md` — ordinals and write floors for handler writes

## Controlled interaction message families

The following limits and epoch rules apply to the optional controlled WebXR
adapter and XR UI intents, not a new general viewer protocol.

### Versioned interaction families and exact byte ceilings

High-rate spatial, navigation, and XR-UI messages need a smaller explicit
contract than the transport's theoretical maximum. Define a ceiling per message
family and put `schemaVersion` in every request, response, and server push in
that family.

Measure the final UTF-8 JSON **envelope**, not character count and not the
payload alone:

```python
def encode_bounded(event_type: str, payload: dict, maximum_wire_bytes: int) -> bytes:
    versioned = {"schemaVersion": 1, **payload}
    encoded = json.dumps(
        {"event_type": event_type, "payload": versioned},
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    if len(encoded) > maximum_wire_bytes:
        raise ValueError(
            f"{event_type} is {len(encoded)} UTF-8 bytes; limit is {maximum_wire_bytes}"
        )
    return encoded
```

Apply the same check inbound before dispatch and outbound before transport
send. Error replies must remain within the response family's ceiling and should
report observed bytes, maximum bytes, event type, and required schema version.
Keep the limits centralized so browser, Python, Rust, and tests cannot drift.

### Focus/input epochs and bounded dedupe

For focus-sensitive pointer, hand, controller, ray, or spatial samples, a
sequence number alone is insufficient. Scope ordering with:

```text
(client_session_id, input_epoch, input identity, sequence)
```

- `client_session_id` identifies one controlled-client lifetime and has a
  documented maximum string length.
- `input_epoch` increases whenever focus, tracking/input-source ownership,
  rendered surface revision, or session ownership invalidates old input.
- Send any required `cancel` under the old epoch, then advance the epoch and
  restart per-input sequences. This lets the server close captures before it
  rejects late packets.
- Ignore epochs older than the accepted epoch. Within an epoch, ignore duplicate
  or reordered sequences without advancing canonical state.
- Validate an entirely new-epoch payload—including focus, schema, coordinate
  space, finite matrices, ray, node/action identity, and document revision—
  **before** committing the new epoch. An unfocused or malformed high epoch must
  not poison the session and make the next valid packet look stale.
- Bound retained sessions and sequence keys with a documented eviction policy.
  When a session is evicted or advances epoch, prune its old per-input keys;
  attacker-controlled session IDs must not create unbounded memory growth.

Stage changes and reconnects should push canonical navigation/UI state again,
advance the relevant document/revision state, and reject packets captured
against the old stage. Tests should cover reconnect sequence restart, stage
change convergence, stale epoch rejection, malformed-future-epoch recovery, and
cache eviction.
