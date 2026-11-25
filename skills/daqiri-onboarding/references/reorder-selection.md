# Reorder Selection and Capability Diagnosis

Read for ordering/placement selection or hardware-reorder failures. Establish
sequence behavior, packet geometry, memory domain, conversion requirements,
queue mode and the receiving adapter from supplied facts.

Raw ibverbs hardware reorder uses NIC first-DMA placement plus host completion
polling/publication. It is separate from application direct polling and requires
a compatible indirect RX queue. Check the matching reference and engine's
initialization constraints: finite cyclic addressing with supported parser
samples, fixed packet/output geometry, compatible memory and no unsupported
conversion. Preserve the sender protocol; silently truncating or wrapping a
monotonic sequence does not make it eligible. Use supported software reorder
when appropriate. Native semantic validation alone does not test these engine
or device prerequisites.

Preserve the acquisition memory path: Spark/GB10 UMA uses `host_pinned`;
IGX dGPU uses `device` packet/payload and aggregate buffers. Raw ibverbs software
reorder requires `reorder_type: gpu`, including on UMA. Establish producer
readiness before host metadata reads; a CUDA stream wait does not synchronize
the host. Follow stream/event and consumer-completion rules before reading or
freeing GPU outputs; CPU metadata does not make device payload CPU-addressable.

Separate receiver identity, NIC generation, firmware/provider/tool versions,
supported/max capabilities, effective current state and next-boot values.
`PROG_PARSE_GRAPH=1` and `FLEX_PARSER_PROFILE_ENABLE=4` are the required parser
settings. Read Current and Next Boot columns explicitly: an asterisk can indicate non-default
values rather than pending activation. Supported settings pending activation
need the documented cold-boot activation and same-adapter capability recheck.
An unavailable effective feature needs a bounded prerequisite diagnosis and
supported fallback; absent capabilities alone do not isolate firmware age.
Do not promise that a reboot or guessed firmware version
fixes it. Platform names never replace capability checks.

Follow the installed timeout contract. `passthrough` can publish a fixed-size
aggregate with a timeout flag and missing-slot bitmap; only received slots are
valid. `drop` withholds an incomplete aggregate. Inspect metadata while the burst
is owned, finish consumption, then free it to rearm hardware credits. Holding
outputs can drop the next cycle; early freeing violates ownership. Retain limits
around late packets and cyclic reuse when proposing bounded payload/reuse checks.

Sources: `docs/concepts.md` RX Packet Aggregation and Reorder;
`docs/api-reference/configuration.md` RX Reorder Configs;
`docs/tutorials/system_configuration.md` Enable Programmable Flex Parsing;
`docs/benchmarks/raw_benchmarking.md` Hardware reorder benchmark;
`docs/api-reference/cpp.md` Reordered RX bursts.
