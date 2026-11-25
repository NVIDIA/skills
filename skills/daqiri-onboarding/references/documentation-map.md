# Documentation and Helper Map

## Resolve the source

Locate the DAQIRI checkout from the user's path or the current workspace (`git rev-parse --show-toplevel` where applicable), and confirm it contains DAQIRI's `CMakeLists.txt`, `docs/` and `scripts/`. A copied skill need not reside inside that checkout. Ask for the checkout only when it cannot be discovered. Paths below are relative to the DAQIRI root; read the relevant section rather than every page.

Prefer the matching checkout's documentation and actual helper `--help` over a newer website. If only an installed runtime is available, inspect its version and compiled engines, then use the published guides below as a fallback. State when those guides describe a different/unknown version and verify available commands before execution. Do not invent a release-specific documentation URL. Concept answers can use the published docs without requiring a checkout.

## Technical sources

Published guides use `https://nvidia.github.io/daqiri/`. The links below are browsing fallbacks; the source column identifies matching local content.

| Need | Checkout source and section | Published guide |
|---|---|---|
| Requirements, build, runtime smoke | `docs/getting-started.md`: System Requirements, Build, Run A Benchmark | [Getting Started](https://nvidia.github.io/daqiri/getting-started/) |
| Stream/engine choice | `docs/benchmarks/index.md`; `docs/concepts.md`: Stream Types | [Benchmarking](https://nvidia.github.io/daqiri/benchmarks/) |
| GPUDirect, bursts, segments, queues, memory, ownership, reorder | `docs/concepts.md`: relevant named concept | [Concepts](https://nvidia.github.io/daqiri/concepts/) |
| NIC/GPU topology, GB10 versus discrete GPU, hugepages, tuning | `docs/tutorials/system_configuration.md`: applicable platform tab | [System Configuration](https://nvidia.github.io/daqiri/tutorials/system_configuration/) |
| Generator inputs, roles and validation | `docs/config-generation.md`: raw pair, socket roles, generic rendering, native validation | [Configuration Generation](https://nvidia.github.io/daqiri/config-generation/) |
| YAML fields and semantics | `docs/api-reference/configuration.md`; `docs/tutorials/configuration-walkthrough.md` | [YAML Reference](https://nvidia.github.io/daqiri/api-reference/configuration/), [Walkthrough](https://nvidia.github.io/daqiri/tutorials/configuration-walkthrough/) |
| Polling selection and latency boundaries | `docs/concepts.md`: Queue Polling Modes; `docs/api-reference/cpp.md`: direct API contracts; `docs/benchmarks/raw_benchmarking.md`: Direct-polling latency sweep | [Raw Benchmarking](https://nvidia.github.io/daqiri/benchmarks/raw_benchmarking/) |
| Reorder eligibility, effective capabilities and timeouts | `docs/api-reference/configuration.md`: RX Reorder Configs; `docs/tutorials/system_configuration.md`: Enable Programmable Flex Parsing; `docs/api-reference/cpp.md`: Reordered RX bursts | [YAML Reference](https://nvidia.github.io/daqiri/api-reference/configuration/) |
| Raw first run, loopback, counters and steering triage | `docs/benchmarks/raw_benchmarking.md`: selected loopback/run section | [Raw Ethernet Benchmarking](https://nvidia.github.io/daqiri/benchmarks/raw_benchmarking/) |
| UDP/TCP/RoCE, namespaces, route proof and restoration | `docs/benchmarks/socket_benchmarking.md`: selected protocol and Restore host networking | [Socket and RDMA Benchmarking](https://nvidia.github.io/daqiri/benchmarks/socket_benchmarking/) |
| Application buffer lifecycle | `docs/api-reference/cpp.md`; `docs/concepts.md`: Zero-Copy Ownership | [C++ API](https://nvidia.github.io/daqiri/api-reference/cpp/) |
| TensorRT application specifically requested | `docs/tutorials/daqiri-resnet-inference.md` | [ResNet Inference](https://nvidia.github.io/daqiri/tutorials/daqiri-resnet-inference/) |

## Reuse existing tools

Inspect the selected script before execution; some setup helpers mutate network state. Invoke commands in the correct host/container context from the guide.

| Task | Existing tool | Boundary |
|---|---|---|
| Container build | `scripts/build-container.sh` | Reuse a compatible existing image before proposing a build |
| Hardware/system checks | `python/tune_system.py --check <property>` | Select relevant diagnostics. `--check all` also writes a topology image; use a fresh `--schematic-output` path. Does not supply a complete reversible tuning transaction |
| Config generation | `scripts/gen_daqiri_config.py` | `raw-pair`, `socket-pair`, generic rendering; inspect `--help` |
| Native semantic validation | `daqiri_config_validate` in the actual build/install | Hardware-free; `--list-engines` discovers compiled engines; successful validation is not a hardware test |
| Cabled namespace setup | `scripts/setup_cabled_loopback_netns.sh --platform <platform>` | Choose `dgx-spark` or `igx-thor` for the documented topology; preserve existing networking |
| Spark RDMA setup | `scripts/setup_spark_rdma_loopback.sh` | Read requirements and namespace behavior first |
| Spark cross-host networking | `scripts/setup_spark_xhost_net.sh` | Confirm role, peer and interface selection |
| Optional performance sweeps | `examples/run_cabled_loopback_bench.sh --platform <platform>`, `examples/run_spark_mq_bench.sh` | Choose `dgx-spark` or `igx-thor` for the cabled harness; the multi-queue harness remains Spark-specific. Not required for one first-run test |

Use current benchmark executables and shipped example configs described by the selected guide. Avoid adding skill-private clones of configuration, validation or setup scripts. If a checkout lacks a mapped tool, inspect that version's documented alternative or report the prerequisite; do not silently pull another release's script into it.
