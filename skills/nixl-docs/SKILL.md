---
name: nixl-docs
description: Find and read NIXL documentation for APIs, architecture, backends, installation and builds, environment variables, telemetry, troubleshooting, examples, benchmarks, and contribution.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NIXL Team <sronen@nvidia.com>
  tags:
    - nixl
    - docs
---

# NIXL Documentation Index

## Purpose

Index of NIXL documentation for agents working in a NIXL repository clone.

## Prerequisites

A clone of the NIXL repository.

## Instructions

Find relevant files in the index and read them. Paths are relative to this skill directory; `../../` is the repository root.

## Repository root
Project overview, contribution and community policies, security reporting, and third-party license attributions.
- [NVIDIA Inference Xfer Library (NIXL)](../../README.md): Project overview, supported platforms, source builds, Python wheel, Rust bindings, Docker, ROCm, examples, and etcd.
- [Contribution Guidelines](../../CONTRIBUTING.md): Development environment, code standards, contribution process, plugin development, testing, documentation, pull requests, and DCO.
- [Contributor Covenant Code of Conduct](../../CODE_OF_CONDUCT.md): Community behavior standards, scope, reporting, and enforcement guidelines.
- [Report a Security Vulnerability](../../SECURITY.md): Security vulnerability reporting via NVIDIA PSIRT email, web form, PGP key, and required details.
- [C++ third-party attributions](../../ATTRIBUTIONS-CPP.md): Third-party C and C++ libraries, repository links, and license texts.
- [Python third-party attributions](../../ATTRIBUTIONS-Python.md): Generated third-party Python package attributions, project links, and license texts.
- [Rust third-party attributions](../../ATTRIBUTIONS-Rust.md): Generated third-party Rust crate attributions, repository links, and license texts.

## Core documentation
Architecture, backend plugin API, C++ style, Python API, telemetry, and tracing.
- [Backend Plugin Interface](../../docs/BackendGuide.md): Southbound backend plugin API, plugin manager, capabilities, memory registration, transfers, notifications, and UCX.
- [NIXL Code Style](../../docs/CodeStyle.md): C++ naming, class layout, headers, formatting, Doxygen comments, and coding practices.
- [NIXL Doxygen Overview](../../docs/doxygen/nixl_doxygen.md): Doxygen architecture overview of agents, memory sections, backends, metadata exchange, etcd, and Redis.
- [NIXL Architecture](../../docs/nixl.md): Architecture, agents, memory sections, backends, metadata exchange with etcd and Redis, and TOML configuration.
- [NIXL Python API](../../docs/python_api.md): Python API features, installation, backend parameters via get_plugin_params, QueryMem, UCX, and GDS.
- [NIXL Telemetry System](../../docs/telemetry.md): Telemetry events, metrics, shared-memory buffer, Prometheus, prometheus_mp, DOCA exporters, and readers.
- [NIXL Tracing System](../../docs/tracing.md): Span tracing with NVTX, Nsight Systems, sampling, backend plugins, and correlation.
- [NIXL Usage Examples](../../docs/uses.md): Designs and frameworks that use NIXL to move model weights during RL post-training.

## Getting Started (Fern)
NIXL overview, component architecture, and install-to-transfer quick start.
- [Overview](../../fern/docs/pages/getting-started/overview.md): NIXL is a high-performance point-to-point data transfer library for AI inference frameworks.
- [Architecture](../../fern/docs/pages/getting-started/architecture.md): How NIXL's core components work together to enable high-performance data transfers.
- [Quick Start](../../fern/docs/pages/getting-started/quick-start.md): Install NIXL and learn the complete transfer workflow -- from agent initialization to transfer completion.

## User Guide (Fern)
Transfer backends, etcd metadata exchange, telemetry events and exporters, NIXLBench, and KVBench.
- [Backend Selection](../../fern/docs/pages/user-guide/backends/index.md): How to choose the right NIXL transfer backend for your memory types and use case.
- [UCX](../../fern/docs/pages/user-guide/backends/ucx.md): UCX transfer backend for high-performance VRAM and DRAM transfers via RDMA and TCP.
- [Libfabric](../../fern/docs/pages/user-guide/backends/libfabric.md): Libfabric (OFI) network backend for VRAM-to-VRAM, VRAM-to-DRAM, and DRAM-to-DRAM transfers as an alternative to UCX.
- [Mooncake](../../fern/docs/pages/user-guide/backends/mooncake.md): Mooncake Transfer Engine backend for VRAM and DRAM transfers in Mooncake-enabled deployments.
- [UCCL-P2P](../../fern/docs/pages/user-guide/backends/uccl.md): UCCL-P2P network backend for VRAM and DRAM transfers.
- [DOCA GPUNetIO](../../fern/docs/pages/user-guide/backends/gpunetio.md): DOCA GPUNetIO backend for high-performance GPU-to-GPU transfers using GPUDirect Async.
- [GPUDirect Storage](../../fern/docs/pages/user-guide/backends/gds.md): GPUDirect Storage backend for direct GPU-to-file transfers without CPU bounce buffers.
- [GPUDirect Storage MT](../../fern/docs/pages/user-guide/backends/gds-mt.md): Multi-threaded GPUDirect Storage backend for higher throughput on parallel file operations.
- [POSIX](../../fern/docs/pages/user-guide/backends/posix.md): Standard POSIX I/O backend for DRAM-to-file transfers.
- [HF3FS](../../fern/docs/pages/user-guide/backends/hf3fs.md): HuggingFace 3FS backend for file operations on 3FS fuse-based filesystems.
- [OBJ](../../fern/docs/pages/user-guide/backends/obj.md): Object storage backend supporting S3-compatible stores with multiple executor variants.
- [Azure Blob](../../fern/docs/pages/user-guide/backends/azure-blob.md): Azure Blob Storage backend for DRAM-to-object-storage transfers via Azure REST API.
- [INFINIA](../../fern/docs/pages/user-guide/backends/infinia.md): DDN INFINIA backend for asynchronous transfers between DRAM or VRAM and object storage.
- [DOCA_MEMOS](../../fern/docs/pages/user-guide/backends/doca-memos.md): DOCA_MEMOS backend for key-value transfers between DRAM and an NVMe KV device on BlueField.
- [GUSLI](../../fern/docs/pages/user-guide/backends/gusli.md): Block-level I/O backend for transfers between block storage devices and DRAM.
- [Metadata Exchange with etcd](../../fern/docs/pages/user-guide/etcd-metadata-exchange.md): Configure etcd-based distributed metadata exchange for NIXL agents -- key prefix scheme, watcher lifecycle, and Prometheus integration.
- [Telemetry Guide](../../fern/docs/pages/user-guide/telemetry.md): Understanding NIXL's telemetry system -- event types, metrics, configuration, and monitoring with Prometheus.
- [Logging Guide](../../fern/docs/pages/user-guide/logging.md): Configuring NIXL's logging -- verbosity, per-process log files, filename templates, size limits, and failure behavior.
- [NIXLBench](../../fern/docs/pages/user-guide/benchmarking/nixlbench/index.md): A benchmarking tool for measuring NIXL data transfer performance across network and storage backends.
- [Building NIXLBench](../../fern/docs/pages/user-guide/benchmarking/nixlbench/build.md): Build NIXLBench using Docker or natively from source with Meson.
- [Using NIXLBench](../../fern/docs/pages/user-guide/benchmarking/nixlbench/usage.md): How to run NIXLBench benchmarks and troubleshoot common issues.
- [KVBench](../../fern/docs/pages/user-guide/benchmarking/kvbench/index.md): A KV cache benchmarking utility that generates and runs NIXLBench commands to profile LLM inference transfer performance.
- [Building KVBench](../../fern/docs/pages/user-guide/benchmarking/kvbench/build.md): Install KVBench using the NIXLBench Docker container or a Python virtual environment.
- [Using KVBench](../../fern/docs/pages/user-guide/benchmarking/kvbench/commands.md): KVBench command reference, model configuration guide, and LLM example configurations.

## Developer Guide (Fern)
C++, Python, Rust, and Docker source builds, plus contribution guidance.
- [Building NIXL from Source](../../fern/docs/pages/user-guide/building-nixl/index.md): Build NIXL from source -- C++ library, Python bindings, Rust bindings, or Docker container.
- [Docker](../../fern/docs/pages/user-guide/building-nixl/docker.md): Build and run NIXL in a Docker container.
- [NIXL C++ (Meson)](../../fern/docs/pages/user-guide/building-nixl/nixl-cpp.md): Build the NIXL C++ library from source using the Meson build system.
- [Python Bindings](../../fern/docs/pages/user-guide/building-nixl/python-bindings.md): Build and install NIXL Python bindings from source.
- [Rust Bindings](../../fern/docs/pages/user-guide/building-nixl/rust-bindings.md): Build NIXL Rust bindings from source using Meson or Cargo.
- [Contribution Guide](../../fern/docs/pages/development/contribution-guide.md): How to contribute to the NIXL project -- development setup, code style guidelines, testing requirements, and the contribution process.

## Resources (Fern)
Environment variables for core, etcd, telemetry, and backends, plus troubleshooting.
- [Environment Variables](../../fern/docs/pages/resources/environment-variables.md): Complete reference for all NIXL environment variables covering core configuration, etcd, telemetry, and backend-specific settings.
- [Troubleshooting](../../fern/docs/pages/resources/troubleshooting.md): Common errors, debugging steps, and solutions for NIXL build, runtime, and transfer issues.

## Fern snippets and placeholder pages
Placeholder Fern pages and reusable backend environment-variable tables.
- [NIXL Documentation placeholder](../../fern/docs/pages/empty.md): Placeholder Fern page containing only the NIXL Documentation title.
- [Second Page placeholder](../../fern/docs/pages/second-page.md): Placeholder Fern page containing only the Second Page title.
- [Azure Blob environment variables](../../fern/snippets/env-vars-azure-blob.mdx): AZURE_STORAGE account, container, and connection-string variables, plus AZURE_CA_BUNDLE.
- [GDS environment variables](../../fern/snippets/env-vars-gds.mdx): CUFILE_ENV_PATH_JSON for the GPUDirect Storage cuFile configuration path.
- [Libfabric environment variables](../../fern/snippets/env-vars-libfabric.mdx): NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG for NUMA-aware DRAM rail bandwidth and dynamic provider-specific NIXL_LIBFABRIC_ variables.
- [Mooncake environment variables](../../fern/snippets/env-vars-mooncake.mdx): NIXL_MOONCAKE_IP_ADDR override for the Mooncake Transfer Engine network interface.
- [OBJ environment variables](../../fern/snippets/env-vars-obj.mdx): AWS_DEFAULT_BUCKET, AWS_ENDPOINT_OVERRIDE for S3-compatible stores, and AWS SDK credential and region variables.
- [UCX environment variables](../../fern/snippets/env-vars-ucx.mdx): NIXL_UCX_WARNING_TIMEOUT for UCX memory registration warnings.

## Network and peer-transfer plugins
RDMA, GPU-initiated, and peer-to-peer transfer backends.
- [DOCA GPUNetIO Plugin](../../src/plugins/gpunetio/README.md): DOCA GPUNetIO GDAKI backend for GPU-initiated RDMA on attached or pooled CUDA streams.
- [GPUDirect TCPXO Plugin](../../src/plugins/gpudirect_tcpxo/README.md): Build and run the GPUDirect TCPXO backend for GPU-to-GPU transfers on Google Cloud A3 VMs.
- [NIXLBench TCPXO Guide](../../src/plugins/gpudirect_tcpxo/docs/nixlbench_guide.md): Run NIXLBench on two Google Cloud A3 VMs to check a GPUDirect TCPXO setup.
- [Libfabric Plugin](../../src/plugins/libfabric/README.md): Libfabric OFI multi-rail RDMA backend with GPU Direct, EFA, Neuron, and NUMA-aware rails.
- [Mooncake Backend Plugin](../../src/plugins/mooncake/README.md): Mooncake Transfer Engine backend for TCP, RDMA, CXL shared memory, and NVMe-oF transfers.
- [UCCL Backend Plugin](../../src/plugins/uccl/README.md): UCCL P2P backend for RDMA, TCP, TCP-X, EFA, intranode transfers, and vLLM KV-cache disaggregation.

## Storage backend plugins
File, block, object, and key-value storage backends.
- [Azure Blob Storage Plugin](../../src/plugins/azure_blob/README.md): Azure Blob backend using Azure SDK, Entra ID, Azurite, and asynchronous blob I/O.
- [DOCA_MEMOS Plugin](../../src/plugins/doca_memos/README.md): Build, configure, and troubleshoot the DOCA_MEMOS key-value plugin on BlueField.
- [GDS Plugin](../../src/plugins/cuda_gds/README.md): GPUDirect Storage GDS and GDS_MT backends using cuFile batch and TaskFlow engines.
- [GUSLI Plugin](../../src/plugins/gusli/README.md): GUSLI client-server block I/O backend for shared-memory SPDK, networked, and kernel block devices.
- [HF3FS Plugin](../../src/plugins/hf3fs/README.md): HF3FS backend using hf3fs_usrbio for 3FS file I/O and shared-memory transfers.
- [Infinia Plugin](../../src/plugins/infinia/README.md): DDN Infinia coroutine object storage backend for DRAM, VRAM, and OBJ transfers.
- [Object Storage Plugin](../../src/plugins/obj/README.md): AWS S3 OBJ backend with standard and CRT clients plus vendor-accelerated engines.
- [CUObject Client Config for S3 Accelerated](../../src/plugins/obj/s3_accel/README.md): Configure CUObject RDMA addresses for accelerated S3 engines and speed up their startup.
- [Dell ObjectScale S3 Engine](../../src/plugins/obj/s3_accel/dell/README.md): Dell ObjectScale S3-over-RDMA OBJ engine using CUDA CUObject for DRAM and VRAM.
- [POSIX Plugin](../../src/plugins/posix/README.md): POSIX file backend using Linux AIO, libaio, io_uring, liburing, and FILE_SEG.
- [Redis Plugin](../../src/plugins/redis/README.md): Redis key-value backend using hiredis, libevent, connection pooling, asynchronous reads and writes, and queryMem.

## Telemetry exporter plugins
Custom exporter development plus DOCA and Prometheus telemetry plugins.
- [Custom Telemetry Plugin Development Guide](../../src/plugins/telemetry/README.md): Custom telemetry exporter interface, built-in events, metric filtering, CSV example, and Meson build.
- [DOCA Telemetry Exporter](../../src/plugins/telemetry/doca/README.md): DOCA exporter with Prometheus scrape and DTS IPC delivery backends, histograms, labels, and metric filtering.
- [Prometheus Telemetry Exporter](../../src/plugins/telemetry/prometheus_mp/README.md): Single Prometheus endpoint aggregating all ranks via mmap stores, flock election, and stale-store reaping.

## Bindings and utilities
Python and Rust bindings, utility tests, device proxy, and file backend helpers.
- [nixl Python Meta Package](../../src/bindings/python/nixl-meta/README.md): PyPI meta package selecting CUDA backends at runtime through PyTorch.
- [Rust Bindings for NIXL](../../src/bindings/rust/README.md): Idiomatic Rust bindings, Cargo and stub-API builds, test suite, and CUDA memory-view tests.
- [Utility Tests](../../src/utils/README.md): Unit tests for serialization, deserialization, UCX utility functions, and UCX active messages.
- [Device Proxy](../../src/utils/device/proxy/README.md): Host CPU proxy turning GPU command rings into backend transfers; proxyTransport ownership, threading, and requests.
- [File Utilities](../../src/utils/file/README.md): Shared QueryMem and path-mode FILE_SEG helpers for POSIX, HF3FS, GDS, and GDS_MT.

## Examples
Runnable Python, C++, Rust, storage, and expert-parallel transfer examples.
- [NIXL Examples](../../examples/README.md): Python, C++, and Rust overview; two-peer registration, prepped and combined transfers, reposting, and notifications.
- [C++ Examples](../../examples/cpp/README.md): C++ examples for nixlAgent transfers, etcd metadata exchange, and telemetry reading.
- [NIXL EP: Expert-Parallel Communication Example](../../examples/device/ep/README.md): Device-API MoE dispatch and combine over RDMA and NVLink, Buffer, connect_ranks, and elastic scaling.
- [Elastic Test Suite](../../examples/device/ep/tests/elastic/README.md): NIXL EP elastic.py single-node and multi-node rank expansion and contraction test plans.
- [Python Examples](../../examples/python/README.md): Python two-peer tensor transfers, GDS, partial metadata, QueryMem, notifications, and telemetry.
- [NIXL Storage Transfer Sample](../../examples/python/remote_storage_example/README.md): Client-server POSIX and GDS storage transfers with UCX networking and pipelined remote I/O.
- [Rust Examples](../../examples/rust/README.md): Single-process NIXL transfer example using Rust bindings.

## Benchmarks
KVBench workloads, NIXLBench backend benchmarks, and etcd coordination runtime.
- [NIXL KVBench](../../benchmark/kvbench/README.md): LLM KV-cache transfer workloads, custom traffic matrices, RDMA, storage, and etcd coordination.
- [Adding a New Model Architecture](../../benchmark/kvbench/docs/adding-a-new-model-architecture.md): KVBench BaseModelArch subclasses, factory registration, YAML configuration, and KV-cache sizing.
- [Creating a Model Configuration](../../benchmark/kvbench/docs/creating-a-model-config.md): KVBench YAML for parallelism, quantization, sequence lengths, hardware, and access patterns.
- [Custom Traffic Performance Test](../../benchmark/kvbench/docs/ct-perftest.md): ct-perftest and sequential-ct-perftest traffic matrices for asymmetric KV-cache flows.
- [KVCache GDS Profiling](../../benchmark/kvbench/docs/tutorial-gds.md): KVBench and NIXLBench GDS profiling for KV-cache transfers between files and GPU memory.
- [NIXL Benchmark](../../benchmark/nixlbench/README.md): ASIO- or etcd-coordinated UCX, GPUNETIO, GDS, POSIX, and OBJ benchmarks with NIXL/NVSHMEM workers and communication patterns.
- [ETCD Python Runtime](../../benchmark/nixlbench/src/runtime/etcd/README.md): etcd-based MPI replacement for KVBench via C++, pybind11, and _EtcdDistUtils; point-to-point, collectives, barriers, ranks.

## CI, tests, contrib, and GitHub
CI jobs, crash debugging, GPU RDMA setup, tests, Conda builds, AWS testing, and pull requests.
- [Debugging CI Core Dumps](../../.ci/docs/DebugCoreDumps.md): Finding CI sqsh exports, Enroot import via Slurm, locating core files, gdb, and debug symbols.
- [Build Wheel Matrix CI](../../.ci/docs/build-wheel-matrix-ci.md): Jenkins manylinux wheel matrix, Docker stages, UCX plugins, auditwheel, and Artifactory publishing.
- [NIXL CI Overview](../../.ci/docs/ci-overview.md): GitHub Actions, Blossom-CI, Jenkins jobs, triggers, PR gating, nightly runs, and image caching.
- [CI Crash Debug Reference](../../.ci/docs/crash-debug.md): Slurm crash-triggered Enroot exports to NFS, gdb availability, and cleanup.
- [GPU RDMA Test Setup](../../.ci/docs/setup_nvidia_gpu_with_rdma_support_on_ubuntu.md): Ubuntu GPU host setup: drivers, MLNX_OFED, nvidia-peermem, GDRCopy, GDS, MPS, and Docker.
- [Test Descriptions](../../test/README.md): Agent, descriptor, metadata, UCX, Python binding, telemetry, and Google Test suites.
- [DOCA_MEMOS Unit Tests](../../test/unit/plugins/doca_memos/README.md): Run the DOCA_MEMOS unit tests with mocked DOCA calls, so no BlueField hardware is needed.
- [DOCA GPUNetIO Unit Test](../../test/unit/plugins/gpunetio/README.md): DOCA GPUNetIO GDAKI GPU transfer test with attached and pooled CUDA streams.
- [Conda Build Options](../../contrib/README.md): Conda C++ and Python binding builds and installs, plus manylinux wheel builds with OpenSSL 4.
- [NIXL AWS Testing](../../contrib/aws-efa/README.md): AWS Batch, EKS, and GitHub Actions tests for EFA and the S3 plugin.
- [Pull Request Template](../../.github/pull_request_template.md): What, Why, and How prompts for the change, its justification, and design.

## C++ and Python APIs
Public C++ headers and Python modules for agents, descriptors, backends, telemetry, and logging.
- [nixlAgent (nixl.h)](../../src/api/cpp/nixl.h): Declares nixlAgent for backends, memory registration, transfers, memory views, notifications, and peer and etcd metadata.
- [Descriptors (nixl_descriptors.h)](../../src/api/cpp/nixl_descriptors.h): Declares nixlBasicDesc, nixlStrideDesc, nixlBlobDesc, nixlRemoteDesc, nixlDescList, and descriptor-list aliases.
- [nixlAgentConfig (nixl_params.h)](../../src/api/cpp/nixl_params.h): Declares nixlAgentConfig for progress and listener threads, synchronization, telemetry, and etcd.
- [Core types (nixl_types.h)](../../src/api/cpp/nixl_types.h): Declares nixl_mem_t, nixl_xfer_op_t, nixl_status_t, nixl_thread_sync_t, nixlAgentOptionalArgs, nixlMemViewH, and nixlXferTelemetry.
- [Backend auxiliary types (backend_aux.h)](../../src/api/cpp/backend/backend_aux.h): Declares nixlBackendInitParams, nixlBackendOptionalArgs, nixlBackendReqH, nixlBackendMD, nixlMetaDesc, and nixlMetaStrideDescList.
- [nixlBackendEngine (backend_engine.h)](../../src/api/cpp/backend/backend_engine.h): Declares nixlBackendEngine and its capability flags for registration, connections, transfers, notifications, queryMem, and telemetry events.
- [nixlBackendPlugin (backend_plugin.h)](../../src/api/cpp/backend/backend_plugin.h): Declares nixlBackendPlugin, creator template, plugin API, and dynamic loading entry points.
- [nixlTelemetryExporter (telemetry_exporter.h)](../../src/api/cpp/telemetry/telemetry_exporter.h): Declares nixlTelemetryExporter, exporter initialization, batch hooks, and bind-failure handling.
- [nixlTelemetryPlugin (telemetry_plugin.h)](../../src/api/cpp/telemetry/telemetry_plugin.h): Declares nixlTelemetryPlugin, exporter creator, plugin API, and dynamic loading entry points.
- [Python package exports (__init__.py)](../../src/api/python/__init__.py): Re-exports nixl_agent, nixl_agent_config, nixl_backend_handle, nixl_prepped_dlist_handle, nixl_xfer_handle, nixl_thread_sync_t, and DEFAULT_COMM_PORT.
- [Python nixl_agent API (_api.py)](../../src/api/python/_api.py): Implements nixl_agent, configuration, descriptor helpers, memory registration, transfers, notifications, and metadata.
- [Python logging (logging.py)](../../src/api/python/logging.py): Configures Python logging, get_logger, NIXL_LOG_LEVEL, and NIXL_DEBUG_LOGGING.
