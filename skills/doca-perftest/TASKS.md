# DOCA Perftest: tasks

Source for every flag and behavior: the DOCA 3.5 guide, [DOCA Perftest](https://docs.nvidia.com/doca/archive/3-5-0/doca-perftest/index.html). Run `doca_perftest -h` on the installed binary; the flag set can change between releases.

## point-to-point

Preconditions: same DOCA Perftest version on both hosts, RDMA device visible (`ibv_devinfo` or `ibstat`), passwordless SSH from client to server for auto-launch.

Smoke (auto-launch):

```bash
# server host is the target
doca_perftest -d mlx5_0 -n <server-host-name>
```

Equivalent explicit run (use when SSH auto-launch is not configured):

```bash
# server
doca_perftest -d mlx5_0 -N 1 -c RC -v write -m bw -s 65536 -D 10
# client
doca_perftest -d mlx5_0 -N 1 -c RC -v write -m bw -s 65536 -D 10 -n <server-host-name>
```

Flags used above: `-d` device, `-N` process count (`-C <core>` to pin), `-c RC` transport, `-v write` verb, `-m bw` metric (`lat` for latency), `-s` message size in bytes, `-D` duration in seconds, `-n` remote host (client only).

To stop the server from auto-launch: add `--launch_server disable` on the client and start the server by hand.

Done when: both sides report the same parameters and the client prints a bandwidth result. Record the device, verb, size, duration and DOCA version with the number.

## select-driver

- Default: `-r ibv` (libibverbs, always available).
- `-r dv`: DOCA Verbs backend. Needs the DOCA Verbs package; if it is missing the tool exits with an error instead of falling back.

Run the same command with both drivers and compare; state the driver in every reported number.

## select-memory

- `-M host` (default), `-M nullmr`, `-M device`.
- GPU memory: `-M cuda -G <gpu-id>`. Use `cuda_auto_detect` (the recommended syntax in the guide) or `cuda_peermem` / `cuda_dmabuf` / `cuda_data_direct` for an explicit path. The deprecated `cuda_auto_detect` alias forms still work but do not copy them into new scripts.

Check that `nvidia_peermem` (or the DMA-BUF path) is loaded before running GPU memory; without it the run fails at memory registration.

## multi-node

Prerequisites: OpenMPI bundled with DOCA on every node, the same binary path on every node, and passwordless SSH.

Use one of:

- **CLI traffic pattern**: `--traffic_pattern` (one-to-one, one-to-many, many-to-one, all-to-all, bisection, ring).
- **JSON scenario**: a file that defines several tests, devices, hosts and patterns; one run produces the aggregate report.

Launch from one host; results are collected to a single report (CLI: a results file; JSON: the scenario report). Use `-h` or the guide section "Common Arguments and Use-Cases" for the exact keys.

Done when: the aggregate report lists every node and every test with a non-zero result. A partial list usually means one node failed to launch; check its SSH and MPI reachability first.

## latency

- `-m lat` with `-v write` or `-v send`. Read the latency mode section of the guide before comparing: one-way uses local send-completion timing, which is not the same as round-trip.
- Keep the message size and core placement identical across runs.

## gpunetio

Only for GPU-initiated RDMA through DOCA GPUNetIO.

```bash
doca_perftest -d mlx5_0 -n <server-host-name> -M cuda -G 0 -r dv --engine gpunetio
```

Limits: Write verb only, unidirectional only, needs `-r dv`, CUDA memory, `doca-gpunetio`, and a Volta+ GPU.

## debug

Work the layers in order. Stop at the first failure and fix it before going on.

1. **Device open fails.** Check the device name (`ibdev2netdev`, `ibv_devices`); another process may hold it (`fuser` on the device node or a stale `doca_perftest`). Do not kill another team's process without asking.
2. **Client cannot launch server.** Check passwordless SSH to the server, the same binary path and version on both sides, and that the server port is free. Run the server by hand with the same parameters to separate the two problems.
3. **DV backend error.** Confirm the DOCA Verbs package is installed and that `-r dv` is supported on this device. The tool reports this explicitly; read the message rather than retrying.
4. **GPU memory registration fails.** Check `nvidia_peermem` (or DMA-BUF support) and the CUDA device id.
5. **Numbers are far below line rate.** Check the core and NUMA placement (`-C`), message size (small messages are message-rate bound, not bandwidth bound), queue-pair count (`-q` or the QP histogram option in the guide), and whether both ends run the same version.
6. **Latency numbers disagree between runs.** Confirm the same latency mode, message size and CPU pinning; reboot-free noise sources are frequency scaling and IRQ affinity.

Exit gate for a debug session: one run with the parameters recorded and a result that matches the expected order of magnitude for the link. Name the layer that was wrong.
