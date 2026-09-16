# Container Readiness

Run after self-hosting is approved. `preflight.md` proves the host has a GPU. This file
proves containers can use it.

Complete every check in this file before post-approval `list-model-profiles` or any large
image or model pull. Profile discovery may pull the selected NIM image, so storage and
cache checks are part of this gate.

## 1. Docker

Require all three commands to succeed:

```bash
docker version
docker info
docker compose version
```

`docker version` must show a reachable server, not only a client. Stop on daemon,
permission, or Compose-plugin errors. Report the failing layer. Do not silently add
`sudo` or change Docker configuration.

## 2. NVIDIA Container Toolkit

Follow the current
[NVIDIA Container Toolkit sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html).
Its Docker check is:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

This intentionally pulls a small image when it is not cached. Require the container output
to list the same intended GPU devices seen on the host. A working host `nvidia-smi` alone
does not pass this check.

If the command fails, stop before any NIM, vLLM, or speech container pull. Point to NVIDIA
Container Toolkit installation or Docker runtime configuration based on the actual error.

## 3. Storage and Cache Paths

Get Docker's storage root with:

```bash
docker info --format '{{.DockerRootDir}}'
```

Check free space on that filesystem and every host path that the generated
`compose.yaml` will mount. Depending on the routed stack these include the locked NIM
cache, the Hugging Face cache, the compiled kernel cache, and the speech model tree.

For each host-mounted cache:

1. resolve the exact path from the approved deployment source
2. create it only when it does not exist and the parent is approved
3. require it to be a directory and writable by the host account and container user
   required by the source launch command
4. check free space before pulling

Do not replace an existing cache, change ownership recursively, or delete files to make
space. Report the required path and available space when the check fails.

## 4. Credentials and Registry

Reconfirm the required keys from `preflight.md` without printing them. Use the current
build.nvidia.com login instructions for NGC. A registry authentication failure is a hard
stop before model pulls.

## 5. Speech Model Tree

Required when the routed stack is the single-GPU stack in `platforms/single-gpu.md`, which
covers DGX Spark, Jetson Thor, and a low-concurrency workstation. The speech service loads
GGUF files from a read-only mount, so it cannot create or repair anything itself.

Require all of the following before the one-time download:

- `hf auth whoami` succeeds for the locked LLM repository and the speech model
  repositories
- a writable parent path outside the generated project, with space for the ASR GGUF, the
  Magpie GGUF and its extracted tokenizer, the codec GGUF, and the normalization grammars
- space for the Hugging Face cache and the compiled kernel cache, which are separate from
  the model tree and much larger
- on Jetson Thor, the JetPack release the current platform documentation supports

Two ownership rules decide whether the service can read the models.

**Create the model tree before the first `compose up`.** When Compose starts with the bind
mount missing, Docker creates that host path as root. The download then cannot write to
it, and the failure reads like a tooling problem rather than a permissions one. If it has
already happened, reclaim ownership for the user account and retry.

**Do not run the download with `sudo` to work around it.** Elevating changes the PATH, so
the user's Hugging Face CLI disappears and the files land owned by root again. Fix the
ownership, then run the download as the account that owns the path.

After the download, require the tree and every file in it to be readable by the container
user, which is not the host account that downloaded them. Assert the expected files exist
by name, including the grammar files under each language directory, rather than trusting
that an archive extracted correctly.

## Pass Condition

Proceed to exact local profile discovery, `output-contract.md`, and the routed deployment
guide only when:

- Docker daemon and Compose are reachable
- a container can see every assigned NVIDIA GPU
- all required host-mounted paths are writable
- Docker storage and model caches have enough free space
- required registries and model sources can authenticate
- the speech model tree checks above pass when the routed stack is single-GPU

Record the commands, expected GPU assignment, and cache paths in the generated README.
