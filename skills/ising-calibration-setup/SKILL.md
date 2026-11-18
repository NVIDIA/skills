---
name: ising-calibration-setup
description: >-
  Use when connecting to the NVIDIA Ising Calibration model via the hosted
  API or a self-hosted NIM. Covers auth, keys and connectivity checks only.
license: Apache-2.0
version: "0.1.0"
permissions: [env, network, shell]
tags: [nvidia, ising, quantum, calibration, nim, api, setup]
metadata:
  author: "NVIDIA"
  version: "0.1.0"
  github-url: "https://github.com/NVIDIA/Ising"
  tags: [nvidia, ising, quantum, calibration, nim, api, setup]
---

# Ising Calibration Setup

## Purpose

Get a user from zero to a **verified working connection** to the NVIDIA
Ising Calibration model (Ising-Calibration-1.5-31B), in either of two ways:

- **Hosted API** on build.nvidia.com -- fastest, no GPU required.
- **Self-hosted NIM** container -- for on-premises, data-residency, or
  high-volume use, requires a supported NVIDIA GPU.

This skill is plumbing-scoped: getting a key, getting a `200` back, and
confirming that connection actually works. It stops there -- it does not
teach how to use the model (prompting it for calibration analysis, building
an agent workflow around it, etc.). See "Next steps" below for where that
guidance lives.

## Instructions

Ask the user which they want (or infer it: "I don't have a GPU" -> hosted API;
"I need this on-prem / air-gapped / high volume" -> self-hosted NIM). Note that
the self-hosted container still needs internet access on its first run to
download the model weights. Then follow the matching section below
top-to-bottom. Full command sequences live in `references/`; this file only
handles routing, prerequisites, and the connectivity checklist.

Getting the credential is where users most often get stuck, and it happens in a
browser where you cannot see what they see. Read the relevant reference doc in
full before starting, then take them through one screen at a time, saying what
they should expect to see and waiting for confirmation before moving on, rather
than handing over a URL and waiting for them to return with a key. Never ask the
user to paste a key into the conversation; have them load it into their own
shell and verify it there.

Name the two setups when talking to the user: "the hosted API at
build.nvidia.com" and "a self-hosted NIM container". Refer to each step by what
it does ("generating an NGC personal key", "pulling the NIM container") rather
than by its number in the lists below.

### When to use

- "How do I connect to Ising Calibration?"
- "Set up the Ising Calibration model"
- "Get an API key for build.nvidia.com / Ising"
- "Run / self-host the Ising Calibration NIM" (own hardware or a cloud GPU instance)
- "My Ising Calibration API calls are failing, check my setup"
- "/ising-calibration-setup"

### Prerequisites (either setup)

- An NVIDIA account (create one at build.nvidia.com or ngc.nvidia.com if the
  user does not have one).
- `curl` available locally for connectivity checks.
- `jq`, only if the user wants the optional hello-world request at the end. The
  reference docs give a `python3` fallback if it is not installed.

## Hosted API (build.nvidia.com)

No GPU needed; requests are billed/rate-limited against the hosted endpoint.

1. Sign in at `https://build.nvidia.com`.
2. Generate a personal API key: profile menu -> Settings -> API keys ->
   **Generate API Key** (or go directly to `https://build.nvidia.com/settings/api-keys`).
   You will be asked to name the key and set an expiration. The key starts
   with `nvapi-` and is shown only once -- store it immediately without
   letting it land in shell history (see
   [`references/hosted-api-quickstart.md`](references/hosted-api-quickstart.md)
   for a `read -rs`-based method).
3. The model page, for rate limits and current terms:
   `https://build.nvidia.com/nvidia/ising-calibration-1.5-31b`.
4. Follow [`references/hosted-api-quickstart.md`](references/hosted-api-quickstart.md)
   for the endpoint, the request format, and a working `curl` example.

Governed by the NVIDIA API Trial Terms of Service.

## Self-hosted NIM container

Requires an NVIDIA Ada Lovelace, Hopper, or Blackwell GPU with enough memory (see
the reference doc for tested configurations), plus Docker with the **NVIDIA
Container Toolkit** installed -- `docker run --gpus all` fails without it.

Walk the user through these rather than summarizing them; the full walkthrough
is in [`references/self-hosted-nim-quickstart.md`](references/self-hosted-nim-quickstart.md),
which you should read before starting.

1. Sign in at `https://org.ngc.nvidia.com`.
2. Go to `https://org.ngc.nvidia.com/setup/personal-keys` (or: account icon ->
   Setup -> Generate Personal Key) and click **Generate Personal Key**.
3. Fill in the dialog's three required fields: **Key Name**, **Expiration**, and
   **Services Included** -- a dropdown of per-service checkboxes. **NGC Catalog**
   is the only scope needed; it grants pulls from the public nvcr.io catalog,
   and without it `docker login` fails with an auth error even though the key is
   valid.
4. The key is shown **once**. Have the user copy it and load it into their shell
   with `read -rs NGC_API_KEY` (not `export NGC_API_KEY=...`, which lands the
   raw key in shell history), then confirm it is set with a check that does not
   print it: `test -n "$NGC_API_KEY" && echo set`.

   This is a **separate credential** from the `nvapi-` key used for the hosted
   API. If the user already has an `nvapi-` key, tell them it will not work here
   and they need to generate this one as well.
5. Log Docker in to the NVIDIA container registry -- this is also the check that
   the key and its scopes are right, so do it before pulling anything:
   `echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin`
   Expect `Login Succeeded`. If it fails, fix the key here rather than
   continuing to the pull.
6. Pull and run the container, then confirm it is serving. Use the `docker run`
   from the quickstart as written, including `-e NGC_API_KEY=$NGC_API_KEY`,
   which passes the key into the container. Full command, GPU/memory
   requirements, health check, and example request are in
   [`references/self-hosted-nim-quickstart.md`](references/self-hosted-nim-quickstart.md).
7. Run the connectivity checklist. You can then offer the optional hello-world
   request below, the same as for the hosted API.

## Connectivity checklist

Run this for either setup to confirm the connection actually works, and to tell
an auth failure apart from a network/model-not-ready failure. Full checklist
with copy-pasteable commands: [`references/connectivity-checklist.md`](references/connectivity-checklist.md).

Summary:

1. Hosted API: confirm `NVIDIA_API_KEY` is set and non-empty. Self-hosted NIM:
   there is no request-time key -- `NGC_API_KEY` is used for the image pull and
   passed into the container at startup, not sent with requests -- so confirm
   the container is running and ready instead.
2. Send one lightweight request against the endpoint.
3. A `401`/`403` means different things per setup. Hosted API: the key is
   missing, wrong, or expired -- regenerate it. Self-hosted NIM: the container
   itself never returns 401, so a proxy, gateway, or load balancer in front of
   it is rejecting the request -- check that layer, not the NGC key.
4. A connection-refused/timeout against a self-hosted NIM means the container
   is not up yet or the port mapping is wrong -- check `docker ps` and
   container logs before assuming an auth problem.

## Hello world: an optional first real request

A `200` from the connectivity checklist is the bar for a working connection --
setup is done at that point. A text ping does not exercise the multimodal path
though, so a real calibration-plot request is a useful way to see the connection
do the model's actual job. Offer it for either setup and let the user decide
whether to run it. Each reference has a "Hello world" section with the request:

- **Hosted API**: [`references/hosted-api-quickstart.md`](references/hosted-api-quickstart.md)
- **Self-hosted NIM**: [`references/self-hosted-nim-quickstart.md`](references/self-hosted-nim-quickstart.md)

Both pull a real sample image from the public QCalEval benchmark dataset, so no
local test image is needed. The request shape is the same either way; only the
endpoint and model ID differ -- read the self-hosted ID from `/v1/models` rather
than copying the hosted string across.

## Examples

- "I don't have a GPU, how do I try Ising Calibration?" -> hosted API path:
  generate an nvapi- key, store it as `NVIDIA_API_KEY`, verify with the curl
  ping in `references/hosted-api-quickstart.md`.
- "Run the Ising Calibration NIM on our H200 box" -> self-hosted path: generate
  an NGC personal key with the NGC Catalog scope, `docker login`, pull and run
  the container, wait for `/v1/health/ready`.
- "My calls return 401" -> connectivity checklist: establish which setup first,
  then follow the matching Troubleshooting entry below.

## Next steps

Once the connectivity checklist passes, this skill's job is done -- you have a
working, authenticated connection. What to do with it is out of scope here;
go to:

- **Building an agentic calibration workflow?** Start with the
  [Quantum Calibration Agent Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint/)
  -- an end-to-end workflow built on this model.
- **Want the other models in this family (decoders, other calibration
  sizes)?** Browse the
  [Hugging Face collection](https://huggingface.co/collections/nvidia/nvidia-ising).
- **Training or optimizing a predecoder?** See the
  [Ising-Decoding cookbook](https://github.com/NVIDIA/Ising-Decoding).

## Limitations

- Hosted API rate limits and pricing are not published in this skill; check
  the build.nvidia.com model page for current terms.
- GPU memory requirements vary by GPU family -- see
  `references/self-hosted-nim-quickstart.md` for the tested configurations.

## Troubleshooting

- **Error**: `401 Unauthorized` on the hosted API. **Cause**: missing/expired
  `NVIDIA_API_KEY`, or using the NGC key instead of the `nvapi-` key.
  **Solution**: regenerate at `build.nvidia.com/settings/api-keys`.
- **Error**: `docker login` or `docker pull` fails with auth error. **Cause**:
  missing/expired `NGC_API_KEY` (or set in a different shell than the one
  running `docker login`); using the `nvapi-` key instead of the NGC key; or a
  key whose service scopes omit **NGC Catalog**. **Solution**: check the scope
  before assuming the key is bad -- a wrongly-scoped key fails identically to an
  invalid one. Regenerate at `org.ngc.nvidia.com/setup/personal-keys` with NGC
  Catalog included.
- **Error**: `docker login` succeeds but `docker pull` returns 401 or
  "not found". **Cause**: the credential is fine but the image is not visible
  to the account, typically an unaccepted access agreement on the catalog page.
  **Solution**: have the user open the catalog page while signed in to confirm
  the image is visible to their account.
- **Error**: `401`/`403` from a self-hosted NIM endpoint. **Cause**: the
  container does not authenticate requests, so the rejection comes from a
  reverse proxy, API gateway, or load balancer in front of it. **Solution**:
  check that layer's configuration and credentials. To confirm, bypass it:
  on the machine running Docker, curl `localhost:<host-port>`, where the host
  port is the left side of the `-p` flag shown by `docker ps` (8000 if the
  quickstart command was used unchanged). A `200` there proves the container
  is fine and the 401 comes from the layer in front. Regenerating the NGC key
  does not help here.
- **Error**: container starts but the startup weight download fails.
  **Cause**: the weights come from public Hugging Face, so the usual cause is
  outbound network access from the container (proxy, firewall, DNS, or
  huggingface.co blocked). **Solution**: check `docker logs <container>` for
  the failing URL and fix egress.
- **Error**: connection refused on `localhost:8000` (or whichever host port
  was mapped). **Cause**: NIM container still loading weights, crashed, or the
  `-p` host port does not match the one being curled.
  **Solution**: check `docker ps` and `docker logs <container>`; wait for
  `/v1/health/ready` to return `200` before sending requests.
- **Error**: `Insufficient GPU memory` / OOM on container start. **Cause**: the
  GPU does not have enough memory for any configuration the container can run.
  **Solution**: use a GPU with more memory, or more GPUs; see the tested
  configurations in `references/self-hosted-nim-quickstart.md`.
