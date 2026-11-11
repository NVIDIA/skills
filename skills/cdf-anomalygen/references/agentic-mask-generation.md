# Agentic mask generation reference — Step 2, `mask_backend=amg` (the default)

Progressive-disclosure detail for **Step 2 — mask placement** in the parent `SKILL.md`. Read this for
model resolution, the model-service bootstrap and its lifecycle, what a run costs, resume, and how a
run reports its own deviations. If this file conflicts with `SKILL.md`, `SKILL.md` wins.

## What it is, in one paragraph

AMG is an agentic alternative to allocate → pair → place: a VLM **proposes**
candidate masks on the clean image, **refines** them over several rounds, and an independent
**judge** scores each on six dimensions; only masks that clear the acceptance bar are delivered. It
reads the same dataset layout as AMP (clean images + per-defect example images and masks), needs no
`defect_spec` routing, and writes the same deliverable: placed masks + `testcase.jsonl`. Downstream
steps cannot tell which backend produced their input.

## Model resolution — two cases

A **VLM API is a base, a key and a model name**, wherever those come from. AMG makes no distinction
between one vendor's endpoint and another's, or between a hosted service and one running on this
host: they are the same three settings, and all three live in ONE place — **`~/.anomalygen/endpoints.yaml`**,
the model registry (`--model-registry <path>` to use another file; the `$HOME` one is picked up
automatically from any directory, and stays writable inside the read-only product containers).
Nothing is built into the code: the template `scripts/agentic_mask_generation_serving/endpoints.example.yaml` carries the known-good
entries (the cloud recipe and the shipped local gemma recipe), and copying it to
`~/.anomalygen/endpoints.yaml` is the whole setup (the file lives outside the repo, so a real key never enters version control or a docker build context).

| #   | State                              | What to do                                                                                                                                                                                |
| --- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | The models resolve in the registry | **Use it.** An entry (`model` + `base_url` + `key`) per model name is the whole requirement; `--gen-model`/`--judge-model` say which entries a run uses.                                  |
| 2   | Anything less                      | **Stop and ask the user** — supply an API, let the skill start the model service, or switch to `mask_backend=amp`. Never decide for them; the two backends do not produce the same masks. |

`--gen-model` and `--judge-model` are **required flags — there is no default model** — and every
name they carry must resolve in the registry: the entry point resolves all of them before it builds
the transport and **refuses to start when one does not** — there is no environment fallback to
silently reroute a name. The key's *value* is ignored
by an endpoint that does not check keys — so `key: EMPTY` (or omitting `key:`) is enough for an
unlocked local server. The run's opening `model routing:` log lines say where each role resolved;
they carry only whether a key is set, never its value.

### Which endpoint a model name reaches

**Exactly two routing layers**, answering different questions:

- **The base picks the envelope.** Every base is an OpenAI-compatible endpoint and every call to it
  goes out as `POST <base_url>/chat/completions` with `authorization: Bearer <key>`, **whatever the
  model on it is called** — a model *named* `anthropic/claude-*` (OpenRouter's id style) included.
  The one exception is a property of the world, not configuration: **the public Anthropic API**
  (`https://api.anthropic.com/v1`, the one base constant in the code) serves exactly one protocol —
  the native `/messages` wire format — so every model on it takes the Anthropic route,
  authenticated the way that endpoint requires (`x-api-key: <key>` +
  `anthropic-version: 2023-06-01`, not a Bearer header). Only Claude models live there; a wrong
  name fails at the capability smoke before the run spends.
- **The model name picks the cache markers.** A claude-named model (an id containing `claude` or
  `anthropic`) carries the transport's `cache_control` breakpoints whatever base it resolves to:
  natively on `/messages`, and embedded in OpenAI block form on `/chat/completions` — the
  LiteLLM/OpenRouter gateway convention, which such gateways forward to their Anthropic backends,
  so prompt caching survives the chat envelope. Every other model gets the clean payload a plain
  OpenAI-compatible endpoint expects, with no marker anywhere in it.

`<base_url>` carries the version segment, exactly like the OpenAI SDK's `base_url` — the transport
appends **only the route suffix** above, never a `/v1` of its own.

**A user's own OpenAI-compatible endpoint therefore needs nothing added to this codebase** — one
`endpoints.yaml` entry. Put the provider's **documented OpenAI base_url into `base_url:` verbatim**
— the same string their own `OpenAI(base_url=...)` example uses, whatever version path it carries
(`https://.../v1`, Gemini's `https://generativelanguage.googleapis.com/v1beta/openai`, ...) — put
their key in `key:`, pass their model name to `--gen-model`/`--judge-model`, and the call goes out
as `POST <their base_url>/chat/completions` with `authorization: Bearer <their key>` — the model
name plays no part in picking the envelope. That payload carries the output cap as
`max_completion_tokens` and no sampling parameters, which is what OpenAI's own endpoint requires of
a reasoning model such as GPT-6 Astra (it rejects `max_tokens` and `temperature` with HTTP 400);
vLLM reads `max_completion_tokens` into the same field as its deprecated `max_tokens`, and LiteLLM
gateways translate it per backend — so the shipped cloud model works from `https://api.openai.com/v1`
directly with nothing but its registry entry (the example file carries it, commented out).

Routing is **per model name**, so generation and judging can sit on different endpoints in one
command (a local Gemma generating while a cloud judge scores) — each name simply carries its own
entry. When entries for a cloud model **and** a local service both exist, the **model name
decides**: the required `--gen-model`/`--judge-model` flags name exactly which entry each role
uses — never a detector, so the same command always reaches the same endpoint.

The judge matters more than the generator: a weak judge rates bad masks highly, and every
quality-improving option in the engine routes through the judge.

## Starting the model service (option b)

Needs a GPU. The weights — `google/gemma-4-31b-it`, ~62.6 GB, apache-2.0, ungated — are a
checkpoint like every other model: `scripts/download_checkpoints.sh gemma` fetches them into
`checkpoints/hf`, and preflight's checkpoint group verifies them. The serving recipe
defaults `HF_HUB_CACHE` to `checkpoints/hf`, so a start finds them there.

The `gemma` argument matters here and nowhere else in this skill: bare, the script provisions all
eleven groups. Named, it fetches this model only.

**The service is a Step 2 thing and only a Step 2 thing.** Both mask sets — validation and generation
— are Step 2 products, so it is started once, both runs go against it, and it is stopped before the
step ends. It holds a large share of the GPU (`--gpu-memory-utilization 0.9`), and Step 3's training
and Step 4's SDG need that GPU: leaving it up is how a fine-tune OOMs for a reason nothing in its own
log explains. Starting is idempotent — probe `serverctl.sh health` first and start only if it is down
— and stopping is part of the step's gate (parent `SKILL.md`, Step 2).

### In a container — the design is ONE container, no sidecar

The shape is one container, no sidecar: vLLM runs as a background process **inside the running
container**, and AMG in that same container reaches it on plain `localhost:8902` — so nothing is
exposed, no second container is orchestrated, and the base URL needs no host name. vLLM lives in the
app venv: vllm shares the image's torch (`docker/Dockerfile` pins both), so the `vllm` on
the default PATH is `/opt/venv/bin/vllm` — the same interpreter as everything else.

Weights reach the service through the repo bind-mount itself: they live in `checkpoints/hf` with
every other checkpoint (see above), and the recipe defaults `HF_HUB_CACHE` there — so the documented
`docker run` needs **no weight flags at all**: no extra mount, no `HF_HOME`.

The image and its `docker run` invocation are owned by the **`cdf-anomalygen-release`** skill —
`skills/cdf-anomalygen-release/SKILL.md` and its `references/running.md`; take both from there rather
than reconstructing them here. The serve command inside the container is the same recipe as on a dev
host below: the image carries the whole repo at `/workspace/cdf-anomalygen`,
`scripts/agentic_mask_generation_serving/` included. Everything above about **when** the service runs — Step 2 only,
idempotent start, stop before the step's gate — holds identically in a container.

### On a dev host

```shell
GPUS=$(nvidia-smi --list-gpus | wc -l)
[ "${GPUS:-0}" -ge 1 ] || GPUS=1                       # one device is the floor, not a failure
mkdir -p results
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS - 1)))" TP="$GPUS" \
    setsid scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh > "results/gemma_server_${NAME}.log" 2>&1 &
scripts/agentic_mask_generation_serving/serverctl.sh wait      # 0 up · 1 no such process · 2 still loading, call again
scripts/agentic_mask_generation_serving/serverctl.sh health    # /v1/models AND a real completion
```

The log carries `${NAME}` for the same reason the two AMG logs carry `${AMG_MODE}`: a fixed filename
means the next run silently overwrites this one's startup record — and that record is the only evidence
of *why* a start failed, which is exactly what you go looking for when `wait` reports the process gone.

**`wait` is called repeatedly, like `train_control.sh wait`.** Each call is a ~9.5-minute slice
(`--timeout`, default 570s) and reports state as its exit code:

| exit | meaning                                 | what to do                                                |
| ---- | --------------------------------------- | --------------------------------------------------------- |
| 0    | `/v1/models` answered                   | run `health`, then Step 2                                 |
| 1    | no `vllm serve --port 8902` process     | read `results/gemma_server_${NAME}.log`; it is not coming |
| 2    | process alive, socket not accepting yet | call `wait` again                                         |

**A cold start is 10–15 minutes on one GPU and `health` reports `LOADING` for all of it.** Measured on a
single RTX PRO 6000 at `MAXLEN=65536`: weights 352s, `torch.compile` 96s, CUDA-graph capture 16s, then
chat-template detection — first successful `/v1/models` about **14 minutes** in. Roughly 3.5 of those
minutes came *after* the log printed `Supported tasks: ['generate']`, which is emitted before uvicorn
binds the socket. Treat exit **2** as progress, not a stall: restarting on it leaves two vLLMs on one
GPU, each asking for `--gpu-memory-utilization 0.9`, and the second dies in a way that reads like the
original failure.

`TP` must equal the number of devices `CUDA_VISIBLE_DEVICES` names. `MAXLEN` (default 32768) is
the serving window every request must fit prompt + answer inside. The engine's agent output budget
is a FIXED constant (8000, like every other per-call budget — no env, no registry field), sized so
prompt + budget fit the default window with the measured prompt maximum (~23.4k); a window smaller
than 32768 rejects big prompts — which shows up as whole types stalling on
`degraded[api_abort:bad_request]` — and the remedy is a larger `MAXLEN`, never a smaller budget.
`setsid` because a job backgrounded from an agent's shell is signalled when that shell goes away.

Then run Step 2 against it — the wiring is `endpoints.yaml`'s gemma entry (from the template),
which resolves `nvidia/google/gemma-4-31b-it` to `http://localhost:8902/v1`. Serve on any other
port/host, or lock the server (`API_KEY=<secret>` at startup), and it is the same entry: point its
`base_url:` at the new address and put the matching value in `key:`:

```shell
M=nvidia/google/gemma-4-31b-it
python -m anomalygen.scripts.agentic_mask_generation.generate \
    --dataset-path "${DATASET_DIR}" --anomaly-types <texture>+<defect> [...] \
    --num-sdg "${NUM_SDG}" --gen-model "$M" --judge-model "$M" \
    --beam-multi 4 --beam-pairwise \
    --exemplar-captions --exemplar-visual-prompt --coverage-note \
    --visual-prompt auto --defect-library auto --out-dir "${AMG_DIR}"
```

**Gemma always runs with the in-loop beam on — `--beam-multi 4 --beam-pairwise` — and with the three information channels open: `--exemplar-captions --exemplar-visual-prompt --coverage-note` — as above.** The
defaults leave both off because a strong proposer's single refine turn is enough; a
smaller local model's is not. `--beam-multi 4` asks each refine turn for 4 distinct-strategy fix
options in one call and keeps the best, and `--beam-pairwise` has the selector compare those options
A-vs-B instead of scoring each alone — the easier judgment for the same model that is also judging.

The three information channels ride with the beam for the same reason — they hand the smaller model
what a strong proposer infers on its own, and each defaults OFF so the cloud recipe's prompts stay
byte-identical:

- `--exemplar-captions` — each real example travels with its own distill-time caption (what its
  annotation sits on and how fully it covers it), so the card's one-line summary is never the only
  reading of the evidence.
- `--exemplar-visual-prompt` — each example is also shown as the visual prompt's own painted map
  with its REAL mask outlined: where annotations live is a picture, not a phrase that can be read
  against the wrong feature.
- `--coverage-note` — the judge, the refiner and the pairwise selector all read the same measured
  coverage note (the candidate and the real annotations measured with ONE scale-free yardstick),
  so "how full is full" is a number, not an impression.

Together they cost roughly a quarter more wall-clock per mask; leave them off for cloud models.

`scripts/agentic_mask_generation_serving/serverctl.sh stop` ends it. The recipe and the control script are
`scripts/agentic_mask_generation_serving/`; the serving limits and their failure modes — the context window, the
per-prompt image cap, why `health` asks for a completion rather than trusting `/v1/models` — are
the "Serving notes" in the tutorial's Prerequisites
(`docs/tutorial/texture/3-agentic-mask-generation.md`). One of them bears repeating here: `stop`
matches the port and never the model name (`pkill -f gemma` also matches the shell that launched
it). One more lives in the recipe instead: it prepends `$CUDA_HOME/bin` to `PATH` whenever `CUDA_HOME` is set, which is how
a vLLM that wants a CUDA toolkit at startup finds one on a host with no system install — export
`CUDA_HOME` at the bundled toolkit inside the venv vLLM runs from before the launch above.

**That last one is a dev-host concern only — skip it in the container.** The image ships no CUDA
toolkit binaries at all (`/usr/local/cuda` carries `compat/` and `lib64/`, no `bin/`, no `nvcc`), and
vLLM starts there without one because the recipe disables every FlashInfer path it would otherwise
take — the sampler, and at TP>1 the allreduce fusion pass and all-reduce communicator — since those
kernels are JIT-compiled and need `nvcc`. Leave `CUDA_HOME` unset rather than hunting for a toolkit
that was never installed.

## Before it spends: the capability probe

Every run sends **one call per distinct model** in `{--gen-model, --judge-model}` — two tiny
generated images and a request for `{"n_images": <int>}` — down the same transport that model will
use for the real work, before the first defect card is distilled. A model that will not take images
or will not answer in JSON fails every call AMG makes, and this is what turns that into a refusal
that costs one call instead of a batch:

```text
degraded[model_smoke_failed]: <model>: <what it could not do>
```

and a non-zero exit whose message names the fix — a credential the endpoint refused, a model that is
not serving images, or a model that would not answer in JSON. `--skip-smoke` turns the probe off,
which moves that same failure into the middle of the batch.

## Cost and time

Defaults are quality-first: the throughput levers (`--beam-prune --accept-continue
--judge-cache-order`) are OFF. Offer them when the user asks for throughput or the batch is large —
on the `azure/anthropic/claude-opus-5` measurement below they cut the calls per delivered mask to
roughly a third — and state that they trade measured quality for that saving when you do.

**A small per-type share is dearer per mask.** Each type pays a fixed setup (defect-card
distillation, valid-region enumeration, visual-prompt precompute) that amortizes over however many
masks the type delivers, and the batch overshoot the trim discards is proportionally larger when the
target is small. Measured on `azure/anthropic/claude-opus-5`: a 40-masks-per-type run pays **~10
calls per delivered mask** (20–30 minutes wall); at 5 masks per type across 5 types it is **~13–15
calls per delivered mask** (368 calls for 25 masks, then 324 for another 25), each type overshooting
to 10–14 accepted masks before trimming to 5. Budget accordingly when a run is many types × few
masks rather than few types × many.

## Flags that matter (everything else has working defaults)

- `--num-sdg N` — delivered masks for the WHOLE run, split across the types the way AMP splits its own
  `--num_sdg`. Total rows = **exactly `N`**. This is what `SKILL.md` Step 2 runs.
- `--mode inference|validation` — how that split is made, matching AMP's `--mode`. `inference`
  (default, the generation set): even, and a type may get 0. `validation`: proportional to each type's
  real mask count, with a zero share refused — a type absent from the validation set drops out of the
  per-type KPI that chooses the best checkpoint.
- `--per-defect-counts '{"T+D": 12}'` — exact targets per type; omitted types get 0. `inference` only,
  and its sum wins over `--num-sdg`. AMP's `--per_defect_counts` by another spelling.
- Omit both `--num-sdg` and `--per-defect-counts` for one pass over every clean image, with no
  target at all.
- `--shape flexible|auto|fixed` — how each type's mask geometry is made. `auto` (default): a per-type
  judgment picks between synthesizing the shape (flexible) and stamping a real training-mask shape
  (fixed) — except a type whose card says the defect IS a whole host structure, which never stamps
  (the mask must take its geometry from the target board). `flexible`/`fixed` force one path for
  every type; `fixed` wins over the card.
- `--gen-model` / `--judge-model` — REQUIRED, per role; every name carries its own registry entry
  (`--model-registry` / `endpoints.yaml`), and a name with none refuses the run.
- `--out-dir` — one run per output set (validation, generation), like AMP's two seeds.
- `--skip-smoke` — skip the pre-run capability probe above.
- `--overwrite` — **destroys** the out-dir's masks and starts over. Rarely what you want; see resume.

## Resume — interruption is cheap

Re-running the same command into the same `--out-dir` **continues the run**: already-delivered masks
are counted, mask filenames never collide, and only the shortfall is produced. A run whose settings
differ from what the out-dir holds is refused with the differing fields named. There is no `--resume`
flag — matching settings are the switch. This also means a dead endpoint mid-run costs only the
shortfall: the run stops loudly, keeps what it earned, and says how to continue.

## The run reports its own deviations — read them

Every time the run does something other than what was asked it logs one line:
`degraded[KIND]: detail`. Before trusting a batch, summarize them:

```shell
for f in "${LOG_DIR}"/amg_*_"${NAME}".log; do
  echo "== $(basename "$f")"
  grep -o 'degraded\[[^]]*\]' "$f" | sort | uniq -c | sort -rn
done
```

Per log, not merged: the validation and the generation set are separate deliverables, and a clean
validation run must not dilute a dirty generation one into looking acceptable.

Frequent kinds and what they mean: `json_repaired:*` (a truncated model reply was salvaged — benign
in small numbers) · `vp_auto_no_card` (**the run had no visual cue** — the score is not comparable;
pass `--defect-library auto`) · `api_abort:bad_request` (the model's context cannot fit the request —
on a served model raise `MAXLEN`) · `model_smoke_failed` (a model failed the capability probe and the
run stopped before spending) · `type_failed` (a type delivered nothing; the run says why and exits
non-zero).

## `spatial_dependency` and AMP interplay

AMG does not read `defect_spec`'s `spatial_dependency` — it decides placement by looking at the
board, per candidate, and the judge enforces it. Every type in `--anomaly-types` is placed by AMG, so
the run's `testcase.jsonl` is the whole deliverable. When the field is present, log the notice (parent
`SKILL.md` has the exact line) and leave the default alone; `mask_backend=amp` is the supported way
to honour `free`/`text`/`cad` routing.
