# The list of compatible models

`nimselect.py list` answers "which models can run on these GPUs?" for the source the Locality step
chose (SKILL.md): for every LLM and VLM NIM of that source it checks the images select checks,
against an inspected environment, with select's own tag, fit and ranking rules
(`references/fit-rules.md`), so each row shows the image and verdict select gives. Read this page
to explain a row, a mark or a count, the region, the cache, a China NIM distributor (CND) list, or
why a row and select can differ.

No NGC API returns "the NIMs that fit these GPUs", the GPU tags of the catalog are on few NIMs, and
the NIM Metadata API answers one NIM and tag per call. So the list reads the catalog, then each
model in turn. It reads only public endpoints, anonymously (`references/security.md`), and pulls,
runs and hands off nothing.

## How an NGC list is made
1. One catalog search of the NIM containers (`GET /search/catalog/resources/CONTAINER?q=...`,
   filtered to NIMs, 1000 a page), and more pages while `resultPageTotal`, or `resultTotal`
   against the NIMs the pages gave, says more remain (an empty page does not end that). The
   catalog is read within the list's 120 s. A catalog read only in part (a later page failed,
   more pages than 5, or a row with no `resourceId` or a blank one, which cannot be checked)
   makes the list `partial`, with a note.
2. `--query TEXT` keeps only the models whose `TEAM/REPO` holds every word of it, ignoring case,
   before any request about a model. A word of letters is found anywhere (`llama` also matches
   `codellama` and `deepseek-r1-distill-llama-8b`); a word with a digit never inside a longer
   number, a size or the minor part of a version (`llama 4` matches `llama-4-scout`, not
   `nemotron-super-49b`, `nano-4b` or `405b`; `8b` not `128b`; `llama 2` not `llama-3.2`), but a
   version written onto the name counts (`llama 3` matches `llama3-70b` and `llama-3.1-8b`). Each
   word is matched on its own, so `llama 2` also matches `llama-3.1-typhoon2`. A leading `nim/` or
   `nvcr.io/nim/` is dropped.
3. Org `nim` only. Left out, and counted in `facts.skipped`: production branches (`-pb6`,
   `-pb25h2`: `production_branch`), containers that load a model given at run time
   (`model_free`), and NIMs that serve no LLM or VLM (`non_llm`: speech, embedding, reranking,
   image or biology NIMs, by the skill's own name lists, written from the catalogs of 2026-11-11;
   after the metadata read, a NIM whose profiles name only other backends than an LLM's counts
   here too). A NIM whose repository record NGC says it does not show (a 401 or 403 with NGC's
   error envelope, such as "Not a catalog artifact!", or a 404) is `not_public`: it cannot be
   checked or picked. A 401 or 403 without that envelope (a proxy's or a firewall's page) is an
   error for that model, never `not_public`. NGC refusing the record of every NIM checked (3 or
   more) fails the list (exit 4, `ngc_http_error`): that is NGC failing, not a source with no NIM.
4. Hardware variants (`<repo>-dgx-spark`, `<repo>-rtx` and the other suffixes select knows,
   `references/fit-rules.md`) fold under their base model.
5. Per model, 8 at a time, the images select checks, in select's order (`references/fit-rules.md`,
   Image choice): the newest version tag (by select's tag rules, never `latest`); when it does not
   fit, the hardware variants, the repository's tag for this GPU (`1.8.0-RTX` on an RTX GPU), and
   the older versions: the earlier builds of the newest's major.minor line, newest first, in the
   places left after the newest build of the two older major.minor lines, which keep theirs; at
   most 6, and no more after the first that fits, unless the tests the repository's other images
   record for the same profile ids could change the row: then it reads every image of the
   repository that select checks and fits each with the others' records, as select does
   (`tested_in_other_image`, `references/fit-rules.md`). When NGC stops answering during those
   reads, the images read are fitted with the records read, as select fits them; when one that
   fits is select's pick whatever the others record (no image before it could fit by a record:
   one whose metadata records no VRAM need cannot), the row is kept, with `records_unread` true,
   and the stop still counts as a model that failed (Time and failures). Each repository record
   gives the tags, CPU architectures, `canGuestPull` and the
   card's Deployment Geography; the image list is read when
   the NIM can be pulled without a key. The row
   shows the image select would pick (choose.pick), with its verdict and `kind`, so a row and
   select agree (a test holds them to that). When the newest image's metadata cannot be read, only
   an older or variant image that fits makes a row; else the model is in `facts.not_checked`.

## A row
| field | meaning |
|---|---|
| `model` | NGC: `TEAM/REPO` of the base model; select takes it at the pick. CND: the image without its tag (`io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b`) |
| `image` | the image the row shows, with its tag: select's choice. A CND image with no tag but latest is shown with no tag (`pinned` false) |
| `verdict` | `fits`, `unknown`, `does_not_fit`, `no_metadata` (NGC publishes no profiles for the images checked) or `ngc_build_unknown` (CND: no known NGC build) |
| `kind` | `newest`; `variant` (a hardware build: another repository or a tag such as `1.8.0-RTX`; `variant_of` names the newest image); `older` (an older version: the newer images checked are not confirmed to fit, even with the tests the repository's other images record for the same profile ids; on 2026-11-19, on 8x H100 NVL, llama-3.1-8b-instruct shows 2.1.7, the newest, whose fp8 profile 2.1.5 records as tested there). An older build whose metadata records less (no VRAM need, no CPU architectures) is `unknown` where a newer image's own records rule it out, and is then the row (`references/fit-rules.md`, Image choice): on 2026-11-11 one GB10 row and two L40S rows, which did not fit with only the newest checked |
| `reasons` | `unknown`: the checks the profile shown left open (`gpu_not_tested`, `vram_not_recorded` ...), and any of the notes below. `does_not_fit`: the rejection codes, the most frequent first. `fits`: notes only. The notes are `pci_inferred` on Kubernetes, and `tested_in_other_image` (the row's image records no test of the profile on this GPU; another image of its repository records one for the same profile id): a note is no open check, so never report one as a check left to confirm. `ngc_build_unknown`: that code |
| `tier` | how the profile shown matched the GPU: `direct`, `tested` or `agnostic` |
| `gpus_min` | `fits`: the fewest GPUs among the profiles that fit; `unknown`: among the best unconfirmed ones; null for the other verdicts |
| `gpu_type` | the GPU type the row judged; null when no profile was read |
| `precision`, `backend` | the precision and inference backend of that profile; null when the metadata records none |
| `pinned` | false when the image has no fixed version tag: NGC, a moving alias such as `1.8` (the repository lists no build); CND, no tag but latest is known for the image, so it is shown with no tag. select at the pick names the version, or asks for it |
| `needs_ngc_key` | NGC: `canGuestPull` is not true, so pulling the image needs an NGC key. nim-select uses none and never asks for one; `nim-deploy` pulls. CND rows: false |
| `needs_busy_gpus` | host: the row needs more GPUs than are idle now (How rows are judged) |
| `geography`, `region_excluded` | the card's Deployment Geography, and whether it excludes `--region` (Region) |
| `variant_of` | the base model's newest image, when the row shows a hardware variant |
| `records_unread` | only on a row kept when NGC stopped answering while the list read the repository's other images for their records: true. The image is select's pick, but select reads them all and may recommend another of its profiles; a note says so |
| `next` | `select`: the step for the row the user picks |
| `ngc_image`, `registry_gpus` | CND rows only (A CND list) |

No row has a profile id or a hand-off: the list names no profile to deploy. Rows come in this
order: `fits` (fewest GPUs first), `unknown` (grouped by `reasons`), `does_not_fit`,
`no_metadata`, `ngc_build_unknown`; by model within each.

## What the list prints
To keep the agent's context small, the JSON object on stdout carries every fact but, in `rows`,
only the `fits` and `unknown` rows, brief: `model`, `image`, `verdict`, `reasons`, `gpus_min`,
`gpu_type` (`fits`: also `precision`, `backend`), and only the marks that say something
(`needs_ngc_key`, `needs_busy_gpus`, `region_excluded` and `records_unread` when true, `kind` when
not `newest`, `variant_of`, `pinned` when false, `geography` when not `Global`, `ngc_image`,
`registry_gpus`).
`facts.other_rows` names the models of the other rows, by verdict. `--out FILE` saves every row
with every field, and `facts.rows_file` names that file: read it only when the user asks about a
row that was not printed, or for a list from before this conversation (SKILL.md, Locality). On
2026-11-11 the full NGC list on 8x H100 NVL printed 29 KB; every row with every field would have
been 68 KB.

## How rows are judged
- Strictly, as select: a requirement the metadata leaves out, or a profile neither built for nor
  tested on this GPU, is `unknown`, never `fits`. Only `fits` rows are compatible; an `unknown`
  row is not confirmed, and nothing is estimated from a parameter count.
- On a host the list counts every whole GPU of each type as idle: it answers "which models can
  these GPUs run". `facts.gpus_judged` gives, per type, `count`, `judged` (within `--max-gpus`)
  and `idle` now. A row that needs more GPUs than are idle now is `needs_busy_gpus`; select, at the
  pick, judges the idle GPUs. On Kubernetes the rows judge the allocatable GPUs of one node,
  within the namespace quota, as select does; GPU usage there is unknown (`idle` null), so no row
  is marked, and a note says so.
- Each GPU type is judged on its own; `gpu_type` names the type of the row.
- `gpus_min` is the smallest count that fits. select recommends the most GPUs allowed among equal
  profiles (`references/fit-rules.md`, Ranking) and lists the smaller ones in `facts.shortlist`
  (`facts.unconfirmed` for an `unknown` verdict); with `--max-gpus N` it recommends within N.
- `--max-gpus`, `--need-lora`, `--backend` and `--prefer` work as for select; pass the same ones
  to select at the pick.
- MIG, shared GPUs or no GPUs stop the list with `gpus_unknown` (exit 3) before any request.

## Region
- `geography` is the card's "Deployment Geography" text (at most 80 characters: it is the card's
  text, shown as data), or null when the card has none. On 2026-11-11 the cards that had one said
  `Global`, except `meta/llama-4-maverick-17b-128e-instruct` and
  `meta/llama-4-scout-17b-16e-instruct`: `Global, except EU`.
- `--region R` is where the NIM will run, in the user's words: `eu`, a member state (`germany`,
  `fr`), `Berlin, Germany`, `us`, `india`, `uk`, or the code the `country` command gives (`DE`,
  `US`: any ISO 3166 code is known). `region_excluded` is true when the geography
  excludes it (`Global, except EU` excludes the EU and each member state), false when it does not,
  and null when that cannot be told: no geography, a text the script cannot read, no `--region`,
  or a region it does not know (a city, `europe`, which is more than the EU, `eu-west-1`).
- Only the rows the agent shows (`fits`, `unknown`) count. A note asks for the user's country when
  such a row's geography is not `Global` and `--region` is not given, or was given but could not
  be judged against it (`Paris`, `europe`); list again with `--region COUNTRY`. A geography the
  script cannot read gets its own note: show it as the card says it.
- The country lookup, `nimselect.py country`, runs only after the user agrees to it; otherwise
  ask the country. It sends one GET to Cloudflare's trace page,
  `https://www.cloudflare.com/cdn-cgi/trace`, which names the country of this machine's public IP
  address, or of the proxy's when `HTTPS_PROXY` sends the request through one (Cloudflare sees the
  address; only the country code is kept). Pass `facts.country` as
  `--region`. It checks the machine the agent runs on: when the GPUs are elsewhere (Kubernetes,
  another host), say so. State `country_unknown` (no answer, or no country named): tell the user
  and ask the country. It never chooses the source: China is the Locality step's, with no lookup.
- select gives the same `geography` and `region_excluded` for the model it checks, takes the same
  `--region`, and its notes ask for the country the same way.
- Listing again with `--region`: the cache answers the NGC part in about a second (within the
  hour; after it, the repository records are read again, about two thirds of a cold list's
  requests); a CND list reads its catalog again.
- The region marks where a NIM may be deployed. It does not choose the source: the Locality step
  does (SKILL.md).

## The cache
The list keeps NGC's answers in `~/.cache/nim-select/ngc-cache.json`: metadata for 24 h;
repository records, image lists, the catalog and 401, 403 or 404 answers for 1 h; never a 5xx or a
429. It keeps only answers their reader accepts: a 200 in another shape (a catalog page with no
results list, metadata with no profiles list) is an error (`ngc_bad_response`), never kept, so the
next list asks again; a 401, 403 or 404 is kept only with NGC's error envelope. CND answers are not
kept. The file is written private (folder `0700`, file `0600`), as a new file that replaces the old
one, and read back only when it is this user's regular file that no one else can write, not a
symlink, in a private folder (`references/security.md`, Files); otherwise, or when it is not a
nim-select cache, the list runs without it and `facts.cache.problem` says why. `facts.cache` also
counts `hits`, `misses`, `dropped` (entries of the wrong shape, with a status the cache does not
keep, or past their lifetime) and says whether it was `saved`. `--no-cache` neither reads nor
writes it: use it only when the user wants answers fresher than the cache keeps: a NIM or tag
published within the hour, or profile metadata changed within the last 24 h (a profile or a GPU
test added to a tag the list read earlier that day). Without it, such a change shows only once the
entry expires. select and fit never read the cache, so they always see NGC's current metadata.

## Time and failures
The whole list stops after 120 s, the catalog read included, and keeps the rows read; each NGC
request has 15 s and 2 attempts, and at most 8 are open at a time. One model that could not be
read (a reset, a timeout, an odd answer) is in `facts.not_checked` with its error, and the list
goes on; three models in a row that NGC or the CND does not answer stop it (exit 4), and so does
NGC's first 429 left after the retries (`ngc_rate_limited`): no model starts after it. Models the
120 s cut are in `facts.not_checked` with `why` `deadline`, and those not started after a stop with
`stopped`. Listing again is faster, as the cache answers what NGC answered. On 2026-11-19 a full NGC
list (154 models) took 78 to 94 s and 434 to 539 requests with an empty cache, by GPU type, and
under a second with a full one.

## A CND list (`--source tgcr`, `turingcm` or `lichan`)
- Only the chosen CND, never NGC's catalog or another CND: TGC's resource list (page by page),
  TuringCM's simple-list, or the repositories of Lichan's project `cnd`; at most 4 requests at a
  time, to fixed endpoints (`references/china-distributors.md`). `--query` matches the catalog
  name (TGC, TuringCM) or the repository path (Lichan).
- A list reads at most TGC's first 300 items and Lichan's first 500 repositories (pages of 100).
  A catalog that lists more is not checked past them, and listing again reads the same pages: a
  list with rows is `partial`, with a note (never `facts.not_checked`); one with no row is
  `no_candidates`, whose message says only the part read was checked and whose fix offers select
  on the full image of a model the user names (`HOST/PROJECT/REPO:TAG`).
- It keeps the LLM and VLM NIM images: an image is left out when its name (the name lists), the
  catalog's `model_type` or the image labels' category says it is no LLM or VLM NIM. Left out, in
  `facts.skipped`: images that are no NIM (`not_a_nim`: etcd, minio, milvus ...), `non_llm`,
  `model_free`, and catalog items whose detail names no image (`no_image`). A catalog detail that
  could not be read is in `facts.not_checked` (the list is `partial`), never `no_image`.
- Each repository is checked as select checks it without a tag: the newest image with a version
  tag and a known NGC build, which only its labels or the TGC catalog name, never its name; that
  build's profiles give the verdict, and `ngc_image` names the build. A tag list that fails is read
  once more. `needs_ngc_key` is false and `kind` is `newest`.
- No known build: an `ngc_build_unknown` row, with no profile check before the pull. select on it
  offers the after-pull check, and `--nim-version` when that can map the image. These images can
  be picked: offer to name them.
- `registry_gpus`: the GPUs the CND's catalog lists the image for (TGC, TuringCM; `Generic` and
  `通用` mean any GPU), only short GPU names (an entry of other text is left out). Show it as
  "listed by the registry for: ...": it is the registry's own list, never a verdict.
- `pinned` false: no tag but `latest` is known for the image (on 2026-11-11, 8 of Lichan's 13
  `ngc_build_unknown` rows), so the row shows it with no tag; select at the pick asks for its own
  version tag before any hand-off.
- A catalog that answers only after a login cannot be listed: exit 4 `cnd_unreachable`, with a
  fix that says not to retry; nim-select never logs in.

On 2026-11-11, on 8x H100 NVL, nothing fit on a CND: Lichan gave 15 rows (1 `unknown`:
`io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0`, checked with
`nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant`, the build its version label names; 1
`does_not_fit`; 13 `ngc_build_unknown`) and left out 34 images, 12 of them no NIM. TGC gave 73
rows (16, 7, 50), TuringCM 52 (6, 1, 45).

## States
| state | exit | `next` | what to do |
|---|---|---|---|
| `listed` | 0 | `ask_user` | show the list (SKILL.md, List) and ask which model to check |
| `none_fit` | 0 | `ask_user` | the same, saying plainly that no model was confirmed to fit |
| `partial` | 0 | `ask_user` | not every model was checked (`facts.not_checked`: the 120 s limit, one model's error, a catalog page or detail not read; `facts.notes`, also for a CND catalog longer than a list reads): show the rows read as a partial list, name the models not checked, never say that nothing fits, and offer to list again (not for that catalog length: listing again reads the same pages) |
| `no_candidates` | 3 | `ask_user` | the source lists no LLM or VLM NIM that matches (`facts.skipped` counts what was left out): ask for another `--query`, or list without one. For a CND catalog longer than a list reads (the message and `facts.notes` say so), none matched in the part read and the rest was not checked, so never say the source has none; listing again reads the same pages: offer select on the full image of a model the user names (`HOST/PROJECT/REPO:TAG`), as the fix says |
| `gpus_unknown` | 3 | `ask_user` | MIG, shared or no GPUs: follow the fix, as for select |
| `ngc_unreachable`, `cnd_unreachable` | 4 | `fix_errors` | NGC or the CND stopped answering (the catalog, three models in a row, or a 429: error `ngc_rate_limited`), no model could be checked, or NGC refused the record of every NIM checked (error `ngc_http_error`); the rows read before the failure are kept (`facts.partial`, `facts.not_checked`): show them as partial, check the network or proxy, retry once (after a 429, wait a few minutes; after a login fix, do not retry) |

A bad option (`--max-gpus 0`, no `--source`) or an `--out` file that cannot be written is exit 2:
fix the command.

## Facts
| key | meaning |
|---|---|
| `schema` | `nim-select/list/v1`. select's facts have no schema, so another skill tells a list from a select file and never takes a list as a hand-off |
| `source` | `ngc`, `tgcr`, `turingcm` or `lichan` |
| `environment_summary`, `gpus_judged` | the inspected GPUs, and how many of each type the rows judge |
| `options` | the `--query`, `--region` and fit options used |
| `counts`, `rows` | rows per verdict, and the rows in order (printed: `fits` and `unknown` only, brief; What the list prints) |
| `other_rows`, `rows_file` | printed only: the models of the other rows by verdict, and the `--out` file with every row (null without `--out`) |
| `skipped` | catalog entries left out, by why |
| `not_checked`, `partial` | models not checked, with `why`: `deadline`, `stopped` (after a stop), or an error code with its `message`; `partial` is also true when the catalog was not read in full |
| `cache`, `elapsed_s` | the cache's use, and the seconds the list took |
| `notes` | what to tell the user besides the rows |

`--out FILE` saves the facts with every row; the list writes no select file.
