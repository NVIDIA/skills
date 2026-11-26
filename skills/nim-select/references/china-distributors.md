# China NIM distributors (CNDs)

A China NIM distributor (CND) serves NIM images inside China. Its image is a NIM that NVIDIA
publishes to NGC first, under another registry, path and tag. nim-select checks a CND image with
the NIM Metadata API profiles of the NGC build it was made from, and hands off the CND image
itself. Read this page for a CND source or image, `ngc_build_unknown`, a tgcr login, or the
location check.

## The three CNDs
| source | images on | catalog | how the scripts find a name (anonymous GET) | image labels (Harbor v2) |
|---|---|---|---|---|
| `tgcr` | `tgcr.turing-agi.com` and its public mirror `tgcr-gz.turing-agi.com` (the same repositories) | TGC | `https://tgc.turing-agi.com/tgc/api/resources?page=1&size=100&search=WORD`, then `.../resources/{id}` for each match: its `image_tags` (`docker_url`, `ngc_url`, `arch`, `digest`). A tgcr image named by its path: the items named like its repository, else every item (TGC searches names only, and `GLM-5.1` is `z-ai/glm5_1`) | `tgcr.turing-agi.com` answers 401 without a login: the labels need a tgcr login |
| `turingcm` | `resource.turingcm.com` (`PROJECT/REPO`, the project is the team) | TuringCM | `https://nim.turingcm.com/server/resource/nimImages/simple-list`, then `getNimImagesById?id=ID` for each match: the images are in its `deploy` text, which can name a DGX Spark build too | `resource.turingcm.com`, anonymous |
| `lichan` | `io.chancloud.com`, project `cnd` | Lichan (its Harbor) | `https://io.chancloud.com/api/v2.0/projects/cnd/repositories?page_size=100&page=N`, then the newest tagged images of each match | `io.chancloud.com`, anonymous for project `cnd` |

The three CNDs have different APIs (the TGC and TuringCM catalogs have different schemas; Lichan has
only its Harbor) and list different NIMs, so the scripts have one small reader per CND
(`scripts/cnd.py`) and use only the reader of the CND the user picked. Every request is a public,
anonymous GET; nothing logs in, and no key or token is sent. At most 4 requests are open at a time,
counted across threads (a read a deadline gave up on keeps its place until it ends), and an answer
over 4 MB is not read. Budgets: 5 s for one Harbor request, 8 s for one image's labels, 12 s for a
repository's tags, 10 s for one catalog request and 30 s for a whole catalog search. The tags of a
tgcr image come from the TGC catalog, where only an item's detail lists its image tags: when the
items found by name list no tag of the repository (or not the tag asked for), every other TGC
item, about 90, is read too. For a tgcr image named by its path, the search and that read share
the 30 s; after a name search, that read has 30 s of its own: up to about 60 s for the two. The CNDs
can be slow or blocked from outside China: a CND that does not answer in time is `cnd_unreachable`
(exit 4), never a reason to use NGC or another CND instead. A catalog that answers 401 or 403 needs
a login, which nim-select never does: `cnd_unreachable` with a fix that starts "Do not retry".

## Location check: `nimselect.py locate`
When neither the image host nor the user names the source or says whether they are in China, and no
source was settled earlier in the conversation, the agent runs it straight away, with no question
first, then confirms the source with the user (SKILL.md, Locality step 3). It checks the machine the
agent runs on, which may not be where the GPUs are; when they are elsewhere (Kubernetes or another
host), the agent says so when it reports the source. It reads:
- the timezone: `TZ`, else `/etc/timezone`, else the `/etc/localtime` link. A path (`TZ=:/etc/localtime`)
  is followed to the zone file it links to; `posix/` and `right/` zones are the same zones;
- the language: `LC_ALL`, else `LANG`;
- whether `https://www.google.com/generate_204` answers 204 within 3 seconds (a block page, a proxy
  page, a redirect, which is not followed, or a timeout does not count).

`facts.locality = {timezone, lang, google_reachable, result}`:
| result | when | next |
|---|---|---|
| `china` | Google is not reachable, and the timezone is a mainland China zone (Asia/Shanghai, Asia/Chongqing, Asia/Harbin, Asia/Urumqi, PRC, or the aliases Asia/Chungking, Asia/Kashgar) or the language starts with `zh_CN` | ask "You seem to be in China. Use TGC, TuringCM or Lichan?" (NGC stays a valid answer); never use a CND without the user's answer |
| `outside_china` | Google is reachable and the timezone is not a China zone | `facts.source` is `ngc`: use it, and say so in the next message to the user (the report, a question or an error): "NGC, because this machine looks to be outside China; say if you pull from a China registry" |
| `unknown` | anything else, such as a China timezone with Google reachable (a VPN) | ask "NGC, TGC, TuringCM or Lichan?" |

Ask the question in the same message as any other question. Its one request is a plain GET to
google.com from this machine; it carries no data about the user.

## The source of a model reference
- An image reference names its source by its host; `--source` is then not needed. A host that
  contradicts `--source` is `source_conflict` (exit 2). An image on another registry (a mirror) is
  checked as NGC.
- A name (`qwen3.6-27b`, `qwen/qwen3.6-27b`, `qwen 3.6 27b`) goes to `--source`, default `ngc`.

## Finding a name on a CND
The words of the name (`TEAM/REPO` counts as `REPO`) must all be in a catalog name (TGC,
TuringCM) or a repository path (Lichan). At most 10 matches have their image looked up. A search
reads the whole catalog of that CND: TuringCM's one list, and TGC's results and Lichan's
repositories 100 a page, up to 300 and 500 (Lichan's project `cnd` had 49 repositories on
2026-11-11). So `model_not_found` means the CND lists no such name. TGC results past the first 300
and Lichan repositories past the first 500 are not searched: when there are more, the search is
`cnd_unreachable`, even when a name read is the query exactly (one past them could have the same
name), with a fix that starts "Do not retry" (a search reads the same pages): ask for the full
image. A full tgcr image is looked up by its repository, in the items named like it and then the
first 300 of the catalog only (an item's name need not hold the path: GLM-5.1 is `z-ai/glm5_1`).
An image listed past them is `ngc_build_unknown` unless `--nim-version` maps it (Mapping, item 4).
- One match, or one whose name is the query exactly (`qwen3.6-27b` and `Qwen3.6-27B`): that image.
  Catalog items that name the same image count as one match.
- One TuringCM item whose deploy text names several images (a DGX Spark build too): the one whose
  repository is the name; `facts.notes` names the others, which were not checked.
- Several: exit 3 `model_ambiguous` with `facts.candidates` (`image`, `repository`,
  `catalog_name`, `newest_tag`, and the catalog's `architectures` and `model_type` when it gives
  them; `newest_tag` is null on TuringCM, whose deploy text names `latest`, not the newest tag): ask,
  then rerun with the chosen `repository`, adding `--tag` only if the user named a version, or with
  the chosen `newest_tag` when another candidate has the same repository (TGC lists
  `z-ai/glm-5.3-flash:2.1.2-variant` and its DGX Spark tag `2.1.2-variant-spark` as two items).
- None: exit 3 `model_not_found`, naming the catalog, with `facts.other_sources`: the other two
  CNDs. Tell the user this CND lists no such NIM and offer to search one of the others
  (`--source`) with the name `facts.query`; search it only when the user says so. Never search
  another CND or NGC on your own. An image named on a CND host whose repository is missing is
  `model_not_found` the same way, with its repository name as `facts.query`: another CND is searched
  by that name, never the image, whose host names its own CND (`source_conflict`).

Names the query names exactly are looked up first; at most 10 matches have their image looked up.

## Mapping a CND image to its NGC build
Labels, the TGC catalog or the user's version only, never by the name and tag alone. In order,
the first that NGC has wins (`facts.mapping`):
1. `ngc_url`: the `com.nvidia.nim.ngc_url` label, or the TGC catalog's `ngc_url` for that image
   tag, when it is `nvcr.io/nim/TEAM/REPO:TAG` with a version tag. A path in another NGC org
   (not public) or `latest` is not used. A moving NGC tag (`1.7`) is
   replaced by the build it stands for today, with a warning.
2. `version_label`: the `com.nvidia.nim.version` label is the NGC tag; the NGC repository is the
   CND path's `TEAM/REPO` once the project (`cnd/`, `cnd-enterprise/`, `public/`) and a leading
   `nim/` are stripped. A warning says the repository was taken from the path.
3. `user_version`: `--nim-version V`, the NGC version the user gives for one image named with its
   own tag (for example when tgcr labels need a login). The repository comes from the path, as in 2.
4. Otherwise: `ngc_build_unknown` (exit 3). The profile can only be confirmed after the pull, by
   the image's own `list-model-profiles`, which `nim-deploy` runs; or the user gives the image's NGC
   version (`--nim-version`). As `--nim-version` takes the NGC repository from the CND path, the
   `fix` offers it only where it can map the image: not when the path names no TEAM/REPO, when NGC
   does not show that repository, or when a version the user gave that NGC has still clashed.

Each NGC build must exist on NGC (repository and tag), and the CND image's CPU architecture must be
one the NGC image has; an architecture clash is named in `facts.mappings_tried`, which lists every
value tried and why it was not used. When either side's architectures are not known, the build is
used and a note says which side is unknown. NGC hides some image lists: then the CND image's own
architectures are checked. The TGC catalog gives none for some entries: an entry of the same image
(one digest) that gives one is used; else fit takes the NGC build's only when it has one, as a build
for amd64 and arm64 says nothing about the CND image (a DGX Spark image is arm64), and its profiles
that record none are `unknown` (`arch_not_recorded`). The `com.nvidia.nim.model` label is never
used: it pointed DGX Spark (arm64) images at the amd64 build.

Never map by the repository name and the CND tag: CND tags are not NGC tags, and there is no
consistent rule between them. The TGC catalog shows why (2026-11-03):
- `tgcr-gz.turing-agi.com/public/z-ai/glm-5.2:latest` is `nvcr.io/nim/zai-org/glm-5.2:1.11.1-variant`;
- `tgcr-gz.turing-agi.com/public/z-ai/glm5_1:1.0` is `nvcr.io/nim/zai-org/glm-51:1.8.0-turbo`.

## Tags and the hand-off
- `facts.image` and `facts.handoff.image` are the CND image with its own tag; `facts.ngc_image` and
  `facts.mapping` say which NGC build lent the profiles. Never hand off `facts.ngc_image`.
- `facts.cnd_repository` is the CND image without its tag: select takes it after resolve, as it
  takes `facts.resource_id` on NGC, with `--tag` only when the user named a version: `--tag`
  `facts.cnd_tag`, the CND image's own tag, never `facts.tag`, which is the NGC build's.
- A tag the user names is checked as named; a moving tag of the same image (`1` for `1.0.0`, one
  digest) is handed off as the most specific tag of that image.
- No tag, or `latest`: the repository's tags are listed (Harbor; for tgcr, the TGC catalog's image
  tags) and the newest image with a version tag and a known NGC build is taken, with a note. When
  the CND lists the image only as `latest`, the verdict is still given for the build `latest`
  maps to today, but `facts.pinned` is false and `facts.handoff.image` is null: never hand off
  `latest`; ask the user for the image's version tag (the warning names the digest of `latest`).
  On tgcr the warning offers `--nim-version` only when the path names the build's NGC repository:
  `z-ai/glm-5.2` would give `nim/z-ai/glm-5.2`, not the catalog's `nim/zai-org/glm-5.2`, so a tag TGC
  does not list is then confirmed after the pull. When nothing names the build of an image listed
  only as `latest`, `ngc_build_unknown` gives `facts.image` with no tag: ask for the image's own
  version tag and, when the fix offers it, its NGC version, then select `IMAGE:TAG` with
  `--nim-version`.
- A tgcr tag TGC does not list, mapped with `--nim-version`, has a warning: nothing anonymous shows
  the tag exists, so the user confirms it before the hand-off.
- select checks only that one image on a CND: no older versions or hardware variants. Name another
  tag of the image to check it.
- A CND tag that does not exist is `tag_not_found`; a tag whose multi-arch index lacks the image for
  this platform is not: its labels could not be read (`ngc_build_unknown`). NGC publishing no
  profiles for the build is `no_metadata`; `tags_to_try` is not given, as NGC tags are not CND tags.

## What the CNDs showed on 2026-11-05 (anonymous)
| image | labels or catalog | result |
|---|---|---|
| `io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0` (also `1.0`, `1`, `latest`), amd64 | version `1.7.0-variant`; ngc_url in another NGC org (not public) | `version_label`: `nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant`. On 8x H100 NVL: `unknown` (GPU-agnostic profiles, no memory minimum) |
| `resource.turingcm.com/qwen/qwen3.5-35b-a3b:1.0.0` | the same labels | the same build |
| `resource.turingcm.com/qwen/qwen3.6-27b:1.0` (also `1`, `latest`), a multi-arch index | no NIM label | `ngc_build_unknown` |
| `resource.turingcm.com/nim/llama-3.1-8b-instruct:latest` (TuringCM: llama-3.1-8b-instruct, whose deploy text names `megacube/llama-3.1-8b-instruct-dgx-spark` first) | version `1.15.3` | the path names no TEAM/REPO: `ngc_build_unknown`, after-pull check only |
| `tgcr-gz.turing-agi.com/public/z-ai/glm5_1:1.0` (TGC: GLM-5.1) | TGC ngc_url `nvcr.io/nim/zai-org/glm-51:1.8.0-turbo`; Harbor 401 | `ngc_url`, the TGC item found by its repository |
| `tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:latest` (TGC: Qwen3.6-27B) | TGC ngc_url `nvcr.io/nim/qwen/qwen3.6-27b:1.7.0-variant`; Harbor 401 | `ngc_url`, but listed only as `latest`: no hand-off until the user names a tag |

No anonymous CND image label names a public NGC image yet; many Lichan and TuringCM images carry
the version label. On 2026-11-03, 56 of the 90 TGC items with an image were listed only as
`latest`, and 31 of their 98 image tags had a public `ngc_url`. On 2026-11-05, 55 of the 98 tags
were under an item whose name does not hold the repository name, and 8 of the 76 TuringCM deploy
texts named several repositories, 4 with the DGX Spark build first.

## Errors
| code | exit | meaning | do |
|---|---|---|---|
| `ngc_build_unknown` | 3 | no label, TGC entry or user version names an NGC build NGC has | offer the after-pull check, and `--nim-version` when the fix offers it |
| `model_ambiguous` | 3 | several catalog names, or images of one item, match | show `facts.candidates`, ask, rerun with the chosen `repository` |
| `model_not_found` | 3 | the catalog lists no such name, or the CND has no such repository | say so, offer the other two CNDs (`facts.other_sources`) with the name `facts.query`; search one only when the user says so |
| `tag_not_found` | 3 | the CND has no such tag | ask, or leave the tag out |
| `cnd_unreachable` | 4 | the CND or its catalog did not answer in time, needs a login, or (TGC, Lichan) lists more than a search reads | check the network or proxy, retry once (a fix that starts "Do not retry": never retry, follow the fix); never switch on your own |
| `source_conflict` | 2 | the image's host and `--source` disagree | drop `--source`, or ask |
| `bad_input` | 2 | `--nim-version` without a CND image named with its own tag, or not a version | pass the image with its tag (`facts.image`) |
