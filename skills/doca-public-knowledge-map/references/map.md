# doca-public-knowledge-map — routing tables

Full enumerated routing tables moved out of `SKILL.md` to keep the loader within the per-file size budget. The agent opens this file when it needs the concrete docs.nvidia.com / on-disk / GitHub entries. The version-pinned DOCA 3.5 docs are under `docs.nvidia.com/doca/archive/3-5-0/` (they redirect to NVIDIA's networking docs site).

Companion routing tables (open only the file that matches the question; do not load every file):

- Library guides — [libraries.md](libraries.md)
- DOCA services — [services.md](services.md)
- DOCA tools — [tools.md](tools.md)
- Externally-productized DPU / service products — [external-dpu.md](external-dpu.md)
- Externally-productized fabric / orchestration / firmware — [external-fabric.md](external-fabric.md)

## Example questions this skill answers well

These are the question SHAPES this skill is designed to route, with one worked
example each. A productive A/B test against this skill probes the *shape*, not
the literal wording.

- **"Where can I read about DOCA &lt;library/service/tool/concept&gt;?"** —
  worked example: *"Where can I read about DOCA Flow Connection Tracking?"*
  Answered by walking the `## Library- and module-specific guides`,
  `## DOCA services`, or `## DOCA tools` routing tables to the matching
  `docs.nvidia.com/doca/sdk/...` entry.
- **"Which DOCA libraries do I have installed and at what version?"** —
  worked example: *"How do I confirm the box has DOCA Flow 3.3 installed?"*
  Answered by combining `## Layout of an installed DOCA package` and
  `## Where to find the version`, with cross-link to the layered
  version-detection rules in
  doca-setup ## Capabilities and modes (`doca-setup/CAPABILITIES.md`).
- **"Where is sample &lt;X&gt; on disk and where is its source on GitHub?"** —
  worked example: *"Where is the doca_flow sample that exercises ACL pipes?"*
  Answered by combining `## Layout of an installed DOCA package` (local)
  with `## Public source code: GitHub` (remote) — both sections name the
  canonical paths.
- **"What does this on-disk path mean, what should I cite from it?"** —
  worked example: *"What's in `/opt/mellanox/doca/applications/` that the
  customer can actually run?"* Answered by `## Layout of an installed DOCA package`,
  plus the "no source-tree paths" ground rule above.
- **"This URL I have 404s — what's the new one?"** — worked example: *"The
  Comm Channel page is gone in DOCA 2.5+."* Answered by the URL-rename rule
  at the top of `## Public documentation entry points`, plus the
  `## URL audit` footer at the bottom of this file.
- **"Where is the customer-facing place to ask for help on this?"** —
  worked example: *"What's the developer forum's DOCA category?"* Answered
  from the developer-forum entry in `## Public documentation entry points`,
  never internal NVIDIA channels (see ground rule above).

If the question fits a different shape (how to write code, how to set up an
env, how to debug a crash), route to the matching skill instead — see
bundle `AGENTS.md` for the routing table.

## Public source code: GitHub

NVIDIA publishes a small, deliberately scoped set of DOCA-adjacent code on
GitHub. These are the public, customer-visible repositories.

| Repository | URL | What you find there |
| --- | --- | --- |
| DOCA Platform Framework | <https://github.com/NVIDIA/doca-platform> | DPU provisioning, Kubernetes operator pieces, deployment manifests. Read this when the user is operating DPUs at scale or running them under Kubernetes. |

**Important:** the bulk of DOCA — libraries, samples, applications, build
system — is **not on GitHub**. It ships as packages and, for licensed users,
as source archives downloaded from the Developer Zone. If the user asks for
"the DOCA library X source on GitHub", correct them: the published sample for
library X lives under `/opt/mellanox/doca/samples/doca_<library>/` on an
installed system, or in the downloadable source archive referenced from the
Downloads page.

## Layout of an installed DOCA package

When DOCA is installed from the official packages on a Linux host or on
BlueField, an agent can rely on this on-disk layout. Use it instead of asking
the user to share source code.

| Path | What is there | How to use it |
| --- | --- | --- |
| `/opt/mellanox/doca` | Install root. | Use as `${DOCA_DIR}` in any command you suggest. |
| `/opt/mellanox/doca/samples` | One subdirectory per library, each containing a self-contained sample (typical files: `<library>_main.c`, `meson.build`). | The authoritative example for that library on the installed version. Read these before answering "show me a sample of X". |
| `/opt/mellanox/doca/applications` | Full reference applications (e.g. `doca_pcc`, `doca_dpi`). | Larger, integrated examples. Inspect their `meson.build` to see how they declare DOCA dependencies. |
| `/opt/mellanox/doca/tools` | Shipped CLIs (e.g. `doca_caps`, `doca_socket_relay`). | Use them for runtime introspection before answering capability questions. |
| `/opt/mellanox/doca/infrastructure` *(legacy / split-profile installs only)* OR `/opt/mellanox/doca/lib/<arch>-linux-gnu/` + `/opt/mellanox/doca/include/` *(default on DOCA 3.3+)* | Headers, libraries, and `pkg-config` files used to build against DOCA. The `infrastructure/` subtree is present in some legacy / split installer profiles; on most current DOCA 3.3+ installs the layout has flattened and `infrastructure/` may NOT exist. ALWAYS resolve via `pkg-config --variable=includedir doca-common` and `pkg-config --variable=libdir doca-common` rather than hard-coding either path. | Inspect `*.pc` to verify the exact Meson dependency name for a library (`doca-flow`, `doca-common`, etc.). |
| `/opt/mellanox/doca/services` | Bundled services (DTS, telemetry agents, etc.). | Read service-specific README files inside each subdirectory before suggesting service-level changes. |

Useful enumeration commands to suggest to the user (read-only, safe):

```bash
ls /opt/mellanox/doca
ls /opt/mellanox/doca/samples
ls /opt/mellanox/doca/applications
pkg-config --list-all | grep -i doca
```

To build against DOCA from a user-owned directory, the canonical environment
hint is to expose the DOCA `pkg-config` directory before running `meson setup`:

```bash
PC=$(find /opt/mellanox/doca -name doca-common.pc -print -quit 2>/dev/null)
if [ -n "$PC" ]; then
  export PKG_CONFIG_PATH="${PC%/*}:$PKG_CONFIG_PATH"  # commonly /opt/mellanox/doca/lib/<arch>-linux-gnu/pkgconfig on DOCA 3.3+; or /opt/mellanox/doca/infrastructure/lib/pkgconfig on legacy / split-profile installs
else
  echo "doca-common.pc not found: partial install, see doca-setup" >&2
fi
```

If `pkg-config --modversion doca-common` fails after that, treat it as a real
environment problem (wrong package, wrong arch, missing dev package) and stop;
do not silently change the user's environment. Load
`doca-setup/SKILL.md` with the captured command output to
diagnose, repair, and prepare the environment.

## Where to find the version

Load `doca-version/SKILL.md` and run its
canonical four-source detection chain; do not pick one convenient
source and stop:

1. `pkg-config --modversion doca-common`.
2. `cat /opt/mellanox/doca/applications/VERSION`.
3. `doca_caps --version`.
4. On BlueField, the BFB-image version from the documented
   BlueField-side sources in
   `doca-version TASKS.md ## configure` (`doca-version/TASKS.md#configure`)
   (including `bfver` / `/etc/mlnx-release` where available).

Run every source available on the target and compare all observed
values. If any available source disagrees, stop and route to
`doca-version TASKS.md ## debug` (`doca-version/TASKS.md#debug`);
do not select a preferred value. If none of the canonical sources is
available, stop and report that the version cannot be established.
Always quote the versions actually observed; never assume "latest"
and never route documentation by an assumed latest release.

## URL audit

Every URL referenced by this routing map is HEAD-checked against the
public NVIDIA documentation surface on every commit, so bundle
releases always ship a routing map whose every URL was reachable at
release time. If a URL on this page no longer resolves, that is the
release's bug, not yours — fall back to the umbrella entry points
listed in *Public documentation entry points* (DOCA SDK index, DOCA
Libraries / Services / Tools umbrella pages) and search from there.

## Ground rules for any agent using this skill

1. **Prefer the local install over the web.** If DOCA is installed on the
   machine, the files in `/opt/mellanox/doca` (Linux) are the exact bits the
   user is running. Web docs describe a release; local files *are* the release.
2. **Check the version when quoting installed artifacts.** When a DOCA install
   is present and the answer quotes installed API names, options, or sample
   filenames, find the installed version (see "Where to find the version"
   above) and use the matching documentation release. For a pure documentation
   lookup or a system without a local install, route from the public docs
   without claiming an installed version.
3. **Never invent URLs, file paths, package names, `pkg-config` modules,
   library names, or sample names.** If you do not see it in this map, in the
   user's local install, or in the official docs you fetched, say so and ask.
4. **Public sources only.** Reference NVIDIA documentation only on the
   public hosts listed in `AGENTS.md` ground rule #1 (bundle `AGENTS.md`).
   Anything else is not available to a customer agent.
5. **No source-tree paths.** Cite installed paths (under `/opt/mellanox/doca/`
   or standard system paths) or public URLs only.

> **Hardware-safety meta-policy.** Every per-artifact `## Safety policy`
> in this bundle overlays the cross-cutting hardware-safety
> meta-policy. When the user's question touches a change to live DPU /
> NIC hardware state (`mlxconfig` writes, firmware burn, BlueField
> mode flip, BAR window change, IOMMU / hugepages kernel boot
> parameter, BFB reflash), load
> `doca-hardware-safety/SKILL.md` alongside
> the per-artifact skill. Cross-cutting meta-policy lives there
> (pre-flight inventory, out-of-band access requirement, maintenance
> window, replica-first validation, rollback discipline,
> observability-before-workload, refuse-and-escalate when no rollback
> exists). Per-artifact overlays MUST NOT redefine it.

> **Upgrade / downgrade discipline.** When the user is contemplating
> or recovering a DOCA upgrade or downgrade — moving a host to the
> next release, refreshing the BFB, bumping an NGC container tag,
> rolling back, or reacting to an accidental `apt upgrade` drift —
> load `doca-upgrade/SKILL.md`. Its headline
> contract is **detect → report → ASK → only-then guided upgrade;
> never auto-upgrade.** It routes version detection to
> `doca-version/SKILL.md`, every hardware /
> firmware / reboot step to
> `doca-hardware-safety/SKILL.md`, and
> sunset / deprecation lifecycle lookups back to this map's
> Release Notes entry below. It does not redefine version detection
> or the hardware-touching discipline.

## Public documentation entry points

Start here for any conceptual question. These are the canonical NVIDIA-hosted
documents.

> **Rule when a URL in this skill returns 404.** NVIDIA periodically renames
> doc pages and library slugs (the most recent example: *Comm Channel* →
> *Comch* in DOCA 2.5). If a URL listed here 404s during a real session:
> (1) tell the user explicitly that the skill's URL is stale; (2) try
> `https://docs.nvidia.com/doca/sdk/` and look for a
> renamed link; (3) **do not invent a replacement URL** and do not silently
> point at an `archive/` URL — those are version-pinned to old releases.
> If the index is also unreachable, state that the public documentation
> surface could not be reached, ask the user to retry after checking
> connectivity, and offer the Developer Forum entry point without inventing a
> replacement URL. File a fix against this skill (see "URL audit" footer for
> the last verification date and DOCA version).

| Topic | URL | When to use |
| --- | --- | --- |
| DOCA SDK documentation index | <https://docs.nvidia.com/doca/sdk/index.html> | The top of the documentation tree. Always start here when the user asks an open-ended "how does DOCA do X?" question. |
| DOCA Overview | <https://docs.nvidia.com/doca/sdk/doca-overview/index.html> | High-level architecture: what DOCA is, what runs on the BlueField DPU vs. the host, libraries vs. applications vs. services vs. tools. |
| DOCA Installation Guide for Linux | <https://docs.nvidia.com/doca/sdk/NVIDIA+DOCA+Installation+Guide+for+Linux> | Install steps, supported OSes, package layout, post-install verification. Read this whenever the user is setting DOCA up for the first time. |
| DOCA Programming Guide (index) | <https://docs.nvidia.com/doca/sdk/doca-programming-guide/index.html> | SDK architecture, common patterns, how libraries connect, how to wire up a sample. Read this before drilling into a specific library guide. |
| DOCA Developer Quick Start Guide | <https://docs.nvidia.com/doca/sdk/doca-developer-quick-start-guide/index.html> | NVIDIA's official "how do I bring DOCA up and run my first reference application" walkthrough. Cite this for any beginner asking "how do I start?" *once they have BlueField + host hardware*. For users **without** hardware (macOS / Windows / Linux without a NIC), route via `doca-setup ## no-install` (`doca-setup/TASKS.md#no-install`) (NGC container) instead — the Quick Start assumes hardware is already present. |
| DOCA Release Notes | <https://docs.nvidia.com/doca/sdk/doca-release-notes/index.html> | What changed in a release, supported hardware, supported OSes, known issues. Read whenever the user's symptom could be a known issue. |
| DOCA Compatibility Policy | <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html> | NVIDIA's authoritative statement of DOCA version semantics — quarterly GA cadence, October LTS designation (3-year support, 7-update LTS train), the semantic-versioning scheme (X.Y.Z), the three compatibility types (source / binary / behavioral), and the two compatibility directions (backward / forward). Cite this whenever the user asks "what does the version string mean?", "is this LTS release still supported?", or any host ↔ DPU compatibility question. |
| Developer Zone — DOCA landing | <https://developer.nvidia.com/networking/doca> | Marketing-and-onboarding view (videos, tutorials, blog posts). Use only as a fallback when the official SDK docs do not have the topic yet. |
| DOCA Downloads | <https://developer.nvidia.com/doca-downloads> | Where customers actually download DOCA packages, BFB images, host packages. Use this to answer "which package do I need?". |
| NGC Catalog (containers and resources) | <https://catalog.ngc.nvidia.com> | Find DOCA containers, model artifacts, and DPU images. Search for `doca` to enumerate. |
| DOCA Developer Forum | <https://forums.developer.nvidia.com/c/infrastructure/doca/370> | Last-resort discovery for undocumented behavior, real customer questions, NVIDIA staff answers. Always include the disclaimer that forum threads can age and may not match the user's installed version. |

