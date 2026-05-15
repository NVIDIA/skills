---
name: nemo-gym-docs
description: >
  Maintain the NeMo Gym Fern docs site — add, update, move, or remove pages
  under fern/. Use for any documentation change. Triggered by: "edit docs",
  "add doc page", "update docs", "rename page", "fix broken link", "add
  redirect", "preview docs", "publish docs", any request that touches `fern/`.
---

# NeMo Gym Docs Maintenance

Unified skill for adding, updating, moving, and removing pages on the NeMo Gym Fern documentation site.

## Scope Rule

**ALL docs edits happen under `fern/`.** All new pages — including release notes and migration guides — belong under `fern/`.

**Bleeding-edge tree + GA snapshots.** Gym keeps one bleeding-edge content tree at `fern/versions/latest/` (the folder name is historical — it's mounted under the `main` slug via `main.yml`) and one frozen GA snapshot per shipped release (currently `fern/versions/v0.2.1/`). All new edits land in `fern/versions/latest/`. The `latest.yml` file is a GA *alias* — a symlink to the current GA's yml — so `/latest/...` and `/<current-ga>/...` serve the same pages. Back-ports into a frozen GA snapshot are deliberate and rare; default to editing `fern/versions/latest/` only.

## Layout at a Glance

```
fern/
├── fern.config.json          # Org + Fern CLI version (currently 4.80.3)
├── package.json              # `npm run dev|check|generate` — wraps `npx -y fern-api@latest`
├── docs.yml                  # Site config: instances, versions, tabs, redirects, libraries
├── versions/
│   ├── latest.yml            # GA alias — symlink to the current GA's yml (currently v0.2.1.yml)
│   ├── main.yml              # Nav tree for the bleeding-edge train (mounts ./latest/pages)
│   ├── latest/pages/         # MDX content for the bleeding-edge train (slug: main)
│   ├── v0.2.1.yml            # Nav tree for the 0.2.1 GA snapshot
│   └── v0.2.1/pages/         # MDX content for the 0.2.1 GA snapshot
├── components/               # Custom TSX (CTAButtons, NavButton, CustomFooter)
├── assets/                   # Images, SVGs, favicon
├── main.css                  # Global theme overrides (NVIDIA green, badge spacing, etc.)
└── product-docs/             # GENERATED library reference (gitignored)
```

```
File system                                            Published URL
─────────────────────────────────────────────────────  ──────────────────────────────────
fern/versions/latest/pages/get-started/quickstart.mdx  docs.nvidia.com/nemo/gym/main/get-started/quickstart
fern/versions/v0.2.1/pages/get-started/quickstart.mdx  docs.nvidia.com/nemo/gym/v0.2.1/get-started/quickstart
                                                       docs.nvidia.com/nemo/gym/latest/get-started/quickstart  (GA alias)
```

## Operations

### Add a Page

1. Gather: page title, target section, filename (kebab-case `.mdx`), subdirectory under `fern/versions/latest/pages/`.
2. Create `fern/versions/latest/pages/<subdirectory>/<filename>.mdx`:

   ```mdx
   ---
   title: "<Page Title>"
   description: "One-line SEO description (or empty string)"
   position: 3
   ---

   # <Page Title>

   <content>
   ```

3. If the parent folder is mounted in `main.yml` with `title-source: frontmatter`, the page is auto-discovered — no nav edit needed. Otherwise add a `- page:` entry under the right `section:` in `fern/versions/main.yml`.
4. Do **not** mirror into the current GA snapshot folder (e.g., `v0.2.1/`) — frozen GA snapshots only get back-ports on explicit request.

### Update a Page

1. Locate by path, title, or keyword (`grep -rn` in `fern/versions/latest/pages/`).
2. **Content only** — edit the MDX in `fern/versions/latest/pages/`. Don't touch `v0.2.1/` unless this is an explicit back-port.
3. **Title change** — update the frontmatter `title:` and (if the parent uses `title-source: frontmatter`) nothing else; otherwise update the nav `- page:` entry too.
4. **Section move** — `git mv` the file, update its `path:` in `main.yml`, fix all incoming links inside `fern/versions/latest/pages/`.
5. **Slug change** — folders use the page filename for the slug. Renaming the file changes the URL; add a redirect in `fern/docs.yml` so the old URL keeps working.

### Remove a Page

1. Find incoming links: `grep -rn "<filename>" fern/versions/latest/pages --include="*.mdx"`.
2. `git rm` the file from `fern/versions/latest/pages/`.
3. Remove the matching `- page:` block from `main.yml` if it was a manual entry.
4. Fix or remove all incoming links.
5. Add a redirect in `fern/docs.yml` if the URL was public.

### Back-port to a GA Snapshot

Only back-port when the user explicitly asks ("back-port to v0.2.1"). Apply the same change inside the GA snapshot's `pages/` folder (e.g., `fern/versions/v0.2.1/pages/`) and update its yml if needed. `latest.yml` is a symlink to the current GA's yml, so nav changes propagate automatically.

### Worked Example: Adding a Page

Request: *"Add a how-to for collecting rollouts under Get Started."*

1. Create `fern/versions/latest/pages/get-started/rollout-collection.mdx`:

   ```mdx
   ---
   title: "Collect Rollouts"
   description: "Run the agent against your dataset and write results to JSONL"
   position: 4
   ---

   # Collect Rollouts

   <content>
   ```

2. The `get-started` folder in `main.yml` uses `title-source: frontmatter`, so the page appears automatically. `position: 4` controls ordering — it's optional; without it, pages sort alphabetically by filename.
3. `cd fern && npm run check && npm run dev`, verify `/main/get-started/rollout-collection` renders.

### Worked Example: Renaming a Slug (with Redirect)

Request: *"Rename `/main/get-started/setup` to `/main/get-started/detailed-setup`."*

1. `git mv fern/versions/latest/pages/get-started/setup.mdx fern/versions/latest/pages/get-started/detailed-setup.mdx`.
2. Add a redirect in `fern/docs.yml`:

   ```yaml
   redirects:
     - source: "/main/get-started/setup"
       destination: "/main/get-started/detailed-setup"
   ```

3. `grep -rn "/get-started/setup" fern/versions/latest/` and update any incoming links.

---

## Content Guidelines

NeMo Gym uses **Fern-native MDX components directly**. Do not use GitHub `> [!NOTE]` syntax — it will not render.

| Purpose | Component |
|---|---|
| Neutral aside | `<Note>...</Note>` |
| Helpful tip | `<Tip>...</Tip>` |
| Informational callout | `<Info>...</Info>` |
| Warning | `<Warning>...</Warning>` |
| Error / danger | `<Error>...</Error>` |
| Card grid on index pages | `<Cards>` with `<Card title="..." href="...">` children |
| Status/scope tag inside a Card | `<Badge minimal outlined>tag</Badge>` (see below) |

Images live in `fern/assets/` (shared) or under a version's `pages/` (version-scoped). Reference with root-relative paths.

### Cards and Badges

Index pages use a `<Cards>` grid. Each `<Card>` can carry one or more `<Badge>` tags to indicate scope, status, or read-time.

Valid intents: `success`, `note`, `tip`, `warning`, `error`, `info`, `launch`, `check`. Use `<Badge minimal outlined>...</Badge>` (no intent) for neutral tags.

Place badges as the last line inside the `<Card>`, separated by a blank line from the body text. The CSS in `main.css` (`.fern-card .fern-docs-badge`) handles spacing — do not add inline `style=` props.

```mdx
<Cards>
  <Card title="Quickstart" href="/main/get-started/quickstart">
    Install, start servers, and collect your first rollouts in one page.

    <Badge intent="success" minimal outlined>start here</Badge> <Badge minimal outlined>5 min</Badge>
  </Card>
</Cards>
```

## Frontmatter Fields

```yaml
---
title: "<Page Title>"        # required — used for nav and <h1>
description: ""              # required (may be empty string) — SEO
position: 1                  # optional — orders auto-discovered pages within a folder
---
```

The MDX body should still open with `# <Page Title>` matching the frontmatter title. Folders using `title-source: frontmatter` in the version YAML pull the nav label from `title:`.

## Validate

First-time setup: authenticate the CLI against the `nvidia` Fern org via Google SSO (one-time, browser flow):

```bash
npx -y fern-api@latest login    # opens browser → sign in with your @nvidia.com Google account
```

Run from `fern/` (no install step — scripts shell out to `npx -y fern-api@latest`):

```bash
npm run check       # `fern check` — YAML + frontmatter validation
npm run dev         # `fern docs dev` — localhost:3000 hot-reload preview
```

`fern check` must pass before commit. The dev server's broken-link warnings for cross-version links like `/latest/about` are **false positives** — Fern's local validator does not resolve the version slug from `docs.yml` against `latest.yml`. The published site renders them correctly.

To regenerate the autodoc library reference (gitignored under `product-docs/`):

```bash
npm run generate:library    # `fern docs md generate`
```

## Commit & Preview

```bash
git add fern/
git commit -s -m "docs: <add|update|remove> <page-title>"
```

PRs that touch `fern/**` get an automatic Fern preview URL posted as a comment by `.github/workflows/fern-docs-preview-comment.yml`. No manual step needed.

```
                    ┌─ fern-docs-ci.yml                  → fern check
PR (touches fern/) ─┼─ fern-docs-preview-build.yml       → upload fern/ artifact (no secrets)
                    └─ fern-docs-preview-comment.yml     → 🌿 preview URL comment

Push to main (touches docs/** or fern/**) → publish-fern-docs.yml → docs.nvidia.com/nemo/gym
Tag push (docs/v*)                        → publish-fern-docs.yml → docs.nvidia.com/nemo/gym
Manual dispatch                           → publish-fern-docs.yml → docs.nvidia.com/nemo/gym
```

The preview-comment + publish jobs require the `DOCS_FERN_TOKEN` repository or organization secret (from `fern token`).

## Publishing to Production

Production publishes on three triggers (see `.github/workflows/publish-fern-docs.yml`):

1. **Push to `main`** when `docs/**` or `fern/**` changes — continuous staging.
2. **Tag push** matching `docs/v*` — versioned release.
3. **Manual dispatch** from the Actions tab.

Tag format must be `docs/v<MAJOR>.<MINOR>.<PATCH>`. Do not push a tag unless the user asks.

```bash
git tag docs/v0.3.0
git push origin docs/v0.3.0
```

URL → version mapping:

```
docs.nvidia.com/nemo/gym/latest/...   → GA alias (currently mirrors v0.2.1)
docs.nvidia.com/nemo/gym/main/...     → bleeding-edge train (folder: ./latest/pages)
docs.nvidia.com/nemo/gym/v0.2.1/...   → 0.2.1 GA snapshot
```

## Cutting a New Version Train

When the user ships a new version (e.g. `v0.3.0`):

1. `cp -r fern/versions/latest fern/versions/v0.3.0` — frozen snapshot of the bleeding-edge folder.
2. `cp fern/versions/main.yml fern/versions/v0.3.0.yml` and rewrite `./latest/` path prefixes to `./v0.3.0/`.
3. Retarget the `fern/versions/latest.yml` symlink at the new GA's yml: `cd fern/versions && ln -sfn v0.3.0.yml latest.yml`.
4. In `fern/docs.yml` `versions:`, add the new GA snapshot entry (`display-name: "0.3.0"`, `slug: v0.3.0`, `availability: stable`) and demote/remove the previous GA snapshot per the support policy.
5. `fern/versions/latest/pages/` keeps moving forward as the bleeding-edge tree. `main.yml` is unchanged.
6. Tag `docs/v0.3.0` and push to publish.

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML file |
| Page 404 in preview | `slug:` missing/duplicated in the same section; or `position:` collision in an auto-discovered folder |
| Broken-link warning for `/latest/...` cross-version link | False positive in `fern docs dev`; works on published site |
| `JSX expressions must have one parent element` | Wrap multi-element MDX content in `<>...</>` or a `<div>` |
| Old URL breaks (page renamed/moved) | Add a `redirects:` entry in `fern/docs.yml` |
| Library reference missing | `npm run generate:library` in `fern/` |
| Broken image | Path is relative to the MDX file; check `fern/assets/` exists |
| Card badges have no spacing | Don't add inline styles — `main.css` `.fern-card .fern-docs-badge` rules handle it |
| `/latest/<page>` and `/<current-ga>/<page>` show different content | They shouldn't — `latest.yml` should mirror the current GA version's yml and point at the same `./<current-ga>/pages/...` content. Sync them. |

## Key References

| File | Purpose |
|---|---|
| `fern/docs.yml` | Site config, versions, redirects, libraries |
| `fern/versions/latest.yml` | GA alias — symlink to the current GA's yml |
| `fern/versions/main.yml` | Nav tree for the bleeding-edge train (mounts `./latest/pages`) |
| `fern/versions/v0.2.1.yml` | Nav tree for the 0.2.1 GA snapshot |
| `fern/versions/<ver>/pages/` | MDX content for a version |
| `fern/components/` | Custom TSX (CTAButtons, NavButton, CustomFooter) |
| `fern/assets/` | Shared images, SVGs, favicon |
| `fern/main.css` | Global theme overrides — NVIDIA green, card/badge spacing |
| `fern/package.json` | `npm run check|dev|generate|generate:library` — each wraps `npx -y fern-api@latest` |
| `.github/workflows/fern-docs-*.yml` | CI: check, preview build, preview comment |
| `.github/workflows/publish-fern-docs.yml` | CI: publish to docs.nvidia.com/nemo/gym |

---
