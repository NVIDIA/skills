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

**ALL docs edits happen under `fern/`.** The legacy `docs/` directory is the original Sphinx source — do not add new pages there. Release notes, migration guides, and every new page belong under `fern/`.

**Two-version mirror.** Gym maintains two real version trees in parallel: `fern/versions/latest/` and `fern/versions/v0.2/`. Unless the user says otherwise, **every change to `latest` must be mirrored to `v0.2`** (and vice versa for back-ports). The PM is particular about fidelity between versions; do not let them drift.

## Layout at a Glance

```
fern/
├── fern.config.json          # Org + Fern CLI version (currently 4.80.3)
├── package.json              # `npm run dev|check|generate` — wraps `npx -y fern-api@latest`
├── docs.yml                  # Site config: instances, versions, tabs, redirects, libraries
├── versions/
│   ├── _nav_order.yml        # Cross-version nav ordering
│   ├── latest.yml            # Nav tree for current train
│   ├── latest/pages/         # MDX content for current train
│   ├── v0.2.yml              # Nav tree for the 0.2 train
│   └── v0.2/pages/           # MDX content for the 0.2 train
├── components/               # Custom TSX (CTAButtons, NavButton, CustomFooter)
├── assets/                   # Images, SVGs, favicon
├── main.css                  # Global theme overrides (NVIDIA green, badge spacing, etc.)
└── product-docs/             # GENERATED library reference (gitignored)
```

```
File system                                         Published URL
──────────────────────────────────────────────────  ──────────────────────────────────
fern/versions/latest/pages/get-started/quickstart.mdx  docs.nvidia.com/nemo/gym/latest/get-started/quickstart
fern/versions/v0.2/pages/get-started/quickstart.mdx    docs.nvidia.com/nemo/gym/v0.2/get-started/quickstart
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

3. If the parent folder is mounted in `latest.yml` with `title-source: frontmatter`, the page is auto-discovered — no nav edit needed. Otherwise add a `- page:` entry under the right `section:` in `fern/versions/latest.yml`.
4. **Mirror to `v0.2`**: copy the MDX to `fern/versions/v0.2/pages/<subdirectory>/<filename>.mdx` and update any `/latest/...` links in its body to `/v0.2/...`. Update `v0.2.yml` if a manual nav entry was needed.

### Update a Page

1. Locate by path, title, or keyword (`grep -rn` in `fern/versions/latest/pages/`).
2. **Content only** — edit the MDX directly, then mirror the same edit to the `v0.2` copy.
3. **Title change** — update the frontmatter `title:` and (if the parent uses `title-source: frontmatter`) nothing else; otherwise update the nav `- page:` entry too.
4. **Section move** — `git mv` the file, update its `path:` in the nav, fix all incoming links, mirror to `v0.2`.
5. **Slug change** — folders use the page filename for the slug. Renaming the file changes the URL; add a redirect in `fern/docs.yml` so the old URL keeps working.

### Remove a Page

1. Find incoming links: `grep -rn "<filename>" fern/versions/latest/pages fern/versions/v0.2/pages --include="*.mdx"`.
2. `git rm` the file from both `latest/` and `v0.2/`.
3. Remove the matching `- page:` block from `latest.yml` and `v0.2.yml` if it was a manual entry.
4. Fix or remove all incoming links.
5. Add a redirect in `fern/docs.yml` if the URL was public.

### Back-port to an Older Version

When `latest` and `v0.2` diverge intentionally (e.g. an API only exists in `latest`), do not mirror — but call out the divergence in the PR description so the PM can confirm.

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

2. The `get-started` folder in `latest.yml` uses `title-source: frontmatter`, so the page appears automatically. `position: 4` controls ordering.
3. Mirror to `fern/versions/v0.2/pages/get-started/rollout-collection.mdx`. Replace any `/latest/...` links in the body with `/v0.2/...`.
4. `cd fern && npm run check && npm run dev`, verify both `/latest/get-started/rollout-collection` and `/v0.2/get-started/rollout-collection` render.

### Worked Example: Renaming a Slug (with Redirect)

Request: *"Rename `/latest/get-started/setup` to `/latest/get-started/detailed-setup`."*

1. `git mv fern/versions/latest/pages/get-started/setup.mdx fern/versions/latest/pages/get-started/detailed-setup.mdx`.
2. Mirror the rename to `v0.2`.
3. Add redirects in `fern/docs.yml`:

   ```yaml
   redirects:
     - source: "/latest/get-started/setup"
       destination: "/latest/get-started/detailed-setup"
     - source: "/v0.2/get-started/setup"
       destination: "/v0.2/get-started/detailed-setup"
   ```

4. `grep -rn "/get-started/setup" fern/versions/` and update any incoming links in both versions.

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

### Cards and Badges (PM is particular about fidelity)

Every `<Card>` on an index page should carry the same scope/status badges that the original Sphinx docs in `docs/` had. Mapping:

| Original `{bdg-*}` | Fern equivalent |
|---|---|
| `{bdg-primary}` | `<Badge intent="success" minimal outlined>...</Badge>` |
| `{bdg-warning}` | `<Badge intent="warning" minimal outlined>...</Badge>` |
| `{bdg-secondary}` | `<Badge minimal outlined>...</Badge>` (no intent) |

Valid intents: `success`, `note`, `tip`, `warning`, `error`, `info`, `launch`, `check`. Place badges as the last line inside the `<Card>`, separated by a blank line from the body text. The CSS in `main.css` (`.fern-card .fern-docs-badge`) handles vertical spacing from the description and horizontal spacing between adjacent badges — do not add inline `style=` props.

```mdx
<Cards>
  <Card title="Quickstart" href="/latest/get-started/quickstart">
    Install, start servers, and collect your first rollouts in one page.

    <Badge intent="success" minimal outlined>start here</Badge> <Badge minimal outlined>5 min</Badge>
  </Card>
</Cards>
```

When adding or editing a Card, **check the original `docs/<same-path>/index.md` for the badges that were on the corresponding `:::{grid-item-card}` directive** and reproduce them. Dropping badges silently is a regression.

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
docs.nvidia.com/nemo/gym/latest/...   → latest train
docs.nvidia.com/nemo/gym/v0.2/...     → 0.2 train
```

## Cutting a New Version Train

When the user ships a new version (e.g. `v0.3`):

1. Copy `fern/versions/latest/pages/` → `fern/versions/v0.3/pages/` (frozen snapshot of the previous "latest").
2. Copy `fern/versions/latest.yml` → `fern/versions/v0.3.yml` and rewrite all `./latest/` path prefixes to `./v0.3/`.
3. Replace `/latest/` link prefixes in the new `v0.3/pages/` body MDX with `/v0.3/`.
4. Add the version to `fern/docs.yml` `versions:` list with `slug: v0.3` and `availability: stable`. Keep the `latest` entry pointing at `versions/latest.yml`.
5. `latest/pages/` continues forward as the current dev train.
6. Tag `docs/v0.3.0` and push to publish.

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML file |
| Page 404 in preview | `slug:` missing/duplicated in the same section; or `position:` collision in an auto-discovered folder |
| Broken-link warning for `/latest/...` cross-version link | False positive in `fern docs dev`; works on published site |
| `JSX expressions must have one parent element` | Wrap multi-element MDX content in `<>...</>` or a `<div>` |
| Old Sphinx URL breaks | Add a `redirects:` entry in `fern/docs.yml` |
| Library reference missing | `npm run generate:library` in `fern/` |
| Broken image | Path is relative to the MDX file; check `fern/assets/` exists |
| Card badges have no spacing | Don't add inline styles — `main.css` `.fern-card .fern-docs-badge` rules handle it; if missing, restore from the badge spacing commit |
| `latest` and `v0.2` show different content for the same page | Mirror the change you made to `latest` over to `v0.2` (or call out the intentional divergence in the PR) |

## Key References

| File | Purpose |
|---|---|
| `fern/docs.yml` | Site config, versions, redirects, libraries |
| `fern/versions/latest.yml` | Nav tree for the latest train |
| `fern/versions/v0.2.yml` | Nav tree for the 0.2 train |
| `fern/versions/_nav_order.yml` | Cross-version nav ordering |
| `fern/versions/<ver>/pages/` | MDX content for a version |
| `fern/components/` | Custom TSX (CTAButtons, NavButton, CustomFooter) |
| `fern/assets/` | Shared images, SVGs, favicon |
| `fern/main.css` | Global theme overrides — NVIDIA green, card/badge spacing |
| `fern/package.json` | `npm run check|dev|generate|generate:library` — each wraps `npx -y fern-api@latest` |
| `.github/workflows/fern-docs-*.yml` | CI: check, preview build, preview comment |
| `.github/workflows/publish-fern-docs.yml` | CI: publish to docs.nvidia.com/nemo/gym |
| `docs/` | Legacy Sphinx source (read-only reference for badge fidelity) |

---
