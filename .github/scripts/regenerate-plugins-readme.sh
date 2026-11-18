#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
#
# Regenerate the README's Plugin Catalog table. Plugin counterpart of
# regenerate-readme.sh; used by sync-plugins.yml and runnable locally.
#
# Reads:
#   plugins/<dir>/plugin.json   — name, version, description (source of truth
#                                 for what is actually published)
#   components.d/*.yml          — `plugins:` entries map a catalog dir back to
#                                 its product (name, repo) for the Source column
#   catalog-exceptions.yml      — `plugin_exceptions:` dirs are catalog-owned
#                                 and link to this repo
#   plugins.d/<name>.yml        — presence means the plugin is BUILT by
#                                 build-plugins.py from catalog skills, not
#                                 mirrored; the plugin sync ignores it entirely
#                                 (e.g. nvidia-skills)
#
# Writes:
#   README.md (in place) — content between this marker pair is replaced:
#     <!-- plugins-table-start --> ... <!-- plugins-table-end -->
#   Only that section is touched, so this script and regenerate-readme.sh
#   (skills tables) can run in separate workflows without conflicting.
#   If the markers are absent the script warns and exits 0.

set -euo pipefail

README="README.md"
START='<!-- plugins-table-start -->'
END='<!-- plugins-table-end -->'

if ! grep -qF "$START" "$README" || ! grep -qF "$END" "$README"; then
  echo "::warning::README has no ${START} / ${END} markers — plugin table not regenerated"
  exit 0
fi

# catalog_dir|product name|owner/repo  — from components.d plugins: entries
map=$(mktemp)
for f in components.d/*.yml; do
  yq e 'true' "$f" >/dev/null 2>&1 || continue
  yq -r '(.plugins // [])[] | .catalog_dir + "|" + (parent | parent | .name) + "|" + (parent | parent | .repo)' "$f" 2>/dev/null >> "$map" || true
done
if [ -f catalog-exceptions.yml ]; then
  yq -r '(.plugin_exceptions // [])[] | .dir + "|" + (.component // "NVIDIA Skills") + "|"' catalog-exceptions.yml >> "$map" || true
fi

table=$(python3 - "$map" <<'PY'
import json, sys
from pathlib import Path

mapping = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    d, name, repo = (line.split("|") + ["", ""])[:3]
    mapping[d] = (name, repo)

rows = []
for pdir in sorted(Path("plugins").glob("*/")):
    manifest = pdir / "plugin.json"
    if not manifest.is_file():
        continue
    if (Path("plugins.d") / f"{pdir.name}.yml").is_file():
        continue  # catalog-built plugin, outside the plugin sync
    try:
        m = json.loads(manifest.read_text())
    except Exception:
        continue
    name = m.get("name", pdir.name)
    version = str(m.get("version", "—"))
    desc = str(m.get("description", "")).replace("|", "\\|").replace("\n", " ").strip()
    skills = sorted(p.parent.name for p in (pdir / "skills").glob("*/SKILL.md"))
    product, repo = mapping.get(pdir.name, ("", ""))
    if repo:
        source = f"[{product}](https://github.com/{repo})"
    elif product:
        source = f"{product} (this repo)"
    else:
        source = "—"
    rows.append(f"| `{name}` | {desc} | `{version}` | {len(skills)} | {source} |")

print("| Plugin | Description | Version | Skills | Source |")
print("|--------|-------------|---------|--------|--------|")
if rows:
    print("\n".join(rows))
else:
    print("| _No plugins published yet_ | | | | |")
PY
)

tmp=$(mktemp)
table_file=$(mktemp)
printf '%s\n' "$table" > "$table_file"
awk -v start="$START" -v end="$END" -v tf="$table_file" '
  $0 == start { print; while ((getline line < tf) > 0) print line; close(tf); skipping=1; next }
  $0 == end   { skipping=0 }
  !skipping   { print }
' "$README" > "$tmp"
mv "$tmp" "$README"
rm -f "$table_file" "$map"
echo "✓ README plugin table regenerated ($(echo "$table" | tail -n +3 | grep -c '^| `' || true) plugin(s))"
