#!/usr/bin/env bash
# Regenerate the README's Available Skills and Getting Help & Contributing
# tables from components.d/*.yml. Used by the sync-skills workflow; can also
# be run locally to preview the result.
#
# Reads:
#   components.d/*.yml      — per-component catalog files (source of truth)
#   skills/<catalog_dir>/   — to count SKILL.md files per component
#   /tmp/sync-versions.txt  — optional, populated by the sync workflow with
#                             upstream short SHA, full SHA, and committer
#                             date per component. When absent, the Version
#                             cell is rendered as an em dash.
#
# Writes:
#   README.md (in place) — content between marker pairs is replaced:
#     <!-- skills-table-start --> ... <!-- skills-table-end -->
#     <!-- help-table-start  --> ... <!-- help-table-end  -->

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

VERSIONS_FILE="${VERSIONS_FILE:-/tmp/sync-versions.txt}"

# TEMPORARY — remove after Computex 2026. Manual catalog exceptions live
# in a separate file (see header of manual-components.yml for why).
MANUAL_CONFIG=.github/scripts/manual-components.yml

# Aggregate per-component files into a single config so the existing
# yq queries can index into a flat .components list.
CONFIG=/tmp/components.aggregated.yml
yq ea '[.] | {"components": .}' components.d/*.yml > "$CONFIG"

sorted_indices=$(yq -r '.components | to_entries | sort_by(.value.name | downcase) | .[].key' "$CONFIG")

# Only render components with at least one verified SKILL.md in the catalog.
# Enforcement (sync workflow) removes non-compliant skills and orphan product
# dirs before this script runs, so a zero count means the product currently
# has no skill present in the repo that passes compliance.
component_skill_count() {
  local idx=$1
  local total=0
  while read -r catalog_dir; do
    if [ -d "skills/$catalog_dir" ]; then
      cnt=$(find "skills/$catalog_dir" -name SKILL.md -type f 2>/dev/null | wc -l | tr -d ' ')
      total=$((total + cnt))
    fi
  done < <(yq -r ".components[$idx].skills[].catalog_dir" "$CONFIG")
  echo "$total"
}

kept_indices=""
for i in $sorted_indices; do
  if [ "$(component_skill_count "$i")" -gt 0 ]; then
    kept_indices="$kept_indices $i"
  fi
done

# Available Skills table
SKILLS_ROWS=/tmp/skills-rows.tsv
truncate -s 0 "$SKILLS_ROWS"

for i in $kept_indices; do
  name=$(yq -r ".components[$i].name" "$CONFIG")
  description=$(yq -r ".components[$i].description" "$CONFIG" | tr -d '\n' | sed 's/  */ /g; s/^ //; s/ $//')
  repo=$(yq -r ".components[$i].repo" "$CONFIG")
  ref=$(yq -r ".components[$i].ref // \"main\"" "$CONFIG")
  primary_path=$(yq -r ".components[$i].skills[0].path" "$CONFIG")
  primary_path=${primary_path%/}
  primary_catalog=$(yq -r ".components[$i].skills[0].catalog_dir" "$CONFIG")

  skill_count=$(component_skill_count "$i")

  slug=$(echo "$name" | tr 'A-Z ' 'a-z-')
  version_cell="—"
  if [ -s "$VERSIONS_FILE" ]; then
    if version_line=$(grep "^${slug}|" "$VERSIONS_FILE"); then
      IFS='|' read -r _ short_sha full_sha date sha_repo <<< "$version_line"
      version_cell="[\`${short_sha}\`](https://github.com/${sha_repo}/commit/${full_sha}) · ${date}"
    fi
  fi

  sort_key=$(echo "$name" | tr 'A-Z' 'a-z')
  row=$(printf '| **%s** | %s | %d | [`skills/%s/`](skills/%s) | [Source](https://github.com/%s/tree/%s/%s) | %s |' \
    "$name" "$description" "$skill_count" "$primary_catalog" "$primary_catalog" "$repo" "$ref" "$primary_path" "$version_cell")
  printf '%s\t%s\n' "$sort_key" "$row" >> "$SKILLS_ROWS"
done

# TEMPORARY — remove after Computex 2026. Append rows for manually-staged
# products (no upstream sync); Source and Version cells render as em dashes.
# These bypass the kept_indices verified-skills filter intentionally — they
# live in this catalog only as a stopgap until their upstream goes public.
manual_count=0
if [ -f "$MANUAL_CONFIG" ]; then
  manual_count=$(yq '.components | length' "$MANUAL_CONFIG")
  for i in $(seq 0 $((manual_count - 1))); do
    name=$(yq -r ".components[$i].name" "$MANUAL_CONFIG")
    description=$(yq -r ".components[$i].description" "$MANUAL_CONFIG" | tr -d '\n' | sed 's/  */ /g; s/^ //; s/ $//')

    dir_count=$(yq -r ".components[$i].catalog_dirs | length" "$MANUAL_CONFIG")
    catalog_cell=""
    skill_count=0
    for j in $(seq 0 $((dir_count - 1))); do
      d=$(yq -r ".components[$i].catalog_dirs[$j]" "$MANUAL_CONFIG")
      catalog_cell="${catalog_cell}[\`${d}/\`](skills/${d}) · "
      if [ -d "skills/$d" ]; then
        cnt=$(find "skills/$d" -name SKILL.md -type f 2>/dev/null | wc -l | tr -d ' ')
        skill_count=$((skill_count + cnt))
      fi
    done
    catalog_cell=${catalog_cell% · }

    sort_key=$(echo "$name" | tr 'A-Z' 'a-z')
    row=$(printf '| **%s** | %s | %d | %s | — | — |' \
      "$name" "$description" "$skill_count" "$catalog_cell")
    printf '%s\t%s\n' "$sort_key" "$row" >> "$SKILLS_ROWS"
  done
fi

{
  echo "| Product | Description | Skills | Catalog | Source | Version |"
  echo "|---------|-------------|:------:|---------|--------|---------|"
  sort -t$'\t' -k1,1 "$SKILLS_ROWS" | cut -f2-
} > /tmp/skills-table.md
rm -f "$SKILLS_ROWS"

# Getting Help & Contributing table
HELP_ROWS=/tmp/help-rows.tsv
truncate -s 0 "$HELP_ROWS"

for i in $kept_indices; do
  name=$(yq -r ".components[$i].name" "$CONFIG")
  repo=$(yq -r ".components[$i].repo" "$CONFIG")
  ref=$(yq -r ".components[$i].ref // \"main\"" "$CONFIG")
  contrib=$(yq -r ".components[$i].links.contributing // \"CONTRIBUTING.md\"" "$CONFIG")
  discussions=$(yq -r ".components[$i].links.discussions // true" "$CONFIG")
  security=$(yq -r ".components[$i].links.security // true" "$CONFIG")

  issues_cell="[Issues](https://github.com/${repo}/issues)"
  if [ "$discussions" = "true" ]; then
    discussions_cell="[Discussions](https://github.com/${repo}/discussions)"
  else
    discussions_cell="—"
  fi
  contributing_cell="[Contributing](https://github.com/${repo}/blob/${ref}/${contrib})"
  if [ "$security" = "true" ]; then
    security_cell="[Security](https://github.com/${repo}/blob/${ref}/SECURITY.md)"
  else
    security_cell="—"
  fi

  sort_key=$(echo "$name" | tr 'A-Z' 'a-z')
  row=$(printf '| **%s** | %s | %s | %s | %s |' \
    "$name" "$issues_cell" "$discussions_cell" "$contributing_cell" "$security_cell")
  printf '%s\t%s\n' "$sort_key" "$row" >> "$HELP_ROWS"
done

# TEMPORARY — remove after Computex 2026. Manual products have no public
# upstream, so every link cell renders as an em dash.
if [ -f "$MANUAL_CONFIG" ]; then
  for i in $(seq 0 $((manual_count - 1))); do
    name=$(yq -r ".components[$i].name" "$MANUAL_CONFIG")
    sort_key=$(echo "$name" | tr 'A-Z' 'a-z')
    row=$(printf '| **%s** | — | — | — | — |' "$name")
    printf '%s\t%s\n' "$sort_key" "$row" >> "$HELP_ROWS"
  done
fi

{
  echo "| Product | Issues | Discussions | Contributing | Security |"
  echo "|---------|--------|-------------|--------------|----------|"
  sort -t$'\t' -k1,1 "$HELP_ROWS" | cut -f2-
} > /tmp/help-table.md
rm -f "$HELP_ROWS"

# Replace content between markers
awk -v skills_file=/tmp/skills-table.md -v help_file=/tmp/help-table.md '
  BEGIN { in_skills = 0; in_help = 0 }
  /<!-- skills-table-start -->/ {
    print
    while ((getline line < skills_file) > 0) print line
    close(skills_file)
    in_skills = 1
    next
  }
  /<!-- skills-table-end -->/ {
    in_skills = 0
    print
    next
  }
  /<!-- help-table-start -->/ {
    print
    while ((getline line < help_file) > 0) print line
    close(help_file)
    in_help = 1
    next
  }
  /<!-- help-table-end -->/ {
    in_help = 0
    print
    next
  }
  !in_skills && !in_help { print }
' README.md > /tmp/README.md.new

mv /tmp/README.md.new README.md
rm -f /tmp/skills-table.md /tmp/help-table.md

total=$(echo "$sorted_indices" | wc -w | tr -d ' ')
kept=$(echo "$kept_indices" | wc -w | tr -d ' ')
echo "Regenerated README tables for $kept of $total synced components (filtered out $((total - kept)) with no verified skills) + $manual_count manual"
