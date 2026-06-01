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
#
# Row aggregation:
#   Components that share the same display `name` are rendered as a single
#   README row regardless of whether they come from components.d/ (synced)
#   or manual-components.yml (manually-staged). When both sources contribute
#   to the same name, skill counts are summed and the synced row's metadata
#   (description, source, version) wins over the manual row's em-dash
#   defaults. Enables one logical product to span multiple yml entries
#   (e.g., the Physical AI product pulling some skills from a public
#   sub-repo and keeping the rest in the internal-staged manual list).

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

# Available Skills table — emit structured rows, aggregate by name, then format.
# TSV columns (tab-separated):
#   sort_key | name | description | skills_cell | is_manual
# `skills_cell` is a backtick-quoted, comma-separated list of skill catalog
# directories, each rendered as a markdown link to skills/<dir>/. Replaces
# the prior Source + Version columns; Source links pointed at only the
# primary skill (misleading for multi-skill products) and Version cells
# rendered as em dashes whenever VERSIONS_FILE was unpopulated.
SKILLS_ROWS=/tmp/skills-rows.tsv
truncate -s 0 "$SKILLS_ROWS"

for i in $kept_indices; do
  name=$(yq -r ".components[$i].name" "$CONFIG")
  description=$(yq -r ".components[$i].description" "$CONFIG" | tr -d '\n' | sed 's/  */ /g; s/^ //; s/ $//')

  # Build the comma-separated skills list: one link per catalog_dir that
  # currently has at least one verified SKILL.md in the catalog.
  skills_cell=""
  while read -r catalog_dir; do
    if [ -d "skills/$catalog_dir" ] && [ -n "$(find "skills/$catalog_dir" -name SKILL.md -type f 2>/dev/null | head -1)" ]; then
      skills_cell="${skills_cell}[\`${catalog_dir}\`](skills/${catalog_dir}), "
    fi
  done < <(yq -r ".components[$i].skills[].catalog_dir" "$CONFIG")
  skills_cell=${skills_cell%, }

  sort_key=$(echo "$name" | tr 'A-Z' 'a-z')
  printf '%s\t%s\t%s\t%s\t%d\n' \
    "$sort_key" "$name" "$description" "$skills_cell" 0 \
    >> "$SKILLS_ROWS"
done

# TEMPORARY — remove after Computex 2026. Append rows for manually-staged
# products (no upstream sync). These bypass the kept_indices verified-skills
# filter intentionally — they live in this catalog only as a stopgap until
# their upstream goes public.
manual_count=0
if [ -f "$MANUAL_CONFIG" ]; then
  manual_count=$(yq '.components | length' "$MANUAL_CONFIG")
  for i in $(seq 0 $((manual_count - 1))); do
    name=$(yq -r ".components[$i].name" "$MANUAL_CONFIG")
    description=$(yq -r ".components[$i].description" "$MANUAL_CONFIG" | tr -d '\n' | sed 's/  */ /g; s/^ //; s/ $//')

    dir_count=$(yq -r ".components[$i].catalog_dirs | length" "$MANUAL_CONFIG")
    skills_cell=""
    for j in $(seq 0 $((dir_count - 1))); do
      d=$(yq -r ".components[$i].catalog_dirs[$j]" "$MANUAL_CONFIG")
      if [ -d "skills/$d" ]; then
        skills_cell="${skills_cell}[\`${d}\`](skills/${d}), "
      fi
    done
    skills_cell=${skills_cell%, }

    sort_key=$(echo "$name" | tr 'A-Z' 'a-z')
    printf '%s\t%s\t%s\t%s\t%d\n' \
      "$sort_key" "$name" "$description" "$skills_cell" 1 \
      >> "$SKILLS_ROWS"
  done
fi

# Aggregation pass: group rows by sort_key (lowercase name). Concatenate
# skill links when synced + manual entries share the same product name
# (e.g., Physical AI = 5 manual + 2 synced — one row, all 7 listed).
# Prefer the synced row's description over the manual row's.
{
  echo "| Product | Description | Skills |"
  echo "|---------|-------------|--------|"
  sort -t$'\t' -k1,1 "$SKILLS_ROWS" | awk -F'\t' '
    {
      sk = $1; name = $2; desc = $3; skills = $4; man = $5 + 0
      if (!(sk in seen)) {
        seen[sk] = 1
        order[++n] = sk
        s_name[sk] = name
        s_desc[sk] = desc
        s_skills[sk] = skills
        s_man[sk] = man
      } else {
        # Merge skill lists across entries sharing this name.
        if (s_skills[sk] == "" || s_skills[sk] == "—") {
          s_skills[sk] = skills
        } else if (skills != "" && skills != "—") {
          s_skills[sk] = s_skills[sk] ", " skills
        }
        # Prefer non-manual entry for display name and description.
        if (man == 0 && s_man[sk] == 1) {
          s_name[sk] = name
          s_desc[sk] = desc
          s_man[sk] = 0
        }
      }
    }
    END {
      for (i = 1; i <= n; i++) {
        sk = order[i]
        printf "| **%s** | %s | %s |\n", \
          s_name[sk], s_desc[sk], s_skills[sk]
      }
    }
  '
} > /tmp/skills-table.md
rm -f "$SKILLS_ROWS"

# Getting Help & Contributing table — same aggregation pattern.
# TSV columns:
#   sort_key | name | issues_cell | discussions_cell | contributing_cell | security_cell | is_manual
HELP_ROWS=/tmp/help-rows.tsv
truncate -s 0 "$HELP_ROWS"

for i in $kept_indices; do
  name=$(yq -r ".components[$i].name" "$CONFIG")
  repo=$(yq -r ".components[$i].repo" "$CONFIG")
  ref=$(yq -r ".components[$i].ref // \"main\"" "$CONFIG")
  contrib=$(yq -r ".components[$i].links.contributing // \"CONTRIBUTING.md\"" "$CONFIG")
  # yq `//` defaults on both null and false, so an explicit `discussions: false`
  # would otherwise be silently treated as the default `true`. Read the raw
  # value, then default to "true" only when it is missing/null.
  discussions=$(yq -r ".components[$i].links.discussions" "$CONFIG")
  [ "$discussions" = "null" ] && discussions="true"
  security=$(yq -r ".components[$i].links.security" "$CONFIG")
  [ "$security" = "null" ] && security="true"

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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%d\n' \
    "$sort_key" "$name" "$issues_cell" "$discussions_cell" "$contributing_cell" "$security_cell" 0 \
    >> "$HELP_ROWS"
done

# TEMPORARY — remove after Computex 2026. Manual products have no public
# upstream, so every link cell renders as an em dash.
if [ -f "$MANUAL_CONFIG" ]; then
  for i in $(seq 0 $((manual_count - 1))); do
    name=$(yq -r ".components[$i].name" "$MANUAL_CONFIG")
    sort_key=$(echo "$name" | tr 'A-Z' 'a-z')
    printf '%s\t%s\t—\t—\t—\t—\t%d\n' "$sort_key" "$name" 1 >> "$HELP_ROWS"
  done
fi

# Aggregation: prefer the synced row's link cells when both synced + manual
# rows share the same name. The link cells aren't combined because there's
# one logical issues / discussions / contributing / security link per
# product, not per skill.
{
  echo "| Product | Issues | Discussions | Contributing | Security |"
  echo "|---------|--------|-------------|--------------|----------|"
  sort -t$'\t' -k1,1 "$HELP_ROWS" | awk -F'\t' '
    {
      sk = $1; name = $2; iss = $3; dis = $4; contrib = $5; sec = $6; man = $7 + 0
      if (!(sk in seen)) {
        seen[sk] = 1
        order[++n] = sk
        s_name[sk] = name
        s_iss[sk] = iss
        s_dis[sk] = dis
        s_contrib[sk] = contrib
        s_sec[sk] = sec
        s_man[sk] = man
      } else if (man == 0 && s_man[sk] == 1) {
        # Replace manual rows em-dash cells with the synced rows real links.
        s_name[sk] = name
        s_iss[sk] = iss
        s_dis[sk] = dis
        s_contrib[sk] = contrib
        s_sec[sk] = sec
        s_man[sk] = 0
      }
    }
    END {
      for (i = 1; i <= n; i++) {
        sk = order[i]
        printf "| **%s** | %s | %s | %s | %s |\n", \
          s_name[sk], s_iss[sk], s_dis[sk], s_contrib[sk], s_sec[sk]
      }
    }
  '
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
