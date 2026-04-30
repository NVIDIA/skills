#!/usr/bin/env bash
# Regenerate the README's Available Skills and Getting Help & Contributing
# tables from components.yml. Used by the sync-skills workflow; can also
# be run locally to preview the result.
#
# Reads:
#   components.yml          — component catalog (source of truth)
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

sorted_indices=$(yq -r '.components | to_entries | sort_by(.value.name | downcase) | .[].key' components.yml)

# Available Skills table
{
  echo "| Product | Description | Skills | Catalog | Source | Version |"
  echo "|---------|-------------|:------:|---------|--------|---------|"
  for i in $sorted_indices; do
    name=$(yq -r ".components[$i].name" components.yml)
    description=$(yq -r ".components[$i].description" components.yml | tr -d '\n' | sed 's/  */ /g; s/^ //; s/ $//')
    repo=$(yq -r ".components[$i].repo" components.yml)
    ref=$(yq -r ".components[$i].ref // \"main\"" components.yml)
    primary_path=$(yq -r ".components[$i].skills[0].path" components.yml)
    primary_path=${primary_path%/}
    primary_catalog=$(yq -r ".components[$i].skills[0].catalog_dir" components.yml)

    skill_count=0
    while read -r catalog_dir; do
      if [ -d "skills/$catalog_dir" ]; then
        cnt=$(find "skills/$catalog_dir" -name SKILL.md -type f 2>/dev/null | wc -l | tr -d ' ')
        skill_count=$((skill_count + cnt))
      fi
    done < <(yq -r ".components[$i].skills[].catalog_dir" components.yml)

    slug=$(echo "$name" | tr 'A-Z ' 'a-z-')
    version_cell="—"
    if [ -s "$VERSIONS_FILE" ]; then
      if version_line=$(grep "^${slug}|" "$VERSIONS_FILE"); then
        IFS='|' read -r _ short_sha full_sha date sha_repo <<< "$version_line"
        version_cell="[\`${short_sha}\`](https://github.com/${sha_repo}/commit/${full_sha}) · ${date}"
      fi
    fi

    printf '| **%s** | %s | %d | [`skills/%s/`](skills/%s) | [Source](https://github.com/%s/tree/%s/%s) | %s |\n' \
      "$name" "$description" "$skill_count" "$primary_catalog" "$primary_catalog" "$repo" "$ref" "$primary_path" "$version_cell"
  done
} > /tmp/skills-table.md

# Getting Help & Contributing table
{
  echo "| Product | Issues | Discussions | Contributing | Security |"
  echo "|---------|--------|-------------|--------------|----------|"
  for i in $sorted_indices; do
    name=$(yq -r ".components[$i].name" components.yml)
    repo=$(yq -r ".components[$i].repo" components.yml)
    ref=$(yq -r ".components[$i].ref // \"main\"" components.yml)
    contrib=$(yq -r ".components[$i].links.contributing // \"CONTRIBUTING.md\"" components.yml)
    discussions=$(yq -r ".components[$i].links.discussions // true" components.yml)
    security=$(yq -r ".components[$i].links.security // true" components.yml)

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

    printf '| **%s** | %s | %s | %s | %s |\n' \
      "$name" "$issues_cell" "$discussions_cell" "$contributing_cell" "$security_cell"
  done
} > /tmp/help-table.md

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

echo "Regenerated README tables for $(echo "$sorted_indices" | wc -l | tr -d ' ') components"
