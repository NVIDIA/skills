#!/usr/bin/env python3
"""
discover_assets.py — Skill Card Asset Discoverer

Given a path to a skill directory (e.g. <repo>/.agents/skills/<name>/),
walks up to find the repo root and emits a signal summary the agent uses
to fill the skill card context. The agent does not need to issue
additional Read calls to fill the card — every signal it needs is in
this output.

Usage: python3 discover_assets.py <skill_directory>
"""

import json
import re
import subprocess
import sys
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────

FILE_CHAR_LIMIT = 4000
TOTAL_CHAR_LIMIT = 30000
REPO_README_CHAR_LIMIT = 2500
EVAL_DOC_CHAR_LIMIT = 3000
CHANGELOG_BODY_CHAR_LIMIT = 3000
GIT_COMMAND_TIMEOUT_SECONDS = 3
LICENSE_SCAN_LINE_LIMIT = 5
LICENSE_IDENTIFIER_CHAR_LIMIT = 120
MCP_REFERENCE_LIMIT = 10
CONSTRAINT_CHAR_LIMIT = 300
CONSTRAINT_LIMIT = 25
H1_SCAN_LINE_LIMIT = 40
EVAL_DOC_LIMIT = 2
URLS_PER_PLATFORM_LIMIT = 10
DOC_INDEX_LIMIT = 30
SUMMARY_WIDTH = 70
USAGE_ERROR_EXIT = 1
SKILL_DEF_FULL = True  # Skill definition always extracted in full

REPO_ROOT_MARKERS = [".git", "pyproject.toml", "package.json", "LICENSE", "LICENSE.md"]

LICENSE_FILENAMES = {"license", "license.md", "license.txt", "copying", "notice", "notice.md"}

KNOWN_AGENTS = [
    "Amp", "Astra", "Blackbox", "Claude Code", "Codex", "Cursor",
    "Gemini Command Line Interface", "Gemini CLI", "GitHub Copilot",
    "Goose", "Junie", "OpenCode", "OpenClaw", "Hermes", "Kiro", "Roo Code",
]

PLATFORM_DOMAINS = {
    "Build.Nvidia.com": ["build.nvidia.com", "nvcr.io"],
    "GitHub": ["github.com"],
    "Hugging Face": ["huggingface.co", "hf.co"],
    "NGC": ["ngc.nvidia.com", "catalog.ngc.nvidia.com"],
}

API_KEY_PATTERNS = [
    r"\b[A-Z][A-Z0-9_]{2,}_API_KEY\b",
    r"\bHF_TOKEN\b",
    r"\bNGC_API_KEY\b",
    r"\bOPENAI_API_KEY\b",
    r"\bANTHROPIC_API_KEY\b",
    r"\bGITHUB_TOKEN\b",
    r"\bAWS_[A-Z_]+_KEY\b",
]

MCP_PATTERNS = [r"\bmcp__[a-z0-9_\-]+", r"MCP\s+server"]

CONSTRAINT_KEYWORDS = [
    "not supported", "not yet available", "must be disabled", "only supported",
    "cannot", "unsupported", "requires", "limited to",
]

EVAL_KEYWORDS = ["eval", "benchmark", "performance", "accuracy", "testing", "metric"]

# Legal/process links that should NOT be emitted as release channels.
LEGAL_URL_FRAGMENTS = [
    "sharepoint.com",
    "confluence.nvidia.com",
    "nvbugspro.nvidia.com",
    "forms.office.com",
    "app.intigriti.com",
    "nvidia.com/object/submit",
    "psirt",
]

# ─── Helpers ──────────────────────────────────────────────────────────────

def find_repo_root(start: Path) -> Path:
    """Walk up from start until we find a repo-root marker. Fall back to start."""
    current = start.resolve()
    while current != current.parent:
        for marker in REPO_ROOT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent
    return start


def reject_path_traversal(raw_path: str) -> None:
    """Reject explicit parent-directory traversal in user-supplied paths."""
    if ".." in Path(raw_path).expanduser().parts:
        print(f"Error: path must not contain '..': {raw_path}", file=sys.stderr)
        sys.exit(USAGE_ERROR_EXIT)


def has_yaml_frontmatter(path: Path) -> bool:
    try:
        text = path.read_text(errors="ignore")
        if not text.startswith("---"):
            return False
        end = text.find("\n---", 3)
        if end == -1:
            return False
        header = text[3:end]
        return "name:" in header and "description:" in header
    except Exception:
        return False


def read_content(path: Path, limit=None) -> str:
    try:
        text = path.read_text(errors="ignore")
        if limit is None or len(text) <= limit:
            return text
        return text[:limit] + f"\n... [truncated at {limit} chars]"
    except Exception:
        return "[unreadable]"


def parse_frontmatter(path: Path) -> dict:
    out = {}
    try:
        text = path.read_text(errors="ignore")
        if not text.startswith("---"):
            return out
        end = text.find("\n---", 3)
        if end == -1:
            return out
        header = text[3:end]
        for line in header.splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
            if m:
                key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
                if val:
                    out[key] = val
    except Exception:
        pass
    return out


def parse_license_identifier(license_path: Path) -> str | None:
    """Identify the license from the first non-empty line of a LICENSE file."""
    try:
        text = license_path.read_text(errors="ignore")
        for line in text.splitlines()[:LICENSE_SCAN_LINE_LIMIT]:
            line = line.strip()
            if not line:
                continue
            # Common short-form identifiers
            patterns = [
                (r"BSD[- ]?2[- ]?Clause", "BSD 2-Clause"),
                (r"BSD[- ]?3[- ]?Clause", "BSD 3-Clause"),
                (r"Apache\s+License.*2\.0", "Apache 2.0"),
                (r"MIT License", "MIT"),
                (r"GNU GENERAL PUBLIC LICENSE.*Version 3", "GPL-3.0"),
                (r"GNU GENERAL PUBLIC LICENSE.*Version 2", "GPL-2.0"),
                (r"Mozilla Public License", "MPL-2.0"),
                (r"NVIDIA AI Foundation Models Community License", "NVIDIA AI Foundation Models Community License"),
            ]
            for pat, name in patterns:
                if re.search(pat, line, re.IGNORECASE):
                    return name
            # If no pattern hits, return the first line verbatim (capped)
            return line[:LICENSE_IDENTIFIER_CHAR_LIMIT]
    except Exception:
        return None
    return None


def parse_pyproject_version(pyproject_path: Path) -> str | None:
    try:
        text = pyproject_path.read_text(errors="ignore")
        m = re.search(r'^\s*version\s*=\s*["\'](.+?)["\']', text, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def parse_package_json_version(pkg_path: Path) -> str | None:
    try:
        data = json.loads(pkg_path.read_text(errors="ignore"))
        return data.get("version")
    except Exception:
        return None


def parse_changelog_top_entry(changelog_path: Path) -> dict:
    """Return {version, date, body} from the top entry of a Keep-a-Changelog file."""
    out = {}
    try:
        text = changelog_path.read_text(errors="ignore")
        # Match first version header: ## [1.2.3] - 2026-03-03  (or similar)
        m = re.search(
            r"^##\s*\[?([0-9][^\]\s]*)\]?\s*[-–]\s*(\d{4}-\d{2}-\d{2})",
            text, re.MULTILINE,
        )
        if m:
            out["version"] = m.group(1)
            out["date"] = m.group(2)
            # Body: from end of header line until next ## or EOF.
            start = m.end()
            next_heading = re.search(r"\n##\s", text[start:])
            body_end = start + next_heading.start() if next_heading else len(text)
            body = text[start:body_end].strip()
            out["body"] = body[:CHANGELOG_BODY_CHAR_LIMIT]
    except Exception:
        pass
    return out


def git_info(root: Path) -> dict:
    out = {}
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--always"],
            capture_output=True, text=True, timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
        if r.returncode == 0 and r.stdout.strip():
            out["describe"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%H|%ai"],
            capture_output=True, text=True, timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split("|", 1)
            out["last_commit_sha"] = parts[0]
            if len(parts) > 1:
                out["last_commit_date"] = parts[1]
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
        if r.returncode == 0 and r.stdout.strip():
            out["remote_url"] = r.stdout.strip()
    except Exception:
        pass
    return out


def find_urls(text: str) -> list:
    return re.findall(r"https?://[^\s)>\]\"'`]+", text)


def group_urls_by_platform(urls: list) -> dict:
    groups = {p: [] for p in PLATFORM_DOMAINS}
    groups["Other"] = []
    for url in urls:
        if any(frag in url for frag in LEGAL_URL_FRAGMENTS):
            continue  # Legal boilerplate URLs are not release channels
        matched = False
        for platform, domains in PLATFORM_DOMAINS.items():
            if any(d in url for d in domains):
                if url not in groups[platform]:
                    groups[platform].append(url)
                matched = True
                break
        if not matched and url not in groups["Other"]:
            groups["Other"].append(url)
    return groups


def find_agents(text: str) -> list:
    found = []
    for agent in KNOWN_AGENTS:
        if re.search(r"\b" + re.escape(agent) + r"\b", text, re.IGNORECASE):
            if agent not in found:
                found.append(agent)
    return found


def find_api_keys(text: str) -> list:
    keys = []
    for pat in API_KEY_PATTERNS:
        for m in re.findall(pat, text):
            if m not in keys:
                keys.append(m)
    return keys


def find_mcp_refs(text: str) -> list:
    refs = []
    for pat in MCP_PATTERNS:
        for m in re.findall(pat, text, re.IGNORECASE):
            if m not in refs:
                refs.append(m)
    return refs[:MCP_REFERENCE_LIMIT]


def find_constraints(text: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    hits = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) > CONSTRAINT_CHAR_LIMIT:
            continue
        lower = s_clean.lower()
        if any(kw in lower for kw in CONSTRAINT_KEYWORDS):
            if s_clean not in hits:
                hits.append(s_clean)
    return hits[:CONSTRAINT_LIMIT]


def count_arguments_usage(text: str) -> int:
    return len(re.findall(r"\$ARGUMENTS", text))


# ─── Skill-dir categorization (unchanged role logic, repo-scope added) ───

def categorize_skill_dir(skill_root: Path) -> dict:
    roles = {
        "Skill definition": [], "Documentation": [], "Reference material": [],
        "Scripts": [], "Config": [], "Other": [],
    }
    for path in sorted(skill_root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(skill_root)
        parts = rel.parts
        if any(p in {"__pycache__", ".git", ".venv", "node_modules"} for p in parts):
            continue
        suffix = path.suffix.lower()
        if "references" in parts[:-1]:
            roles["Reference material"].append(path)
            continue
        if "scripts" in parts[:-1] or suffix in {".py", ".sh", ".js", ".ts", ".bash"}:
            roles["Scripts"].append(path)
            continue
        if suffix in {".md", ".yaml", ".yml"} and has_yaml_frontmatter(path):
            roles["Skill definition"].append(path)
            continue
        if suffix in {".md", ".rst", ".txt"}:
            roles["Documentation"].append(path)
            continue
        if suffix in {".yaml", ".yml", ".toml", ".json", ".ini", ".env", ".cfg"}:
            roles["Config"].append(path)
            continue
        roles["Other"].append(path)
    return roles


# ─── Repo-root signal collection ──────────────────────────────────────────

def collect_repo_signals(repo_root: Path, skill_root: Path) -> dict:
    """Pull governance-relevant signals from the repo above the skill."""
    out = {
        "repo_root": str(repo_root),
        "is_nested": repo_root != skill_root,
    }

    # LICENSE file (first match)
    for fname in ["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"]:
        lic = repo_root / fname
        if lic.exists():
            out["license_file"] = str(lic.relative_to(repo_root))
            out["license_identifier"] = parse_license_identifier(lic)
            break

    # Version signals — try multiple sources, report all
    versions = {}
    py = repo_root / "pyproject.toml"
    if py.exists():
        v = parse_pyproject_version(py)
        if v:
            versions["pyproject"] = v
    pkg = repo_root / "package.json"
    if pkg.exists():
        v = parse_package_json_version(pkg)
        if v:
            versions["package_json"] = v
    cl = repo_root / "CHANGELOG.md"
    if cl.exists():
        entry = parse_changelog_top_entry(cl)
        if entry.get("version"):
            versions["changelog"] = entry["version"]
            out["changelog_top_entry"] = entry
    if versions:
        out["versions"] = versions

    # Git
    git = git_info(repo_root)
    if git:
        out["git"] = git

    # Known-issue / docs scan
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        doc_files = []
        eval_docs = []
        for p in sorted(docs_dir.rglob("*.md")):
            rel = p.relative_to(repo_root)
            title = _first_h1(p) or p.stem
            entry = {"path": str(rel), "title": title}
            doc_files.append(entry)
            # Flag as evaluation-relevant by name or title
            name_lower = p.stem.lower()
            title_lower = title.lower()
            if any(kw in name_lower or kw in title_lower for kw in EVAL_KEYWORDS):
                eval_docs.append(entry)
        if doc_files:
            out["docs"] = doc_files
        if eval_docs:
            out["evaluation_docs"] = eval_docs

    # README at repo root
    for fname in ["README.md", "README.rst", "README.txt"]:
        rm = repo_root / fname
        if rm.exists():
            out["readme"] = str(rm.relative_to(repo_root))
            break

    # Security policy
    sec = repo_root / "SECURITY.md"
    if sec.exists():
        out["security_md"] = str(sec.relative_to(repo_root))

    # Third-party license file presence (useful for Database Type context)
    for fname in ["third_party_oss_license.txt", "third_party_licenses.txt", "NOTICE"]:
        tp = repo_root / fname
        if tp.exists():
            out.setdefault("third_party_license_files", []).append(str(tp.relative_to(repo_root)))

    return out


def _first_h1(path: Path) -> str | None:
    try:
        for line in path.read_text(errors="ignore").splitlines()[:H1_SCAN_LINE_LIMIT]:
            m = re.match(r"^#\s+(.+?)\s*$", line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


# ─── Content extraction for the agent ─────────────────────────────────────

def extract_skill_contents(roles: dict) -> list:
    """Extract skill-local file contents, prioritized and budgeted."""
    extracted = []
    total = 0
    priority = [
        "Skill definition", "Documentation", "Reference material",
        "Scripts", "Config",
    ]
    for role in priority:
        for path in roles.get(role, []):
            if role == "Skill definition" and SKILL_DEF_FULL:
                content = read_content(path, limit=None)
                extracted.append((role, path, content))
                total += len(content)
                continue
            if total >= TOTAL_CHAR_LIMIT:
                break
            remaining = TOTAL_CHAR_LIMIT - total
            content = read_content(path, min(FILE_CHAR_LIMIT, remaining))
            extracted.append((role, path, content))
            total += len(content)
        if total >= TOTAL_CHAR_LIMIT and role != "Skill definition":
            break
    return extracted


def extract_repo_contents(repo_signals: dict, repo_root: Path) -> list:
    """Extract a small set of repo-root governance files in full."""
    extracted = []
    # CHANGELOG top entry is already parsed; don't re-emit full file.
    # README: enough for description + audience.
    if readme := repo_signals.get("readme"):
        extracted.append(("Repo README", repo_root / readme,
                          read_content(repo_root / readme, limit=REPO_README_CHAR_LIMIT)))
    # Evaluation docs: a small sample, capped per file.
    for d in repo_signals.get("evaluation_docs", [])[:EVAL_DOC_LIMIT]:
        p = repo_root / d["path"]
        extracted.append(("Repo eval doc", p, read_content(p, limit=EVAL_DOC_CHAR_LIMIT)))
    return extracted


# ─── Output ───────────────────────────────────────────────────────────────

def emit_signal_summary(
    skill_root: Path, repo_root: Path, roles: dict,
    skill_extracted: list, repo_extracted: list, repo_signals: dict,
) -> None:
    print("\n" + "=" * SUMMARY_WIDTH)
    print("\n=== STRUCTURED SIGNAL SUMMARY ===")
    print("# These are the pre-extracted signals for card context assembly.")
    print("# Consult this section before scanning raw file contents.")
    print("=" * SUMMARY_WIDTH + "\n")

    # Skill frontmatter
    fm = {}
    if roles["Skill definition"]:
        fm = parse_frontmatter(roles["Skill definition"][0])
    print("## Skill definition frontmatter")
    if fm:
        for k, v in fm.items():
            print(f"  {k}: {v}")
    else:
        print("  [no parseable frontmatter]")
    print()

    # Repo signals
    print("## Repo-root signals")
    if repo_signals.get("is_nested"):
        print(f"  repo_root: {repo_signals['repo_root']}")
    else:
        print("  [skill directory IS the repo root — no nesting]")
    if lic := repo_signals.get("license_identifier"):
        print(f"  license_identifier: {lic}  (from {repo_signals.get('license_file')})")
    if versions := repo_signals.get("versions"):
        for src, v in versions.items():
            print(f"  version.{src}: {v}")
    if git := repo_signals.get("git"):
        for k, v in git.items():
            print(f"  git.{k}: {v}")
    if cl := repo_signals.get("changelog_top_entry"):
        print(f"  changelog.version: {cl.get('version')}")
        print(f"  changelog.date: {cl.get('date')}")
        if body := cl.get("body"):
            print("  changelog.body: |")
            for line in body.splitlines()[:H1_SCAN_LINE_LIMIT]:
                print(f"    {line}")
    if readme := repo_signals.get("readme"):
        print(f"  readme: {readme}")
    if sec := repo_signals.get("security_md"):
        print(f"  security_md: {sec}")
    if tp := repo_signals.get("third_party_license_files"):
        for t in tp:
            print(f"  third_party_license_file: {t}")
    print()

    # Collect full text for pattern scans
    all_text = "\n".join(c for _, _, c in skill_extracted + repo_extracted)
    if fm:
        all_text += "\n" + " ".join(f"{k}: {v}" for k, v in fm.items())
    # CHANGELOG top-entry body is extracted separately; include it in the scan
    if cl := repo_signals.get("changelog_top_entry"):
        if body := cl.get("body"):
            all_text += "\n" + body

    # URLs
    urls = find_urls(all_text)
    groups = group_urls_by_platform(urls)
    print("## Detected URLs by platform  (legal/process URLs excluded)")
    any_urls = False
    for platform, items in groups.items():
        if items:
            any_urls = True
            print(f"  {platform}:")
            for u in items[:URLS_PER_PLATFORM_LIMIT]:
                print(f"    - {u}")
    if not any_urls:
        print("  [no release-channel URLs detected]")
    print()

    # Agents
    agents = find_agents(all_text)
    print("## Agents mentioned anywhere in sources")
    if agents:
        for a in agents:
            print(f"  - {a}")
    else:
        print("  [none detected]")
    print()

    # Credentials
    keys = find_api_keys(all_text)
    print("## Detected API-key / credential env vars")
    if keys:
        for k in keys:
            print(f"  - {k}")
    else:
        print("  [none detected]")
    print()

    # MCP references
    mcps = find_mcp_refs(all_text)
    print("## MCP / tool references")
    if mcps:
        for m in mcps:
            print(f"  - {m}")
    else:
        print("  [none detected]")
    print()

    # $ARGUMENTS
    arg_count = count_arguments_usage(all_text)
    print(f"## $ARGUMENTS usage count: {arg_count}")
    print()

    # Constraint sentences
    constraints = find_constraints(all_text)
    print("## Constraint sentences (candidates for Known Technical Limitations)")
    if constraints:
        for c in constraints:
            print(f"  - {c}")
    else:
        print("  [none detected]")
    print()

    # Evaluation docs
    print("## Evaluation artifacts")
    eval_docs = repo_signals.get("evaluation_docs", [])
    if eval_docs:
        for d in eval_docs:
            print(f"  - {d['path']}  ({d['title']})")
    else:
        print("  [none detected — evaluation fields may require human input]")
    print()

    # Docs index
    if docs := repo_signals.get("docs"):
        print("## Repo docs/ index")
        for d in docs[:DOC_INDEX_LIMIT]:
            print(f"  - {d['path']}  ({d['title']})")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 discover_assets.py <skill_directory>", file=sys.stderr)
        sys.exit(USAGE_ERROR_EXIT)

    reject_path_traversal(sys.argv[1])
    skill_root = Path(sys.argv[1]).expanduser().resolve()
    if not skill_root.exists():
        print(f"Error: directory not found: {skill_root}", file=sys.stderr)
        sys.exit(USAGE_ERROR_EXIT)
    if not skill_root.is_dir():
        print(f"Error: not a directory: {skill_root}", file=sys.stderr)
        sys.exit(USAGE_ERROR_EXIT)

    repo_root = find_repo_root(skill_root)
    roles = categorize_skill_dir(skill_root)

    print(f"# Asset Discovery Report — Skill Card")
    print(f"# Skill target: {skill_root}")
    print(f"# Repo root:    {repo_root}")
    if repo_root == skill_root:
        print("# (Skill directory is the repo root — no parent signals.)")
    print()

    for role, files in roles.items():
        if files:
            print(f"## {role} ({len(files)} file{'s' if len(files) != 1 else ''})")
            for f in files:
                print(f"  - {f.relative_to(skill_root)}")
            print()

    if not roles["Skill definition"]:
        print("STOP: No skill definition file found. Cannot proceed.")
        return

    # Repo-root scope
    repo_signals = collect_repo_signals(repo_root, skill_root)

    # Extract contents
    skill_extracted = extract_skill_contents(roles)
    repo_extracted = extract_repo_contents(repo_signals, repo_root)

    print("\n" + "=" * SUMMARY_WIDTH)
    print("\n=== EXTRACTED FILE CONTENTS (skill scope) ===")
    print("=" * SUMMARY_WIDTH + "\n")
    for role, path, content in skill_extracted:
        try:
            rel = path.relative_to(skill_root)
        except ValueError:
            rel = path
        print(f"### [{role}] {rel}")
        print("```")
        print(content)
        print("```\n")

    if repo_extracted:
        print("\n" + "=" * SUMMARY_WIDTH)
        print("\n=== EXTRACTED FILE CONTENTS (repo scope) ===")
        print("=" * SUMMARY_WIDTH + "\n")
        for role, path, content in repo_extracted:
            try:
                rel = path.relative_to(repo_root)
            except ValueError:
                rel = path
            print(f"### [{role}] {rel}")
            print("```")
            print(content)
            print("```\n")

    emit_signal_summary(skill_root, repo_root, roles, skill_extracted, repo_extracted, repo_signals)

    # Append style guide + Jinja template so the agent can proceed in one pass
    skill_dir = Path(__file__).parent.parent
    for label, fname in [("STYLE GUIDE", "style-guide.md"), ("JINJA TEMPLATE", "skill-card.md.j2")]:
        fpath = skill_dir / "references" / fname
        print("\n" + "=" * SUMMARY_WIDTH)
        print(f"\n=== {label} ===")
        print("=" * SUMMARY_WIDTH + "\n")
        if fpath.exists():
            print(fpath.read_text(errors="ignore"))
        else:
            print(f"[{fname} not found — check skill installation at {skill_dir}]")


if __name__ == "__main__":
    main()
