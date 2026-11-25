#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"
SKIP_TREE_NAMES = {
    ".DS_Store",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".py",
    ".sh",
    ".slurm",
    ".toml",
    ".ts",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
FILE_ORDER = {
    "challenge.md": 0,
    "state.md": 1,
    "trajectory.md": 2,
    "dashboard.md": 3,
    "literature.md": 4,
    "candidates.md": 5,
    "baseline.md": 6,
    "experiment.md": 7,
    "result_viewer.html": 8,
    "jobs": 9,
}


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class WebUIHandler(BaseHTTPRequestHandler):
    challenge_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/":
            self.serve_file(STATIC_DIR / "index.html")
            return
        if route.startswith("/static/"):
            target = safe_join(STATIC_DIR, route.removeprefix("/static/"))
            if target is None:
                self.send_error(404, "File not found")
                return
            self.serve_file(target)
            return
        if route == "/api/tree":
            self.send_json(build_file_tree(self.challenge_dir))
            return
        if route == "/api/raw":
            query = parse_qs(parsed.query)
            rel = query.get("path", [""])[0]
            target = safe_join(self.challenge_dir, rel)
            if target is None or not target.exists():
                self.send_error(404, "File not found")
                return
            self.send_text(read_text(target), "text/plain; charset=utf-8")
            return
        if route == "/md":
            query = parse_qs(parsed.query)
            rel = query.get("path", [""])[0]
            target = safe_join(self.challenge_dir, rel)
            if target is None or not target.exists() or target.suffix.lower() != ".md":
                self.send_error(404, "Markdown file not found")
                return
            self.send_text(
                render_markdown_page(target, self.challenge_dir), "text/html; charset=utf-8"
            )
            return
        if route == "/view":
            query = parse_qs(parsed.query)
            rel = query.get("path", [""])[0]
            target = safe_join(self.challenge_dir, rel)
            if target is None or not target.exists() or not target.is_file():
                self.send_error(404, "File not found")
                return
            self.send_text(render_text_page(target, self.challenge_dir), "text/html; charset=utf-8")
            return
        if route.startswith("/files/"):
            rel = route.removeprefix("/files/")
            target = safe_join(self.challenge_dir, rel)
            if target is None or not target.exists() or not target.is_file():
                self.send_error(404, "File not found")
                return
            self.serve_file(target)
            return
        if route == "/asset":
            query = parse_qs(parsed.query)
            rel = query.get("path", [""])[0]
            target = safe_join(self.challenge_dir, rel)
            if target is None or not target.exists() or not target.is_file():
                self.send_error(404, "Asset not found")
                return
            self.serve_file(target)
            return
        self.send_error(404, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: object) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, text: str, content_type: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def safe_join(root: Path, rel: str) -> Path | None:
    rel = unquote(rel).lstrip("/")
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def build_file_tree(root: Path) -> dict[str, object]:
    root = root.resolve()
    file_count = 0

    def visit(directory: Path) -> list[dict[str, object]]:
        nonlocal file_count
        nodes: list[dict[str, object]] = []
        try:
            children = sorted(directory.iterdir(), key=tree_sort_key)
        except OSError:
            return nodes
        for child in children:
            if child.name in SKIP_TREE_NAMES or child.name.startswith(".") or child.is_symlink():
                continue
            rel = rel_path(child, root)
            if child.is_dir():
                nodes.append(
                    {
                        "name": child.name,
                        "path": rel,
                        "type": "directory",
                        "experiment": bool(re.match(r"^exp\d+", child.name, re.IGNORECASE)),
                        "children": visit(child),
                    }
                )
                continue
            if not child.is_file():
                continue
            file_count += 1
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            nodes.append(
                {
                    "name": child.name,
                    "path": rel,
                    "type": "file",
                    "kind": file_kind(child),
                    "size": size,
                }
            )
        return nodes

    tree = visit(root)
    return {
        "name": root.name,
        "root": str(root),
        "file_count": file_count,
        "tree": tree,
    }


def tree_sort_key(path: Path) -> tuple[int, int, str]:
    explicit_order = FILE_ORDER.get(path.name.lower())
    if explicit_order is not None:
        return (0, explicit_order, path.name.lower())
    if path.is_dir():
        return (1, 0, path.name.lower())
    return (2, 0, path.name.lower())


def file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in TEXT_SUFFIXES or not suffix:
        return "text"
    return "binary"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def render_markdown_page(path: Path, root: Path) -> str:
    rel = rel_path(path, root)
    body = render_markdown(read_text(path), path.parent, root)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(rel)}</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #20252b;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1100px, calc(100% - 40px));
      margin: 28px auto 48px;
      background: #fff;
      border: 1px solid #dce1e8;
      border-radius: 8px;
      padding: 24px;
      box-shadow: 0 1px 2px rgba(18, 28, 45, 0.04);
    }}
    .path {{
      margin: 0 0 18px;
      color: #687381;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    h1, h2, h3, h4 {{ line-height: 1.2; }}
    h1 {{ margin-top: 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 22px;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      border: 1px solid #dce1e8;
      padding: 8px 10px;
    }}
    th {{ background: #f8fafc; }}
    pre {{
      overflow-x: auto;
      background: #101828;
      color: #f8fafc;
      padding: 14px;
      border-radius: 8px;
    }}
    code {{
      background: #f1f5f9;
      border-radius: 4px;
      padding: 1px 4px;
    }}
    pre code {{ background: transparent; padding: 0; }}
    img {{ max-width: 100%; height: auto; }}
    a {{ color: #2457d6; }}
  </style>
</head>
<body>
  <main>
    <p class="path">{html.escape(rel)}</p>
    {body}
  </main>
</body>
</html>"""


def render_text_page(path: Path, root: Path) -> str:
    rel = rel_path(path, root)
    text = read_text(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        try:
            if path.suffix.lower() == ".json":
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(rel)}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{
      margin: 0;
      background: #ffffff;
      color: #24292f;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    header {{
      position: sticky;
      top: 0;
      padding: 10px 16px;
      border-bottom: 1px solid #d0d7de;
      background: #f6f8fa;
      color: #57606a;
      font: 12px/1.4 Inter, ui-sans-serif, system-ui, sans-serif;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 0;
      padding: 16px;
      font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      tab-size: 2;
    }}
  </style>
</head>
<body>
  <header>{html.escape(rel)}</header>
  <pre>{html.escape(text)}</pre>
</body>
</html>"""


def render_markdown(text: str, base_dir: Path, root: Path) -> str:
    lines = text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []
    heading_ids: dict[str, int] = {}
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{inline_markdown(' '.join(paragraph), base_dir, root)}</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not in_code and stripped.startswith("<!--"):
            flush_paragraph()
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if stripped.startswith("```"):
            if in_code:
                out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        table = collect_table(lines, i)
        if table:
            flush_paragraph()
            out.append(render_table(table, base_dir, root))
            i += len(table)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            base_id = markdown_heading_id(heading.group(2))
            occurrence = heading_ids.get(base_id, 0)
            heading_ids[base_id] = occurrence + 1
            heading_id = base_id if occurrence == 0 else f"{base_id}-{occurrence}"
            out.append(
                f'<h{level} id="{html.escape(heading_id)}">'
                f"{inline_markdown(heading.group(2), base_dir, root)}</h{level}>"
            )
            i += 1
            continue
        bullet = collect_bullets(lines, i)
        if bullet:
            flush_paragraph()
            out.append(
                "<ul>"
                + "".join(f"<li>{inline_markdown(item, base_dir, root)}</li>" for item in bullet)
                + "</ul>"
            )
            i += len(bullet)
            continue
        paragraph.append(stripped)
        i += 1
    flush_paragraph()
    if in_code:
        out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(out)


def collect_table(lines: list[str], start: int) -> list[str]:
    if start + 1 >= len(lines):
        return []
    if not is_table_line(lines[start]) or not is_separator_line(lines[start + 1]):
        return []
    rows = [lines[start], lines[start + 1]]
    i = start + 2
    while i < len(lines) and is_table_line(lines[i]):
        rows.append(lines[i])
        i += 1
    return rows


def render_table(lines: list[str], base_dir: Path, root: Path) -> str:
    headers = split_row(lines[0])
    body_rows = [split_row(line) for line in lines[2:]]
    header_html = "".join(f"<th>{inline_markdown(cell, base_dir, root)}</th>" for cell in headers)
    rows_html = []
    for row in body_rows:
        cells = []
        for idx in range(len(headers)):
            value = row[idx] if idx < len(row) else ""
            cells.append(f"<td>{inline_markdown(value, base_dir, root)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + header_html
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def collect_bullets(lines: list[str], start: int) -> list[str]:
    items = []
    i = start
    while i < len(lines):
        match = re.match(r"^\s*[-*]\s+(.+)$", lines[i])
        if not match:
            break
        items.append(match.group(1).strip())
        i += 1
    return items


def inline_markdown(value: str, base_dir: Path, root: Path) -> str:
    escaped = html.escape(value)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)

    def image_replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        href = html.unescape(match.group(2))
        return f'<img src="{html.escape(resolve_href(href, base_dir, root, "asset"))}" alt="{alt}">'

    def link_replace(match: re.Match[str]) -> str:
        label = match.group(1)
        href = html.unescape(match.group(2))
        return f'<a href="{html.escape(resolve_href(href, base_dir, root))}">{label}</a>'

    escaped = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_replace, escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_replace, escaped)
    return escaped


def markdown_heading_id(value: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    text = re.sub(r"[`*~]", "", text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text) or "section"


def resolve_href(href: str, base_dir: Path, root: Path, kind: str = "") -> str:
    href = href.strip()
    if re.match(r"^(https?://|#)", href):
        return href
    path_part, separator, anchor = href.partition("#")
    fragment = f"#{anchor}" if separator else ""
    target = Path(unquote(path_part))
    if target.is_absolute():
        try:
            rel = str(target.resolve().relative_to(root.resolve()))
        except ValueError:
            return href
    else:
        rel = rel_path((base_dir / target).resolve(), root)
    if kind == "asset" or not rel.endswith(".md"):
        return "/files/" + quote_path(rel) + fragment
    return "/md?path=" + quote(rel) + fragment


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_separator_line(line: str) -> bool:
    if not is_table_line(line):
        return False
    cells = split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def quote_path(rel: str) -> str:
    return "/".join(quote(part) for part in rel.split("/"))


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def choose_port(host: str, requested: int) -> int:
    if requested == 0:
        return 0
    for port in range(requested, requested + 50):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only NV-MONAI-Autoresearch challenge file browser."
    )
    parser.add_argument(
        "--challenge", default="challenge/challenge_template", help="Challenge folder."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument(
        "--port", type=int, default=8765, help="Bind port. Use 0 for any free port."
    )
    args = parser.parse_args()

    challenge_dir = Path(args.challenge).expanduser()
    if not challenge_dir.is_absolute():
        challenge_dir = (Path.cwd() / challenge_dir).resolve()
    if not challenge_dir.exists():
        raise SystemExit(f"Challenge folder does not exist: {challenge_dir}")

    port = choose_port(args.host, args.port)
    if args.port and port != args.port:
        print(f"Port {args.port} is busy; using port {port or 'auto'} instead.")

    handler = type("ConfiguredWebUIHandler", (WebUIHandler,), {"challenge_dir": challenge_dir})
    server = ReusableThreadingHTTPServer((args.host, port), handler)
    host, actual_port = server.server_address
    print(f"NV-MONAI-Autoresearch files: http://{host}:{actual_port}/")
    print(f"Challenge: {challenge_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
