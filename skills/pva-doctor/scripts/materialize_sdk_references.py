#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an installed PVA SDK documentation package to locked Markdown.

The sync command reads the SDK's local HTML references and the packaged Orin
and Thor Programmer's Guide PDFs. It does not use the network. Each
content-bearing HTML page becomes one Markdown file. Each guide becomes one
Markdown file per chapter. The verify command checks the local sources, lock,
generated file set, and output hashes without regenerating content.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import html
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar
from urllib.parse import quote, unquote, urlsplit

MANIFEST_SCHEMA_VERSION = 5
# Independent integer revisions: input shape, lock shape, and conversion behavior.
# Lock v5 stores the converter revision as an integer; regenerate older caches.
LOCK_SCHEMA_VERSION = 5
CONVERTER_VERSION = 10
LOCK_NAME = "reference-lock.json"
INDEX_NAME = "index.md"
SOURCE_ROOT_LABEL = "$PVA_SDK_ROOT/share/doc"

MARKDOWN_COPYRIGHT_HEADER = """<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
"""


class ReferenceError(RuntimeError):
    """Report a reference generation or verification failure."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferenceError(f"JSON root must be an object: {path}")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceError(f"{label} must be a nonempty string")
    return value.strip()


def require_id(value: Any, label: str) -> str:
    result = require_string(value, label)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", result):
        raise ReferenceError(f"{label} must use lowercase letters, digits, and hyphens")
    return result


def require_relative_pattern(value: Any, label: str) -> str:
    result = require_string(value, label).replace("\\", "/")
    if result.startswith("/") or ".." in PurePosixPath(result).parts:
        raise ReferenceError(f"{label} must stay below the SDK documentation root")
    return result


def validate_chapter(value: Any, guide_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReferenceError(f"{guide_id}: each chapter must be an object")
    number = value.get("chapter")
    if not isinstance(number, int) or number < 1:
        raise ReferenceError(f"{guide_id}: chapter number must be positive")
    filename = require_string(value.get("file"), f"{guide_id} chapter file")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*\.md", filename):
        raise ReferenceError(f"{guide_id}: unsafe chapter file: {filename}")
    return {
        "chapter": number,
        "title": require_string(value.get("title"), f"{guide_id} chapter title"),
        "file": filename,
    }


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ReferenceError(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}")

    raw_html = manifest.get("html")
    if not isinstance(raw_html, dict):
        raise ReferenceError("html must be an object")
    excludes = raw_html.get("exclude", [])
    if not isinstance(excludes, list):
        raise ReferenceError("html.exclude must be an array")
    html_config = {
        "exclude": [require_relative_pattern(item, "html exclude pattern") for item in excludes],
        "article_class": require_string(
            raw_html.get("article_class", "bd-article"), "html article class"
        ),
    }

    raw_guides = manifest.get("guides")
    if not isinstance(raw_guides, list) or not raw_guides:
        raise ReferenceError("guides must be a nonempty array")
    guides: list[dict[str, Any]] = []
    guide_ids: set[str] = set()
    guide_files: set[str] = set()
    for raw in raw_guides:
        if not isinstance(raw, dict):
            raise ReferenceError("each guide must be an object")
        guide_id = require_id(raw.get("id"), "guide id")
        filename = require_string(raw.get("file_name"), f"{guide_id} file_name")
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise ReferenceError(f"{guide_id}: file_name must be a PDF base name")
        if guide_id in guide_ids or filename in guide_files:
            raise ReferenceError(f"duplicate guide id or file name: {guide_id}")
        raw_chapters = raw.get("chapters")
        if not isinstance(raw_chapters, list) or not raw_chapters:
            raise ReferenceError(f"{guide_id}: chapters must be a nonempty array")
        chapters = [validate_chapter(chapter, guide_id) for chapter in raw_chapters]
        numbers = [chapter["chapter"] for chapter in chapters]
        files = [chapter["file"] for chapter in chapters]
        if numbers != list(range(1, len(chapters) + 1)) or len(files) != len(set(files)):
            raise ReferenceError(f"{guide_id}: chapters must be ordered and unique")
        guide_ids.add(guide_id)
        guide_files.add(filename)
        guides.append(
            {
                "id": guide_id,
                "title": require_string(raw.get("title"), f"{guide_id} title"),
                "file_name": filename,
                "document_id": require_string(raw.get("document_id"), f"{guide_id} document id"),
                "chapters": chapters,
            }
        )

    raw_routes = manifest.get("skill_routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise ReferenceError("skill_routes must be a nonempty object")
    routes: dict[str, dict[str, Any]] = {}
    valid_chapters = set.intersection(
        *({chapter["chapter"] for chapter in guide["chapters"]} for guide in guides)
    )
    for raw_skill, raw_route in raw_routes.items():
        skill = require_id(raw_skill, "skill route name")
        if not isinstance(raw_route, dict):
            raise ReferenceError(f"skill route {skill!r} must be an object")
        patterns = raw_route.get("html_patterns", [])
        terms = raw_route.get("search_terms", [])
        chapter_numbers = raw_route.get("guide_chapters", [])
        if not isinstance(patterns, list) or not isinstance(terms, list):
            raise ReferenceError(f"skill route {skill!r} has invalid arrays")
        if not isinstance(chapter_numbers, list) or not all(
            isinstance(number, int) for number in chapter_numbers
        ):
            raise ReferenceError(f"skill route {skill!r} has invalid guide_chapters")
        unknown = sorted(set(chapter_numbers) - valid_chapters)
        if unknown:
            raise ReferenceError(f"skill route {skill!r} has unknown chapters: {unknown}")
        routes[skill] = {
            "html_patterns": [
                require_relative_pattern(item, f"{skill} HTML pattern") for item in patterns
            ],
            "search_terms": [require_string(item, f"{skill} search term") for item in terms],
            "guide_chapters": chapter_numbers,
        }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "html": html_config,
        "guides": guides,
        "skill_routes": routes,
    }


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferenceError("manifest root must be an object")
    return validate_manifest(value), sha256_bytes(raw)


def path_is_excluded(relative: str, patterns: list[str]) -> bool:
    return any(route_matches(pattern, relative) for pattern in patterns)


def ensure_source_file(root: Path, path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReferenceError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ReferenceError(f"{label} escaped the SDK documentation root: {path}")
    if not resolved.is_file():
        raise ReferenceError(f"{label} is not a regular file: {path}")
    return resolved


def read_sdk_version(sdk_doc_root: Path) -> str:
    path = ensure_source_file(sdk_doc_root, sdk_doc_root / "VERSION", "SDK VERSION file")
    value = path.read_text(encoding="utf-8").strip()
    if not value or len(value) > 200 or "\n" in value:
        raise ReferenceError(f"SDK VERSION file has invalid content: {path}")
    return value


def has_article(raw: bytes, article_class: str) -> bool:
    pattern = (
        rb"<article\b[^>]*\bclass=[\"'][^\"']*\b"
        + re.escape(article_class.encode("ascii"))
        + rb"\b[^\"']*[\"']"
    )
    return re.search(pattern, raw, re.IGNORECASE) is not None


def discover_html_sources(
    sdk_doc_root: Path, html_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    sources: list[dict[str, Any]] = []
    skipped: list[str] = []
    output_sources: dict[str, str] = {}
    candidates = (path for path in sdk_doc_root.rglob("*") if path.suffix.lower() == ".html")
    for candidate in sorted(candidates):
        path = ensure_source_file(sdk_doc_root, candidate, "HTML source")
        relative = path.relative_to(sdk_doc_root).as_posix()
        if path_is_excluded(relative, html_config["exclude"]):
            skipped.append(relative)
            continue
        raw = path.read_bytes()
        if not has_article(raw, html_config["article_class"]):
            skipped.append(relative)
            continue
        output = (PurePosixPath("html") / PurePosixPath(relative)).with_suffix(".md")
        if output.as_posix() in output_sources:
            raise ReferenceError(
                "HTML sources map to the same Markdown path: "
                f"{output_sources[output.as_posix()]} and {relative} -> {output}"
            )
        output_sources[output.as_posix()] = relative
        sources.append(
            {
                "relative_path": relative,
                "output_path": output.as_posix(),
                "raw": raw,
                "raw_sha256": sha256_bytes(raw),
            }
        )
    if not sources:
        raise ReferenceError(f"no content-bearing HTML files found below {sdk_doc_root}")
    return sources, skipped


def find_guide_source(sdk_doc_root: Path, guide: dict[str, Any]) -> dict[str, Any]:
    matches = sorted(sdk_doc_root.rglob(guide["file_name"]))
    if len(matches) != 1:
        raise ReferenceError(
            f"{guide['id']}: expected one {guide['file_name']} below {sdk_doc_root}; "
            f"found {len(matches)}"
        )
    path = ensure_source_file(sdk_doc_root, matches[0], f"{guide['id']} guide")
    raw = path.read_bytes()
    return {
        "relative_path": path.relative_to(sdk_doc_root).as_posix(),
        "raw": raw,
        "raw_sha256": sha256_bytes(raw),
    }


class ArticleMarkdownParser(HTMLParser):
    """Convert an NVIDIA Sphinx article body to stable Markdown."""

    BLOCK_TAGS: ClassVar[set[str]] = {"p", "div", "section", "article", "dl", "dd"}
    VOID_TAGS: ClassVar[set[str]] = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
    SKIP_TAGS: ClassVar[set[str]] = {
        "script",
        "style",
        "svg",
        "form",
        "button",
        "noscript",
    }

    def __init__(
        self,
        article_class: str,
        link_resolver: Callable[[str], str | None],
        image_resolver: Callable[[str], str | None],
    ) -> None:
        super().__init__(convert_charrefs=True)
        self.article_class = article_class
        self.link_resolver = link_resolver
        self.image_resolver = image_resolver
        self.parts: list[str] = []
        self.article_depth = 0
        self.skip_depth = 0
        self.pre_depth = 0
        self.pre_parts: list[str] = []
        self.list_stack: list[str] = []
        self.li_depth = 0
        self.link_stack: list[str | None] = []
        self.found_article = False
        self.pending_space = False
        self.table_row_cells = 0
        self.table_row_is_header = False

    @property
    def active(self) -> bool:
        return self.article_depth > 0

    def emit(self, value: str) -> None:
        self.parts.append(value)

    def newline(self, count: int = 1) -> None:
        self.pending_space = False
        self.emit("\n" * count)

    def flush_space(self) -> None:
        if (
            self.pending_space
            and self.parts
            and not self.parts[-1].endswith((" ", "\n", "[", "(", "*", "`"))
        ):
            self.emit(" ")
        self.pending_space = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if not self.active:
            if tag == "article" and self.article_class in classes:
                self.article_depth = 1
                self.found_article = True
            return
        is_void = tag in self.VOID_TAGS
        if not is_void:
            self.article_depth += 1
        if self.skip_depth:
            if not is_void:
                self.skip_depth += 1
            return
        if tag in self.SKIP_TAGS or "headerlink" in classes or "copybutton" in classes:
            self.skip_depth = 1
            return
        if self.pre_depth:
            self.pre_depth += 1
            return
        if tag == "pre":
            self.pre_depth = 1
            self.pre_parts = []
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.newline(2)
            self.emit("#" * int(tag[1]) + " ")
        elif tag == "blockquote":
            self.newline(2)
            self.emit("> ")
        elif tag in self.BLOCK_TAGS and not (tag == "p" and self.li_depth):
            self.newline(2)
        elif tag == "br":
            self.newline()
        elif tag in {"ul", "ol"}:
            self.list_stack.append(tag)
            self.newline()
        elif tag == "li":
            self.newline()
            marker = "1. " if self.list_stack and self.list_stack[-1] == "ol" else "- "
            self.emit("  " * max(0, len(self.list_stack) - 1) + marker)
            self.li_depth += 1
        elif tag in {"strong", "b"}:
            self.flush_space()
            self.emit("**")
        elif tag in {"em", "i"}:
            self.flush_space()
            self.emit("*")
        elif tag == "code":
            self.flush_space()
            self.emit("`")
        elif tag == "a":
            href = attributes.get("href")
            resolved = self.link_resolver(href) if href else None
            self.link_stack.append(resolved)
            if resolved:
                self.flush_space()
                self.emit("[")
        elif tag == "img":
            source = attributes.get("src")
            resolved = self.image_resolver(source) if source else None
            if resolved:
                alt = (attributes.get("alt") or "image").replace("]", "\\]")
                self.emit(f"[Image: {alt}; packaged source: `{resolved}`]")
        elif tag == "tr":
            self.newline()
            self.table_row_cells = 0
            self.table_row_is_header = False
        elif tag in {"th", "td"}:
            self.emit("| ")
            self.table_row_cells += 1
            self.table_row_is_header = self.table_row_is_header or tag == "th"
        elif tag == "dt":
            self.newline(2)
            self.emit("**")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.VOID_TAGS or not self.active:
            return
        if self.skip_depth:
            self.skip_depth -= 1
            self.article_depth -= 1
            return
        if self.pre_depth:
            self.pre_depth -= 1
            if self.pre_depth == 0:
                code = "".join(self.pre_parts).strip("\n")
                fence = "````" if "```" in code else "```"
                self.newline(2)
                self.emit(f"{fence}\n{code}\n{fence}")
                self.newline(2)
            self.article_depth -= 1
            return
        if (
            tag == "blockquote"
            or tag in {"h1", "h2", "h3", "h4", "h5", "h6"}
            or tag in self.BLOCK_TAGS
            and not (tag == "p" and self.li_depth)
        ):
            self.newline(2)
        elif tag in {"ul", "ol"}:
            if self.list_stack:
                self.list_stack.pop()
            self.newline()
        elif tag == "li":
            self.li_depth = max(0, self.li_depth - 1)
        elif tag in {"strong", "b"}:
            self.emit("**")
        elif tag in {"em", "i"}:
            self.emit("*")
        elif tag == "code":
            self.emit("`")
        elif tag == "a":
            resolved = self.link_stack.pop() if self.link_stack else None
            if resolved:
                self.emit(f"]({resolved})")
        elif tag in {"th", "td"}:
            self.emit(" ")
        elif tag == "tr":
            self.emit("|")
            self.newline()
            if self.table_row_is_header and self.table_row_cells:
                self.emit("|" + " --- |" * self.table_row_cells)
                self.newline()
        elif tag == "dt":
            self.emit("**")
        self.article_depth -= 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in self.VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self.active or self.skip_depth:
            return
        if self.pre_depth:
            self.pre_parts.append(data)
            return
        value = re.sub(r"\s+", " ", data.replace("\xa0", " "))
        if not value.strip():
            self.pending_space = True
            return
        core = value.strip()
        if (value.startswith(" ") or self.pending_space) and not core.startswith(
            (".", ",", ":", ";", ")", "]")
        ):
            self.pending_space = True
            self.flush_space()
        self.emit(core)
        self.pending_space = value.endswith(" ")

    def markdown(self) -> str:
        if not self.found_article:
            raise ReferenceError("HTML page has no required Sphinx article element")
        value = "".join(self.parts)
        value = "\n".join(line.rstrip() for line in value.splitlines())
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip() + "\n"


def html_title(raw: bytes) -> str:
    match = re.search(rb"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    if not match:
        return "Untitled NVIDIA documentation page"
    value = re.sub(r"<[^>]+>", "", match.group(1).decode("utf-8", "replace"))
    value = html.unescape(re.sub(r"\s+", " ", value)).strip()
    return value.split(" — ", 1)[0].strip()


def internal_link_resolver(
    current_source: str,
    known_html: set[str],
) -> Callable[[str], str | None]:
    current = PurePosixPath(current_source)
    current_output = PurePosixPath("html") / current.with_suffix(".md")

    def resolve(href: str) -> str | None:
        parsed = urlsplit(href)
        if parsed.scheme:
            return href if parsed.scheme in {"https", "http", "mailto"} else None
        if parsed.netloc:
            return None
        if not parsed.path:
            return f"#{parsed.fragment}" if parsed.fragment else None
        decoded = unquote(parsed.path)
        if decoded.startswith("/"):
            target = PurePosixPath(decoded.lstrip("/"))
        else:
            target = PurePosixPath(posixpath.normpath((current.parent / decoded).as_posix()))
        if ".." in target.parts:
            return None
        target_text = target.as_posix()
        if target_text in known_html:
            target_output = PurePosixPath("html") / target.with_suffix(".md")
            relative = posixpath.relpath(target_output.as_posix(), current_output.parent.as_posix())
            result = quote(relative, safe="/.-_")
            if parsed.fragment:
                result += "#" + quote(parsed.fragment, safe="-._~")
            return result
        return None

    return resolve


def image_source_resolver(current_source: str) -> Callable[[str], str | None]:
    current = PurePosixPath(current_source)

    def resolve(source: str) -> str | None:
        parsed = urlsplit(source)
        if parsed.scheme:
            return source if parsed.scheme in {"https", "http"} else None
        if parsed.netloc:
            return None
        decoded = unquote(parsed.path)
        if decoded.startswith("/"):
            target = PurePosixPath(decoded.lstrip("/"))
        else:
            target = PurePosixPath(posixpath.normpath((current.parent / decoded).as_posix()))
        if ".." in target.parts:
            return None
        return f"{SOURCE_ROOT_LABEL}/{target.as_posix()}"

    return resolve


def html_to_markdown(
    raw: bytes,
    source_path: str,
    known_html: set[str],
    html_config: dict[str, Any],
) -> str:
    parser = ArticleMarkdownParser(
        html_config["article_class"],
        internal_link_resolver(source_path, known_html),
        image_source_resolver(source_path),
    )
    parser.feed(raw.decode("utf-8", "replace"))
    return parser.markdown()


def strip_first_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and re.fullmatch(r"#\s+.+", lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip() + "\n"


def render_html_page(
    source: dict[str, Any],
    known_html: set[str],
    html_config: dict[str, Any],
    sdk_version: str,
) -> str:
    title = html_title(source["raw"])
    body = strip_first_heading(
        html_to_markdown(
            source["raw"],
            source["relative_path"],
            known_html,
            html_config,
        )
    )
    return "\n".join(
        [
            "<!-- Generated by materialize_sdk_references.py. Do not edit. -->",
            "",
            f"# {title}",
            "",
            f"- SDK documentation root: `{SOURCE_ROOT_LABEL}`",
            f"- Packaged source: `{source['relative_path']}`",
            f"- Installed SDK documentation version: `{sdk_version}`",
            f"- Raw HTML SHA-256: `{source['raw_sha256']}`",
            f"- Converter version: `{CONVERTER_VERSION}`",
            "",
            body.rstrip(),
            "",
        ]
    )


def require_pdftotext() -> str:
    executable = shutil.which("pdftotext")
    if not executable:
        raise ReferenceError("pdftotext is required to convert the packaged Programmer's Guides")
    return executable


def pdftotext_identity(executable: str) -> str:
    process = subprocess.run([executable, "-v"], check=False, capture_output=True, text=True)
    output = (process.stderr or process.stdout).splitlines()
    return output[0].strip() if output else Path(executable).name


def extract_pdf_pages(raw: bytes, executable: str, guide_id: str) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
        source.write(raw)
        source.flush()
        process = subprocess.run(
            [executable, "-layout", "-enc", "UTF-8", source.name, "-"],
            check=False,
            capture_output=True,
        )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", "replace").strip()
        raise ReferenceError(f"pdftotext failed for {guide_id}: {detail}")
    pages = process.stdout.decode("utf-8", "replace").replace("\r\n", "\n").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages:
        raise ReferenceError(f"{guide_id}: pdftotext returned no pages")
    return pages


def first_nonempty_line(page: str) -> str:
    for line in page.replace("\x00", "").splitlines():
        if line.strip():
            return re.sub(r"\s+", " ", line).strip()
    return ""


def detect_chapter_ranges(pages: list[str], guide: dict[str, Any]) -> dict[int, tuple[int, int]]:
    starts: dict[int, int] = {}
    for page_number, page in enumerate(pages, start=1):
        match = re.match(r"Chapter\s+(\d+)\.", first_nonempty_line(page), re.IGNORECASE)
        if match:
            starts.setdefault(int(match.group(1)), page_number)
    expected = [chapter["chapter"] for chapter in guide["chapters"]]
    if sorted(starts) != expected:
        missing = sorted(set(expected) - set(starts))
        extra = sorted(set(starts) - set(expected))
        raise ReferenceError(
            f"{guide['id']}: could not detect the declared chapter set; "
            f"missing={missing} extra={extra}"
        )
    ordered_starts = [starts[number] for number in expected]
    if ordered_starts != sorted(ordered_starts) or len(set(ordered_starts)) != len(expected):
        raise ReferenceError(f"{guide['id']}: detected chapter starts are not ordered")
    document_end = len(pages)
    for page_number in range(ordered_starts[-1] + 1, len(pages) + 1):
        if first_nonempty_line(pages[page_number - 1]).lower() == "notice":
            document_end = page_number - 1
            break
    ranges: dict[int, tuple[int, int]] = {}
    for index, number in enumerate(expected):
        start = starts[number]
        end = ordered_starts[index + 1] - 1 if index + 1 < len(expected) else document_end
        if end < start:
            raise ReferenceError(f"{guide['id']}: invalid detected range for chapter {number}")
        ranges[number] = (start, end)
    return ranges


def render_guide_chapter(
    guide: dict[str, Any],
    chapter: dict[str, Any],
    pages: list[str],
    page_range: tuple[int, int],
    source: dict[str, Any],
    sdk_version: str,
    pdf_extractor: str,
) -> str:
    start, end = page_range
    lines = [
        "<!-- Generated by materialize_sdk_references.py. Do not edit. -->",
        "",
        f"# Chapter {chapter['chapter']}. {chapter['title']}",
        "",
        f"- SDK documentation root: `{SOURCE_ROOT_LABEL}`",
        f"- Packaged source: `{source['relative_path']}`",
        f"- Document: `{guide['document_id']}`",
        f"- Installed SDK documentation version: `{sdk_version}`",
        f"- Raw PDF SHA-256: `{source['raw_sha256']}`",
        f"- PDF pages: {start}-{end}",
        f"- PDF extractor: `{pdf_extractor}`",
        f"- Converter version: `{CONVERTER_VERSION}`",
        "",
        "> The official packaged PDF is authoritative. Check it when a figure, table, or equation is unclear.",
        "",
    ]
    for page_number in range(start, end + 1):
        page = "\n".join(
            line.rstrip()
            for line in pages[page_number - 1].replace("\x00", "").strip("\n").splitlines()
        )
        lines.extend(
            [
                f"## PDF page {page_number}",
                "",
                "````text",
                page,
                "````",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(path: Path, value: str, source_paths: list[str]) -> dict[str, Any]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith(MARKDOWN_COPYRIGHT_HEADER):
        normalized = f"{MARKDOWN_COPYRIGHT_HEADER}\n{normalized}"
    data = normalized.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"sha256": sha256_bytes(data), "source_paths": source_paths}


def route_matches(pattern: str, relative: str) -> bool:
    # Only the HTML extension is case-insensitive; preserve directory/base-name case.
    if PurePosixPath(pattern).suffix.lower() == ".html":
        pattern = PurePosixPath(pattern).with_suffix(".html").as_posix()
    if PurePosixPath(relative).suffix.lower() == ".html":
        relative = PurePosixPath(relative).with_suffix(".html").as_posix()
    return fnmatch.fnmatchcase(relative, pattern) or PurePosixPath(relative).match(pattern)


def render_index(
    manifest: dict[str, Any],
    manifest_hash: str,
    sdk_version: str,
    html_sources: list[dict[str, Any]],
    guide_records: list[dict[str, Any]],
    skipped_html: list[str],
    pdf_extractor: str,
) -> str:
    html_by_path = {source["relative_path"]: source for source in html_sources}
    lines = [
        "<!-- Generated by materialize_sdk_references.py. Do not edit. -->",
        "",
        "# Local NVIDIA PVA SDK reference index",
        "",
        f"- SDK documentation root: `{SOURCE_ROOT_LABEL}`",
        f"- Installed SDK documentation version: `{sdk_version}`",
        f"- Manifest SHA-256: `{manifest_hash}`",
        f"- Converter version: `{CONVERTER_VERSION}`",
        f"- PDF extractor: `{pdf_extractor}`",
        f"- HTML reference pages: `{len(html_sources)}`",
        f"- Programmer's Guide chapters: `{sum(len(item['chapters']) for item in guide_records)}`",
        "",
        "Use `rg` before you open a large API page or guide chapter. The generated corpus is the local reference source for an operator run.",
        "",
        "## Skill routes",
        "",
    ]
    for skill in sorted(manifest["skill_routes"]):
        route = manifest["skill_routes"][skill]
        matched = sorted(
            relative
            for relative in html_by_path
            if any(route_matches(pattern, relative) for pattern in route["html_patterns"])
        )
        lines.extend([f"### {skill}", "", "HTML path patterns:", ""])
        for pattern in route["html_patterns"]:
            lines.append(f"- `{pattern}`")
        if not route["html_patterns"]:
            lines.append("- None")
        lines.extend(["", f"Matched HTML pages: `{len(matched)}`", ""])
        lines.append("Suggested `rg` terms:")
        lines.append("")
        for term in route["search_terms"]:
            lines.append(f"- `{term}`")
        if not route["search_terms"]:
            lines.append("- None")
        lines.extend(["", "Programmer's Guide chapters:", ""])
        for record in guide_records:
            selected = [
                chapter
                for chapter in record["chapters"]
                if chapter["chapter"] in route["guide_chapters"]
            ]
            if selected:
                lines.append(f"- {record['title']}:")
                for chapter in selected:
                    lines.append(
                        f"  - [{chapter['chapter']}. {chapter['title']}]"
                        f"(guides/{record['id']}/{chapter['file']})"
                    )
        if not route["guide_chapters"]:
            lines.append("- None")
        lines.append("")

    lines.extend(["## HTML references", ""])
    groups: dict[str, list[dict[str, Any]]] = {}
    for source in html_sources:
        group = (
            source["relative_path"].split("/", 1)[0] if "/" in source["relative_path"] else "root"
        )
        groups.setdefault(group, []).append(source)
    for group in sorted(groups):
        lines.extend([f"### {group}", ""])
        for source in groups[group]:
            title = html_title(source["raw"])
            target = quote(source["output_path"], safe="/.-_")
            path_label = f"`{source['relative_path']}`"
            lines.append(f"- [{title}]({target}) — {path_label}")
        lines.append("")

    lines.extend(["## Programmer's Guides", ""])
    for record in guide_records:
        lines.extend(
            [
                f"### {record['title']}",
                "",
                f"- SDK documentation root: `{SOURCE_ROOT_LABEL}`",
                f"- Packaged source: `{record['relative_path']}`",
                f"- Raw PDF SHA-256: `{record['raw_sha256']}`",
                "",
            ]
        )
        for chapter in record["chapters"]:
            start, end = chapter["pdf_pages"]
            lines.append(
                f"- [{chapter['chapter']}. {chapter['title']}]"
                f"(guides/{record['id']}/{chapter['file']}) — PDF pages {start}-{end}"
            )
        lines.append("")

    if skipped_html:
        lines.extend(
            [
                "## Skipped non-reference HTML",
                "",
                "These packaged HTML files do not contain the configured Sphinx article element or match an explicit exclusion:",
                "",
            ]
        )
        lines.extend(f"- `{path}`" for path in skipped_html)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def safe_replace_directory(staging: Path, destination: Path) -> None:
    destination = destination.resolve()
    if destination == Path(destination.anchor) or destination == destination.parent:
        raise ReferenceError(f"unsafe output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
    if backup.exists():
        raise ReferenceError(f"stale reference backup exists: {backup}")
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def source_inventory(
    sdk_doc_root: Path, manifest: dict[str, Any]
) -> tuple[str, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    sdk_version = read_sdk_version(sdk_doc_root)
    html_sources, skipped = discover_html_sources(sdk_doc_root, manifest["html"])
    guide_sources = []
    for guide in manifest["guides"]:
        source = find_guide_source(sdk_doc_root, guide)
        source["id"] = guide["id"]
        guide_sources.append(source)
    return sdk_version, html_sources, skipped, guide_sources


def sync(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    sdk_doc_root = args.sdk_doc_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(sdk_doc_root) or sdk_doc_root.is_relative_to(output_dir):
        raise ReferenceError(
            f"unsafe output directory: {output_dir} overlaps SDK documentation: {sdk_doc_root}"
        )
    if manifest_path.is_relative_to(output_dir):
        raise ReferenceError(
            f"unsafe output directory: {output_dir} would replace input manifest: {manifest_path}"
        )
    skill_root = Path(__file__).resolve().parents[1]
    if output_dir.is_relative_to(skill_root) or skill_root.is_relative_to(output_dir):
        raise ReferenceError(
            f"unsafe output directory: {output_dir} overlaps installed Doctor skill: {skill_root}"
        )
    for parent in (output_dir, *output_dir.parents):
        if (parent / "SKILL.md").is_file():
            raise ReferenceError(
                f"unsafe output directory: {output_dir} is inside an installed skill: {parent}"
            )
    if not sdk_doc_root.is_dir():
        raise ReferenceError(f"SDK documentation root is not a directory: {sdk_doc_root}")
    manifest, manifest_hash = load_manifest(manifest_path)
    sdk_version, html_sources, skipped, guide_sources = source_inventory(sdk_doc_root, manifest)
    pdftotext = require_pdftotext()
    extractor = pdftotext_identity(pdftotext)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    outputs: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    guide_records: list[dict[str, Any]] = []
    try:
        known_html = {source["relative_path"] for source in html_sources}
        for source in html_sources:
            rendered = render_html_page(source, known_html, manifest["html"], sdk_version)
            result = write_markdown(
                staging / source["output_path"], rendered, [source["relative_path"]]
            )
            result["path"] = source["output_path"]
            outputs.append(result)
            source_records.append(
                {
                    "media_type": "html",
                    "path": source["relative_path"],
                    "output_path": source["output_path"],
                    "raw_sha256": source["raw_sha256"],
                }
            )

        guides_by_id = {guide["id"]: guide for guide in manifest["guides"]}
        for source in guide_sources:
            guide = guides_by_id[source["id"]]
            pages = extract_pdf_pages(source["raw"], pdftotext, guide["id"])
            ranges = detect_chapter_ranges(pages, guide)
            chapters = []
            for chapter in guide["chapters"]:
                page_range = ranges[chapter["chapter"]]
                rendered = render_guide_chapter(
                    guide,
                    chapter,
                    pages,
                    page_range,
                    source,
                    sdk_version,
                    extractor,
                )
                relative = f"guides/{guide['id']}/{chapter['file']}"
                result = write_markdown(staging / relative, rendered, [source["relative_path"]])
                result["path"] = relative
                outputs.append(result)
                chapters.append({**chapter, "pdf_pages": list(page_range)})
            source_records.append(
                {
                    "media_type": "pdf",
                    "path": source["relative_path"],
                    "raw_sha256": source["raw_sha256"],
                    "pdf_pages": len(pages),
                    "chapter_ranges": {
                        str(number): list(page_range) for number, page_range in ranges.items()
                    },
                }
            )
            guide_records.append(
                {
                    "id": guide["id"],
                    "title": guide["title"],
                    "relative_path": source["relative_path"],
                    "raw_sha256": source["raw_sha256"],
                    "chapters": chapters,
                }
            )

        index = render_index(
            manifest,
            manifest_hash,
            sdk_version,
            html_sources,
            guide_records,
            skipped,
            extractor,
        )
        result = write_markdown(staging / INDEX_NAME, index, [])
        result["path"] = INDEX_NAME
        outputs.append(result)
        outputs.sort(key=lambda item: item["path"])
        source_records.sort(key=lambda item: item["path"])
        lock = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "converter_version": CONVERTER_VERSION,
            "manifest_sha256": manifest_hash,
            "sdk_docs_version": sdk_version,
            "pdf_extractor": extractor,
            "skipped_html": skipped,
            "sources": source_records,
            "outputs": outputs,
        }
        (staging / LOCK_NAME).write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        safe_replace_directory(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        f"SYNCED {len(html_sources)} HTML pages and "
        f"{sum(len(guide['chapters']) for guide in manifest['guides'])} guide chapters "
        f"into {len(outputs)} Markdown files at {output_dir}"
    )


def current_source_records(
    html_sources: list[dict[str, Any]], guide_sources: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    records = {
        source["relative_path"]: {
            "media_type": "html",
            "raw_sha256": source["raw_sha256"],
            "output_path": source["output_path"],
        }
        for source in html_sources
    }
    for source in guide_sources:
        records[source["relative_path"]] = {
            "media_type": "pdf",
            "raw_sha256": source["raw_sha256"],
        }
    return records


def verify(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    sdk_doc_root = args.sdk_doc_root.resolve()
    output_dir = args.output_dir.resolve()
    if not sdk_doc_root.is_dir():
        raise ReferenceError(f"SDK documentation root is not a directory: {sdk_doc_root}")
    manifest, manifest_hash = load_manifest(manifest_path)
    sdk_version, html_sources, skipped, guide_sources = source_inventory(sdk_doc_root, manifest)
    lock_path = output_dir / LOCK_NAME
    if not lock_path.is_file():
        raise ReferenceError(f"reference lock is missing: {lock_path}")
    lock = load_json(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ReferenceError("reference lock schema is stale; regenerate with sync, then verify")
    if lock.get("converter_version") != CONVERTER_VERSION:
        raise ReferenceError(
            "reference corpus was made by a different converter version; "
            "regenerate with sync, then verify"
        )
    if lock.get("manifest_sha256") != manifest_hash:
        raise ReferenceError("reference corpus was not made from the current manifest")
    if lock.get("sdk_docs_version") != sdk_version:
        raise ReferenceError("reference corpus was made from a different SDK documentation version")
    if lock.get("skipped_html") != skipped:
        raise ReferenceError("packaged HTML inventory changed")

    sources = lock.get("sources")
    if not isinstance(sources, list):
        raise ReferenceError("reference lock sources must be an array")
    locked_sources: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ReferenceError("reference lock contains an invalid source record")
        locked_sources[source["path"]] = source
    current_sources = current_source_records(html_sources, guide_sources)
    if set(locked_sources) != set(current_sources):
        raise ReferenceError("packaged documentation source set changed")
    for path, current in current_sources.items():
        locked = locked_sources[path]
        for field, value in current.items():
            if locked.get(field) != value:
                raise ReferenceError(f"packaged documentation source changed: {path}")

    outputs = lock.get("outputs")
    if not isinstance(outputs, list):
        raise ReferenceError("reference lock outputs must be an array")
    expected_paths = {INDEX_NAME}
    expected_paths.update(source["output_path"] for source in html_sources)
    for guide in manifest["guides"]:
        expected_paths.update(
            f"guides/{guide['id']}/{chapter['file']}" for chapter in guide["chapters"]
        )
    locked_paths = {output.get("path") for output in outputs if isinstance(output, dict)}
    if locked_paths != expected_paths:
        raise ReferenceError("reference lock output set does not match the packaged sources")

    listed: set[str] = set()
    for output in outputs:
        relative = output.get("path")
        expected_hash = output.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ReferenceError("reference lock contains an invalid output record")
        path = (output_dir / relative).resolve()
        if path.suffix != ".md" or output_dir != path.parent and output_dir not in path.parents:
            raise ReferenceError(f"unsafe generated path in lock: {relative}")
        if not path.is_file():
            raise ReferenceError(f"generated Markdown is missing: {path}")
        if sha256_bytes(path.read_bytes()) != expected_hash:
            raise ReferenceError(f"generated Markdown hash mismatch: {path}")
        listed.add(path.as_posix())

    actual_markdown = {
        path.resolve().as_posix() for path in output_dir.rglob("*.md") if path.is_file()
    }
    untracked = sorted(actual_markdown - listed)
    if untracked:
        raise ReferenceError(f"untracked generated Markdown exists: {untracked[0]}")
    print(
        f"VERIFIED {len(html_sources)} HTML pages, "
        f"{len(manifest['guides'])} guides, and {len(outputs)} Markdown files in {output_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("sync", "Convert local SDK files and replace the generated corpus; requires pdftotext."),
        ("verify", "Check the existing corpus and source hashes without writing; Python only."),
    ):
        command = subparsers.add_parser(name, help=description, description=description)
        command.add_argument(
            "--manifest",
            type=Path,
            required=True,
            help="Path to Doctor's bundled references/sdk-doc-sources.json.",
        )
        command.add_argument(
            "--sdk-doc-root",
            type=Path,
            required=True,
            help="Installed SDK share/doc directory containing VERSION, HTML, and guide PDFs.",
        )
        command.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Workspace Markdown directory to replace (sync) or check (verify).",
        )
        command.set_defaults(func=sync if name == "sync" else verify)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ReferenceError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
