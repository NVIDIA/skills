# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain explicit run files and verify an archive before restoring it."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Expected a nonempty relative POSIX path')
    path = PurePosixPath(name)
    # A drive-qualified name (C:/x) is relative to POSIX but absolute on Windows.
    if (path.is_absolute() or PureWindowsPath(name).drive
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe relative path: {name!r}')
    if name == 'retention-manifest.json':
        raise ValueError('Reserved manifest name')
    return path


def create_archive(root, relative_files, archive):
    """Write a new archive containing only explicit files and return its SHA256."""
    root = Path(root).resolve(strict=True)
    archive = Path(archive)
    names = list(relative_files)
    if not names or len(names) != len(set(names)):
        raise ValueError('Select at least one file, without duplicates')
    entries = []
    for name in names:
        relative = _relative(name)
        source = root.joinpath(*relative.parts)
        if any(root.joinpath(*relative.parts[:i]).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            raise ValueError(f'Symlink not permitted: {name}')
        source.resolve(strict=True).relative_to(root)
        if not source.is_file():
            raise ValueError(f'Not a regular file: {name}')
        if source.resolve() == archive.resolve():
            raise ValueError('Archive cannot contain itself')
        entries.append((name, source))
    created = False
    try:
        with archive.open('xb') as handle:
            created = True
            manifest = []
            with zipfile.ZipFile(handle, 'w', zipfile.ZIP_DEFLATED) as bundle:
                for name, source in entries:
                    # Hash the exact bytes streamed into the archive.
                    digest = hashlib.sha256()
                    size = 0
                    with source.open('rb') as inp, bundle.open(name, 'w', force_zip64=True) as out:
                        for block in iter(lambda: inp.read(1024 * 1024), b''):
                            out.write(block)
                            digest.update(block)
                            size += len(block)
                    manifest.append(dict(path=name, size=size, sha256=digest.hexdigest()))
                bundle.writestr('retention-manifest.json', json.dumps(manifest, sort_keys=True))
        return sha256(archive)
    except BaseException:
        if created:
            archive.unlink(missing_ok=True)
        raise


def restore_archive(archive, destination, expected_sha256, *, max_bytes=10 * 1024**3):
    """Verify checksums and restore to a new directory within a storage budget."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    if sha256(archive) != expected_sha256:
        raise ValueError('Archive SHA256 mismatch')
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        if sum(item.file_size for item in infos) > max_bytes:
            raise ValueError('Archive exceeds restore storage budget')
        if 'retention-manifest.json' not in names:
            raise ValueError('Missing retention manifest')
        if bundle.getinfo('retention-manifest.json').file_size > 10 * 1024**2:
            raise ValueError('Manifest exceeds size limit')
        manifest = json.loads(bundle.read('retention-manifest.json'))
        if not isinstance(manifest, list) or not manifest:
            raise ValueError('Manifest must be a nonempty list')
        expected = {}
        for entry in manifest:
            if not isinstance(entry, dict) or set(entry) != {'path', 'size', 'sha256'}:
                raise ValueError('Malformed manifest entry')
            name = entry['path']
            _relative(name)
            if name in expected or type(entry['size']) is not int or entry['size'] < 0:
                raise ValueError('Duplicate path or invalid size')
            digest = entry['sha256']
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid file checksum')
            expected[name] = entry
        if set(names) != set(expected) | {'retention-manifest.json'}:
            raise ValueError('Unexpected or missing archive members')
        # Reject file/parent collisions before writing anything.
        for name in expected:
            if any(str(parent) in expected for parent in PurePosixPath(name).parents):
                raise ValueError('Archive file/parent collision')
            if bundle.getinfo(name).file_size != expected[name]['size']:
                raise ValueError(f'Size mismatch: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.restore-') as tmp:
            staging = Path(tmp) / 'files'
            staging.mkdir()
            for name, entry in expected.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                if sha256(target) != entry['sha256']:
                    raise ValueError(f'File checksum mismatch: {name}')
            # Exclusive creation preserves existing destinations, including empty ones.
            destination.mkdir()
            try:
                for child in staging.iterdir():
                    shutil.move(str(child), str(destination / child.name))
            except BaseException:
                shutil.rmtree(destination)
                raise
    return destination
