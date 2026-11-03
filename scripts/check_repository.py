#!/usr/bin/env python3
"""Verify portable publication inputs without installing packages or loading models.

These unsigned checks authenticate nothing by themselves. Pin a trusted Git
commit first. Historical model results have their own benchmark verifier.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(base, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe manifest path')
    path = base.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink(), f'Missing or linked file: {name}')
    require(path.resolve().is_relative_to(base.resolve()), f'Escaping file: {name}')
    return path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def verify_adapter():
    base = ROOT / 'weights/coding-4b'
    manifest = load(base / 'manifest.json')
    require(manifest['schema'] == 'ruvllm.adapter-publication.v1', 'Unknown adapter schema')
    require(manifest['sota_verified'] is False and manifest['production_qualified'] is False,
            'Adapter qualification claim changed')
    expected = set(manifest['files']) | {'README.md', 'manifest.json'}
    actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
    require(actual == expected, 'Adapter inventory differs')
    for name, receipt in manifest['files'].items():
        path = safe_file(base, name)
        require(path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256'],
                f'Adapter file differs: {name}')
    for name in ('adapter_model.safetensors', 'mlx/adapters.safetensors'):
        path = base / name
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            require(0 < length <= 1024 * 1024, 'Unexpected tensor header length')
            tensors = json.loads(stream.read(length))
        tensors.pop('__metadata__', None)
        require(len(tensors) == 32, f'Unexpected tensor count: {name}')
        count = sum(math.prod(tensor['shape']) for tensor in tensors.values())
        require(count == manifest['parameters'] == 1507328, f'Parameter count differs: {name}')
    return {'files': len(manifest['files']), 'tensors_per_format': 32, 'parameters': 1507328}


def verify_source_export():
    manifest = load(ROOT / 'provenance/python-and-research-source.json')
    amendments = load(ROOT / 'provenance/publication-amendments.json')
    require(amendments['source_manifest_sha256'] == sha(ROOT / 'provenance/python-and-research-source.json'),
            'Original source manifest differs')
    changes = {row['path']: row for row in amendments['changes']}
    require(len(changes) == len(amendments['changes']), 'Duplicate source amendment')
    seen = set()
    for entry in manifest['files']:
        name = entry['path']
        require(name not in seen, f'Duplicate source path: {name}')
        seen.add(name)
        expected = entry['source_sha256']
        if name in changes:
            change = changes[name]
            require(change['source_sha256'] == expected and bool(change['reason']), f'Invalid amendment: {name}')
            expected = change['publication_sha256']
        require(sha(safe_file(ROOT, name)) == expected, f'Source export differs: {name}')
    require(set(changes) <= seen, 'Amendment does not refer to exported source')
    return {'files': len(seen), 'documented_amendments': len(changes)}


def verify_svg():
    assets = sorted((ROOT / 'docs/assets').glob('*.svg'))
    require(len(assets) == 3, 'Expected three publication diagrams')
    for path in assets:
        text = path.read_text()
        require('<!DOCTYPE' not in text.upper() and '<!ENTITY' not in text.upper(), 'SVG declarations forbidden')
        root = ET.fromstring(text)
        tags = {node.tag.rsplit('}', 1)[-1] for node in root.iter()}
        require({'svg', 'title', 'desc', 'style'} <= tags, f'Missing accessible SVG content: {path.name}')
        require(not tags.intersection({'script', 'foreignObject', 'image'}), f'Active or external SVG: {path.name}')
        require('prefers-reduced-motion' in text and '@keyframes' in text, f'Motion policy missing: {path.name}')
        for node in root.iter():
            for key, value in node.attrib.items():
                name = key.rsplit('}', 1)[-1].lower()
                require(not name.startswith('on'), 'SVG event handlers forbidden')
                if name == 'href':
                    require(value.startswith('#'), 'External SVG reference forbidden')
        for url in re.findall(r'url\(([^)]+)\)', text):
            require(url.strip(' \"\'').startswith('#'), 'External CSS reference forbidden')
    return len(assets)


def verify_links():
    documents = [ROOT / 'README.md', ROOT / 'SECURITY.md', ROOT / 'native/README.md',
                 ROOT / 'weights/coding-4b/README.md', ROOT / 'benchmarks/README.md',
                 ROOT / 'plugins/ruflo-neural-evolution/README.md']
    documents.extend((ROOT / 'docs').rglob('*.md'))
    checked = 0
    for document in documents:
        text = document.read_text()
        links = re.findall(r'\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)', text)
        links += re.findall(r'<img\b[^>]*\bsrc=[\"\']([^\"\']+)', text)
        for link in links:
            target = urlsplit(link.strip('<>'))
            if target.scheme or target.netloc or not target.path:
                continue
            path = (document.parent / unquote(target.path)).resolve()
            require(path.is_relative_to(ROOT) and path.exists(),
                    f'Broken local link in {document.relative_to(ROOT)}: {link}')
            checked += 1
    return checked


def main():
    try:
        report = {'adapter': verify_adapter(), 'source_export': verify_source_export(),
                  'animated_svgs': verify_svg(), 'local_links': verify_links(),
                  'scope': 'Bundled integrity and portable documentation; no inference or source authentication',
                  'verified': True}
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f'Publication verification failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
