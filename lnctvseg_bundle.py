"""Pinned LNCTVSeg assets and per-task technical acceptance checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TASK = 'lnctvseg_nnunet_v1'
APP = Path(__file__).resolve().parent
MANIFEST_PATH = APP / 'models/manifests/lnctvseg.json'


def manifest():
    return json.loads(MANIFEST_PATH.read_text())


def manifest_sha256():
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()


def validate_acceptance(value):
    if not isinstance(value, dict) or any(value.get(key) != expected for key, expected in {
        'task': TASK, 'result': 'passed', 'manifest_sha256': manifest_sha256(),
        'source_commit': manifest()['source_commit'], 'laterality': 'passed',
    }.items()):
        raise ValueError('LNCTVSeg acceptance is absent, stale or inconsistent')


def validate_assets(root):
    """Hash the actual release before loading its trusted publisher pickle files."""
    spec = manifest()
    if not spec.get('nnunet') or not spec.get('files'):
        raise ValueError('LNCTVSeg release assets have not been pinned')
    if spec.get('label_map_sha256') and hashlib.sha256((APP / 'labels/lnctvseg.json').read_bytes()).hexdigest() != spec['label_map_sha256']:
        raise ValueError('LNCTVSeg label-map checksum mismatch')
    for name, expected in spec['files'].items():
        path = Path(root) / name
        if path.stat().st_size != expected['bytes']:
            raise ValueError(f'LNCTVSeg asset size mismatch: {name}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f'LNCTVSeg asset checksum mismatch: {name}')
    return spec


def orientation_ready(path):
    try:
        value = json.loads(Path(path).read_text())
        return isinstance(value, dict) and all(value.get(key) == expected for key, expected in {
            'task': TASK, 'result': 'passed', 'laterality': 'passed',
            'manifest_sha256': manifest_sha256(),
        }.items())
    except (OSError, ValueError, TypeError):
        return False
