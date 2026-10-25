"""Pinned LNCTVSeg assets and per-task technical acceptance checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _stream_sha256(stream):
    # hashlib.file_digest needs Python 3.11; the portal image runs 3.10.
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


TASK = 'lnctvseg_nnunet_v1'
INTEGRATED_TASKS = (TASK,)
APP = Path(__file__).resolve().parent
MANIFEST_PATH = APP / 'models/manifests/lnctvseg.json'


def manifest():
    return json.loads(MANIFEST_PATH.read_text())


def manifest_sha256():
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()


def validate_label_contract():
    """Export the six published IDs unchanged; explicitly omit only class 7."""
    spec = manifest()
    contract = spec.get('label_contract', {})
    label_map = (APP / 'labels/lnctvseg.json').read_bytes()
    published = set(map(int, json.loads(label_map)))
    checkpoint = spec.get('nnunet') or {}
    expected = {
        'status': 'resolved_with_omission',
        'published_foreground_ids': list(range(1, 7)),
        'checkpoint_foreground_ids': list(range(1, 8)),
        'export_label_ids': list(range(1, 7)),
        'unmapped_checkpoint_ids': [7],
        'omitted_checkpoint_ids': [7],
        'policy': 'omit_without_relabelling',
    }
    if (any(contract.get(key) != value for key, value in expected.items())
            or published != set(range(1, 7))
            or hashlib.sha256(label_map).hexdigest() != spec.get('label_map_sha256')
            or checkpoint.get('output_label_ids') != list(range(8))
            or checkpoint.get('num_output_channels') != 8):
        raise ValueError('LNCTVSeg label contract invalid: export published IDs 1-6 unchanged '
                         'and omit undocumented class 7 with a per-job volume note')


def prepare_export(prediction, output, notes_directory):
    """Keep the raw prediction and write a six-label copy plus durable job notes."""
    import nibabel as nib
    import numpy as np

    validate_label_contract()
    if Path(prediction).resolve() == Path(output).resolve():
        raise ValueError('LNCTVSeg export must not overwrite the raw prediction')
    image = nib.load(prediction)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not set(np.unique(data)).issubset(set(range(8))):
        raise ValueError('LNCTVSeg prediction must be 3-D with integer label IDs 0-7')
    voxel_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000
    if not np.isfinite(voxel_ml) or voxel_ml <= 0:
        raise ValueError('LNCTVSeg prediction has invalid voxel volume')
    omitted = data == 7
    voxels = int(np.count_nonzero(omitted))
    volume_ml = voxels * voxel_ml
    note = ('LNCTVSeg class 7 (undocumented by the authors) omitted: '
            f'{voxels} voxels, {volume_ml:.4f} mL')
    exported = data.astype(np.uint8, copy=True)
    exported[omitted] = 0
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    result = image.__class__(exported, image.affine, header)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, output)
    evidence = {'task': TASK, 'manifest_sha256': manifest_sha256(),
                'export_label_ids': list(range(1, 7)), 'omitted_label_ids': [7],
                'class_7_voxels': voxels, 'class_7_ml': volume_ml, 'notes': [note]}
    notes_directory = Path(notes_directory)
    notes_directory.mkdir(parents=True, exist_ok=True)
    (notes_directory / 'conversion.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (notes_directory / 'OMITTED_LABELS.txt').write_text(note + '\n')
    print(note, flush=True)
    return evidence


def validate_acceptance(value):
    validate_label_contract()
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
            digest = _stream_sha256(stream)
        if digest != expected['sha256']:
            raise ValueError(f'LNCTVSeg asset checksum mismatch: {name}')
    return spec


def orientation_ready(path):
    try:
        validate_label_contract()
        value = json.loads(Path(path).read_text())
        return isinstance(value, dict) and all(value.get(key) == expected for key, expected in {
            'task': TASK, 'result': 'passed', 'laterality': 'passed',
            'manifest_sha256': manifest_sha256(),
        }.items())
    except (OSError, ValueError, TypeError):
        return False
