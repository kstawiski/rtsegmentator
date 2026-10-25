"""BiomedParse v2 prompt, artifact and acceptance contracts (no model imports)."""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from prostate_models import file_sha256, installed_source_commit, marker_has_failure

TASK = 'biomedparse_prompted'
COMMIT = 'e02096c03af0d79c6994ffc2d60a49eeb0361e1f'
CHECKPOINT = Path('weights/biomedparse/hf-main-20261025/biomedparse_v2.ckpt')
CHECKPOINT_SHA256 = '6ea0bee43b983a7490e0b48689add96440c4405ca4b50b9f2fe21d5d03fd9440'
SOURCE = Path('sources/BiomedParse')
ENV = Path('runtime-envs/biomedparse-v2')
TOKENIZER = Path('weights/biomedparse/clip-vit-base-patch32')
TOKENIZER_FILES = ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json')
WINDOWS = {'soft_tissue': (400, 40), 'lung': (1500, -160), 'brain': (80, 40), 'bone': (1800, 400)}
SEED = 20261025
CHECKS = ['upstream_reference', 'exact_mask', 'exact_existence', 'repeatability', 'dicom_dispatch',
          'rtstruct_roundtrip', 'source_geometry', 'public_origin', 'same_portal_mask']
APP_FILES = ('biomedparse.py', 'run_biomedparse.py', 'acceptance_biomedparse.py',
             'catalog/biomedparse-prompts.json', 'catalog/biomedparse-source.json', 'catalog/biomedparse-tokenizer.json',
             'run_task.py', 'dicom_series_to_nifti.py', 'convert_validate_rtstruct.py',
             'orientation_contract.py', 'rtstruct_geometry.py', 'rtstruct_roundtrip.py')


def vocabulary():
    return json.loads((Path(__file__).parent / 'catalog/biomedparse-prompts.json').read_text())


def validate_prompt(text, modality, ct_window=None):
    if modality not in {'CT', 'MR', 'PT'}:
        raise ValueError('BiomedParse v2 requires one CT, MR or PET (PT) series')
    if not isinstance(text, str) or not text.strip() or len(text) > 512 or any(not c.isprintable() for c in text):
        raise ValueError('BiomedParse target must be 1–512 printable characters')
    key = ' '.join(text.split()).casefold()
    for row in vocabulary()['targets'][modality]:
        if key in [s.casefold() for s in [row['target'], *row.get('aliases', [])]]:
            window = row.get('ct_window', 'soft_tissue') if ct_window is None else ct_window
            if modality == 'CT' and (not isinstance(window, str) or window not in WINDOWS):
                raise ValueError('CT window must be soft_tissue, lung, brain or bone')
            # Preserve an upstream example sentence when the user selected it.
            prompt = next((s for s in row.get('aliases', []) if s.casefold() == key), row['target'])
            return {'text': prompt, 'target': row['target'], 'modality': modality,
                    'ct_window': window if modality == 'CT' else None}
    choices = ', '.join(r['target'] for r in vocabulary()['targets'][modality])
    raise ValueError(f'Unsupported BiomedParse {modality} target: {text!r}. Supported targets: {choices}')


def roi_name(prompt):
    name = re.sub('[^A-Za-z0-9_]+', '_', prompt).strip('_')
    return (name or 'Prompted_Target')[:64]


def artifact_names(task):
    return [f'{TASK}_RTSTRUCT.dcm.biomedparse.json'] if task == TASK else []


def result_record(prompt, scores, voxel_count, *, slice_axis, preprocessing):
    scores = [float(s) for s in scores]
    if not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Invalid per-slice existence scores')
    decisions = [s > 0.5 for s in scores]
    detected = any(decisions)
    if voxel_count < 0 or (not detected and voxel_count):
        raise ValueError('Existence-rejected slices cannot contain foreground')
    return {'task': TASK, 'prompt': prompt['text'], 'target': prompt['target'],
            'roi_name': roi_name(prompt['text']), 'modality': prompt['modality'],
            'existence': {'scores_per_slice': scores, 'decisions_per_slice': decisions,
                          'score': max(scores), 'decision': detected, 'threshold': 0.5,
                          'comparison': '>', 'slice_axis_in_zyx': int(slice_axis)},
            'voxel_count': int(voxel_count), 'empty': voxel_count == 0,
            'message': ('object not detected' if not detected else
                        'empty segmentation despite positive slice existence' if voxel_count == 0 else 'object detected'),
            'preprocessing': preprocessing, 'seed': SEED,
            'source_commit': COMMIT, 'checkpoint_sha256': CHECKPOINT_SHA256}


def require_device(gpu):
    capability = tuple(gpu.get('compute_capability', []))
    if capability not in {(6, 1), (9, 0)}:
        raise ValueError('BiomedParse acceptance supports only sm61 and sm90')
    if 'compiled_arches' in gpu:
        arches = gpu['compiled_arches'] or ''
        compatible = ('sm_60', 'sm_61') if capability == (6, 1) else ('sm_90',)
        if not any(arch in arches.split() for arch in compatible):
            raise ValueError('Installed PyTorch lacks a compatible compiled CUDA architecture')
    if gpu.get('total_memory', 0) < 23 * 1024**3:
        raise ValueError('Acceptance is scoped to the target hosts with 24 GB or more; peak model memory remains unmeasured')
    return 'sm' + ''.join(map(str, capability))


def probe_device(python):
    code = """import json, torch, hydra, detectron2, ftfy
if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
p=torch.cuda.get_device_properties(0)
print(json.dumps({'compute_capability':list(torch.cuda.get_device_capability(0)),
'name':p.name,'total_memory':p.total_memory,'torch':torch.__version__,
'compiled_arches':torch._C._cuda_getArchFlags()}))
"""
    cp = subprocess.run([str(python), '-c', code], capture_output=True, text=True, check=True, timeout=45)
    gpu = json.loads(cp.stdout.splitlines()[-1])
    require_device(gpu)
    return gpu


def marker_path(root, gpu, folder='validation'):
    return Path(root) / folder / f'{TASK}-{require_device(gpu)}.json'


def requirements():
    return ['@' + p for p in APP_FILES] + [str(ENV / 'bin/python'), str(ENV / 'requirements.lock'),
        str(CHECKPOINT), str(SOURCE / 'inference.py'), *[str(TOKENIZER / p) for p in TOKENIZER_FILES]]


def binding(root, app_root=None):
    root = Path(root)
    app = Path(app_root or Path(__file__).parent)
    source = root / SOURCE
    if installed_source_commit(source) != COMMIT:
        raise ValueError('BiomedParse source commit mismatch')
    pinned = json.loads((app / 'catalog/biomedparse-source.json').read_text())
    for relative, expected in pinned.items():
        if file_sha256(source / relative) != expected:
            raise ValueError(f'BiomedParse upstream source changed: {relative}')
    digest = file_sha256(root / CHECKPOINT)
    if digest != CHECKPOINT_SHA256:
        raise ValueError('BiomedParse checkpoint SHA-256 mismatch')
    tokenizer = {p: file_sha256(root / TOKENIZER / p) for p in TOKENIZER_FILES}
    if tokenizer != json.loads((app / 'catalog/biomedparse-tokenizer.json').read_text())['sha256']:
        raise ValueError('BiomedParse original CLIP tokenizer mismatch')
    return {'source_commit': COMMIT, 'source_hashes': pinned, 'checkpoint_sha256': digest,
            'adapter_hashes': {p: file_sha256(app / p) for p in APP_FILES},
            'environment_lock_sha256': file_sha256(root / ENV / 'requirements.lock'),
            'tokenizer_hashes': tokenizer}


def validate_marker(record, root, gpu, app_root=None):
    if not isinstance(record, dict) or marker_has_failure(record):
        raise ValueError('Failed or malformed BiomedParse acceptance marker')
    if record.get('deviations') or record.get('waivers'):
        raise ValueError('Acceptance cannot waive deviations')
    for key, value in {'schema': 1, 'task': TASK, 'result': 'passed', 'checks': CHECKS,
                       'seed': SEED, 'criterion': 'exact', 'mask_voxel_differences': 0,
                       'device': require_device(gpu), **binding(root, app_root)}.items():
        if record.get(key) != value:
            raise ValueError(f'BiomedParse acceptance mismatch: {key}')
    accepted = record.get('gpu', {})
    for key in ('compute_capability', 'name', 'total_memory', 'torch', 'compiled_arches'):
        if accepted.get(key) != gpu.get(key):
            raise ValueError(f'BiomedParse acceptance GPU mismatch: {key}')
    for mode, runs in (('reference', 1), ('repeatability', 2), ('dicom', 2)):
        evidence = record.get(mode, {})
        if evidence.get('passed') is not True or evidence.get('runs') != runs:
            raise ValueError(f'BiomedParse acceptance lacks {mode} evidence')
    if record.get('dicom', {}).get('rois') != 1 or not record.get('public_origin'):
        raise ValueError('BiomedParse acceptance requires a public, nonempty DICOM test')
