"""Pure contracts for the pinned NV-Segment automatic bundles (no ML imports)."""
from __future__ import annotations
import ast
import json
import re
import unicodedata
from pathlib import Path

SOURCE_COMMIT = 'cb921f5c58837c0f42a713855d68b32af88e1cdd'
CT_SOURCE_COMMIT = 'f9f5f51b589e5dc9c23c453cf5138398e4084056'
MODES = {
    'nv_segment_ct_auto': {'modality':'CT','bundle':'NV-Segment-CT','upstream_mode':None,
        'source':'sources/NV-Segment-CTMR/NV-Segment-CT','source_commit':CT_SOURCE_COMMIT,
        'revision':'afb51518689f71e6abb367ee6301b2cd0225c66a',
        'sha256':'c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b'},
    'nv_segment_ctmr_ct_body': {'modality':'CT','upstream_mode':'CT_BODY'},
    'nv_segment_ctmr_mri_body': {'modality':'MR','upstream_mode':'MRI_BODY'},
    'nv_segment_ctmr_mri_brain': {'modality':'MR','upstream_mode':'MRI_BRAIN'},
}
for _name, _spec in MODES.items():
    if _name != 'nv_segment_ct_auto':
        _spec.update(bundle='NV-Segment-CTMR', source='sources/NV-Segment-CTMR-cb921f5c/NV-Segment-CTMR',
            source_commit=SOURCE_COMMIT, revision='4fb8b4a6b2532be9f1c449a3726fe5440ab4213a',
            sha256='6c6d86a48a7a9d377ddb41d663d487cfedd6456df3b828ecd931e5cba5e71f99')

def check_modality(mode: str, modality: str) -> None:
    if mode not in MODES:
        raise ValueError(f'Unknown NV-Segment mode: {mode}')
    if modality != MODES[mode]['modality']:
        raise ValueError(f'{mode} requires {MODES[mode]["modality"]}; received {modality}')

def selected_indices(config: dict, mappings: dict, mode: str) -> list[int]:
    """Read the official selection, rejecting an unrecognised config expression."""
    spec=MODES[mode]
    if spec['upstream_mode'] in (None, 'CT_BODY'):
        expr=config['everything_labels' if spec['upstream_mode'] is None else 'ct_body']
        match=re.fullmatch(r'\$list\(set\(\[i\+1 for i in range\((\d+)\)\]\) - set\((\[[\d, ]+\])\)\)',expr)
        if not match: raise ValueError(f'Unrecognised official CT selection: {expr}')
        ids=sorted(set(range(1,int(match[1])+1))-set(ast.literal_eval(match[2])))
    else:
        key='TotalSegmentatorMR' if spec['upstream_mode']=='MRI_BODY' else 'LUMIR'
        ids=[int(pair[1]) for pair in mappings[key]]
    if not ids or len(ids)!=len(set(ids)) or any(i<=0 for i in ids):
        raise ValueError('Foreground label selection must be positive and unique')
    return ids

def roi_names(channel_def: dict, indices: list[int]) -> dict[str,str]:
    result={}; used=set()
    for index in sorted(indices):
        text=channel_def[str(index)]
        name=unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode()
        name=' '.join(re.sub(r'[\\\x00-\x1f\x7f]', ' ',name).split())
        if not name: raise ValueError(f'Empty ROI name for {index}')
        if re.search(r'tumou?r|lesion|cyst|metasta|nodule|neoplasm',name,re.I):
            name='Lesion candidate: '+name
        name=name[:64].rstrip()
        if name.casefold() in used:
            suffix=f' [{index}]'; name=name[:64-len(suffix)].rstrip()+suffix
        if name.casefold() in used: raise ValueError(f'ROI name collision: {name}')
        used.add(name.casefold()); result[str(index)]=name
    return result

def generate(source: Path, mode: str) -> tuple[dict[str,str],list[int]]:
    config=json.loads((source/'configs/inference.json').read_text())
    metadata=json.loads((source/'configs/metadata.json').read_text())
    mp=source/'configs/label_mappings.json'
    mappings=json.loads(mp.read_text()) if mp.exists() else {}
    indices=selected_indices(config,mappings,mode)
    return roi_names(metadata['network_data_format']['outputs']['pred']['channel_def'],indices),indices

def validate_acceptance(marker: dict, mode: str) -> None:
    if not isinstance(marker, dict):
        raise ValueError('Acceptance marker must be a JSON object')
    spec=MODES[mode]
    required={'task':mode,'result':'passed','source_commit':spec['source_commit'],
              'checkpoint_revision':spec['revision'],'checkpoint_sha256':spec['sha256']}
    for key,value in required.items():
        if marker.get(key)!=value: raise ValueError(f'Acceptance marker mismatch: {key}')


def acceptance_from_metrics(metrics: dict, orientation: dict, mode: str, fixture: str) -> dict:
    """Build technical acceptance only after inference, conversion and audit succeed."""
    if (metrics.get('mode') != mode or metrics.get('device') != 'cuda'
            or metrics.get('cuda_available') is not True or metrics.get('load_only') is not False
            or metrics.get('exact_input_grid') is not True or metrics.get('output_dtype') != 'uint16'
            or metrics.get('nonempty_foreground_labels', 0) <= 0):
        raise ValueError('Incomplete GPU inference/grid evidence')
    if (orientation.get('task') != mode or orientation.get('result') != 'passed'
            or orientation.get('rois', 0) <= 0 or not orientation.get('artifact_sha256')):
        raise ValueError('Incomplete orientation evidence')
    marker = {'task': mode, 'result': 'passed', 'source_commit': metrics.get('source_commit'),
              'checkpoint_revision': metrics.get('checkpoint_revision'),
              'checkpoint_sha256': metrics.get('checkpoint_sha256'), 'fixture': fixture,
              'checks': ['gpu_inference', 'exact_input_grid', 'rtstruct_roundtrip', 'orientation'],
              'artifact_sha256': orientation['artifact_sha256'],
              'inference_seconds': metrics.get('total_seconds'),
              'peak_allocated_bytes': metrics.get('peak_allocated_bytes'),
              'peak_reserved_bytes': metrics.get('peak_reserved_bytes'),
              'scope': 'Technical execution and geometry; no clinical performance approval'}
    validate_acceptance(marker, mode)
    return marker
