"""Contracts for the pinned NCI and U-MambaMTL bpMRI releases."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import fcntl
import json
import math
import os
import sys
from pathlib import Path
import subprocess

NCI = 'nci_monai_prostate_lesion_bpmri'
UMAMBA = 'umambamtl_cspca_bpmri'
TASKS = {
    NCI: ('nci_monai_lesion', 'nci-monai', 'run_nci_prostate.py'),
    UMAMBA: ('umambamtl', 'umamba', 'run_umambamtl.py'),
}
COMMITS = {
    'nci_monai_lesion': '21ed8e57c7256834d4fbaf19579ca25ad3d135ee',
    'umambamtl': 'bc657e9f45d7d1ed457b071dd099686e6e90a742',
}
REVISIONS = {NCI: 'NCI-v3/organ+fold0-4/model_best+classifier',
             UMAMBA: 'U-MambaMTL/prostate_weights+detector/f0-f4'}
# The released wrapper assigns the ADC directory to key hbv, and vice versa.
UMAMBA_KEYS = ('t2w', 'hbv', 'adc')
UMAMBA_CHANNEL_ROLES = ('t2w', 'adc', 'hbv')
RGA_PARAMETERS = {'threshold': 'dynamic', 'num_lesions_to_extract': 3,
                  'min_voxels_detection': 72, 'dynamic_threshold_factor': 3.0}
NCI_THRESHOLD = 0.6344772701607316


def weights(task):
    folder = Path('weights') / TASKS[task][0]
    if task == NCI:
        return [folder / 'organ/model.ts', folder / 'classifier/model_best.pth.tar',
                *[folder / f'fold{i}/model_best_fold{i}.pth.tar' for i in range(5)]]
    return [folder / 'prostate_crop/prostate_weights.ckpt',
            *[folder / f'detector/f{i}.ckpt' for i in range(5)]]


def source_files(task):
    if task == NCI:
        base = Path('sources/nci_monai_lesion/prostate-mri-lesion-seg')
        return [base / 'LICENSE', *[base / 'prostate_mri_lesion_seg_app' / name for name in (
            'organ_seg_operator.py', 'custom_lesion_seg_operator.py',
            'custom_lesion_classifier_operator.py', 'rrunet3D.py', 'resnet.py', 'common.py', 'app.py')]]
    base = Path('sources/umambamtl')
    return [base / 'LICENSE', *[base / 'gc_algorithms/base_container/models/umamba_mtl' / name
            for name in ('process.py', 'process_prostate.py', 'UMambaBot_3d.py', 'UMambaBot_3d_mtl.py')]]


def requirements(task):
    model, env, runner = TASKS[task]
    return [f'@{runner}', '@held_mri.py', '@held_mri_upstream.py', '@prostate_models.py',
            '@prostate_postprocess.py', '@convert_masks_validate_rtstruct.py', '@models/manifests/held-mri-sources.json',
            f'runtime-envs/{env}/bin/python', f'runtime-envs/{env}/requirements.lock',
            *map(str, weights(task)), *map(str, source_files(task)),
            f'validation/{task}.json', f'orientation-validation/{task}.json']


def require_capability(capability):
    if tuple(capability) < (7, 0):
        raise ValueError('U-MambaMTL requires CUDA compute capability >= 7.0; P6000 (sm_61) is unsupported. Use the H100 worker.')


def device_key(gpu):
    capability = gpu.get('compute_capability')
    name = gpu.get('name', '')
    if capability == [9, 0] and 'H100' in name:
        return 'H100'
    if capability == [6, 1] and 'P6000' in name:
        return 'P6000'
    raise ValueError('Held MRI acceptance requires an identified H100 or P6000 GPU')


def inference_policy(task):
    # NCI uses the Torch defaults, including non-deterministic cuDNN selection.
    return {'cudnn_deterministic': task == UMAMBA, 'cudnn_benchmark': False,
            'cudnn_allow_tf32': True, 'matmul_allow_tf32': False,
            'deterministic_algorithms': False, 'float32_matmul_precision': 'highest'}


def gpu_preflight(task, root):
    """Query the same interpreter and visible CUDA device that will execute."""
    python = Path(root) / 'runtime-envs' / TASKS[task][1] / 'bin/python'
    script = ('import sys,json; sys.path.insert(0,sys.argv[1]); '
              'from held_mri import runtime_gpu; print(json.dumps(runtime_gpu(sys.argv[2])))')
    try:
        from held_mri_runtime import python_command
        with python_command(python, isolated=True) as command:
            result = subprocess.run([*command, '-c', script, str(Path(__file__).parent), task],
                                    capture_output=True, text=True, timeout=120, check=True)
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError('Cannot verify held MRI executing GPU/runtime: ' + str(exc)) from exc


def runtime_gpu(task):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU inference is required; CPU fallback is disabled')
    capability = torch.cuda.get_device_capability()
    if task == UMAMBA:
        require_capability(capability)
    policy = inference_policy(task)
    torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
    torch.backends.cudnn.benchmark = policy['cudnn_benchmark']
    torch.backends.cudnn.allow_tf32 = policy['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32 = policy['matmul_allow_tf32']
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'])
    torch.set_float32_matmul_precision(policy['float32_matmul_precision'])
    gpu = {'name': torch.cuda.get_device_name(), 'compute_capability': list(capability),
           'compiled_architectures': torch.cuda.get_arch_list(), 'cuda': torch.version.cuda,
           **policy}
    device_key(gpu)
    return gpu


def runtime_start(task, root, *, worker_python, acceptance_audit=False):
    """Recheck device, files and acceptance at the model process boundary."""
    from held_mri_runtime import check_python_policy
    check_python_policy()
    python = Path(root) / 'runtime-envs' / TASKS[task][1] / 'bin/python'
    if (Path(sys.prefix).resolve() != python.parent.parent.resolve()
            or not os.path.samefile(sys.executable, python)):
        raise ValueError('Runner interpreter differs from the accepted installed runtime')
    gpu = runtime_gpu(task)
    current = binding(task, root, worker_python=worker_python, uncached=True)
    if not acceptance_audit:
        for folder in ('validation', 'orientation-validation'):
            validate_marker(json.loads((Path(root) / folder / f'{task}.json').read_text()), task, root,
                            gpu=gpu, worker_python=worker_python, expected=current)
    return gpu


@contextmanager
def marker_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_device_records(path):
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        records = payload['devices'] if payload.get('schema_version') == 2 else {}
        return records if isinstance(records, dict) else {}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {}


def invalidate_device(root, task, label):
    """Retain other GPU evidence, remove legacy unscoped records."""
    for folder in ('validation', 'orientation-validation'):
        path = Path(root) / folder / f'{task}.json'
        with marker_lock(path):
            records = read_device_records(path)
            records.pop(label, None)
            if records:
                write_device_records(path, records)
            else:
                path.unlink(missing_ok=True)


def write_device_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps({'schema_version': 2, 'devices': records}, indent=2) + '\n')
    temporary.replace(path)


def store_device_marker(root, task, record, *, worker_python=None):
    # Publication must read actual bytes even in a long-lived acceptance process.
    validate_marker(record, task, root, gpu=record['gpu'], worker_python=worker_python, uncached=True)
    for folder in ('validation', 'orientation-validation'):
        path = Path(root) / folder / f'{task}.json'
        with marker_lock(path):
            records = read_device_records(path)
            records[device_key(record['gpu'])] = record
            write_device_records(path, records)


def pinned_source_hashes(task):
    return json.loads((Path(__file__).parent / 'models/manifests/held-mri-sources.json').read_text())[task]


def verify_release(task, root, *, uncached=False):
    from prostate_models import file_sha256, installed_source_commit
    root = Path(root)
    if installed_source_commit(root / 'sources' / TASKS[task][0]) != COMMITS[TASKS[task][0]]:
        raise ValueError('Installed upstream source commit mismatch')
    actual = {p.as_posix(): file_sha256(root / p, uncached=uncached) for p in source_files(task)}
    if actual != pinned_source_hashes(task):
        raise ValueError('Installed upstream source differs from the pinned file hashes')
    return actual


def binding(task, root, *, worker_python=None, uncached=False):
    from prostate_models import file_sha256, installed_source_commit
    root = Path(root)
    hashes = {p.as_posix(): file_sha256(root / p, uncached=uncached) for p in weights(task)}
    source_hashes = verify_release(task, root, uncached=uncached)
    source_tree = {p.relative_to(root).as_posix(): file_sha256(p, uncached=uncached)
                   for p in (root / 'sources' / TASKS[task][0]).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.pyc', '.json', '.so', '.yaml', '.yml')
                   and '.git' not in p.parts and '__pycache__' not in p.parts}
    app = Path(__file__).parent
    # Include all production Python modules and catalogue/label contracts.
    paths = set(app.glob('*.py')) | set(app.glob('*.pyc'))
    for folder in ('catalog', 'labels', 'models/manifests'):
        paths.update((app / folder).rglob('*.json'))
    adapter_hashes = {p.relative_to(app).as_posix(): file_sha256(p, uncached=uncached) for p in sorted(paths)}
    from held_mri_runtime import runtime_manifest, worker_runtime_manifest, manifest_digest, PYTHON_POLICY
    runtime = runtime_manifest(root / 'runtime-envs' / TASKS[task][1] / 'bin/python', uncached=uncached)
    worker = worker_runtime_manifest(worker_python or sys.executable, uncached=uncached)
    return {'source_commit': installed_source_commit(root / 'sources' / TASKS[task][0]),
            'source_file_hashes': source_hashes, 'source_tree_hashes': source_tree,
            'adapter_hashes': adapter_hashes, 'checkpoint_hashes': hashes,
            'checkpoint_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'checkpoint_revision': REVISIONS[task], 'runtime_manifest': runtime,
            'runtime_manifest_sha256': manifest_digest(runtime), 'inference_policy': inference_policy(task),
            'worker_runtime_manifest': worker, 'worker_runtime_manifest_sha256': manifest_digest(worker),
            'python_policy': PYTHON_POLICY,
            'environment_lock_sha256': file_sha256(root / 'runtime-envs' / TASKS[task][1] / 'requirements.lock', uncached=uncached)}


def validate_marker(record, task, root, *, gpu=None, worker_python=None, uncached=False, expected=None):
    from prostate_models import marker_has_failure
    def require(value, message):
        if not value:
            raise ValueError(message)
    gpu = gpu if gpu is not None else gpu_preflight(task, root)
    key = device_key(gpu)
    if task == UMAMBA:
        require_capability(gpu['compute_capability'])
    if isinstance(record, dict) and record.get('schema_version') == 2:
        record = record.get('devices', {}).get(key)
    require(isinstance(record, dict), f'Missing acceptance for executing device {key}')
    require(record.get('gpu') == gpu, 'Acceptance executing GPU/runtime policy mismatch')
    require(isinstance(record, dict) and record.get('task') == task and record.get('result') == 'passed',
            'Acceptance task/result mismatch')
    require(not marker_has_failure(record), 'Acceptance contains a failure or deviation')
    require(record.get('acceptance_modes') == ['reference', 'dicom'], 'Both acceptance modes are required')
    expected = expected if expected is not None else binding(task, root, worker_python=worker_python, uncached=uncached)
    require(expected['source_commit'] == COMMITS[TASKS[task][0]], 'Installed source commit mismatch')
    for key, value in expected.items():
        require(record.get(key) == value, f'Acceptance {key} mismatch')
    for mode in ('reference', 'dicom'):
        proof = record.get(mode + '_mode', {})
        require(isinstance(proof, dict) and proof.get('passed') is True and proof.get('runs') == 2,
                f'Missing {mode} repeatability evidence')
        digest = proof.get('result_sha256', '')
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                f'Missing {mode} evidence digest')
    required = {'gpu_inference', 'unmodified_reference', 'repeatability', 'exact_input_grid',
                'rtstruct_roundtrip', 'orientation', 'both_acceptance_modes', 'lesion_measurements'}
    checks = record.get('checks', [])
    require(isinstance(checks, list) and all(isinstance(c, str) for c in checks) and required <= set(checks),
            'Incomplete acceptance checks')
    capability = record.get('gpu', {}).get('compute_capability')
    require(capability in ([[6, 1], [9, 0]] if task == NCI else [[9, 0]]), 'GPU acceptance target mismatch')
    name = record.get('gpu', {}).get('name', '')
    require(isinstance(name, str) and ('H100' in name if capability == [9, 0] else 'P6000' in name),
            'GPU acceptance device name mismatch')
    require(record.get('tolerances') == {'map_atol': 0.001 if task == UMAMBA else 0.0,
                                        'mask_voxel_differences': 0, 'grid_atol_mm': 0.0001},
            'Acceptance tolerance mismatch')


def roi_name(task, candidate_id, value):
    if type(candidate_id) is not int or candidate_id < 1:
        raise ValueError('Positive integer lesion identifier required')
    if task == UMAMBA:
        if candidate_id > 3 or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('U-MambaMTL requires at most three candidates with scores in (0,1]')
        name = f'UMMTL_L{candidate_id:02d}_p{value:.2f}'
    else:
        if type(value) is not int or value not in (2, 3, 4, 5):
            raise ValueError('NCI PI-RADS class must be 2 to 5')
        name = f'NCI_L{candidate_id:02d}_PIRADS{value}'
    if len(name) > 64:
        raise ValueError('ROI name exceeds the DICOM limit')
    return name


def extract_umamba(probability_xyz, spacing_xyz, provenance=None):
    from importlib.metadata import version
    import numpy as np
    from report_guided_annotation import extract_lesion_candidates
    if version('report-guided-annotation') != '0.2.4':
        raise RuntimeError('U-MambaMTL extraction requires report-guided-annotation 0.2.4')
    probability = np.asarray(probability_xyz)
    spacing = np.asarray(spacing_xyz)
    if probability.ndim != 3 or not probability.size or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Expected finite 3-D probabilities in [0,1]')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Positive x,y,z spacing required')
    detection, candidates, indexed = extract_lesion_candidates(probability, **RGA_PARAMETERS)
    masks, lesions = {}, []
    for number, score in candidates:
        mask = (indexed == number) & (detection > 0)
        if score <= 0 or not mask.any():
            continue
        name = roi_name(UMAMBA, int(number), float(score))
        masks[name] = mask
        lesions.append({'candidate_id': int(number), 'roi_name': name, 'likelihood': float(score),
                        'voxel_count': int(mask.sum()), 'volume_ml': float(mask.sum() * spacing.prod() / 1000)})
    return detection, masks, {'schema_version': 1, 'task': UMAMBA,
        'status': 'nonempty' if masks else 'empty', 'reference_role': 't2w',
        'channel_order': list(UMAMBA_CHANNEL_ROLES), 'upstream_channel_keys': list(UMAMBA_KEYS),
        'case_likelihood': float(detection.max()), 'lesions': lesions,
        'extraction': {'package': 'report-guided-annotation', 'version': '0.2.4', **RGA_PARAMETERS},
        'provenance': provenance or {}}


def export_masks(work, output, reference, masks_xyz, report):
    """Masks are already on the native T2 grid. Empty lesion sets stay explicit."""
    import nibabel as nib
    import numpy as np
    from prostate_postprocess import write_json
    work, output = Path(work), Path(output)
    directory = work / 'masks'
    directory.mkdir(exist_ok=True)
    mapping = {}
    for name, mask in masks_xyz.items():
        if mask.shape != reference.shape:
            raise ValueError('Mask shape differs from native T2')
        if not np.any(mask):
            raise ValueError(f'Empty ROI {name}; no contour may be invented')
        path = directory / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine), path)
        mapping[name] = str(path)
    write_json(str(output) + '.lesions.json', report)
    write_json(work / 'masks.json', mapping)
    write_json(work / 'result.json', {'empty': not bool(mapping)})
