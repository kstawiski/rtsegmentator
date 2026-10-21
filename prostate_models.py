"""Prostate runner contracts, staged DICOM validation and CPU orchestration."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

CHANNEL_ROLES = ('t2w', 'adc', 'hbv')
TRAINER = 'nnUNetTrainerV2_Loss_FL_and_CE_checkpoints'
UMU_FOLDER = 'Dataset077_ProstateZones/nnUNetTrainer__nnUNetPlans__3d_fullres'
TASKS = {
    'picai_nnunet_supervised_bpmri': ('picai_supervised', 'picai', 'run_picai.py'),
    'picai_nnunet_semisupervised_bpmri': ('picai_semi_supervised', 'picai', 'run_picai.py'),
    'umu_prostate_zones_urethra_t2': ('umu_prostate_zones', 'umu-nnunetv2', 'run_umu_prostate.py'),
    'prostate158_tumor_bpmri': ('prostate158_tumor', 'prostate158', 'run_prostate158.py'),
}
SOURCE_COMMITS = {
    'picai_supervised': 'fbea30efc5f57adab0445e554f3de3c0a2dca64a',
    'picai_semi_supervised': '6129f7e0fe3fc3994064672aa2c7f629584004a3',
    'umu_prostate_zones': 'eacfe6436635f43b74dadf4d31e8424a856014e5',
    'prostate158_tumor': '8187638887409c95ba0b091f249a40c0822c89e0',
}
UMU_LABELS = {'background': 0, 'PZ': 1, 'CZ': 2, 'TZ': 3, 'AFS': 4, 'Urethra': 5}


def artifact_names(task):
    """Finite, task-owned download allowlist; inference work files are excluded."""
    if task not in TASKS:
        return []
    suffixes = ('.lesions.json', '.probability.nii.gz', '.detection.nii.gz') if task.startswith('picai_') else (
        '.provenance.json', '.segmentation.nii.gz')
    if task == 'umu_prostate_zones_urethra_t2':
        suffixes += ('.probabilities.npz',)
    return [f'{task}_RTSTRUCT.dcm{suffix}' for suffix in suffixes]


def model_folder(task):
    model, _, _ = TASKS[task]
    if model.startswith('picai_'):
        number = 2201 if model == 'picai_supervised' else 2203
        return Path('weights') / model / 'results/nnUNet/3d_fullres' / f'Task{number}_picai_baseline' / f'{TRAINER}__nnUNetPlansv2.1'
    if model == 'umu_prostate_zones':
        return Path('weights') / model / UMU_FOLDER
    return Path('weights') / model


_SHA256_CACHE = {}


def file_sha256(path):
    """SHA-256 of a checkpoint, cached per process by path, size and mtime.

    Preflight runs on every catalogue refresh; re-reading several GB each time is
    needlessly slow, while a replaced or modified file changes size or mtime.
    """
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _SHA256_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        _SHA256_CACHE[key] = digest.hexdigest()
    return _SHA256_CACHE[key]


def weight_files(task):
    folder = model_folder(task)
    if task.startswith('picai_'):
        return [folder / 'plans.pkl', *[folder / f'fold_{fold}' / name for fold in range(5)
                for name in ('model_best.model', 'model_best.model.pkl')]]
    if task == 'umu_prostate_zones_urethra_t2':
        return [folder / 'dataset.json', folder / 'plans.json',
                *[folder / f'fold_{fold}/checkpoint_best.pth' for fold in range(5)]]
    return [folder / 'tumor.pt']


def requirements(task):
    model, env, runner = TASKS[task]
    files = [f'@{runner}', '@prostate_models.py', '@prostate_postprocess.py',
             f'runtime-envs/{env}/bin/python', f'sources/{model}/LICENSE',
             *map(str, weight_files(task)), f'validation/{task}.json']
    if task.startswith('picai_'):
        files += ['@convert_masks_validate_rtstruct.py', '@picai_predict.py', f'runtime-envs/{env}/bin/nnUNet_predict']
        files += [f'sources/{model}/{name}' for name in (
            'nnUNetTrainerV2_focalLoss.py', 'nnUNetTrainerV2_Loss_CE_checkpoints.py',
            'nnUNetTrainerV2_Loss_FL_and_CE.py')]
    else:
        files += [f'@labels/{model}.json']
        if model == 'umu_prostate_zones':
            files += [f'runtime-envs/{env}/bin/nnUNetv2_predict_from_modelfolder',
                      '@third_party/umu_prostate_zones/utils_postprocess_nnUNet.py',
                      '@third_party/umu_prostate_zones/LICENSE']
    return files


def marker_has_failure(value):
    """A success label cannot conceal failed checks or a waived failure."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'passed', 'predeclared_passed'} and item is not True:
                return True
            if key in {'result', 'status'} and item not in {'passed', 'nonempty'}:
                return True
            if key in {'error', 'errors', 'fatal_error', 'failure', 'failures', 'failed_checks',
                       'unresolved_failures', 'deviations'} and item:
                return True
            if marker_has_failure(item):
                return True
    elif isinstance(value, list):
        return any(marker_has_failure(item) for item in value)
    return False


def installed_source_commit(source):
    """Commit of a pinned upstream checkout, read from .git without running git.

    The portal runs as root in a container while the bundle belongs to another user,
    so `git rev-parse` refuses the repository ("dubious ownership"). Reading HEAD and
    the ref it names (loose or packed) needs no git binary. Without .git, the install
    manifest records the commit.
    """
    git = Path(source) / '.git'
    if not git.is_dir():
        return json.loads((Path(source) / 'RTSEG_MANIFEST.json').read_text())['commit']
    try:
        head = (git / 'HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = head[5:].strip()
            loose = git / ref
            if loose.is_file():
                head = loose.read_text().strip()
            else:
                packed = (git / 'packed-refs').read_text().splitlines()
                head = next(line.split()[0] for line in packed
                            if not line.startswith(('#', '^')) and line.split()[-1] == ref)
    except (OSError, StopIteration, IndexError) as exc:
        raise ValueError('Cannot verify installed source commit') from exc
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('Cannot verify installed source commit')
    return head


def validate_acceptance(record, task, root):
    """Validate the prostate harness marker against this installed release."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(isinstance(record, dict) and record.get('task') == task and record.get('result') == 'passed',
            'Acceptance task/result mismatch')
    require(not marker_has_failure(record), 'Acceptance contains a failure or deviation')
    modes = record.get('acceptance_modes')
    require(isinstance(modes, list) and len(modes) == 2 and 'reference' in modes and 'dicom' in modes,
            'Both acceptance modes are required')
    proof = record.get('reference_mode')
    require(isinstance(proof, dict) and proof.get('passed') is True and
            isinstance(proof.get('result_sha256'), str) and re.fullmatch('[0-9a-f]{64}', proof['result_sha256']) and
            type(proof.get('runs')) is int and proof['runs'] == (2 if task.startswith('picai_') else 1),
            'Missing bound reference evidence')
    checks = record.get('checks')
    required = {'gpu_inference', 'exact_input_grid', 'rtstruct_roundtrip', 'orientation', 'both_acceptance_modes',
                'author_reference_and_repeatability' if task.startswith('picai_') else 'model_output_contract'}
    require(isinstance(checks, list) and all(isinstance(c, str) for c in checks) and required <= set(checks),
            'Incomplete acceptance checks')
    model = TASKS[task][0]
    source = Path(root) / 'sources' / model
    installed_commit = installed_source_commit(source)
    require(record.get('source_commit') == installed_commit == SOURCE_COMMITS[model], 'Source commit mismatch')
    hashes = {relative.as_posix(): file_sha256(Path(root) / relative) for relative in weight_files(task)}
    require(record.get('checkpoint_hashes') == hashes, 'Checkpoint hashes mismatch')
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(record.get('checkpoint_sha256') == digest, 'Checkpoint digest mismatch')
    revision = (installed_commit if task.startswith('picai_') else
                'Dataset077_ProstateZones/checkpoint_best.pth' if model == 'umu_prostate_zones' else
                'zenodo:7040585/tumor.pt')
    require(record.get('checkpoint_revision') == revision, 'Checkpoint revision mismatch')


def validate_orientation(record, task):
    if (not isinstance(record, dict) or record.get('task') != task or record.get('result') != 'passed'
            or marker_has_failure(record)):
        raise ValueError('Orientation task/result mismatch or failed checks')


def validate_umu_dataset(data):
    if data.get('labels') != UMU_LABELS or data.get('channel_names') != {'0': 'T2'}:
        raise ValueError('Umeå Dataset077 channel/label contract differs from the pinned export')
    if data.get('file_ending') != '.nrrd':
        raise ValueError('Umeå Dataset077 requires its native .nrrd input contract')


def role_directories(job, *, multisequence=True):
    """Reject incomplete or redirected role mappings before any conversion."""
    job = Path(job)
    data = json.loads((job / 'job.json').read_text())
    if data.get('input_preparation_interrupted'):
        raise ValueError('Input preparation was interrupted')
    roles = CHANNEL_ROLES if multisequence else ('t2w',)
    records = data.get('input_roles', {})
    manifest = data.get('input_manifest')
    if multisequence and not isinstance(manifest, dict):
        raise ValueError('Multi-sequence input requires a complete input_manifest')
    result = {}
    for role in roles:
        dirname = 'input' if role == 't2w' else f'input_{role}'
        if multisequence and (not isinstance(records.get(role), dict) or records[role].get('dir') != dirname):
            raise ValueError(f'Missing or invalid input role {role}: expected {dirname}')
        directory = job / dirname
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f'Missing or redirected input directory: {dirname}')
        files = sorted(directory.iterdir())
        if not files or any(p.is_symlink() or not p.is_file() for p in files):
            raise ValueError(f'{dirname} must contain DICOM files only')
        actual = [{'name': p.name, 'size': p.stat().st_size} for p in files]
        if isinstance(manifest, dict) and manifest.get(dirname) != actual:
            raise ValueError(f'Incomplete input manifest for {role}')
        result[role] = directory
    return result


def validate_dicom_roles(directories, records):
    """Check series identity, one-volume geometry and standard diffusion metadata.

    Intensity values and rescale tags are left to the existing DICOM converter.
    Enhanced multi-frame input needs the input adapter's frame extraction first.
    """
    import numpy as np
    import pydicom
    summaries = {}
    for role, directory in directories.items():
        headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in sorted(directory.iterdir())]
        if any(str(getattr(d, 'Modality', '')) != 'MR' or int(getattr(d, 'NumberOfFrames', 1)) != 1 for d in headers):
            raise ValueError(f'{role} requires single-frame MR slices forming one 3-D volume')
        identities = {}
        for tag in ('StudyInstanceUID', 'SeriesInstanceUID', 'FrameOfReferenceUID'):
            values = {str(getattr(d, tag, '')) for d in headers}
            if len(values) != 1 or '' in values:
                raise ValueError(f'{role}: inconsistent or missing {tag}')
            identities[tag] = values.pop()
        if records.get(role, {}).get('series_uid', identities['SeriesInstanceUID']) != identities['SeriesInstanceUID']:
            raise ValueError(f'{role}: selected SeriesInstanceUID does not match the staged series')
        if len({str(d.SOPInstanceUID) for d in headers}) != len(headers):
            raise ValueError(f'{role}: duplicate SOPInstanceUID')
        first = headers[0]
        direction = np.asarray(first.ImageOrientationPatient, float)
        normal = np.cross(direction[:3], direction[3:])
        if (direction.shape != (6,) or not np.isfinite(direction).all()
                or not np.allclose([np.linalg.norm(direction[:3]), np.linalg.norm(direction[3:])], 1, atol=1e-4)
                or not np.isclose(direction[:3] @ direction[3:], 0, atol=1e-4)
                or not np.isclose(np.linalg.norm(normal), 1, atol=1e-4)):
            raise ValueError(f'{role}: invalid slice orientation')
        spacing = np.asarray(first.PixelSpacing, float)
        if (spacing.shape != (2,) or not np.isfinite(spacing).all()
                or np.any(spacing <= 0) or first.Rows < 1 or first.Columns < 1):
            raise ValueError(f'{role}: invalid pixel grid')
        positions = []
        for header in headers:
            if (header.Rows != first.Rows or header.Columns != first.Columns
                    or not np.allclose(header.PixelSpacing, first.PixelSpacing, atol=1e-5, rtol=0)
                    or not np.allclose(header.ImageOrientationPatient, direction, atol=1e-5, rtol=0)):
                raise ValueError(f'{role}: inconsistent slice geometry')
            position = np.asarray(header.ImagePositionPatient, float)
            if not np.isfinite(position).all():
                raise ValueError(f'{role}: nonfinite slice position')
            positions.append(float(position @ normal))
        gaps = np.diff(sorted(positions))
        if len(gaps) and (np.any(gaps < 1e-4) or not np.allclose(gaps, np.median(gaps), atol=1e-3, rtol=1e-3)):
            raise ValueError(f'{role}: duplicate or irregular slice positions; expected one 3-D volume')
        if role == 'hbv':
            selected = records.get(role, {})
            value = selected.get('b_value')
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0 or not selected.get('b_value_source')):
                raise ValueError('HBV requires a recorded b_value and documented b_value_source')
            # Standard scalar tag. Other documented selectors belong to the input adapter.
            for header in headers:
                observed = getattr(header, 'DiffusionBValue', None)
                if observed is not None and not math.isclose(float(observed), value, abs_tol=1e-3):
                    raise ValueError('HBV contains an unselected b-value')
        summaries[role] = identities
    if len({s['StudyInstanceUID'] for s in summaries.values()}) != 1:
        raise ValueError('All prostate input roles must share StudyInstanceUID')
    reference = summaries['t2w']['FrameOfReferenceUID']
    warnings = [f'{role}: FrameOfReferenceUID differs from T2; no registration is applied'
                for role, summary in summaries.items() if summary['FrameOfReferenceUID'] != reference]
    return warnings


def native_input(converted, destination):
    """Undo the shared converter's lossless storage permutation before the model."""
    import nibabel as nib
    from orientation_contract import read_input_record, restore_image
    image = nib.load(converted)
    record = read_input_record(converted)
    if record:
        image, _ = restore_image(image, record['source']['shape'], record['source']['affine'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, destination)


def runtime_provenance(task, root):
    from importlib.metadata import PackageNotFoundError, version
    model, _, _ = TASKS[task]
    packages = {}
    for package in ('torch', 'numpy', 'scipy', 'SimpleITK', 'nnunet', 'nnunetv2', 'monai',
                    'picai-prep', 'picai-baseline', 'picai-eval', 'report-guided-annotation'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    weights = []
    for relative in weight_files(task):
        path = Path(root) / relative
        weights.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': file_sha256(path)})
    return {'task': task, 'source_commit': SOURCE_COMMITS[model], 'packages': packages, 'weights': weights}


def run_task(task, job, app_root, model_root, worker_python, run):
    """Use model-root runtime-envs assets; write only to the job workspace/results."""
    model, environment, script = TASKS[task]
    job = Path(job).absolute()
    work = job / 'external' / task
    output = job / 'output' / f'{task}_RTSTRUCT.dcm'
    output.parent.mkdir(parents=True, exist_ok=True)
    # A retry must not retain a previous positive result after a negative run.
    for artifact in output.parent.glob(output.name + '*'):
        if artifact.is_file():
            artifact.unlink()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    metadata = json.loads((job / 'job.json').read_text())
    dirs = role_directories(job, multisequence=model != 'umu_prostate_zones')
    warnings = validate_dicom_roles(dirs, metadata.get('input_roles', {}))
    for role, directory in dirs.items():
        converted = work / 'converted' / f'{role}.nii.gz'
        run([worker_python, str(app_root / 'dicom_series_to_nifti.py'), str(directory), str(converted)])
        native_input(converted, work / 'native' / f'{role}.nii.gz')
    context = {'input_roles': metadata.get('input_roles', {}), 'warnings': warnings,
               'channel_order': list(dirs), 'reference_role': 't2w'}
    (work / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    run([str(model_root / 'runtime-envs' / environment / 'bin/python'), str(app_root / script),
         '--task', task, '--work', str(work), '--model-root', str(model_root), '--output', str(output)])
    result = json.loads((work / 'result.json').read_text())
    if result['empty']:
        # Existing portal/SSH handling recognises this completed-empty marker.
        shutil.rmtree(work)
        raise SystemExit('empty prediction: no RTSTRUCT created')
    args = ['--model-input', str(work / 'native/t2w.nii.gz'), '--dicom', str(job / 'input'), '--output', str(output)]
    if model.startswith('picai_'):
        run([worker_python, str(app_root / 'convert_masks_validate_rtstruct.py'),
             '--masks-json', str(work / 'masks.json'), *args])
    else:
        run([worker_python, str(app_root / 'convert_validate_rtstruct.py'),
             '--prediction', str(work / 'restored.nii.gz'), '--labels', str(app_root / 'labels' / f'{model}.json'), *args])
    shutil.rmtree(work)
