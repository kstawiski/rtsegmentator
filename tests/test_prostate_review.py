"""Review reproductions using synthetic files and mocked inference/transport."""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

import prostate_models
from prostate_postprocess import fill_umu_zones


TASK = 'umu_prostate_zones_urethra_t2'
SUFFIXES = ('.lesions.json', '.provenance.json', '.probability.nii.gz',
            '.detection.nii.gz', '.probabilities.npz', '.segmentation.nii.gz')


def job_at(portal, models):
    job = portal.BASE / 'review-job'
    (job / 'input').mkdir(parents=True)
    (job / 'output').mkdir()
    (job / 'input/image.dcm').write_bytes(b'synthetic transport input')
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'models': models,
        'status': 'queued', 'input_manifest': portal.job_input_manifest(job)}))
    return job


@pytest.mark.parametrize('mode,routed', [('local', False), ('auto', False),
                                       ('local', True), ('auto', True), ('ssh', False)])
@pytest.mark.parametrize('mixed', [False, True, 'positive_first'])
def test_review_m1_empty_completion_all_backends(portal, monkeypatch, tmp_path, mode, routed, mixed):
    tasks = [TASK, 'prostate158_tumor_bpmri'] if mixed else [TASK]
    if mixed == 'positive_first':
        tasks.reverse()
    job = job_at(portal, tasks)
    remote_output = tmp_path / 'remote-output'
    remote_output.mkdir()
    events = []
    monkeypatch.setattr(portal, 'WORKER_MODE', mode)
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', set(tasks) if routed else set())
    def inference(job, task, *args):
        events.append('runner:' + task)
        root = remote_output if routed or mode == 'ssh' else job / 'output'
        (root / (task + '_RTSTRUCT.dcm.provenance.json')).write_text('{"status":"empty"}')
        (root / (task + '_RTSTRUCT.dcm.segmentation.nii.gz')).write_bytes(b'synthetic map')
        if task == TASK:
            raise RuntimeError('empty prediction: no RTSTRUCT created')
        (root / (task + '_RTSTRUCT.dcm')).write_bytes(b'synthetic positive result')
    def retrieve(job, remote, task=None):
        events.append('retrieved')
        for path in remote_output.iterdir():
            shutil.copy2(path, job / 'output' / path.name)
    def cleanup(*args, **kwargs):
        events.append('remote_deleted')
        for path in remote_output.iterdir():
            path.unlink()
    monkeypatch.setattr(portal, 'prepare_worker_job', lambda *a: events.append('staged'))
    monkeypatch.setattr(portal, 'run_local_task', inference)
    monkeypatch.setattr(portal, 'run_worker_task', lambda job, remote, task, *a: inference(job, task))
    monkeypatch.setattr(portal, 'retrieve_worker_results', retrieve)
    monkeypatch.setattr(portal, 'remove_worker_job', cleanup)
    portal.process(job)
    data = portal.metadata(job)
    assert data['status'] == 'complete', data
    assert data['no_structures_found'] == [TASK]
    assert set(data['completed_models']) == set(tasks)
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as archive:
        assert 'NO_STRUCTURES_FOUND.txt' in archive.namelist()
        for task in tasks:
            assert task + '_RTSTRUCT.dcm.provenance.json' in archive.namelist()
            assert task + '_RTSTRUCT.dcm.segmentation.nii.gz' in archive.namelist()
        assert ('prostate158_tumor_bpmri_RTSTRUCT.dcm' in archive.namelist()) is bool(mixed)
    if routed or mode == 'ssh':
        assert events.index('retrieved') < events.index('remote_deleted')


@pytest.mark.parametrize('mode,routed', [('auto', True), ('ssh', False)])
def test_review_m1_empty_retrieval_failure_retains_remote(portal, monkeypatch, mode, routed):
    job = job_at(portal, [TASK])
    monkeypatch.setattr(portal, 'WORKER_MODE', mode)
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {TASK} if routed else set())
    monkeypatch.setattr(portal, 'prepare_worker_job', lambda *a: None)
    def empty(*a): raise RuntimeError('empty prediction: no RTSTRUCT created')
    def fail(*a): raise OSError('retrieval unavailable')
    monkeypatch.setattr(portal, 'run_worker_task', empty)
    monkeypatch.setattr(portal, 'retrieve_worker_results', fail)
    monkeypatch.setattr(portal, 'remove_worker_job', lambda *a: pytest.fail('Unretrieved output deleted'))
    portal.process(job)
    data = portal.metadata(job)
    assert data['status'] == 'failed'
    assert data['remote_results_retained']
    assert 'retrieval unavailable' in data['error']


def test_review_m2_single_series_remote_payload_reads_runner_metadata(portal, tmp_path):
    job = job_at(portal, [TASK])
    (job / 'prompt.json').write_text('{}')
    archive_path = tmp_path / 'payload.tar'
    with tarfile.open(archive_path, 'w') as archive:
        for member in portal.worker_archive_members(job, [TASK]):
            archive.add(job / member, arcname=member)
    payload = tmp_path / 'payload'
    payload.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(payload, filter='data')
    assert prostate_models.role_directories(payload, multisequence=False) == {'t2w': payload / 'input'}
    assert json.loads((payload / 'job.json').read_text())['models'] == [TASK]
    assert (payload / 'prompt.json').is_file()


@pytest.mark.parametrize('empty', [False, True])
def test_review_m3_all_declared_artifacts_are_packaged(portal, empty):
    tasks = list(prostate_models.TASKS)
    job = job_at(portal, tasks)
    expected = set()
    for task in tasks:
        if task in prostate_models.HELD_MRI_TASKS:
            suffixes = ('.lesions.json', '.probability.nii.gz', '.detection.nii.gz') + (
                ('.gland.nii.gz',) if task == 'nci_monai_prostate_lesion_bpmri' else ())
        elif task.startswith('picai_'):
            suffixes = ('.lesions.json', '.probability.nii.gz', '.detection.nii.gz')
        else:
            suffixes = ('.provenance.json', '.segmentation.nii.gz') + (
                ('.probabilities.npz',) if task == TASK else ())
        for suffix in suffixes + (() if empty else ('', '.orientation.json')):
            name = task + '_RTSTRUCT.dcm' + suffix
            (job / 'output' / name).write_bytes(b'synthetic artifact')
            expected.add(name)
    (job / 'output' / (TASK + '_RTSTRUCT.dcm.undeclared.bin')).write_bytes(b'exclude')
    (job / 'output/unselected_RTSTRUCT.dcm.provenance.json').write_text('{}')
    portal.finalize_results(job, tasks if empty else ())
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as archive:
        assert set(archive.namelist()) == expected | {'rtsegmentator_manifest.json'} | (
            {'NO_STRUCTURES_FOUND.txt'} if empty else set())


def test_review_m3_download_size_bound(portal, monkeypatch):
    job = job_at(portal, [TASK])
    (job / 'output' / (TASK + '_RTSTRUCT.dcm.provenance.json')).write_bytes(b'x' * 32)
    monkeypatch.setattr(portal, 'MAX_RESULT_BYTES', 16, raising=False)
    with pytest.raises(ValueError, match='Result artifacts exceed'):
        portal.finalize_results(job, [TASK])
    assert not (job / 'rtstruct-results.zip').exists()


@pytest.mark.parametrize('seed_labels', [(1,), (1, 2), (2, 4), ()])
def test_review_m4_absent_zones_match_pinned_upstream(seed_labels):
    source = os.environ.get('RTSEG_UMU_SOURCE')
    if not source:
        pytest.skip('Set RTSEG_UMU_SOURCE to compare the pinned upstream postprocessor')
    upstream = runpy.run_path(str(Path(source) / 'nnUNet/utils_postprocess_nnUNet.py'))
    gland = np.ones((9, 9, 3), dtype=bool)
    labels = np.zeros(gland.shape, np.uint8)
    for index, label in enumerate(seed_labels):
        labels[8 - index, 8, 2] = label
    urethra = np.zeros_like(gland)
    expected = upstream['fill_empty_voxels'](~gland, *[(labels == k) & gland for k in range(1, 5)],
                                            urethra, (0.5, 0.5, 3)).argmax(3)
    actual = fill_umu_zones(gland, labels, urethra, (0.5, 0.5, 3))
    np.testing.assert_array_equal(actual, expected)


def test_review_m4_vendored_source_and_license_are_unchanged():
    source = os.environ.get('RTSEG_UMU_SOURCE')
    if not source:
        pytest.skip('Set RTSEG_UMU_SOURCE to compare the pinned upstream files')
    source = Path(source)
    vendor = Path(__file__).resolve().parents[1] / 'third_party/umu_prostate_zones'
    assert (vendor / 'utils_postprocess_nnUNet.py').read_bytes() == (source / 'nnUNet/utils_postprocess_nnUNet.py').read_bytes()
    assert (vendor / 'LICENSE').read_bytes() == (source / 'LICENSE').read_bytes()


def test_review_l1_prostate158_uses_author_autocast_and_device(monkeypatch):
    import run_prostate158
    events = []
    @contextlib.contextmanager
    def autocast():
        events.append('autocast_enter')
        yield
        events.append('autocast_exit')
    fake_torch = SimpleNamespace(no_grad=contextlib.nullcontext,
                                cuda=SimpleNamespace(amp=SimpleNamespace(autocast=autocast)))
    def inferer(image, **kwargs):
        assert events == ['autocast_enter']
        assert 'device' not in kwargs  # Authors aggregate on the input device.
        assert kwargs['roi_size'] == (96, 96, 96) and kwargs['sw_batch_size'] == 4
        assert kwargs['overlap'] == 0.5
        return 'logits'
    assert run_prostate158.predict_logits('image', 'model', amp=True,
        torch_module=fake_torch, inferer=inferer) == 'logits'
    assert events == ['autocast_enter', 'autocast_exit']
    events.clear()
    def full_precision(image, **kwargs):
        assert events == []
        return 'legacy logits'
    assert run_prostate158.predict_logits('image', 'model', amp=False,
        torch_module=fake_torch, inferer=full_precision) == 'legacy logits'


def test_review_l1_main_enters_autocast_before_mocked_inference(tmp_path, monkeypatch):
    """Exercise the real entry point; no network, checkpoint or GPU is used."""
    import run_prostate158 as runner
    if os.environ.get('RTSEG_PROSTATE158_TEST_SOURCE'):
        import importlib.util
        spec = importlib.util.spec_from_file_location('baseline_prostate158',
                                                       os.environ['RTSEG_PROSTATE158_TEST_SOURCE'])
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    active = []
    @contextlib.contextmanager
    def autocast():
        active.append(True)
        yield
        active.pop()
    class Tensor:
        affine = np.eye(4)
        def as_tensor(self): return self
        def unsqueeze(self, *a): return self
        def to(self, *a): return self
        def cpu(self): return self
        def argmax(self, *a): return self
        def __getitem__(self, key): return self
        def numpy(self): return np.zeros((3, 3, 3), np.uint8)
    class Model:
        def to(self, *a): return self
        def eval(self): return self
        def load_state_dict(self, *a, **k): pass
    tensor = Tensor()
    calls = []
    def inferer(image, **kwargs):
        calls.append(kwargs)
        assert active, 'Author evaluator autocast is missing'
        assert 'device' not in kwargs, 'Author evaluator aggregates on the input device'
        return tensor
    fake_torch = SimpleNamespace(cat=lambda *a, **k: tensor, load=lambda *a, **k: {},
        inference_mode=contextlib.nullcontext, no_grad=contextlib.nullcontext,
        cuda=SimpleNamespace(amp=SimpleNamespace(autocast=autocast)))
    modules = {
        'torch': fake_torch,
        'monai.networks.nets': SimpleNamespace(UNet=lambda **k: Model()),
        'monai.inferers': SimpleNamespace(sliding_window_inference=inferer),
        'monai.transforms': SimpleNamespace(KeepLargestConnectedComponent=lambda **k: lambda x: x),
        'monai.utils': SimpleNamespace(get_torch_version_tuple=lambda: (2, 5)),
        'SimpleITK': SimpleNamespace(WriteImage=lambda *a: None,
                                    GetArrayViewFromImage=lambda *a: np.zeros((3, 3, 3))),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner, 'prepare_channels', lambda *a: dict.fromkeys(('t2w', 'adc', 'hbv'), tensor))
    monkeypatch.setattr(runner, 'runtime_provenance', lambda *a: {})
    monkeypatch.setattr(runner, 'restore_segmentation', lambda *a: 'restored')
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'context.json').write_text('{}')
    output = tmp_path / 'result.dcm'
    monkeypatch.setattr(sys, 'argv', ['run_prostate158.py', '--task', runner.TASK,
        '--work', str(work), '--model-root', str(tmp_path / 'unused-models'), '--output', str(output)])
    runner.main()
    assert len(calls) == 1
    precision = json.loads(Path(str(output) + '.provenance.json').read_text())['precision']
    assert precision['autocast_enabled'] and precision['aggregation_device'] == 'input device (cuda)'


@pytest.mark.parametrize('task', list(prostate_models.TASKS))
def test_review_m5_environments_belong_to_model_root(tmp_path, monkeypatch, task):
    import inspect
    import model_preflight
    root = Path(__file__).resolve().parents[1]
    model_root, runtime_root = tmp_path / 'models', tmp_path / 'runtime'
    environment = prostate_models.TASKS[task][1]
    relative = f'runtime-envs/{environment}/bin/python'
    interpreter = model_root / relative
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('synthetic interpreter; never executed')
    runtime_root.mkdir()
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', model_root)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', runtime_root)
    monkeypatch.setattr(model_preflight, 'COMMON', [])
    monkeypatch.setattr(model_preflight, 'REQUIREMENTS', {task: [relative]})
    assert relative not in model_preflight.audit([task])[task]['missing']
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'job.json').write_text('{}')
    monkeypatch.setattr(prostate_models, 'role_directories', lambda *a, **k: {})
    monkeypatch.setattr(prostate_models, 'validate_dicom_roles', lambda *a: [])
    calls = []
    def run(command):
        calls.append(command)
        work = Path(command[command.index('--work') + 1])
        (work / 'result.json').write_text('{"empty": true}')
    # Accept either signature while reproducing the misleading contract before the fix.
    args = [task, job, root, model_root]
    if 'runtime_root' in inspect.signature(prostate_models.run_task).parameters:
        args.append(runtime_root)
    if task in prostate_models.HELD_MRI_TASKS:
        prostate_models.run_task(*args, 'worker-python', run, acceptance_audit=True)
    else:
        with pytest.raises(SystemExit, match='empty prediction'):
            prostate_models.run_task(*args, 'worker-python', run)
    assert calls[0][0] == str(interpreter)
    assert 'runtime_root' not in inspect.signature(prostate_models.run_task).parameters
    assert 'Runtimes resolve from `RTSEG_RUNTIME_ROOT`' not in (root / 'docs/PROSTATE_MODELS.md').read_text()
    interpreter.unlink()
    other = runtime_root / relative
    other.parent.mkdir(parents=True)
    other.write_text('wrong root; never executed')
    assert relative in model_preflight.audit([task])[task]['missing']


def synthetic_runtime_identity(python):
    return {'version':'synthetic', 'packages':[], 'stdlib':'/nonexistent-synthetic-stdlib',
            'prefix': str(Path(python).parent.parent) if str(python) != sys.executable else '/nonexistent-synthetic-worker'}


def synthetic_acceptance_bundle(root, task):
    """Use the full real requirement list; no executable or checkpoint is loaded."""
    model = prostate_models.TASKS[task][0]
    for requirement in prostate_models.requirements(task):
        if not requirement.startswith('@'):
            path = root / requirement
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic placeholder')
    commit = prostate_models.SOURCE_COMMITS[model]
    (root / 'sources' / model / 'RTSEG_MANIFEST.json').write_text(json.dumps({'commit': commit}))
    if task in prostate_models.HELD_MRI_TASKS:
        import held_mri
        for src in held_mri.source_files(task):
            p = root / src
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('synthetic source')
        hashes = {p.as_posix(): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in held_mri.weights(task)}
        src_hashes = {p.as_posix(): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in held_mri.source_files(task)}
        app = Path(held_mri.__file__).parent
        adapter_hashes = {name: hashlib.sha256((app / name).read_bytes()).hexdigest() for name in (
            'held_mri.py', 'held_mri_upstream.py', 'prostate_models.py', 'prostate_postprocess.py',
            held_mri.TASKS[task][2], 'convert_masks_validate_rtstruct.py', 'models/manifests/held-mri-sources.json')}
        env_name = held_mri.TASKS[task][1]
        lock_hash = hashlib.sha256((root / 'runtime-envs' / env_name / 'requirements.lock').read_bytes()).hexdigest()
        record = {
            'task': task, 'result': 'passed', 'source_commit': commit,
            'source_file_hashes': src_hashes, 'adapter_hashes': adapter_hashes,
            'checkpoint_revision': held_mri.REVISIONS[task],
            'checkpoint_hashes': hashes,
            'checkpoint_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'environment_lock_sha256': lock_hash,
            'acceptance_modes': ['reference', 'dicom'], 'deviations': [],
            'reference_mode': {'passed': True, 'result_sha256': 'a' * 64, 'runs': 2, 'deviations': []},
            'dicom_mode': {'passed': True, 'result_sha256': 'b' * 64, 'runs': 2, 'deviations': []},
            'gpu': {'name': 'NVIDIA H100', 'compute_capability': [9, 0]},
            'tolerances': {'map_atol': 0.001 if task == 'umambamtl_cspca_bpmri' else 0.0,
                           'mask_voxel_differences': 0, 'grid_atol_mm': 0.0001},
            'checks': ['gpu_inference', 'unmodified_reference', 'repeatability', 'exact_input_grid',
                       'rtstruct_roundtrip', 'orientation', 'both_acceptance_modes', 'lesion_measurements'],
        }
        import held_mri_runtime
        from unittest.mock import patch
        with patch.object(held_mri, 'pinned_source_hashes', lambda task: src_hashes), patch.object(held_mri_runtime, 'runtime_identity', synthetic_runtime_identity):
            record.update(held_mri.binding(task, root))
        return record, copy.deepcopy(record)
    hashes = {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in prostate_models.weight_files(task)}
    record = {'task': task, 'result': 'passed', 'source_commit': commit,
        'checkpoint_revision': commit if task.startswith('picai_') else
            'Dataset077_ProstateZones/checkpoint_best.pth' if model == 'umu_prostate_zones' else 'zenodo:7040585/tumor.pt',
        'checkpoint_hashes': hashes,
        'checkpoint_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'acceptance_modes': ['reference', 'dicom'], 'deviations': [],
        'reference_mode': {'passed': True, 'result_sha256': 'a' * 64,
                           'runs': 2 if task.startswith('picai_') else 1, 'deviations': []},
        'checks': ['gpu_inference', 'exact_input_grid', 'rtstruct_roundtrip', 'orientation', 'both_acceptance_modes',
                   'author_reference_and_repeatability' if task.startswith('picai_') else 'model_output_contract']}
    orientation = {'task': task, 'result': 'passed'}
    return record, orientation


@pytest.mark.parametrize('task', list(prostate_models.TASKS))
@pytest.mark.parametrize('defect', ['good', 'empty', 'wrong_task', 'failed', 'malformed', 'missing', 'stale_hash',
    'stale_file', 'missing_hash', 'wrong_digest', 'wrong_source', 'installed_source', 'missing_mode',
    'missing_check', 'failed_reference', 'unresolved_failure', 'deviation', 'reference_deviation',
    'orientation_empty', 'orientation_wrong_task', 'orientation_failed', 'orientation_missing', 'no_preflight'])
def test_round2_m1_real_preflight_to_catalogue(portal, tmp_path, monkeypatch, task, defect):
    import model_preflight
    root = tmp_path / 'synthetic-bundle'
    record, orientation = synthetic_acceptance_bundle(root, task)
    if task in prostate_models.HELD_MRI_TASKS:
        import held_mri
        expected_hashes = copy.deepcopy(record['source_file_hashes'])
        monkeypatch.setattr(held_mri, 'pinned_source_hashes', lambda t: copy.deepcopy(expected_hashes))
        monkeypatch.setattr(held_mri, 'gpu_preflight', lambda t, root: {'name':'NVIDIA H100', 'compute_capability':[9,0]})
        import held_mri_runtime
        monkeypatch.setattr(held_mri_runtime, 'runtime_identity', synthetic_runtime_identity)
    model = prostate_models.TASKS[task][0]
    if defect == 'empty': record = {}
    elif defect == 'wrong_task': record['task'] = 'wrong-task'
    elif defect == 'failed': record['result'] = 'failed'
    elif defect == 'stale_hash': record['checkpoint_hashes'][str(prostate_models.weight_files(task)[0])] = '0' * 64
    elif defect == 'stale_file': (root / prostate_models.weight_files(task)[0]).write_text('changed installed bytes')
    elif defect == 'missing_hash': record['checkpoint_hashes'].pop(str(prostate_models.weight_files(task)[0]))
    elif defect == 'wrong_digest': record['checkpoint_sha256'] = '0' * 64
    elif defect == 'wrong_source': record['source_commit'] = '0' * 40
    elif defect == 'installed_source': (root / 'sources' / model / 'RTSEG_MANIFEST.json').write_text('{"commit":"stale"}')
    elif defect == 'missing_mode': record['acceptance_modes'] = ['reference']
    elif defect == 'missing_check': record['checks'].remove('rtstruct_roundtrip')
    elif defect == 'failed_reference': record['reference_mode']['passed'] = False
    elif defect == 'unresolved_failure': record['unresolved_failures'] = ['failed run']
    elif defect == 'deviation': record['deviations'] = [{'check': 'reference.map'}]
    elif defect == 'reference_deviation': record['reference_mode']['deviations'] = [{'check': 'reference.map'}]
    elif defect == 'orientation_empty': orientation = {}
    elif defect == 'orientation_wrong_task': orientation['task'] = 'wrong-task'
    elif defect == 'orientation_failed': orientation['result'] = 'failed'
    marker = root / 'validation' / f'{task}.json'
    marker.write_text('{' if defect == 'malformed' else json.dumps(record))
    if defect == 'missing': marker.unlink()
    marker = root / 'orientation-validation' / f'{task}.json'
    marker.parent.mkdir(exist_ok=True)
    if defect == 'orientation_missing':
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(json.dumps(orientation))
    monkeypatch.setattr(model_preflight, 'MODEL_ROOT', root)
    monkeypatch.setattr(model_preflight, 'RUNTIME_ROOT', root)
    check = model_preflight.audit([task])[task]
    assert check['ready'] == (defect in {'good', 'no_preflight'}), check
    expected_orientation_ready = (not defect.startswith('orientation_')
                                  and not (task in prostate_models.HELD_MRI_TASKS and defect in {'stale_file', 'installed_source'}))
    assert check['orientation_ready'] == expected_orientation_ready
    item = {'name': task, 'enabled': True, **copy.deepcopy(portal.PROSTATE_TASKS[task])}
    portal.apply_model_gates(item, None if defect == 'no_preflight' else check, {task})
    assert item['enabled'] == (defect == 'good' and task != 'picai_nnunet_supervised_bpmri'), item
    if task == 'picai_nnunet_supervised_bpmri':
        assert '390 vs 1,132 voxels' in item['status'] and 'Not enabled' in item['status']


@pytest.mark.parametrize('task,number', [('picai_nnunet_supervised_bpmri', '2201'),
                                        ('picai_nnunet_semisupervised_bpmri', '2203')])
def test_round2_m3_flags_after_trainer_restore_and_unchanged_arguments(monkeypatch, task, number):
    import picai_predict
    import run_picai
    command = run_picai.prediction_command('input path', 'output path', task)
    expected = ['-t', f'Task{number}_picai_baseline', '-i', 'input path', '-o', 'output path',
        '-m', '3d_fullres', '-tr', prostate_models.TRAINER, '--num_threads_preprocessing', '2',
        '--num_threads_nifti_save', '1', '-f', '0', '1', '2', '3', '4', '-chk', 'model_best', '--save_npz']
    assert command == [sys.executable, str(Path(run_picai.__file__).with_name('picai_predict.py')), *expected]
    flags = SimpleNamespace(deterministic=True, benchmark=False)
    restored = object()
    calls = []
    def restore(*args, **kwargs):
        calls.append((args, kwargs))
        flags.deterministic, flags.benchmark = False, True  # Training-time state restored.
        return restored
    predict = SimpleNamespace(load_model_and_checkpoint_files=restore)
    def upstream_main():
        assert sys.argv[1:] == expected
        # Repeated restores must each reapply the policy, preserving all arguments/results.
        for fold in (0, 1):
            assert predict.load_model_and_checkpoint_files('model', [fold], checkpoint_name='model_best') is restored
            assert flags.deterministic is True and flags.benchmark is False
        return 'predicted'
    modules = {'torch': SimpleNamespace(backends=SimpleNamespace(cudnn=flags)),
        'nnunet': SimpleNamespace(), 'nnunet.inference': SimpleNamespace(predict=predict),
        'nnunet.inference.predict_simple': SimpleNamespace(main=upstream_main)}
    for name, module in modules.items(): monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, 'argv', command[1:])
    assert picai_predict.main() == 'predicted'
    assert predict.load_model_and_checkpoint_files is restore
    assert calls == [(('model', [fold]), {'checkpoint_name': 'model_best'}) for fold in (0, 1)]
    assert run_picai.CUDNN_POLICY == {'deterministic': True, 'benchmark': False,
        'applied_after': 'load_model_and_checkpoint_files (trainer restore and initialize)'}


def test_checkpoint_hash_cache_rehashes_changed_files(tmp_path):
    import os
    import prostate_models
    path = tmp_path / 'weights.bin'
    path.write_bytes(b'first')
    first = prostate_models.file_sha256(path)
    assert prostate_models.file_sha256(path) == first
    path.write_bytes(b'second, longer')
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert prostate_models.file_sha256(path) != first


def test_installed_source_commit_reads_git_files_without_git(tmp_path):
    import prostate_models
    commit = 'a' * 40
    detached = tmp_path / 'detached'
    (detached / '.git').mkdir(parents=True)
    (detached / '.git/HEAD').write_text(commit + '\n')
    assert prostate_models.installed_source_commit(detached) == commit
    loose = tmp_path / 'loose'
    (loose / '.git/refs/heads').mkdir(parents=True)
    (loose / '.git/HEAD').write_text('ref: refs/heads/main\n')
    (loose / '.git/refs/heads/main').write_text('b' * 40 + '\n')
    assert prostate_models.installed_source_commit(loose) == 'b' * 40
    packed = tmp_path / 'packed'
    (packed / '.git').mkdir(parents=True)
    (packed / '.git/HEAD').write_text('ref: refs/heads/main\n')
    (packed / '.git/packed-refs').write_text('# pack-refs with: peeled\n' + 'c' * 40 + ' refs/heads/main\n')
    assert prostate_models.installed_source_commit(packed) == 'c' * 40
    manifest = tmp_path / 'manifest'
    manifest.mkdir()
    (manifest / 'RTSEG_MANIFEST.json').write_text('{"commit": "' + 'd' * 40 + '"}')
    assert prostate_models.installed_source_commit(manifest) == 'd' * 40
    broken = tmp_path / 'broken'
    (broken / '.git').mkdir(parents=True)
    (broken / '.git/HEAD').write_text('ref: refs/heads/missing\n')
    with pytest.raises(ValueError, match='Cannot verify installed source commit'):
        prostate_models.installed_source_commit(broken)


@pytest.mark.parametrize('defect',['gpu','interpreter','conversion'])
def test_fix_r1_real_preflight_rejects_changed_execution(tmp_path,monkeypatch,defect):
    import held_mri,held_mri_runtime,model_preflight
    task=held_mri.NCI;root=tmp_path/'bundle'
    record,orientation=synthetic_acceptance_bundle(root,task)
    monkeypatch.setattr(held_mri,'pinned_source_hashes',lambda task:record['source_file_hashes'])
    monkeypatch.setattr(held_mri_runtime,'runtime_identity',synthetic_runtime_identity)
    calls=[]
    def gpu(task,root):
        calls.append(task)
        return {'name':'Quadro P6000','compute_capability':[6,1]} if defect=='gpu' else record['gpu']
    monkeypatch.setattr(held_mri,'gpu_preflight',gpu)
    if defect=='interpreter': (root/'runtime-envs/nci-monai/bin/python').write_text('changed executable, same lock')
    if defect=='conversion':
        fake_app=tmp_path/'app';fake_app.mkdir();(fake_app/'dicom_series_to_nifti.py').write_text('changed conversion')
        monkeypatch.setattr(held_mri,'__file__',str(fake_app/'held_mri.py'))
    for folder,data in [('validation',record),('orientation-validation',orientation)]:
        (root/folder/f'{task}.json').write_text(json.dumps({'schema_version':2,'devices':{'H100':data}}))
    monkeypatch.setattr(model_preflight,'MODEL_ROOT',root)
    monkeypatch.setattr(model_preflight,'RUNTIME_ROOT',root)
    check=model_preflight.audit([task])[task]
    assert calls
    assert check['ready'] is check['acceptance_ready'] is check['orientation_ready'] is False
