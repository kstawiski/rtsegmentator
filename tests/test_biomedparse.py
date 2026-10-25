"""Offline contracts: synthetic DICOM and pinned public-example prompt metadata only."""
import copy
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from fastapi import HTTPException

import biomedparse as bp
from run_biomedparse import preprocess

ROOT = Path(__file__).parents[1]
GPU = {'compute_capability': [9, 0], 'name': 'SYNTHETIC H100', 'total_memory': 80 * 1024**3,
       'torch': 'synthetic', 'compiled_arches': 'sm_60 sm_90'}


@pytest.mark.parametrize('modality,text', [('CT', 'spleen'), ('MR', 'brain tumor'), ('PT', 'whole-body lesion'),
    ('CT', 'CT imaging of the spleen within the abdomen')])
def test_supported_prompts(modality, text):
    assert bp.validate_prompt(text, modality)['text'] == text


@pytest.mark.parametrize('modality,text', [('CT', 'unknown thing'), ('MR', 'kidney cyst'), ('PT', 'spleen'),
    ('CT', 'spleen[SEP]liver'), ('CT', ''), ('CT', None), ('CT', 'liver\n'), ('US', 'spleen')])
def test_unsupported_prompts(modality, text):
    with pytest.raises(ValueError):
        bp.validate_prompt(text, modality)


def test_prompt_http_validation(portal):
    items = {bp.TASK: {'prompt': 'text'}}
    selected = {'columns': 32, 'rows': 32, 'slices': 12, 'modality': 'CT'}
    result = portal.validate_assessed_prompt([bp.TASK], items, selected, '{"text":"spleen"}')
    assert result['biomedparse']['target'] == 'spleen'
    with pytest.raises(HTTPException, match='Unsupported'):
        portal.validate_assessed_prompt([bp.TASK], items, selected, '{"text":"unknown"}')


def test_preprocessing_windows_and_percentiles():
    image = np.array([-200, -160, 40, 240, 300, 500, -100, 10]).reshape(2, 2, 2)
    out, record = preprocess(image, 'CT', 'soft_tissue')
    assert record['bounds'] == [-160.0, 240.0]
    assert out.flatten().tolist()[:5] == [0, 0, 127, 255, 255]
    mr, record = preprocess(image, 'MR')
    assert record['percentiles'] == [0.5, 99.5]
    assert mr.min() == 0 and mr.max() == 255


@pytest.mark.parametrize('modality,lower,upper,window,expected_range', [
    ('PT', 0, 0.9, None, (0, 255)),
    ('MR', 10, 80, None, (0, 255)),
    ('CT', 0, 255, 'lung', (154, 198)),
])
def test_raw_low_values_always_normalized(modality, lower, upper, window, expected_range):
    image = np.linspace(lower, upper, 24).reshape(2, 3, 4)
    out, record = preprocess(image, modality, window)
    assert (out.min(), out.max()) == expected_range
    assert record['method'] == ('CT_window' if modality == 'CT' else 'percentile')
    if modality == 'CT':
        assert record['window'] == 'lung' and record['bounds'] == [-910, 590]


@pytest.mark.parametrize('modality,value', [('CT', 40), ('MR', 10), ('PT', 0.1), ('MR', 300)])
def test_raw_constant_image_rejected(modality, value):
    with pytest.raises(ValueError, match='constant image'):
        preprocess(np.full((2, 3, 4), value), modality, 'soft_tissue' if modality == 'CT' else None)


def test_npz_keeps_preprocessed_pixels(tmp_path, monkeypatch):
    """Exercise main's explicit NPZ branch with model and CUDA calls stubbed."""
    import run_biomedparse as runner
    from types import SimpleNamespace
    image = np.linspace(0, 0.9, 24, dtype=np.float32).reshape(2, 3, 4)
    example, prompt = tmp_path / 'case.npz', tmp_path / 'prompt.json'
    np.savez_compressed(example, imgs=image)
    prompt.write_text('{"text":"spleen"}')
    output, sidecar = tmp_path / 'out.npz', tmp_path / 'out.json'
    args = ['run_biomedparse.py', '--npz', str(example), '--output', str(output),
            '--sidecar', str(sidecar), '--prompt', str(prompt), '--modality', 'CT',
            '--source', str(tmp_path / bp.SOURCE), '--checkpoint', str(tmp_path / bp.CHECKPOINT),
            '--tokenizer', str(tmp_path / bp.TOKENIZER), '--bundle-root', str(tmp_path)]
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(runner, 'binding', lambda *a: {})
    monkeypatch.setattr(runner, 'preprocess', lambda *a: pytest.fail('NPZ must bypass raw normalization'))
    monkeypatch.setattr(runner, 'seed_inference', lambda: None)
    monkeypatch.setattr(runner, 'load_model', lambda *a: object())
    def infer(model, received, text):
        assert np.array_equal(received, image) and received.dtype == image.dtype
        return np.zeros_like(image, dtype=np.uint8), [0.1, 0.2], 0
    monkeypatch.setattr(runner, 'infer', infer)
    cuda = SimpleNamespace(is_available=lambda: True,
        get_device_properties=lambda *a: SimpleNamespace(total_memory=GPU['total_memory']),
        get_device_capability=lambda *a: (9, 0), reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    runner.main()
    assert json.loads(sidecar.read_text())['preprocessing'] == {'method': 'upstream_preprocessed_npz'}
    with np.load(output) as data:
        assert not data['segs'].any()


def test_every_pinned_ct_mr_example_prompt():
    examples = Path(os.environ.get('RTSEG_BIOMEDPARSE_EXAMPLES',
        str(Path(os.environ.get('RTSEG_MODEL_ROOT', '/models')) / bp.SOURCE / 'examples')))
    if not examples.is_dir():
        pytest.skip('Set RTSEG_BIOMEDPARSE_EXAMPLES to the pinned upstream examples directory')
    manifest = bp.vocabulary()
    excluded = manifest['excluded_example_prompts']
    seen_exclusions = set()
    count = 0
    paths = sorted((examples / 'imgs').glob('*.npz'))
    assert paths, 'No pinned NPZ examples found'
    for path in paths:
        modality = path.name.split('_')[0]
        if modality not in {'CT', 'MR'}:
            continue
        assert bp.file_sha256(path) == manifest['examples'][path.name]['sha256']
        # Only load the tiny prompt member. Never deserialize a patient file.
        with np.load(path, allow_pickle=True) as data:
            prompts = data['text_prompts'].item()
        for label, text in prompts.items():
            if label == 'instance_label':
                continue
            exclusion = excluded.get(path.name, {}).get(label)
            if exclusion:
                assert text == exclusion['text'] and exclusion['reason']
                with pytest.raises(ValueError, match='Unsupported'):
                    bp.validate_prompt(text, modality)
                seen_exclusions.add((path.name, label))
                continue
            assert bp.validate_prompt(text, modality)['text'] == text
            count += 1
    assert count == 41  # AMOS 14, TotalSeg 23, BraTS 4 single-target prompts.
    assert seen_exclusions == {(name, label) for name, rows in excluded.items() for label in rows}


def test_existence_threshold_and_empty():
    prompt = bp.validate_prompt('spleen', 'CT')
    result = bp.result_record(prompt, [0.2, 0.5], 0, slice_axis=0, preprocessing={})
    assert result['message'] == 'object not detected'
    assert result['empty'] and result['existence']['decision'] is False
    positive = bp.result_record(prompt, [0.5001], 0, slice_axis=1, preprocessing={})
    assert positive['message'] == 'empty segmentation despite positive slice existence'
    with pytest.raises(ValueError):
        bp.result_record(prompt, [0.1], 2, slice_axis=0, preprocessing={})
    assert bp.roi_name('a/b: test') == 'a_b_test'
    alias = bp.validate_prompt('CT imaging of the spleen within the abdomen', 'CT')
    named = bp.result_record(alias, [0.9], 100, slice_axis=0, preprocessing={})
    assert named['roi_name'] == 'CT_imaging_of_the_spleen_within_the_abdomen'
    assert len(bp.roi_name('x' * 80)) == 64


def test_result_allowlist_and_empty_zip(portal, tmp_path, monkeypatch):
    job = tmp_path / 'job'
    (job / 'output').mkdir(parents=True)
    result = bp.result_record(bp.validate_prompt('spleen', 'CT'), [0.1], 0, slice_axis=0, preprocessing={})
    (job / 'output' / bp.artifact_names(bp.TASK)[0]).write_text(json.dumps(result))
    (job / 'job.json').write_text(json.dumps({'models': [bp.TASK]}))
    monkeypatch.setattr(portal, 'save_metadata', lambda *a, **k: None)
    portal.finalize_results(job, empty=[bp.TASK])
    import zipfile
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as archive:
        assert bp.artifact_names(bp.TASK)[0] in archive.namelist()
        assert 'NO_STRUCTURES_FOUND.txt' in archive.namelist()
        assert 'object not detected' in archive.read('CONVERSION_NOTES.txt').decode()


@pytest.mark.parametrize('empty', [False, True])
def test_dispatch(tmp_path, monkeypatch, synthetic_ct, empty):
    job = tmp_path / 'job'
    (job / 'output').mkdir(parents=True)
    (job / 'input').symlink_to(synthetic_ct)
    (job / 'prompt.json').write_text('{"text":"spleen"}')
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(job), bp.TASK])
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    nifti = job / 'external' / bp.TASK / 'input.nii.gz'
    subprocess.run([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(synthetic_ct), str(nifti)],
                   check=True, capture_output=True, text=True)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            if command[1].endswith('run_biomedparse.py'):
                sidecar = Path(command[command.index('--sidecar') + 1])
                sidecar.write_text(json.dumps(bp.result_record(bp.validate_prompt('spleen', 'CT'),
                    [0.1] if empty else [0.9], 0 if empty else 100, slice_axis=0, preprocessing={})))
            if command[1].endswith('convert_validate_rtstruct.py'):
                assert json.loads(Path(command[command.index('--labels') + 1]).read_text()) == {'1': 'spleen'}
                import shutil
                shutil.copyfile(next(synthetic_ct.iterdir()), command[command.index('--output') + 1])
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, 'tempdir', f'/proc/self/fd/{fd}')
        if empty:
            with pytest.raises(SystemExit, match='object not detected: empty prediction'):
                runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
        else:
            runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    finally:
        os.close(fd)
    expected = ['dicom_series_to_nifti.py', 'run_biomedparse.py']
    assert [Path(c[1]).name for c in calls] == expected + ([] if empty else ['convert_validate_rtstruct.py'])
    command = calls[1]
    assert Path(command[0]) == tmp_path / 'models' / bp.ENV / 'bin/python'
    assert command[command.index('--modality') + 1] == 'CT'
    assert command[command.index('--prompt') + 1] == str(job / 'prompt.json')


@pytest.mark.parametrize('orientation', [[-1, 0, 0, 0, -1, 0], [0, 1, 0, 1, 0, 0]])
@pytest.mark.parametrize('empty', [True, False])
def test_valid_reoriented_input_reaches_inference(tmp_path, monkeypatch, synthetic_ct, orientation, empty):
    import pydicom
    for path in synthetic_ct.iterdir():
        ds = pydicom.dcmread(path)
        ds.ImageOrientationPatient = orientation
        ds.save_as(path, enforce_file_format=True)
    job = tmp_path / 'job'
    (job / 'output').mkdir(parents=True)
    (job / 'input').symlink_to(synthetic_ct)
    (job / 'prompt.json').write_text('{"text":"spleen"}')
    nifti = job / 'external' / bp.TASK / 'input.nii.gz'
    # The real converter stores this valid series as an exact flip/permutation of the DICOM grid.
    subprocess.run([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(synthetic_ct), str(nifti)],
                   check=True, capture_output=True, text=True)
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(job), bp.TASK])
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            if command[1].endswith('run_biomedparse.py'):
                Path(command[command.index('--sidecar') + 1]).write_text(json.dumps(
                    bp.result_record(bp.validate_prompt('spleen', 'CT'), [0.1] if empty else [0.9],
                                     0 if empty else 100, slice_axis=0, preprocessing={})))
            if command[1].endswith('convert_validate_rtstruct.py'):
                import shutil
                shutil.copyfile(next(synthetic_ct.iterdir()), command[command.index('--output') + 1])
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, 'tempdir', f'/proc/self/fd/{fd}')
        if empty:
            with pytest.raises(SystemExit, match='object not detected: empty prediction'):
                runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
        else:
            runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    finally:
        os.close(fd)
    expected = ['dicom_series_to_nifti.py', 'run_biomedparse.py']
    assert [Path(c[1]).name for c in calls] == expected + ([] if empty else ['convert_validate_rtstruct.py'])


def test_shifted_model_input_rejected_before_inference(tmp_path, monkeypatch, synthetic_ct):
    import nibabel as nib
    job = tmp_path / 'job'
    (job / 'output').mkdir(parents=True)
    (job / 'input').symlink_to(synthetic_ct)
    (job / 'prompt.json').write_text('{"text":"spleen"}')
    nifti = job / 'external' / bp.TASK / 'input.nii.gz'
    subprocess.run([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(synthetic_ct), str(nifti)],
                   check=True, capture_output=True, text=True)
    real = Path(nifti).with_name('real.nii.gz')
    Path(nifti).rename(real)
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(job), bp.TASK])
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            if command[1].endswith('dicom_series_to_nifti.py'):
                # A converter output shifted by 5 mm is not an exact flip/permutation.
                image = nib.load(real)
                affine = image.affine.copy()
                affine[0, 3] += 5
                nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine, image.header), command[3])
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, 'tempdir', f'/proc/self/fd/{fd}')
        with pytest.raises(ValueError, match='not an exact permutation/flip'):
            runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    finally:
        os.close(fd)
    assert [Path(c[1]).name for c in calls] == ['dicom_series_to_nifti.py']


@pytest.mark.parametrize('modality', ['CT', 'MR', 'PT'])
@pytest.mark.parametrize('empty', [True, False])
def test_mixed_frame_rejected_before_inference(tmp_path, monkeypatch, synthetic_ct, modality, empty):
    import pydicom
    from pydicom.uid import generate_uid
    for path in synthetic_ct.iterdir():
        ds = pydicom.dcmread(path)
        ds.Modality = modality
        ds.save_as(path, enforce_file_format=True)
    last = sorted(synthetic_ct.iterdir())[-1]
    ds = pydicom.dcmread(last)
    ds.FrameOfReferenceUID = generate_uid()
    ds.save_as(last, enforce_file_format=True)
    job = tmp_path / 'job'
    (job / 'output').mkdir(parents=True)
    (job / 'input').symlink_to(synthetic_ct)
    (job / 'prompt.json').write_text(json.dumps({'text': 'whole-body lesion' if modality == 'PT' else 'spleen'}))
    nifti = job / 'external' / bp.TASK / 'input.nii.gz'
    # The real converter accepts this malformed series; dispatch must reject it
    # before it can consume either an existence-negative or a nonempty result.
    subprocess.run([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(synthetic_ct), str(nifti)],
                   check=True, capture_output=True, text=True)
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(job), bp.TASK])
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            if command[1].endswith('run_biomedparse.py'):
                prompt = bp.validate_prompt('whole-body lesion' if modality == 'PT' else 'spleen', modality)
                Path(command[command.index('--sidecar') + 1]).write_text(json.dumps(
                    bp.result_record(prompt, [0.1] if empty else [0.9], 0 if empty else 100,
                                     slice_axis=0, preprocessing={})))
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, 'tempdir', f'/proc/self/fd/{fd}')
        with pytest.raises(ValueError, match='Inconsistent or multi-frame DICOM source images'):
            runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    finally:
        os.close(fd)
    assert [Path(c[1]).name for c in calls] == ['dicom_series_to_nifti.py']
    assert not list((job / 'output').iterdir())


@pytest.fixture
def installed(tmp_path, monkeypatch):
    source = tmp_path / bp.SOURCE
    source.mkdir(parents=True)
    (source / 'RTSEG_MANIFEST.json').write_text(json.dumps({'commit': bp.COMMIT}))
    (source / 'inference.py').write_bytes(b'synthetic upstream')
    checkpoint = tmp_path / bp.CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic checkpoint')
    monkeypatch.setattr(bp, 'CHECKPOINT_SHA256', bp.file_sha256(checkpoint))
    app = tmp_path / 'app'
    (app / 'catalog').mkdir(parents=True)
    for p in bp.APP_FILES:
        path = app / p
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic adapter: ' + p)
    (app / 'catalog/biomedparse-source.json').write_text(json.dumps({'inference.py': bp.file_sha256(source / 'inference.py')}))
    for relative in [bp.ENV / 'requirements.lock', *[bp.TOKENIZER / p for p in bp.TOKENIZER_FILES]]:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'synthetic installed artifact')
    (app / 'catalog/biomedparse-tokenizer.json').write_text(json.dumps({'sha256': {p: bp.file_sha256(tmp_path / bp.TOKENIZER / p) for p in bp.TOKENIZER_FILES}}))
    return tmp_path, app


def marker(root, app):
    return {'schema': 1, 'task': bp.TASK, 'result': 'passed', 'checks': bp.CHECKS, 'seed': bp.SEED,
            'criterion': 'exact', 'mask_voxel_differences': 0, 'device': 'sm90', 'gpu': GPU,
            'public_origin': {'source': 'synthetic-test'},
            **{mode: {'passed': True, 'runs': runs, 'rois': 1}
               for mode, runs in [('reference', 1), ('repeatability', 2), ('dicom', 2)]},
            **bp.binding(root, app)}


def test_marker_binding(installed):
    root, app = installed
    record = marker(root, app)
    bp.validate_marker(record, root, GPU, app)
    for field, value in [('source_commit', 'f' * 40), ('checkpoint_sha256', '0' * 64),
                         ('device', 'sm61'), ('checks', []), ('deviations', ['waived']),
                         ('criterion', 'dice_0.9'), ('mask_voxel_differences', 1),
                         ('reference', {'passed': True, 'runs': 2}), ('nested', {'passed': False})]:
        changed = copy.deepcopy(record)
        changed[field] = value
        with pytest.raises(ValueError): bp.validate_marker(changed, root, GPU, app)
    (root / bp.CHECKPOINT).write_bytes(b'replaced checkpoint')
    with pytest.raises(ValueError, match='checkpoint'): bp.validate_marker(record, root, GPU, app)


@pytest.mark.parametrize('mode,runs', [('reference', 0), ('reference', 2),
    ('repeatability', 1), ('repeatability', 3), ('dicom', 1), ('dicom', 3)])
def test_each_execution_count_is_required(installed, mode, runs):
    root, app = installed
    record = marker(root, app)
    record[mode]['runs'] = runs
    with pytest.raises(ValueError, match=mode):
        bp.validate_marker(record, root, GPU, app)


def test_gate_uses_persistent_digest_cache(installed, tmp_path_factory, monkeypatch):
    root, app = installed
    import prostate_models as pm
    cache = tmp_path_factory.mktemp('digest-cache')
    cache.chmod(0o700)
    monkeypatch.setenv(pm.PREFLIGHT_CACHE_ENV, str(cache))
    record = marker(root, app)
    with pm.preflight_digest_cache(root, app):
        bp.validate_marker(record, root, GPU, app)
    pm._SHA256_CACHE.clear()
    monkeypatch.setattr(pm, '_hash_file_bytes', lambda *a, **k: pytest.fail('warm preflight reread bytes'))
    with pm.preflight_digest_cache(root, app):
        bp.validate_marker(record, root, GPU, app)


def test_catalogue_and_structure_contract():
    from scripts.build_structure_catalog import registries
    entry = registries()['EXTERNAL_TASKS'][bp.TASK]
    assert entry['requires_preflight'] and entry['status'] == 'acceptance_required'
    assert bp.TASK not in registries()['INFORMATIONAL_MODELS']
    assert bp.artifact_names('other') == []
    assert bp.marker_path('/models', GPU).name == 'biomedparse_prompted-sm90.json'
    with pytest.raises(ValueError): bp.require_device({'compute_capability': [8, 0]})


def test_assessed_job_keeps_text(portal, monkeypatch, synthetic_ct):
    from fastapi.testclient import TestClient
    item = {**portal.EXTERNAL_TASKS[bp.TASK], 'name': bp.TASK, 'enabled': True}
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': [item]})
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda *a, **k: None)
    client = TestClient(portal.app)
    uploaded = client.post('/api/uploads', files={'files': ('image.dcm', next(synthetic_ct.iterdir()).read_bytes())})
    data = uploaded.json()
    response = client.post(f"/api/uploads/{data['upload_id']}/start", data={
        'series_key': data['series'][0]['key'], 'models': bp.TASK, 'prompt_json': '{"text":"spleen"}'})
    assert response.status_code == 200, response.text
    job = portal.BASE / response.json()['id']
    assert json.loads((job / 'prompt.json').read_text())['text'] == 'spleen'
    assert json.loads((job / 'job.json').read_text())['prompt']['biomedparse']['target'] == 'spleen'


@pytest.mark.parametrize('relative', ['runtime-envs/biomedparse-v2/requirements.lock',
    'sources/BiomedParse/inference.py', 'weights/biomedparse/clip-vit-base-patch32/vocab.json', 'app/run_biomedparse.py'])
def test_changed_installation_invalidates_marker(installed, relative):
    root, app = installed
    record = marker(root, app)
    (root / relative).write_text('changed synthetic artifact')
    with pytest.raises(ValueError):
        bp.validate_marker(record, root, GPU, app)


def test_preflight_requires_both_current_device_markers(installed, monkeypatch, tmp_path_factory):
    import model_preflight as preflight
    root, app = installed
    for p in preflight.COMMON:
        (app / p).touch()
    (root / bp.ENV / 'bin').mkdir()
    (root / bp.ENV / 'bin/python').touch()
    monkeypatch.setattr(preflight, 'MODEL_ROOT', root)
    monkeypatch.setattr(preflight, 'APP_ROOT', app)
    monkeypatch.setattr(preflight, 'RUNTIME_ROOT', root)
    monkeypatch.setattr(bp, 'probe_device', lambda python: GPU)
    cache = tmp_path_factory.mktemp('preflight-cache')
    cache.chmod(0o700)
    monkeypatch.setenv('RTSEG_HELD_MRI_PREFLIGHT_CACHE', str(cache))
    record = marker(root, app)
    assert not preflight.audit([bp.TASK])[bp.TASK]['ready']
    for folder in ('validation', 'orientation-validation'):
        p = bp.marker_path(root, GPU, folder)
        p.parent.mkdir()
        p.write_text(json.dumps(record))
    assert preflight.audit([bp.TASK])[bp.TASK]['ready']
    record['device'] = 'sm61'
    bp.marker_path(root, GPU).write_text(json.dumps(record))
    assert not preflight.audit([bp.TASK])[bp.TASK]['ready']


@pytest.mark.parametrize('failure', ['missing_example', 'initial_probe', 'device_mismatch'])
def test_failed_harness_removes_markers(tmp_path, monkeypatch, failure):
    import acceptance_biomedparse as harness
    from types import SimpleNamespace
    monkeypatch.setattr(harness, 'binding', lambda *a: {})
    def probe(*a):
        for folder in ('validation', 'orientation-validation'):
            p = bp.marker_path(tmp_path, GPU, folder)
            assert not p.exists() and not p.with_suffix('.json.tmp').exists()
        if failure == 'initial_probe':
            raise subprocess.CalledProcessError(1, ['python', '-c', 'import detectron2'])
        return {**GPU, 'compute_capability': [6, 1]} if failure == 'device_mismatch' else GPU
    monkeypatch.setattr(harness, 'probe_device', probe)
    expected_error = {'missing_example': FileNotFoundError, 'initial_probe': subprocess.CalledProcessError,
                      'device_mismatch': ValueError}[failure]
    for folder in ('validation', 'orientation-validation'):
        p = bp.marker_path(tmp_path, GPU, folder)
        p.parent.mkdir()
        p.write_text('{"result":"passed"}')
        p.with_suffix('.json.tmp').write_text('{"result":"passed"}')
    # No example is installed. Failure must remove a previous marker, not reuse it.
    with pytest.raises(expected_error):
        harness.acceptance(SimpleNamespace(bundle_root=tmp_path, work=tmp_path / 'work',
                                            worker_python=sys.executable, device='sm90'))
    assert all(not bp.marker_path(tmp_path, GPU, f).exists() for f in ('validation', 'orientation-validation'))
    assert all(not bp.marker_path(tmp_path, GPU, f).with_suffix('.json.tmp').exists()
               for f in ('validation', 'orientation-validation'))


def test_public_fixture_writer_with_synthetic_pixels_and_strict_export(tmp_path):
    import nibabel as nib
    from acceptance_biomedparse import build_dicom
    image = np.arange(8 * 24 * 32, dtype=np.uint8).reshape(8, 24, 32)
    dicom = tmp_path / 'dicom'
    build_dicom(image, dicom)
    model_input = tmp_path / 'input.nii.gz'
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
    subprocess.run([sys.executable, str(ROOT / 'dicom_series_to_nifti.py'), str(dicom), str(model_input)],
                   env=env, check=True, capture_output=True, text=True)
    volume = nib.load(model_input)
    assert nib.aff2axcodes(volume.affine) == ('L', 'P', 'S')
    raw = np.asarray(volume.dataobj).transpose(2, 1, 0)
    assert raw.min() < 0 and raw.max() > 240
    recovered, record = preprocess(raw, 'CT', 'soft_tissue')
    assert record['method'] == 'CT_window'
    assert np.array_equal(recovered, image)
    mask = np.zeros(volume.shape, dtype=np.uint8)
    mask[8:24, 6:18, 2:6] = 1
    prediction = tmp_path / 'prediction.nii.gz'
    nib.save(nib.Nifti1Image(mask, volume.affine), prediction)
    labels = tmp_path / 'labels.json'
    labels.write_text('{"1":"spleen"}')
    cp = subprocess.run([sys.executable, str(ROOT / 'convert_validate_rtstruct.py'), '--prediction', str(prediction),
        '--model-input', str(model_input), '--dicom', str(dicom), '--output', str(tmp_path / 'output.dcm'),
        '--labels', str(labels)], env=env, check=True, capture_output=True, text=True)
    validation = json.loads(cp.stdout.splitlines()[-1])
    assert validation['valid'] and validation['rois'] == 1


@pytest.mark.parametrize('window', [[], {}, False, '', 'unknown'])
def test_rejects_invalid_window(window):
    with pytest.raises(ValueError, match='CT window'):
        bp.validate_prompt('spleen', 'CT', window)


def test_portal_gate_status_tracks_acceptance(portal):
    item = {'name': bp.TASK, **portal.EXTERNAL_TASKS[bp.TASK]}
    portal.apply_model_gates(item, None, {bp.TASK})
    assert not item['enabled'] and item['status'] == 'preflight_unavailable'
    item = {'name': bp.TASK, **portal.EXTERNAL_TASKS[bp.TASK]}
    portal.apply_model_gates(item, {'ready': True, 'orientation_ready': True, 'acceptance_ready': True}, {bp.TASK})
    assert item['enabled'] and item['status'] == 'technical_checks_recorded'
