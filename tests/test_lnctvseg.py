"""LNCTVSeg label contract, six-ROI conversion, job notes and acceptance gates."""
import json
import os
import pickle
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import pytest

import lnctvseg_bundle as bundle

ROOT = Path(__file__).parents[1]


def accepted():
    return dict(task=bundle.TASK, result='passed', source_commit=bundle.manifest()['source_commit'],
                manifest_sha256=bundle.manifest_sha256(), laterality='passed')


def test_numeric_label_map_and_catalog(portal):
    labels = json.loads((ROOT / 'labels/lnctvseg.json').read_text())
    assert labels == {'1': 'LN CTV R II-III-Va', '2': 'LN CTV L II-III-Va',
                      '3': 'LN CTV R IV-Vb', '4': 'LN CTV L IV-Vb',
                      '5': 'LN CTV R Ib', '6': 'LN CTV L Ib'}
    assert all(name.isascii() and len(name) <= 64 for name in labels.values())
    row = portal.EXTERNAL_TASKS[bundle.TASK]
    assert row['modality'] == 'CT' and row['requires_preflight']
    assert 'not individual levels' in row['description']
    assert 'contrast' in row['limitations'] and 'institutional' in row['limitations']
    assert 'undocumented seventh class that is not exported' in row['limitations']
    assert 'blocked_status' not in row
    for suffix in ('swinunetr', '3duxnet', 'repuxnet'):
        entry = portal.EXTERNAL_TASKS[f'lnctvseg_{suffix}']
        assert entry['enabled'] is False and entry['limitations']


def test_public_fixture_assignments():
    fixtures = json.loads((ROOT / 'catalog/orientation-audit-fixtures.json').read_text())
    for task in (bundle.TASK, 'hnlnl_2d3d', 'lnsegfm_nnunet', 'cads_open_558_headneck_oars'):
        assert fixtures[task] == 'lnctvseg-example-hn-ct/dicom'
    assert bundle.TASK in json.loads((ROOT / 'catalog/orientation-audit-tasks.json').read_text())


def test_preflight_checks_each_release_file_and_marker(tmp_path, monkeypatch):
    import model_preflight as module
    model, runtime = tmp_path / 'model', tmp_path / 'runtime'
    monkeypatch.setattr(module, 'MODEL_ROOT', model)
    monkeypatch.setattr(module, 'RUNTIME_ROOT', runtime)
    monkeypatch.setattr(module, 'APP_ROOT', ROOT)
    requirements = module.REQUIREMENTS[bundle.TASK]
    assert 'envs/nnunet-v1/bin/python' in requirements
    assert '@labels/lnctvseg.json' in requirements
    for requirement in requirements:
        if requirement.startswith('@') or requirement.startswith('validation/'):
            continue
        path = (runtime if requirement.startswith('envs/') else model) / requirement
        path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    row = module.audit([bundle.TASK])[bundle.TASK]
    assert not row['ready'] and not row['acceptance_ready'] and not row['orientation_ready']
    marker = model / 'validation' / f'{bundle.TASK}.json'; marker.parent.mkdir()
    for value in ['null', '{', json.dumps({**accepted(), 'manifest_sha256': 'stale'}),
                  json.dumps({**accepted(), 'laterality': 'failed'})]:
        marker.write_text(value)
        assert not module.audit([bundle.TASK])[bundle.TASK]['ready']
    marker.write_text(json.dumps(accepted()))
    row = module.audit([bundle.TASK])[bundle.TASK]
    assert row['ready'] and row['acceptance_ready'] and not row['orientation_ready']
    (runtime / 'envs/nnunet-v1/bin/python').unlink()
    assert not module.audit([bundle.TASK])[bundle.TASK]['ready']


def test_assets_are_hashed_before_loading(tmp_path, monkeypatch):
    import hashlib
    file = tmp_path / 'weights/model'; file.parent.mkdir(); file.write_bytes(b'pinned weights')
    spec = {'nnunet': {'folds': [0]}, 'files': {'weights/model': {
        'bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}}}
    monkeypatch.setattr(bundle, 'manifest', lambda: spec)
    assert bundle.validate_assets(tmp_path) == spec
    file.write_bytes(b'changed weight')
    with pytest.raises(ValueError, match='checksum'):
        bundle.validate_assets(tmp_path)


def dispatch(tmp_path, monkeypatch, synthetic_ct, audit=False, marker=True, modality='CT'):
    job, model = tmp_path / 'job', tmp_path / 'model'
    (job / 'output').mkdir(parents=True)
    (job / 'input').symlink_to(synthetic_ct)
    for file in synthetic_ct.iterdir():
        ds = pydicom.dcmread(file); ds.Modality = modality; ds.save_as(file, enforce_file_format=True)
    if marker:
        path = model / 'validation' / f'{bundle.TASK}.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps(accepted()))
    for key, value in {'RTSEG_APP_ROOT': ROOT, 'RTSEG_MODEL_ROOT': model,
                       'RTSEG_RUNTIME_ROOT': tmp_path / 'runtime', 'RTSEG_SCRATCH_ROOT': tmp_path / 'scratch'}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(signal, 'signal', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(job), bundle.TASK] +
                        (['--nodal-acceptance-audit'] if audit else []))
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            if command[1] == str(ROOT / 'run_lnctvseg.py'):
                prediction = Path(command[command.index('--output') + 1]) / 'case.nii.gz'
                nib.save(nib.Nifti1Image(np.arange(8, dtype=np.uint8).reshape(2, 2, 2),
                                       np.diag([2., 2., 3., 1.])), prediction)
            if command[1] == str(ROOT / 'convert_validate_rtstruct.py'):
                ds = pydicom.dcmread(next(synthetic_ct.iterdir()))
                ds.save_as(command[command.index('--output') + 1], enforce_file_format=True)
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, 'tempdir', f'/proc/self/fd/{fd}')
        runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    finally:
        os.close(fd)
    return job, calls


def test_dispatch_reuses_v1_runtime_and_converter(tmp_path, monkeypatch, synthetic_ct):
    job, calls = dispatch(tmp_path, monkeypatch, synthetic_ct)
    assert [Path(command[1]).name for command, _ in calls] == [
        'dicom_series_to_nifti.py', 'run_lnctvseg.py', 'convert_validate_rtstruct.py']
    assert calls[1][0][0].endswith('/envs/nnunet-v1/bin/python')
    assert calls[2][0][-1] == str(ROOT / 'labels/lnctvseg.json')
    assert Path(calls[2][0][calls[2][0].index('--prediction') + 1]).parts[-2:] == ('export', 'case.nii.gz')
    assert (job / 'output' / bundle.TASK / 'OMITTED_LABELS.txt').read_text().strip() == (
        'LNCTVSeg class 7 (undocumented by the authors) omitted: 1 voxels, 0.0120 mL')
    assert not (job / 'external' / bundle.TASK).exists()
    assert pydicom.dcmread(job / 'output' / f'{bundle.TASK}_RTSTRUCT.dcm').StructureSetName == f'RTSEG: {bundle.TASK}'


def test_audit_bootstrap_retains_predictions(tmp_path, monkeypatch, synthetic_ct):
    job, calls = dispatch(tmp_path, monkeypatch, synthetic_ct, audit=True, marker=False)
    assert len(calls) == 3 and (job / 'external' / bundle.TASK).is_dir()
    work = job / 'external' / bundle.TASK
    raw = np.asarray(nib.load(work / 'prediction/case.nii.gz').dataobj)
    exported = np.asarray(nib.load(work / 'export/case.nii.gz').dataobj)
    assert set(np.unique(raw)) == set(range(8))
    assert set(np.unique(exported)) == set(range(7))
    np.testing.assert_array_equal(exported[raw != 7], raw[raw != 7])
    assert np.all(exported[raw == 7] == 0)


def test_dispatch_refuses_missing_acceptance(tmp_path, monkeypatch, synthetic_ct):
    with pytest.raises(FileNotFoundError):
        dispatch(tmp_path, monkeypatch, synthetic_ct, marker=False)


def test_dispatch_refuses_mri(tmp_path, monkeypatch, synthetic_ct):
    with pytest.raises(ValueError, match='requires one CT'):
        dispatch(tmp_path, monkeypatch, synthetic_ct, modality='MR')


def test_v1_runner_uses_release_settings_and_writable_model_directories(tmp_path, monkeypatch):
    import run_lnctvseg
    from types import ModuleType, SimpleNamespace
    model = tmp_path / 'bundle'; source = model / 'weights/lnctvseg/release'
    source.mkdir(parents=True)
    (source / 'plans.pkl').write_bytes(b'plans')
    (source / 'model_final_checkpoint.model').write_bytes(b'checkpoint')
    metadata = {'init': ('/training/plans.pkl', 'all', '/training/output', '/training/data',
                         True, 1, True, False, True), 'plans': {'stage': 1}}
    original = pickle.dumps(metadata)
    (source / 'model_final_checkpoint.model.pkl').write_bytes(original)
    release = {'model_directory': 'weights/lnctvseg/release', 'folds': ['all'],
               'checkpoint': 'model_final_checkpoint', 'tta': True, 'output_label_ids': list(range(8))}
    monkeypatch.setattr(run_lnctvseg, 'validate_assets', lambda root: {
        'source_commit': 'pinned', 'nnunet': release, 'label_contract': {'status': 'unresolved'}})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda n: None,
                        cuda=SimpleNamespace(is_available=lambda: False)))
    predict = ModuleType('nnunet.inference.predict'); calls = []
    def inference(*args, **kwargs):
        calls.append((args, kwargs))
        staged = Path(args[0])
        assert not staged.is_symlink() and not (staged / 'all').is_symlink()
        assert not (staged / 'plans.pkl').is_symlink()
        assert (staged / 'all/model_final_checkpoint.model').is_symlink()
        local = pickle.loads((staged / 'all/model_final_checkpoint.model.pkl').read_bytes())
        assert local['plans'] == metadata['plans']
        assert local['init'][0] == str(staged / 'plans.pkl')
        assert local['init'][2] == str(staged)
        assert local['init'][3] == str(tmp_path / 'work/preprocessed')
        assert local['init'][4:] == metadata['init'][4:]
        (staged / 'all/runtime-output.json').write_text('{}')
    predict.predict_from_folder = inference
    for name in ('nnunet', 'nnunet.inference'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, 'nnunet.inference.predict', predict)
    for key in ('RESULTS_FOLDER', 'nnUNet_raw_data_base', 'nnUNet_preprocessed'):
        monkeypatch.setenv(key, 'old')
    monkeypatch.setattr(sys, 'argv', ['run_lnctvseg.py', '--input', str(tmp_path / 'input'),
        '--output', str(tmp_path / 'output'), '--work', str(tmp_path / 'work'), '--model-root', str(model)])
    run_lnctvseg.main()
    assert calls[0][0][3] == ['all']
    assert calls[0][1]['checkpoint_name'] == 'model_final_checkpoint'
    assert calls[0][1]['tta'] and calls[0][1]['num_threads_preprocessing'] == 1
    assert calls[0][1]['mixed_precision'] is False
    assert not (source / 'all').exists()
    assert (source / 'model_final_checkpoint.model.pkl').read_bytes() == original
    with pytest.raises(FileExistsError):
        run_lnctvseg.main()


def test_orientation_marker_is_bound_to_task_laterality_and_manifest(tmp_path):
    marker = tmp_path / 'orientation.json'
    assert not bundle.orientation_ready(marker)
    good = {'task': bundle.TASK, 'result': 'passed', 'laterality': 'passed',
            'manifest_sha256': bundle.manifest_sha256()}
    for field in good:
        marker.write_text(json.dumps({**good, field: 'wrong'}))
        assert not bundle.orientation_ready(marker)
    marker.write_text(json.dumps(good))
    assert bundle.orientation_ready(marker)


def test_real_release_accepts_current_marker_and_rejects_stale_marker(tmp_path):
    spec = bundle.manifest()
    assert spec['nnunet']['normalization']['scheme'] == 'nonCT'
    assert spec['nnunet']['normalization']['intensityproperties'] is None
    assert spec['nnunet']['output_label_ids'] == list(range(8))
    assert spec['label_contract']['omitted_checkpoint_ids'] == [7]
    bundle.validate_acceptance(accepted())
    for field in accepted():
        with pytest.raises(ValueError, match='stale or inconsistent'):
            bundle.validate_acceptance({**accepted(), field: 'wrong'})
    marker = tmp_path / 'orientation.json'
    marker.write_text(json.dumps(accepted()))
    assert bundle.orientation_ready(marker)


@pytest.mark.parametrize('field,value', [
    ('status', 'unresolved'), ('export_label_ids', list(range(1, 8))),
    ('export_label_ids', list(range(1, 6))), ('omitted_checkpoint_ids', []),
    ('omitted_checkpoint_ids', [6, 7]), ('policy', 'merge_into_6'),
    ('published_foreground_ids', list(range(1, 8))),
    ('checkpoint_foreground_ids', list(range(1, 7))),
    ('unmapped_checkpoint_ids', []),
])
def test_invalid_contract_blocks_acceptance_and_orientation(tmp_path, monkeypatch, field, value):
    spec = bundle.manifest()
    spec['label_contract'][field] = value
    monkeypatch.setattr(bundle, 'manifest', lambda: spec)
    with pytest.raises(ValueError, match='label contract invalid'):
        bundle.validate_acceptance(accepted())
    marker = tmp_path / 'orientation.json'
    marker.write_text(json.dumps(accepted()))
    assert not bundle.orientation_ready(marker)


@pytest.mark.parametrize('field,value', [('num_output_channels', 7), ('output_label_ids', list(range(7)))])
def test_checkpoint_channels_cannot_be_redefined(monkeypatch, field, value):
    spec = bundle.manifest()
    spec['nnunet'][field] = value
    monkeypatch.setattr(bundle, 'manifest', lambda: spec)
    with pytest.raises(ValueError, match='label contract invalid'):
        bundle.validate_label_contract()


def test_label_map_checksum_is_required_for_acceptance(monkeypatch):
    spec = bundle.manifest()
    spec['label_map_sha256'] = 'stale'
    monkeypatch.setattr(bundle, 'manifest', lambda: spec)
    with pytest.raises(ValueError, match='label contract invalid'):
        bundle.validate_acceptance(accepted())


def test_output_directories_cannot_resolve_inside_bundle(tmp_path):
    from run_lnctvseg import job_output, stage_model
    root = tmp_path / 'bundle'
    root.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='read-only model bundle'):
        job_output(alias / 'prediction', root)
    with pytest.raises(ValueError, match='read-only model bundle'):
        stage_model(root, root / 'work', {})
    assert list(root.iterdir()) == []


def test_dispatch_blocks_invalid_contract_before_export(tmp_path, monkeypatch, synthetic_ct):
    spec = bundle.manifest()
    spec['label_contract']['omitted_checkpoint_ids'] = []
    monkeypatch.setattr(bundle, 'manifest', lambda: spec)
    with pytest.raises(ValueError, match='label contract invalid'):
        dispatch(tmp_path, monkeypatch, synthetic_ct, audit=True, marker=False)
    assert (tmp_path / 'job/external' / bundle.TASK).is_dir()
    assert not list((tmp_path / 'job/output').glob('*.dcm'))


@pytest.mark.parametrize('class_7_voxels', [0, 2])
@pytest.mark.parametrize('empty', [False, True])
def test_omission_note_survives_job_cleanup_and_result_zip(tmp_path, portal, class_7_voxels, empty):
    job = tmp_path / 'job'
    raw = tmp_path / 'raw.nii.gz'
    exported = tmp_path / 'export.nii.gz'
    data = np.zeros((8, 8, 8), dtype=np.uint8)
    if not empty:
        data[1:3, 1:3, 1:3] = 1
    data[5, 5, :class_7_voxels] = 7
    nib.save(nib.Nifti1Image(data, np.diag([2., 2., 3., 1.])), raw)
    evidence = bundle.prepare_export(raw, exported, job / 'output' / bundle.TASK)
    expected = ('LNCTVSeg class 7 (undocumented by the authors) omitted: '
                f'{class_7_voxels} voxels, {class_7_voxels * .012:.4f} mL')
    assert evidence['notes'] == [expected]
    assert evidence['class_7_voxels'] == class_7_voxels
    assert evidence['class_7_ml'] == pytest.approx(class_7_voxels * .012)
    (job / 'job.json').write_text(json.dumps({'models': [bundle.TASK]}))
    # Finalization also preserves the note when only the omitted class was found.
    if not empty:
        (job / 'output' / f'{bundle.TASK}_RTSTRUCT.dcm').write_bytes(b'synthetic result')
    portal.finalize_results(job, empty=[bundle.TASK] if empty else [])
    meta = portal.metadata(job)
    assert meta['conversion_notes'] == [expected]
    assert expected in meta['detail']
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as archive:
        assert archive.read(f'{bundle.TASK}/OMITTED_LABELS.txt').decode().strip() == expected
        assert json.loads(archive.read(f'{bundle.TASK}/conversion.json')) == evidence


def test_six_published_rois_export_without_relabelling(tmp_path, synthetic_ct, monkeypatch, capsys):
    import convert_validate_rtstruct as converter
    from rt_utils import RTStructBuilder

    data = np.zeros((32, 32, 12), dtype=np.uint8)
    for label in range(1, 7):
        x = 3 + ((label - 1) % 3) * 9
        y = 3 + ((label - 1) // 3) * 12
        data[x:x + 5, y:y + 6, 3:7] = label
    data[3:8, 3:9, 8:11] = 7
    raw, exported = tmp_path / 'raw.nii.gz', tmp_path / 'export.nii.gz'
    output = tmp_path / 'result.dcm'
    nib.save(nib.Nifti1Image(data, np.diag([-1., -1., 2., 1.])), raw)
    original = raw.read_bytes()
    bundle.prepare_export(raw, exported, tmp_path / 'notes')
    assert raw.read_bytes() == original
    result = nib.load(exported)
    np.testing.assert_array_equal(result.affine, nib.load(raw).affine)
    np.testing.assert_array_equal(np.asarray(result.dataobj), np.where(data == 7, 0, data))
    monkeypatch.setattr(sys, 'argv', ['converter', '--prediction', str(exported),
        '--model-input', str(raw), '--dicom', str(synthetic_ct), '--output', str(output),
        '--labels', str(ROOT / 'labels/lnctvseg.json')])
    converter.main()
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary['rois'] == 6 and summary['valid']
    assert all(row['roundtrip_dice'] == 1 for row in summary['physical_roundtrip'].values())
    rt = RTStructBuilder.create_from(dicom_series_path=str(synthetic_ct), rt_struct_path=str(output))
    labels = json.loads((ROOT / 'labels/lnctvseg.json').read_text())
    assert set(rt.get_roi_names()) == set(labels.values())
    for label, name in labels.items():
        np.testing.assert_array_equal(rt.get_roi_mask_by_name(name), (data == int(label)).transpose(1, 0, 2))


@pytest.mark.parametrize('bad_label', [8, -1, .5, np.nan, np.inf])
def test_omission_does_not_hide_other_invalid_labels(tmp_path, bad_label):
    raw, exported = tmp_path / 'raw.nii.gz', tmp_path / 'export.nii.gz'
    nib.save(nib.Nifti1Image(np.full((2, 2, 2), bad_label, dtype=np.float32), np.eye(4)), raw)
    with pytest.raises(ValueError, match='integer label IDs 0-7'):
        bundle.prepare_export(raw, exported, tmp_path / 'notes')
    assert not exported.exists() and not (tmp_path / 'notes').exists()


def test_omission_cannot_overwrite_raw_prediction(tmp_path):
    raw = tmp_path / 'raw.nii.gz'
    with pytest.raises(ValueError, match='must not overwrite'):
        bundle.prepare_export(raw, raw, tmp_path / 'notes')


def test_catalog_gates_allow_current_acceptance_but_require_orientation(portal, monkeypatch):
    monkeypatch.setattr(portal, 'WORKER_MODE', 'local')
    for orientation_ready in (False, True):
        row = {**portal.EXTERNAL_TASKS[bundle.TASK], 'name': bundle.TASK, 'enabled': True}
        portal.apply_model_gates(row, {'ready': True, 'acceptance_ready': True,
                                      'orientation_ready': orientation_ready}, {bundle.TASK})
        assert row['enabled'] is orientation_ready


def test_routed_task_retrieves_omission_notes_with_rtstruct(tmp_path, portal, monkeypatch):
    job = tmp_path / 'job'
    job.mkdir()
    calls = []
    monkeypatch.setattr(portal.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    portal.retrieve_worker_results(job, '/jobs/synthetic', task=bundle.TASK)
    assert calls == [['scp', '-q', '-r',
        f'{portal.WORKER}:/jobs/synthetic/output/{bundle.TASK}_RTSTRUCT.dcm*',
        f'{portal.WORKER}:/jobs/synthetic/output/{bundle.TASK}', str(job / 'output') + '/']]
