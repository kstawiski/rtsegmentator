"""Catalog, preflight and real dispatcher control flow with CPU-only subprocess fakes."""
import json
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pydicom
import pytest

from nv_segment_logic import MODES

ROOT = Path(__file__).parents[1]


def accepted(mode):
    spec = MODES[mode]
    return dict(task=mode, result='passed', source_commit=spec['source_commit'],
                checkpoint_revision=spec['revision'], checkpoint_sha256=spec['sha256'])


def install_fake_bundle(tmp_path, monkeypatch):
    import model_preflight as module
    model, runtime = tmp_path / 'models', tmp_path / 'runtimes'
    monkeypatch.setattr(module, 'MODEL_ROOT', model)
    monkeypatch.setattr(module, 'RUNTIME_ROOT', runtime)
    monkeypatch.setattr(module, 'APP_ROOT', ROOT)
    for mode in MODES:
        for requirement in module.REQUIREMENTS[mode]:
            if requirement.startswith('@') or requirement.startswith('validation/'):
                continue
            path = (runtime if requirement.startswith('envs/') else model) / requirement
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return module, model, runtime


@pytest.mark.parametrize('mode', MODES)
def test_preflight_requires_individual_acceptance_and_orientation(tmp_path, monkeypatch, mode):
    module, model, runtime = install_fake_bundle(tmp_path, monkeypatch)
    row = module.audit([mode])[mode]
    assert not row['ready'] and not row['acceptance_ready'] and not row['orientation_ready']
    assert row['missing'] == [f'validation/{mode}.json']
    marker = model / 'validation' / f'{mode}.json'
    marker.parent.mkdir(exist_ok=True)
    for invalid in ['{', 'null', json.dumps({**accepted(mode), 'task': 'vista3d_prompted_ct'}),
                    json.dumps({**accepted(mode), 'checkpoint_sha256': 'wrong'}),
                    json.dumps({**accepted(mode), 'checkpoint_revision': 'old'}),
                    json.dumps({**accepted(mode), 'source_commit': 'old'})]:
        marker.write_text(invalid)
        assert not module.audit([mode])[mode]['ready']
    marker.write_text(json.dumps(accepted(mode)))
    row = module.audit([mode])[mode]
    assert row['ready'] and row['acceptance_ready'] and not row['orientation_ready']
    orientation = model / 'orientation-validation' / f'{mode}.json'
    orientation.parent.mkdir(exist_ok=True)
    orientation.write_text('{"result":"passed"}')
    assert module.audit([mode])[mode]['orientation_ready']
    (runtime / 'envs/lnq/bin/python').unlink()
    assert not module.audit([mode])[mode]['ready']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('state', ['ready', 'no_acceptance', 'no_orientation', 'no_preflight', 'not_enabled'])
def test_local_catalog_gate(portal, monkeypatch, mode, state):
    monkeypatch.setattr(portal, 'COLAB_ORCHESTRATOR', 'direct')
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', set())
    monkeypatch.setattr(portal, 'ENABLED_EXTERNAL_MODELS', set(MODES) if state != 'not_enabled' else set())
    monkeypatch.setattr(portal, 'ORIENTATION_VALIDATION', ROOT / 'absent-test-markers')
    def run(command, **kwargs):
        if any('model_preflight.py' in str(arg) for arg in command):
            result = {} if state == 'no_preflight' else {mode: dict(
                ready=state != 'no_acceptance', acceptance_ready=state != 'no_acceptance',
                orientation_ready=state != 'no_orientation', missing=['acceptance'] if state == 'no_acceptance' else [])}
        else:
            result = {'tasks': ['total'], 'licensed': []}
        return SimpleNamespace(stdout=json.dumps(result))
    monkeypatch.setattr(portal.subprocess, 'run', run)
    item = next(item for item in portal.build_catalog()['tasks'] if item['name'] == mode)
    assert item['enabled'] is (state == 'ready')
    assert item['modalities'] == [MODES[mode]['modality']]
    assert not item.get('prompt')
    assert 'Tiny spurious' in item['limitations']
    if mode != 'nv_segment_ct_auto':
        assert 'Noncommercial' in item['license_note']


@pytest.mark.parametrize('mode', MODES)
def test_remote_catalog_uses_remote_markers(portal, monkeypatch, mode):
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {mode})
    monkeypatch.setattr(portal, 'ENABLED_EXTERNAL_MODELS', {mode})
    remote = {mode: dict(ready=False, orientation_ready=False, missing=['remote acceptance'])}
    monkeypatch.setattr(portal, 'remote_preflight', lambda: remote)
    def run(command, **kwargs):
        value = ({mode: dict(ready=True, orientation_ready=True, missing=[])}
                 if any('model_preflight.py' in str(x) for x in command) else {'tasks': ['total'], 'licensed': []})
        return SimpleNamespace(stdout=json.dumps(value))
    monkeypatch.setattr(portal.subprocess, 'run', run)
    item = next(x for x in portal.build_catalog()['tasks'] if x['name'] == mode)
    assert not item['enabled'] and item['execution_backend'] == 'ssh'
    remote[mode] = dict(ready=True, orientation_ready=True, missing=[])
    assert next(x for x in portal.build_catalog()['tasks'] if x['name'] == mode)['enabled']


def dispatch(tmp_path, monkeypatch, synthetic_ct, mode, marker=True, modality=None, audit=False):
    job, model = tmp_path / 'job', tmp_path / 'models'
    (job / 'output').mkdir(parents=True)
    (job / 'input').symlink_to(synthetic_ct)
    for path in synthetic_ct.iterdir():
        ds = pydicom.dcmread(path)
        ds.Modality = modality or MODES[mode]['modality']
        ds.save_as(path, enforce_file_format=True)
    if marker:
        path = model / 'validation' / f'{mode}.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(accepted(mode)))
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(model))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT', str(tmp_path / 'runtimes'))
    monkeypatch.delenv('RTSEG_SCRATCH_ROOT', raising=False)
    monkeypatch.setattr(sys, 'argv', ['run_task.py', str(job), mode] + (['--nv-segment-acceptance-audit'] if audit else []))
    monkeypatch.setattr(signal, 'signal', lambda *args: None)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            if command[1] == str(ROOT / 'convert_validate_rtstruct.py'):
                shutil.copyfile(next(synthetic_ct.iterdir()), command[command.index('--output') + 1])
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    runpy.run_path(str(ROOT / 'run_task.py'), run_name='__main__')
    return job, calls


@pytest.mark.parametrize('mode', MODES)
def test_dispatches_all_modes_without_prompts(tmp_path, monkeypatch, synthetic_ct, mode):
    job, calls = dispatch(tmp_path, monkeypatch, synthetic_ct, mode)
    assert [Path(command[1]).name for command, _ in calls] == [
        'dicom_series_to_nifti.py', 'run_nv_segment.py', 'convert_validate_rtstruct.py']
    command = calls[1][0]
    assert command[command.index('--mode') + 1] == mode
    assert command[command.index('--input-modality') + 1] == MODES[mode]['modality']
    assert command[command.index('--device') + 1] == 'cuda'
    assert command[command.index('--work') + 1].startswith(str(job))
    assert all(options['start_new_session'] for _, options in calls)
    assert all(options['env']['TMPDIR'].startswith(str(job)) for _, options in calls)
    ds = pydicom.dcmread(job / 'output' / f'{mode}_RTSTRUCT.dcm')
    assert ds.StructureSetName == f'RTSEG: {mode}'


def test_dispatch_rejects_unaccepted_task(tmp_path, monkeypatch, synthetic_ct):
    with pytest.raises(FileNotFoundError):
        dispatch(tmp_path, monkeypatch, synthetic_ct, 'nv_segment_ct_auto', marker=False)


def test_dispatch_operator_audit_bootstraps_without_marker(tmp_path, monkeypatch, synthetic_ct):
    _, calls = dispatch(tmp_path, monkeypatch, synthetic_ct, 'nv_segment_ct_auto', marker=False, audit=True)
    assert len(calls) == 3


def test_dispatch_rejects_wrong_modality(tmp_path, monkeypatch, synthetic_ct):
    with pytest.raises(ValueError, match='requires CT'):
        dispatch(tmp_path, monkeypatch, synthetic_ct, 'nv_segment_ct_auto', modality='MR')


def test_public_audit_fixtures_and_no_prompts():
    fixtures = json.loads((ROOT / 'catalog/orientation-audit-fixtures.json').read_text())
    tasks = json.loads((ROOT / 'catalog/orientation-audit-tasks.json').read_text())
    prompts = json.loads((ROOT / 'catalog/orientation-audit-prompts.json').read_text())
    assert set(MODES) <= set(tasks)
    assert not set(MODES) & prompts.keys()
    assert fixtures['nv_segment_ctmr_mri_brain'] == 'hippunfold-ds002168-T1w/dicom'
    assert fixtures['nv_segment_ctmr_mri_body'] == 'totalsegmentator-mri/s0052/dicom'
    assert fixtures['nv_segment_ct_auto'] == 'msd-lung/dicom'
    assert fixtures['nv_segment_ctmr_ct_body'] == 'kits19-case00000/dicom'
