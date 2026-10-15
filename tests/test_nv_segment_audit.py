"""Operator audit success/failure and timeout paths; no GPU or external fixtures."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nv_segment_logic import MODES, acceptance_from_metrics

ROOT = Path(__file__).parents[1]


def load_audit():
    spec = importlib.util.spec_from_file_location('orientation_audit', ROOT / 'scripts/run-orientation-audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(mode):
    spec = MODES[mode]
    return dict(mode=mode, device='cuda', cuda_available=True, load_only=False, exact_input_grid=True,
                output_dtype='uint16', nonempty_foreground_labels=2, source_commit=spec['source_commit'],
                checkpoint_revision=spec['revision'], checkpoint_sha256=spec['sha256'])


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('failure', [None, 'inference', 'conversion', 'orientation', 'metrics', 'fixture'])
def test_markers_only_after_fresh_success(tmp_path, monkeypatch, mode, failure):
    audit = load_audit()
    root, runs = tmp_path / 'model', tmp_path / 'runs'
    fixture = root / 'testdata/public/dicom'
    fixture.mkdir(parents=True)
    (fixture / 'synthetic.txt').write_text('synthetic fixture placeholder')
    config = tmp_path / 'fixtures.json'
    config.write_text(json.dumps({} if failure == 'fixture' else {mode: 'public/dicom'}))
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(root))
    monkeypatch.setenv('RTSEG_APP_ROOT', str(ROOT))
    monkeypatch.setenv('RTSEG_ORIENTATION_RUN_ROOT', str(runs))
    monkeypatch.setenv('RTSEG_ORIENTATION_FIXTURES', str(config))
    # These old markers must not survive a failed repeat.
    for dirname in ['validation', 'orientation-validation']:
        marker = root / dirname / f'{mode}.json'
        marker.parent.mkdir()
        marker.write_text('{"old":true}')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1].endswith('run_task.py'):
            assert command[-1] == '--nv-segment-acceptance-audit'
            if failure in ['inference', 'conversion']:
                raise subprocess.CalledProcessError(1, command)
            metrics_path = Path(command[2]) / 'external' / mode / 'prediction.nii.gz.metrics.json'
            metrics_path.parent.mkdir(parents=True)
            metrics = evidence(mode)
            if failure == 'metrics': metrics['checkpoint_sha256'] = 'different checkpoint'
            metrics_path.write_text(json.dumps(metrics))
        else:
            if failure == 'orientation': raise subprocess.CalledProcessError(1, command)
            Path(command[-1]).write_text(json.dumps(dict(task=mode, result='passed', rois=2,
                                                        artifact_sha256='synthetic artifact')))
    monkeypatch.setattr(audit, 'run_command', run)
    assert audit.main(['--nv-segment-acceptance', mode]) == int(failure is not None)
    for dirname in ['validation', 'orientation-validation']:
        assert (root / dirname / f'{mode}.json').exists() is (failure is None)
    summary = json.loads((runs / 'summary.jsonl').read_text())
    assert summary['status'] == ('passed' if failure is None else 'failed')
    if failure is None:
        marker = json.loads((root / 'validation' / f'{mode}.json').read_text())
        assert marker['checkpoint_sha256'] == MODES[mode]['sha256']
        assert marker['fixture'] == 'public/dicom'


@pytest.mark.parametrize('key,value', [('mode', 'other'), ('device', 'cpu'), ('cuda_available', False),
    ('load_only', True), ('exact_input_grid', False), ('output_dtype', 'uint8'), ('nonempty_foreground_labels', 0)])
def test_acceptance_requires_gpu_prediction_evidence(key, value):
    mode = 'nv_segment_ct_auto'
    metrics = {**evidence(mode), key: value}
    orientation = dict(task=mode, result='passed', rois=1, artifact_sha256='test')
    with pytest.raises(ValueError): acceptance_from_metrics(metrics, orientation, mode, 'public/dicom')


def test_audit_rejects_other_tasks_for_nv_acceptance():
    with pytest.raises(SystemExit):
        load_audit().main(['--nv-segment-acceptance', 'vista3d_prompted_ct'])


def test_audit_timeout_terminates_process_group(tmp_path):
    audit = load_audit()
    with (tmp_path / 'log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            audit.run_command([sys.executable, '-c', 'import time; time.sleep(60)'],
                              env=None, log=log, timeout=.05)
