"""CPU-only checks for the manager audit's fail-closed marker publication."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def module():
    spec = importlib.util.spec_from_file_location('nodal_gpu', ROOT / 'scripts/run-nodal-gpu-checks.py')
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('failure', [None, 'inference', 'export', 'laterality', 'orientation', 'missing_labels', 'label_contract', 'dice'])
def test_publish_only_after_all_checks_pass(tmp_path, monkeypatch, failure):
    audit = module()
    model = tmp_path / 'models'
    fixture = model / 'testdata' / audit.FIXTURE
    fixture.mkdir(parents=True); (fixture / 'slice.dcm').touch()
    monkeypatch.setenv('RTSEG_MODEL_ROOT', str(model))
    monkeypatch.setenv('RTSEG_WORKER_PYTHON', '/fake/worker/python')
    monkeypatch.setattr(audit.lnctvseg_bundle, 'validate_assets', lambda root: {})
    spec = audit.lnctvseg_bundle.manifest()
    if failure != 'label_contract':
        spec['label_contract']['status'] = 'verified'
        spec['nnunet']['output_label_ids'] = list(range(7))
    monkeypatch.setattr(audit.lnctvseg_bundle, 'manifest', lambda: spec)
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='fake GPU\n'))
    def measured(*args):
        if failure == 'inference': raise ValueError('inference failed')
        job, task = Path(args[0][2]), args[0][3]
        prediction = job / 'external' / task / 'prediction/case.nii.gz'
        prediction.parent.mkdir(parents=True); prediction.touch()
        if failure == 'export': raise ValueError('export refused after raw prediction')
        return {'seconds': 10, 'peak_vram_mib': 512}
    monkeypatch.setattr(audit, 'measured_run', measured)
    def dice(prediction, model, output):
        if failure == 'dice': raise ValueError('reference grid mismatch')
        result = {'labels': [{'label': n, 'dice': 0.5} for n in range(1, 7)]}
        output.write_text(json.dumps(result))
        return result
    monkeypatch.setattr(audit, 'fixture_dice', dice)
    def run(command, **kwargs):
        output = Path(command[command.index('--output') + 1])
        kind = 'laterality' if 'audit_nodal_laterality.py' in command[1] else 'orientation'
        if failure == kind: raise ValueError(f'{kind} failed')
        payload = {'result': 'passed', 'components': [{'label': n} for n in range(1, 7)]}
        if failure == 'missing_labels': payload['components'] = []
        if kind == 'orientation': payload['task'] = command[command.index('--task') + 1]
        output.write_text(json.dumps(payload))
    monkeypatch.setattr(audit.orientation_audit, 'run_command', run)
    for task in audit.TASKS:
        for parent in ('validation', 'orientation-validation'):
            marker = model / parent / f'{task}.json'
            marker.parent.mkdir(exist_ok=True); marker.write_text('{"result":"old"}')
    assert audit.main(['--run-root', str(tmp_path / 'run')]) == int(failure is not None)
    for task in audit.TASKS:
        for parent in ('validation', 'orientation-validation'):
            marker = model / parent / f'{task}.json'
            expected = failure is None or (failure in {'missing_labels', 'label_contract', 'dice'} and task == 'hnlnl_2d3d')
            assert marker.exists() is expected
            if marker.exists():
                value = json.loads(marker.read_text())
                assert value['task'] == task and value['result'] == 'passed'
                assert value['laterality'] == 'passed'
    if failure not in {'inference', 'dice'}:
        assert (tmp_path / 'run' / audit.TASKS[0] / 'dice.json').is_file()
    if failure is None:
        audit.lnctvseg_bundle.validate_acceptance(json.loads(
            (model / 'validation' / f'{audit.TASKS[0]}.json').read_text()))
        assert audit.lnctvseg_bundle.orientation_ready(model / 'orientation-validation' / f'{audit.TASKS[0]}.json')
