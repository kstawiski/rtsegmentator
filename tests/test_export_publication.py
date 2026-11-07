"""Publication, retry and remote retrieval boundaries, with synthetic data and no inference."""
import ast
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import pytest

from rtstruct_geometry import dicom_grid

ROOT = Path(__file__).parents[1]
REPORT = 'nv_reason_ct_report'


def job_at(portal, synthetic_ct, models, completed=()):
    job = portal.BASE / 'publication-retry'
    (job / 'output').mkdir(parents=True)
    shutil.copytree(synthetic_ct, job / 'input')
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'status': 'failed', 'models': models,
        'completed_models': list(completed), 'input_manifest': {'input': portal.directory_manifest(job / 'input')}}))
    return job


@pytest.mark.parametrize('backend', ['local', 'ssh'])
def test_report_retry_clears_rerun_but_preserves_completed_models(portal, synthetic_ct, monkeypatch, backend):
    job = job_at(portal, synthetic_ct, ['keep', 'total', REPORT], completed=['keep'])
    keep = job / 'output/keep_RTSTRUCT.dcm'; keep.write_bytes(b'completed synthetic output')
    stale = job / 'output/total_RTSTRUCT.dcm'; stale.write_bytes(b'unvalidated synthetic output')
    Path(str(stale) + '.orientation.json').write_text('{}')
    old_report = job / 'output/report'; old_report.mkdir(); (old_report / 'old.txt').write_text('old draft')
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda *a: None)
    monkeypatch.setattr(portal, 'WORKER_MODE', backend)
    monkeypatch.setattr(portal, 'prepare_worker_job', lambda *a: None)
    monkeypatch.setattr(portal, 'retrieve_worker_results', lambda *a: None)
    monkeypatch.setattr(portal, 'cleanup_worker_job', lambda *a: None)
    # Remote reset is exercised, without a real SSH connection.
    calls = []
    monkeypatch.setattr(portal.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    def run(job_, *args):
        task = args[1] if backend == 'ssh' else args[0]
        assert task != 'keep'
        assert keep.read_bytes() == b'completed synthetic output'
        assert not stale.exists() and not Path(str(stale) + '.orientation.json').exists()
        if task == 'total':
            raise RuntimeError('empty prediction: no RTSTRUCT created')
        assert not old_report.exists()
        old_report.mkdir()
        (old_report / (REPORT + '.json')).write_text('{"status":"draft","reason":""}')
    monkeypatch.setattr(portal, 'run_worker_task' if backend == 'ssh' else 'run_local_task', run)
    portal.retry_job(job.name)
    portal.process(job)
    assert portal.metadata(job)['status'] == 'complete'
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as package:
        names = package.namelist()
        assert keep.name in names and stale.name not in names
        manifest = json.loads(package.read('rtsegmentator_manifest.json'))
        assert {m['name']: m['outcome'] for m in manifest['models']}['total'] == 'no_structures_found'
        assert '- total' in package.read('NO_STRUCTURES_FOUND.txt').decode()
    if backend == 'ssh':
        assert any(cmd[0] == 'ssh' and 'total_RTSTRUCT.dcm' in cmd[-1] for cmd in calls)


def test_empty_finalizer_excludes_the_structure_set_and_keeps_the_runs_own_results(portal, synthetic_ct):
    job = job_at(portal, synthetic_ct, ['total'])
    for name in ('total_RTSTRUCT.dcm', 'total_RTSTRUCT.dcm.orientation.json'):
        (job / 'output' / name).write_bytes(b'stale synthetic artifact')
    # A model that finds nothing still reports it: lesion tables carry its case likelihood.
    (job / 'output' / 'total_RTSTRUCT.dcm.lesions.json').write_text('{"case_likelihood": 0.02}')
    portal.finalize_results(job, empty=['total'])
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as package:
        names = package.namelist()
        assert 'total_RTSTRUCT.dcm' not in names and 'total_RTSTRUCT.dcm.orientation.json' not in names
        assert 'total_RTSTRUCT.dcm.lesions.json' in names
        manifest = json.loads(package.read('rtsegmentator_manifest.json'))
        assert manifest['models'][0]['outcome'] == 'no_structures_found'


@pytest.mark.parametrize('writer', ['convert_validate_rtstruct', 'convert_masks_validate_rtstruct'])
@pytest.mark.parametrize('sig', [signal.SIGTERM, signal.SIGKILL])
def test_kill_after_save_cannot_publish_or_package(portal, synthetic_ct, tmp_path, writer, sig):
    job = job_at(portal, synthetic_ct, ['total'])
    image = tmp_path / 'input.nii.gz'
    data = np.zeros((32, 32, 12), np.uint8); data[5:20, 5:20, 2:8] = 1
    _, affine, _ = dicom_grid(synthetic_ct, nib.Nifti1Image(data, np.eye(4)))
    nib.save(nib.Nifti1Image(data, affine), image)
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'A': str(image)} if 'masks' in writer else {'1': 'A'}))
    extra = ['--masks-json', str(mapping)] if 'masks' in writer else ['--prediction', str(image), '--labels', str(mapping)]
    output = job / 'output/total_RTSTRUCT.dcm'; marker = job / 'saved.txt'
    # Pause at the real saved-file validation boundary, before any validation or publication.
    code = ('import importlib, pathlib, signal, sys; m=importlib.import_module(sys.argv.pop(1)); '
            'marker=pathlib.Path(sys.argv.pop(1)); '
            'm.read_rtstruct=lambda p: (marker.write_text(str(p)), signal.pause())[1]; m.main()')
    proc = subprocess.Popen([sys.executable, '-B', '-c', code, writer, str(marker),
        '--model-input', str(image), '--dicom', str(synthetic_ct), '--output', str(output), *extra],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists(), f'converter exited {proc.poll()} before saved-file validation'
        partial = Path(marker.read_text())
        assert partial.is_file() and partial.stat().st_size > 0
        assert partial.parent.parent == job
        assert partial.stat().st_dev == output.parent.stat().st_dev
        os.killpg(proc.pid, sig); proc.wait(timeout=10)
        assert proc.returncode == -sig and not output.exists()
        # Deliberately keep the SIGKILL orphan during real archive and finalizer checks.
        assert not any('.rtstruct-partial-' in name for name in portal.worker_archive_members(job, [REPORT]))
        portal.finalize_results(job, empty=['total'])
        with zipfile.ZipFile(job / 'rtstruct-results.zip') as package:
            assert all(not name.endswith('.dcm') for name in package.namelist())
        assert partial.exists()
        import run_nv_reason_ct
        summary = {}
        run_nv_reason_ct.headers_and_summary(job, summary)
        assert summary['segmentation_summary'] == []
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for directory in job.glob('.rtstruct-partial-*'):
            shutil.rmtree(directory)


@pytest.mark.parametrize('status', [3, 255])
@pytest.mark.parametrize('task', ['total', REPORT])
def test_task_retrieval_distinguishes_no_match_from_transport_error(portal, synthetic_ct, monkeypatch, status, task):
    job = job_at(portal, synthetic_ct, [task])
    portal.save_metadata(job, remote_results_retained=True, remote_task_active=task)
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == 'scp':
            pytest.fail('scp must not run after an empty or failed probe')
        return subprocess.CompletedProcess(cmd, status, '', 'synthetic status')
    monkeypatch.setattr(portal.subprocess, 'run', run)
    monkeypatch.setattr(portal, 'cleanup_worker_job', lambda j, r: portal.save_metadata(j, remote_results_retained=False))
    if status == 3:
        portal.recover_remote_results(job, '/jobs/test', task)
        assert not portal.metadata(job)['remote_results_retained']
    else:
        with pytest.raises(subprocess.CalledProcessError):
            portal.recover_remote_results(job, '/jobs/test', task)
        assert portal.metadata(job)['remote_results_retained']


def test_scp_failure_is_not_treated_as_empty(portal, synthetic_ct, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total'])
    def run(cmd, **kw):
        if cmd[0] == 'scp':
            raise subprocess.CalledProcessError(1, cmd, stderr='Connection lost')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(portal.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError, match='scp'):
        portal.recover_remote_results(job, '/jobs/test', 'total')
    assert portal.metadata(job)['remote_results_retained']


def test_optional_remote_prefix_absence_keeps_available_notes(portal, synthetic_ct, monkeypatch):
    job = job_at(portal, synthetic_ct, ['lnctvseg_nnunet_v1'])
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        status = 3 if cmd[0] == 'ssh' and 'RTSTRUCT.dcm' in cmd[-1] else 0
        return subprocess.CompletedProcess(cmd, status, '', '')
    monkeypatch.setattr(portal.subprocess, 'run', run)
    portal.retrieve_worker_results(job, '/jobs/test', 'lnctvseg_nnunet_v1')
    scp = next(cmd for cmd in calls if cmd[0] == 'scp')
    assert len(scp) == 5 and scp[-2].endswith('/output/lnctvseg_nnunet_v1')


@pytest.mark.parametrize('present', [False, True])
def test_remote_existence_script_with_real_globs_and_quoted_paths(portal, synthetic_ct, tmp_path, monkeypatch, present):
    import shlex
    job = job_at(portal, synthetic_ct, ['total'])
    remote = tmp_path / 'remote with spaces'; (remote / 'output').mkdir(parents=True)
    if present:
        (remote / 'output/total_RTSTRUCT.dcm.orientation.json').write_text('{}')
    real_run = subprocess.run
    scp = []
    def run(cmd, **kw):
        if cmd[0] == 'ssh':
            return real_run(shlex.split(cmd[-1]), **kw)
        assert cmd[0] == 'scp'
        scp.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(portal, 'REMOTE_PYTHON', sys.executable)
    monkeypatch.setattr(portal.subprocess, 'run', run)
    portal.retrieve_worker_results(job, str(remote), 'total')
    assert len(scp) == int(present)


def remote_tree(root):
    for name in ('external/total/work.bin', 'output/total/part.nii.gz', 'output/prostate_sbrt_crop/crop.dcm',
                 'output/total_RTSTRUCT.dcm', 'output/total_RTSTRUCT.dcm.orientation.json',
                 'output/keep_RTSTRUCT.dcm', 'output/totalseg_RTSTRUCT.dcm'):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b'synthetic')


def run_as_login_shell(shell, calls):
    real_run = subprocess.run
    def run(cmd, **kw):
        calls.append(cmd)
        # OpenSSH hands the joined command string to the account's login shell.
        return real_run([*shell, '-c', cmd[-1]], **kw)
    return run


def ssh_options(calls):
    return all(cmd[:5] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5'] for cmd in calls)


# bash with failglob refuses an unmatched pattern, as zsh and csh do by default.
@pytest.mark.parametrize('shell', [['sh'], ['bash'], ['bash', '-O', 'failglob']], ids=['sh', 'bash', 'failglob'])
@pytest.mark.parametrize('stale', [True, False])
def test_remote_reset_does_not_depend_on_the_login_shell(portal, synthetic_ct, tmp_path, monkeypatch, shell, stale):
    job = job_at(portal, synthetic_ct, ['total'])
    remote = tmp_path / 'remote with spaces'
    remote_tree(remote)
    if not stale:
        for name in ('total_RTSTRUCT.dcm', 'total_RTSTRUCT.dcm.orientation.json'):
            (remote / 'output' / name).unlink()
    calls = []
    monkeypatch.setattr(portal.subprocess, 'run', run_as_login_shell(shell, calls))
    portal.reset_task_output(job, 'total', str(remote))
    left = sorted(p.relative_to(remote).as_posix() for p in remote.rglob('*') if p.is_file())
    assert left == ['output/keep_RTSTRUCT.dcm', 'output/totalseg_RTSTRUCT.dcm']
    assert len(calls) == 1 and ssh_options(calls)


def test_remote_reset_of_a_new_job_succeeds(portal, synthetic_ct, tmp_path, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total'])
    monkeypatch.setattr(portal.subprocess, 'run', run_as_login_shell(['bash', '-O', 'failglob'], []))
    portal.reset_task_output(job, 'total', str(tmp_path / 'not created yet'))


def test_remote_reset_fails_when_the_output_folder_cannot_be_listed(portal, synthetic_ct, tmp_path, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total'])
    remote = tmp_path / 'remote'
    remote_tree(remote)
    (remote / 'output/total_RTSTRUCT.dcm.orientation.json').unlink()
    (remote / 'output').chmod(0o300)
    try:
        if os.access(remote / 'output', os.R_OK):
            pytest.skip('running with privileges that ignore directory permissions')
        monkeypatch.setattr(portal.subprocess, 'run', run_as_login_shell(['sh'], []))
        # A stale structure set that cannot be found must stop the task rather than be retrieved later.
        with pytest.raises(subprocess.CalledProcessError):
            portal.reset_task_output(job, 'total', str(remote))
    finally:
        (remote / 'output').chmod(0o700)
    assert (remote / 'output/total_RTSTRUCT.dcm').exists()


def test_remote_reset_tolerates_an_unremovable_work_folder_only(portal, synthetic_ct, tmp_path, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total'])
    remote = tmp_path / 'remote'
    remote_tree(remote)
    locked = remote / 'external/total/locked'
    locked.mkdir()
    (locked / 'runtime.bin').write_bytes(b'synthetic')
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip('running with privileges that ignore directory permissions')
        monkeypatch.setattr(portal.subprocess, 'run', run_as_login_shell(['sh'], []))
        portal.reset_task_output(job, 'total', str(remote))
        assert not (remote / 'output/total_RTSTRUCT.dcm').exists() and not (remote / 'output/total').exists()
        output = remote / 'output/total'
        output.mkdir()
        (output / 'sub').mkdir()
        (output / 'sub/part.bin').write_bytes(b'synthetic')
        (output / 'sub').chmod(0o500)
        try:
            # A result folder that cannot be cleared must stop the task: its files would be retrieved.
            with pytest.raises(subprocess.CalledProcessError):
                portal.reset_task_output(job, 'total', str(remote))
        finally:
            (output / 'sub').chmod(0o700)
    finally:
        locked.chmod(0o700)


def test_absent_structures_are_listed_but_kept_out_of_the_status_line(portal, synthetic_ct):
    job = job_at(portal, synthetic_ct, ['total'])
    (job / 'output/total_RTSTRUCT.dcm').write_bytes(b'synthetic result')
    omitted = [{'name': 'spleen', 'voxels': 0, 'reason': 'empty input label'},
               {'name': 'liver', 'voxels': 0, 'reason': 'empty input label'},
               {'name': 'rib_left_1', 'voxels': 3, 'reason': 'no contour returned for nonempty mask'}]
    (job / 'total.log').write_text('RTSTRUCT_CONVERSION_NOTE ' + json.dumps(
        {'output': 'total_RTSTRUCT.dcm', 'omitted_rois': omitted}) + '\n')
    portal.finalize_results(job)
    record = portal.metadata(job)
    dropped = 'total: omitted ROI rib_left_1 (3 voxels; no contour returned for nonempty mask).'
    absent = 'total: not present in the prediction: spleen, liver.'
    assert dropped in record['detail'] and 'spleen' not in record['detail']
    assert record['conversion_notes'] == [dropped, absent]
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as package:
        assert package.read('CONVERSION_NOTES.txt').decode() == dropped + '\n' + absent + '\n'
        manifest = json.loads(package.read('rtsegmentator_manifest.json'))
        assert dropped in manifest['notes'] and absent in manifest['notes']


def test_read_only_work_folder_does_not_fail_the_task_reset(portal, synthetic_ct, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total'])
    work = job / 'external' / 'total'
    work.mkdir(parents=True)
    real, calls = shutil.rmtree, []
    def refuse(path, ignore_errors=False, **kwargs):
        calls.append((Path(path).name, ignore_errors))
        if not ignore_errors:
            raise PermissionError(path)
    monkeypatch.setattr(portal.shutil, 'rmtree', refuse)
    portal.reset_task_output(job, 'total')
    assert ('total', True) in calls


def test_same_task_retained_result_is_not_fetched_back(portal, synthetic_ct, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total'])
    portal.save_metadata(job, remote_results_retained=True, remote_task_active='total')
    calls = []
    monkeypatch.setattr(portal, 'recover_remote_results', lambda j, r, t: calls.append(('recover', t)))
    monkeypatch.setattr(portal, 'reset_task_output', lambda j, t, r=None: calls.append(('reset', t, r)))
    monkeypatch.setattr(portal, 'prepare_worker_job', lambda *a: calls.append(('prepare',)))
    monkeypatch.setattr(portal, 'run_worker_task', lambda *a: None)
    portal.run_remote_gpu_task(job, 'total', 1, 1)
    # Reset reaches the worker before the job is prepared; the only recovery is the new run's own retrieval.
    assert calls[0][0] == 'reset' and calls[0][2] and calls[1] == ('prepare',)
    assert calls[2:] == [('recover', 'total')]


def test_other_task_retained_result_is_still_recovered(portal, synthetic_ct, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total', 'lung_nodules'])
    portal.save_metadata(job, remote_results_retained=True, remote_task_active='lung_nodules')
    calls = []
    monkeypatch.setattr(portal, 'recover_remote_results', lambda j, r, t: calls.append(t))
    monkeypatch.setattr(portal, 'reset_task_output', lambda *a: None)
    monkeypatch.setattr(portal, 'prepare_worker_job', lambda *a: None)
    monkeypatch.setattr(portal, 'run_worker_task', lambda *a: None)
    portal.run_remote_gpu_task(job, 'total', 1, 1)
    assert calls == ['lung_nodules', 'total']


def test_total_reset_removes_its_derived_crop(portal, synthetic_ct):
    job = job_at(portal, synthetic_ct, ['total'])
    crop = job / 'output' / 'prostate_sbrt_crop'
    crop.mkdir(parents=True)
    (crop / 'total_RTSTRUCT_cropped.dcm').write_bytes(b'stale')
    portal.reset_task_output(job, 'lung_nodules')
    assert crop.exists()
    portal.reset_task_output(job, 'total')
    assert not crop.exists()


def test_crop_rerun_replaces_an_earlier_crop(portal, synthetic_ct, monkeypatch):
    job = job_at(portal, synthetic_ct, ['total'])
    crop = job / 'output' / 'prostate_sbrt_crop'
    crop.mkdir(parents=True)
    (crop / 'stale.dcm').write_bytes(b'stale')
    seen = []
    monkeypatch.setattr(portal, 'discover_dicom', lambda path: ([], None))
    monkeypatch.setattr(portal, 'crop_ct_rtstruct', lambda *a, **k: seen.append(crop.exists()))
    portal.postprocess_prostate_crop(job)
    assert seen == [False]


def test_staging_falls_back_inside_output_and_packaging_skips_it(portal, synthetic_ct, tmp_path, monkeypatch):
    from rtstruct_geometry import staged_rtstruct
    output = tmp_path / 'made-here' / 'x_RTSTRUCT.dcm'
    monkeypatch.setattr('rtstruct_geometry.os.access', lambda *a: False)
    with staged_rtstruct(output) as partial:
        assert partial.parent.parent == output.parent
        assert partial.parent.name.startswith('.rtstruct-partial-')
    job = job_at(portal, synthetic_ct, ['total'])
    (job / 'output' / 'total_RTSTRUCT.dcm').write_bytes(b'x')
    leftover = job / 'output' / '.rtstruct-partial-abc'
    leftover.mkdir()
    (leftover / 'partial.dcm').write_bytes(b'unvalidated')
    portal.finalize_results(job, empty=['total'])
    with zipfile.ZipFile(job / 'rtstruct-results.zip') as archive:
        assert not any('partial' in name for name in archive.namelist())


def identify_function():
    """The real function body from run_task.py, which is a script and cannot be imported."""
    tree = ast.parse((ROOT / 'run_task.py').read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'identify_rtstruct')
    namespace = {'pydicom': pydicom, 'tempfile': tempfile, 'os': os, 'Path': Path}
    exec(compile(ast.Module([node], []), 'run_task.py', 'exec'), namespace)
    return namespace['identify_rtstruct']


def rtstruct_file(path):
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.StructureSetLabel = 'RTSTRUCT', 'before'
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


def test_identity_rewrite_publishes_and_leaves_no_staging(tmp_path):
    path = rtstruct_file(tmp_path / 'job' / 'output' / 'total_RTSTRUCT.dcm')
    identify_function()(path, 'total')
    assert pydicom.dcmread(path).StructureSetName == 'RTSEG: total'
    assert sorted(p.name for p in (tmp_path / 'job').iterdir()) == ['output']
    assert [p.name for p in path.parent.iterdir()] == ['total_RTSTRUCT.dcm']


def test_identity_rewrite_interrupted_keeps_the_validated_file(tmp_path, monkeypatch):
    path = rtstruct_file(tmp_path / 'job' / 'output' / 'total_RTSTRUCT.dcm')
    before = path.read_bytes()
    def fail(self, target, **kwargs):
        Path(target).write_bytes(b'half written')
        raise OSError('disk full')
    monkeypatch.setattr(pydicom.dataset.Dataset, 'save_as', fail)
    with pytest.raises(OSError, match='disk full'):
        identify_function()(path, 'total')
    assert path.read_bytes() == before


@pytest.mark.parametrize('layout', ['unwritable-job', 'other-filesystem'])
def test_identity_rewrite_stages_inside_output_when_the_job_folder_is_unusable(tmp_path, monkeypatch, layout):
    path = rtstruct_file(tmp_path / 'job' / 'output' / 'total_RTSTRUCT.dcm')
    job, staged = path.parent.parent, []
    original = tempfile.TemporaryDirectory
    def track(*args, **kwargs):
        staged.append(Path(kwargs['dir']))
        return original(*args, **kwargs)
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', track)
    if layout == 'unwritable-job':
        monkeypatch.setattr(os, 'access', lambda target, mode: Path(target) != job)
    else:
        real = Path.stat
        def stat(self, **kwargs):
            result = real(self, **kwargs)
            if self == job:
                return SimpleNamespace(st_dev=result.st_dev + 1)
            return result
        monkeypatch.setattr(Path, 'stat', stat)
    identify_function()(path, 'total')
    assert staged == [path.parent]
    assert pydicom.dcmread(path).StructureSetName == 'RTSEG: total'
