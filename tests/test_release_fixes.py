"""Release-review regressions. Synthetic input and harmless CPU processes only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import dicom_series
from test_fifo_progress_integration import make_job, pause_dispatcher, stale_time
from test_multi_series import batch, series
from test_job_progress import fake_monitor


class PortalStopped(BaseException):
    """Abrupt loss bypasses ordinary exception rollback."""


def test_restart_during_valid_slice_moves_restores_upload(batch, synthetic_ct, monkeypatch):
    portal = batch.portal
    files = [('files', (p.name, p.read_bytes())) for p in sorted(synthetic_ct.iterdir())]
    assessed = batch.client.post('/api/uploads', files=files).json()
    uid, key = assessed['upload_id'], assessed['series'][0]['key']
    original_move = portal.shutil.move
    calls = 0
    def interrupt(source, target):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise PortalStopped()
        return original_move(source, target)
    monkeypatch.setattr(portal.shutil, 'move', interrupt)
    with pytest.raises(PortalStopped):
        portal.create_assessed_jobs(uid, [key], ['total'])
    job = next(p.parent for p in portal.BASE.glob('*/job.json'))
    assert portal.metadata(job)['status'] == 'staging'
    assert len(list((job / 'input').iterdir())) == 6
    assert len(portal.metadata(job)['input_manifest']['input']) == 12
    monkeypatch.setattr(portal.shutil, 'move', original_move)
    portal.recover_interrupted_jobs()
    data = portal.metadata(job)
    assert data['status'] == 'failed' and data['input_rolled_back'] is True
    assert data['error'] == portal.INPUT_PREPARATION_ERROR
    assert portal.validate_series(portal.BASE / uid / 'input') == (12, 'CT')
    assert not list((job / 'input').iterdir()) and not batch.enqueued
    assert batch.client.post(f'/api/v1/jobs/{job.name}/retry').status_code == 409
    result = portal.create_assessed_jobs(uid, [key], ['total'])
    assert len(result['ids']) == 1 and len(batch.enqueued) == 1
    assert portal.input_manifest_complete(batch.enqueued[0], portal.metadata(batch.enqueued[0]))


def test_restart_during_pet_companion_copy_rolls_back_entire_batch(batch, monkeypatch):
    portal = batch.portal
    rows = [series('ct'), series('pet', 'PT')]
    uid = batch.stage(rows)
    original = portal.shutil.copy2
    def interrupt(source, target):
        Path(target).write_bytes(b'partial')
        raise PortalStopped()
    monkeypatch.setattr(portal.shutil, 'copy2', interrupt)
    with pytest.raises(PortalStopped):
        portal.create_assessed_jobs(uid, ['ct', 'pet'], ['total', 'pet_pair'])
    jobs = [p.parent for p in portal.BASE.glob('*/job.json')]
    assert len(jobs) == 2 and all(portal.metadata(j)['status'] == 'staging' for j in jobs)
    monkeypatch.setattr(portal.shutil, 'copy2', original)
    portal.recover_interrupted_jobs()
    assert all(portal.metadata(j)['input_rolled_back'] for j in jobs)
    assert all(portal.metadata(j)['status'] == 'failed' for j in jobs)
    assert all(not (j / 'input_ct').exists() for j in jobs)
    assert sorted(p.name for p in (portal.BASE / uid / 'input').iterdir()) == ['ct.dcm', 'pet.dcm']
    assert not batch.enqueued


def test_unsafe_staging_rollback_fails_without_overwriting_upload(batch, monkeypatch):
    portal = batch.portal
    uid = batch.stage([series('ct')])
    move = portal.shutil.move
    def interrupt(source, target):
        move(source, target)
        Path(source).write_bytes(b'conflicting source')
        raise PortalStopped()
    monkeypatch.setattr(portal.shutil, 'move', interrupt)
    with pytest.raises(PortalStopped):
        portal.create_assessed_jobs(uid, ['ct'], ['total'])
    job = next(p.parent for p in portal.BASE.glob('*/job.json'))
    portal.recover_interrupted_jobs()
    assert portal.metadata(job)['status'] == 'failed'
    assert not portal.metadata(job)['input_rolled_back']
    assert portal.metadata(job)['error'] == portal.INPUT_PREPARATION_ERROR
    assert (portal.BASE / uid / 'input/ct.dcm').read_bytes() == b'conflicting source'
    assert not batch.enqueued


def test_restart_during_batch_commit_restores_queued_and_staging_siblings(batch, monkeypatch):
    portal = batch.portal
    uid = batch.stage([series('ct'), series('mr', 'MR')])
    save = portal.save_metadata
    commits = 0
    def interrupt(job, **changes):
        nonlocal commits
        if changes.get('status') == 'queued':
            commits += 1
            if commits == 2:
                raise PortalStopped()
        return save(job, **changes)
    monkeypatch.setattr(portal, 'save_metadata', interrupt)
    with pytest.raises(PortalStopped):
        portal.create_assessed_jobs(uid, ['ct', 'mr'], ['total', 'total_mr'])
    jobs = [p.parent for p in portal.BASE.glob('*/job.json')]
    assert sorted(portal.metadata(j)['status'] for j in jobs) == ['queued', 'staging']
    monkeypatch.setattr(portal, 'save_metadata', save)
    portal.recover_interrupted_jobs()
    assert all(portal.metadata(j)['status'] == 'failed' for j in jobs)
    assert sorted(p.name for p in (portal.BASE / uid / 'input').iterdir()) == ['ct.dcm', 'mr.dcm']
    assert not batch.enqueued


@pytest.mark.parametrize('damage', ['missing_slice', 'truncated_companion'])
def test_queued_manifest_rejects_partial_input_on_restart(batch, damage):
    portal = batch.portal
    uid = batch.stage([series('ct'), series('pet', 'PT')])
    portal.create_assessed_jobs(uid, ['pet'], ['pet_pair'])
    job = batch.enqueued.pop()
    if damage == 'missing_slice':
        (job / 'input/pet.dcm').unlink()
    else:
        (job / 'input_ct/ct.dcm').write_bytes(b'x')
    portal.recover_interrupted_jobs()
    assert portal.metadata(job)['status'] == 'failed' and not batch.enqueued
    assert portal.metadata(job)['error'] == portal.INPUT_PREPARATION_ERROR
    assert batch.client.post(f'/api/v1/jobs/{job.name}/retry').status_code == 409


def test_dispatch_checks_manifest_again(portal, monkeypatch):
    job = make_job(portal, 'changed-after-recovery', 1)
    portal.save_metadata(job, input_manifest=portal.job_input_manifest(job))
    (job / 'input/synthetic.txt').write_bytes(b'truncated')
    starts = pause_dispatcher(portal, monkeypatch)
    portal.enqueue_jobs([job], target=lambda job: pytest.fail('Partial input dispatched'))
    starts[0].target(*starts[0].args)
    assert portal.metadata(job)['status'] == 'failed'


def test_legacy_queued_job_with_incomplete_assessed_slice_count_is_not_recovered(portal, synthetic_ct, monkeypatch):
    job = make_job(portal, 'legacy-partial', 1)
    (job / 'input/synthetic.txt').unlink()
    for p in sorted(synthetic_ct.iterdir())[:6]:
        (job / 'input' / p.name).write_bytes(p.read_bytes())
    portal.save_metadata(job, input_manifest=None, series={'slices': 12})
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda *a, **k: pytest.fail('Legacy partial input queued'))
    portal.recover_interrupted_jobs()
    assert portal.metadata(job)['status'] == 'failed'


def test_interrupted_resumable_staging_never_dispatches_nonempty_input(portal, synthetic_ct, monkeypatch):
    client = TestClient(portal.app)
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': [{'name': 'total', 'enabled': True, 'modality': 'CT'}]})
    sid = client.post('/api/v1/upload-sessions').json()['session_id']
    for i, p in enumerate(sorted(synthetic_ct.iterdir())):
        payload = p.read_bytes()
        assert client.put(f'/api/v1/upload-sessions/{sid}/files/{i}',
                          params={'name': p.name, 'offset': 0, 'size': len(payload)}, content=payload).status_code == 200
    starts = pause_dispatcher(portal, monkeypatch)
    result = portal.queue_session_job(sid, {'models': ['total']}, False)
    job = portal.BASE / result['id']
    assert portal.metadata(job)['status'] == 'staging'
    session = portal.BASE / sid
    (session / 'files/0').replace(job / 'input/slice-00.dcm')
    # Simulate a new portal: no queue reservations or staging thread survive.
    portal.pending_jobs.clear(); portal.queue_thread = None; portal.queue_running = False
    starts.clear()
    portal.recover_interrupted_jobs()
    assert portal.metadata(job)['status'] == 'failed' and not starts
    assert portal.metadata(job)['error'] == portal.INPUT_PREPARATION_ERROR


@pytest.mark.parametrize('recovery_count', [0, 3])
def test_restart_stops_surviving_idle_runner_tree_before_dispatch(portal, monkeypatch, tmp_path, recovery_count):
    marker = tmp_path / 'child.pid'
    child = ('import os,signal,time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
             f'Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)')
    parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session=True); time.sleep(60)'
    runner = subprocess.Popen([sys.executable, '-c', parent])
    child_pid = child_started = None
    monkeypatch.setattr(portal, 'TASK_STOP_GRACE_SECONDS', .05)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        child_pid = int(marker.read_text()); child_started = portal.proc_stat(child_pid)[2]
        job = make_job(portal, 'survivor', 1, 'processing', task_pid=runner.pid,
                       task_pid_started=portal.proc_stat(runner.pid)[2], recovery_count=recovery_count)
        assert not portal.pending_jobs and not portal.job_workers and portal.queue_current is None
        order = []
        def run(job):
            assert runner.poll() is not None
            child_stat = portal.proc_stat(child_pid)
            assert child_stat is None or child_stat[0] == 'Z'
            order.append(job)
            portal.save_metadata(job, status='complete')
        monkeypatch.setattr(portal, 'process', run)
        # Real startup recovery and dispatcher. The later job cannot pass a survivor.
        portal.recover_interrupted_jobs()
        later = make_job(portal, 'later', 2)
        portal.enqueue_jobs([later])
        portal.queue_thread.join(10)
        assert not portal.queue_thread.is_alive()
        assert order == ([job, later] if recovery_count == 0 else [later])
        assert portal.metadata(job)['recovery_count'] == recovery_count + 1
        if recovery_count == 3:
            assert portal.metadata(job)['status'] == 'failed'
    finally:
        if child_pid:
            stat = portal.proc_stat(child_pid)
            if stat and stat[0] != 'Z' and stat[2] == child_started:
                os.kill(child_pid, signal.SIGKILL)
        if runner.poll() is None:
            runner.kill()
        runner.wait()


def test_pid_start_mismatch_never_signals_unrelated_process(portal, monkeypatch):
    monkeypatch.setattr(portal, 'proc_stat', lambda pid: ('S', 1, 'new-start'))
    monkeypatch.setattr(portal.os, 'kill', lambda *a: pytest.fail('Signalled a reused PID'))
    portal.stop_recorded_task_tree({'task_pid': 42, 'task_pid_started': 'old-start'})


def test_unconfirmed_stop_blocks_dispatch_until_reconciliation_succeeds(portal, monkeypatch):
    old = make_job(portal, 'old', 1, 'processing')
    later = make_job(portal, 'later', 2)
    starts = pause_dispatcher(portal, monkeypatch)
    def stop(data):
        raise RuntimeError('Synthetic stop failure')
    monkeypatch.setattr(portal, 'stop_recorded_task_tree', stop)
    with pytest.raises(RuntimeError, match='Synthetic stop failure'):
        portal.recover_interrupted_jobs()
    portal.enqueue_jobs([later])
    assert not starts and old in portal.recovery_barriers
    monkeypatch.setattr(portal, 'stop_recorded_task_tree', lambda data: None)
    portal.recover_interrupted_jobs()
    assert len(starts) == 1 and not portal.recovery_barriers
    assert [entry[0] for entry in portal.pending_jobs] == [old, later]


def test_stall_retry_keeps_original_absolute_deadline(portal, monkeypatch):
    job = make_job(portal, 'timeout-across-retry', 1)
    clock, launched, killed = fake_monitor(portal, monkeypatch)
    monkeypatch.setattr(portal, 'TASK_TIME_LIMIT_SECONDS', 40)
    def cpu(pid):
        return {(pid, 'start'): int(clock.now) if len(launched) == 2 else 0}
    monkeypatch.setattr(portal, 'process_cpu_times', cpu)
    with pytest.raises(subprocess.TimeoutExpired):
        portal.run_local_task(job, 'total', 1, 1)
    data = portal.metadata(job)
    assert len(launched) == len(killed) == 2
    assert clock.now == 140
    assert data['task_timings']['total']['attempt_started'] > data['task_timings']['total']['started']
    assert data['task_timings']['total']['duration'] == 40
    assert data['task_timings']['total']['status'] == 'failed'
    assert [e['action'] for e in data['stall_events']] == ['retry']


def test_manual_retry_joins_after_already_waiting_jobs(portal, monkeypatch):
    old = make_job(portal, 'old-failed', 1, 'failed')
    waiting = make_job(portal, 'already-waiting', 2)
    starts = pause_dispatcher(portal, monkeypatch)
    order = []
    def run(job):
        order.append(job)
        portal.save_metadata(job, status='complete')
    monkeypatch.setattr(portal, 'process', run)
    portal.enqueue_jobs([waiting])
    assert portal.retry_job(old.name)['status'] == 'queued'
    later = make_job(portal, 'later', 3)
    portal.enqueue_jobs([later])
    assert [entry[0] for entry in portal.pending_jobs] == [waiting, old, later]
    assert [portal.metadata(j)['queue_sequence'] for j in [waiting, old, later]] == [1, 2, 3]
    starts[0].target(*starts[0].args)
    assert order == [waiting, old, later]


def test_recovery_does_not_reorder_existing_manual_retry_reservations(portal, monkeypatch):
    old = make_job(portal, 'old-failed', 1, 'failed')
    orphan = make_job(portal, 'orphan', 2, 'processing')
    waiting = make_job(portal, 'already-waiting', 3)
    pause_dispatcher(portal, monkeypatch)
    portal.enqueue_jobs([waiting])
    portal.retry_job(old.name)
    portal.recover_interrupted_jobs([orphan], orphaned=True)
    assert [entry[0] for entry in portal.pending_jobs] == [orphan, waiting, old]


def test_dicom_order_cache_reuses_scan_returns_fresh_list_and_invalidates(synthetic_ct, monkeypatch):
    reader = dicom_series.sitk.ImageSeriesReader
    scan, names = reader.GetGDCMSeriesIDs, reader.GetGDCMSeriesFileNames
    calls = []
    monkeypatch.setattr(reader, 'GetGDCMSeriesIDs', lambda source: (calls.append(source), scan(source))[1])
    dicom_series._ordered_dicom_files.cache_clear()
    expected = list(names(str(synthetic_ct), scan(str(synthetic_ct))[0]))
    first = dicom_series.ordered_dicom_files(synthetic_ct)
    assert first == expected
    first.clear()
    assert dicom_series.ordered_dicom_files(synthetic_ct / '..' / synthetic_ct.name) == expected
    assert len(calls) == 1
    before = synthetic_ct.stat().st_mtime_ns
    (synthetic_ct / 'ignored.txt').write_text('Synthetic ignored file')
    # Entry count also invalidates when directory timestamps happen to match.
    os.utime(synthetic_ct, ns=(synthetic_ct.stat().st_atime_ns, before))
    assert dicom_series.ordered_dicom_files(synthetic_ct) == expected
    assert len(calls) == 2
    os.utime(synthetic_ct, ns=(synthetic_ct.stat().st_atime_ns, before + 1_000_000))
    assert dicom_series.ordered_dicom_files(synthetic_ct) == expected
    assert len(calls) == 3


def test_explicit_viewer_retry_rebuilds_failed_index(portal, synthetic_ct, monkeypatch):
    import numpy as np
    import viewer
    from test_viewer import make_job as viewer_job, make_rtstruct
    job = viewer_job(portal, synthetic_ct)
    make_rtstruct(job / 'input', job / 'output/model_RTSTRUCT.dcm', np.ones((32, 32, 12), dtype=bool))
    original = viewer.read_structure
    reads = []
    def transient_failure(path):
        reads.append(path)
        if len(reads) == 1:
            raise ValueError('Synthetic transient indexing error')
        return original(path)
    monkeypatch.setattr(viewer, 'read_structure', transient_failure)
    client = TestClient(portal.app)
    base = f'/api/v1/jobs/{job.name}/viewer'
    assert client.get(base).status_code == 200
    with pytest.raises(ValueError, match='Synthetic transient'):
        next(iter(portal.viewer.overlay_cache.items.values())).result(5)
    assert client.get(base).json()['contours_state'] == 'failed'
    assert client.get(base).json()['contours_state'] == 'failed'
    assert len(reads) == 1
    assert client.get(base + '?retry_contours=true').status_code == 200
    next(iter(portal.viewer.overlay_cache.items.values())).result(5)
    assert client.get(base).json()['contours_ready'] is True
    assert len(reads) == 2


@pytest.mark.parametrize('reuse_during', ['initial_scan', 'shutdown_scan'])
def test_process_scan_reuse_never_adopts_unrelated_descendants(portal, monkeypatch, reuse_during):
    phase = {'reused': False, 'scans': 0}
    signals = []
    def stat(pid):
        if pid == 42:
            return ('S', 1, 'new' if phase['reused'] else 'original')
        return ('S', 42, 'unrelated')
    def tree(pid):
        phase['scans'] += 1
        if reuse_during == 'initial_scan':
            phase['reused'] = True
            return {42: 'original', 43: 'unrelated'}
        if phase['scans'] == 1:
            return {42: 'original'}
        phase['reused'] = True
        return {42: 'new', 43: 'unrelated'}
    monkeypatch.setattr(portal, 'proc_stat', stat)
    monkeypatch.setattr(portal, 'process_tree', tree)
    monkeypatch.setattr(portal.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(portal, 'TASK_STOP_GRACE_SECONDS', 0)
    portal.stop_recorded_task_tree({'task_pid': 42, 'task_pid_started': 'original'})
    assert signals == ([] if reuse_during == 'initial_scan' else [(42, signal.SIGTERM)])


@pytest.mark.parametrize('remote', [None, '/jobs/synthetic'])
def test_expired_task_budget_never_launches_an_attempt(portal, monkeypatch, remote):
    job = make_job(portal, 'expired-budget', 1)
    monkeypatch.setattr(portal, 'TASK_TIME_LIMIT_SECONDS', -1)
    monkeypatch.setattr(portal.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Launched with an expired budget'))
    with pytest.raises(subprocess.TimeoutExpired):
        portal.run_monitored_task(job, 'total', 1, 1, ['synthetic-runner'], remote=remote)
    assert portal.metadata(job)['task_timings']['total']['status'] == 'failed'


@pytest.mark.parametrize('slices', [None, 1])
def test_legacy_input_without_manifest_is_not_assumed_complete(portal, monkeypatch, slices):
    job = make_job(portal, 'legacy-unknown-input', 1)
    portal.save_metadata(job, input_manifest=None, series={'slices': slices} if slices else None)
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda *a, **kw: pytest.fail('Unverified legacy input queued'))
    portal.recover_interrupted_jobs()
    assert portal.metadata(job)['status'] == 'failed'
    assert portal.metadata(job)['error'] == portal.INPUT_PREPARATION_ERROR
    with pytest.raises(portal.HTTPException) as error:
        portal.retry_job(job.name)
    assert error.value.status_code == 409


def test_pre_manifest_jobs_keep_retry_with_their_complete_retained_series(portal, synthetic_ct, monkeypatch):
    # Jobs written before input manifests existed had their metadata saved only after the series was in place.
    job = portal.BASE / 'pre-manifest-failed'
    (job / 'input').mkdir(parents=True); (job / 'output').mkdir()
    for p in sorted(synthetic_ct.iterdir()):
        (job / 'input' / p.name).write_bytes(p.read_bytes())
    (job / 'job.json').write_text(json.dumps(dict(id=job.name, models=['total'], status='failed', created=1,
                                                  updated=1, error='synthetic failure')))
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda jobs, **kw: None)
    assert portal.retry_job(job.name)['status'] == 'queued'
    # An empty retained input is still refused, and a recorded series size must match.
    portal.save_metadata(job, status='failed', series={'slices': 99})
    with pytest.raises(portal.HTTPException):
        portal.retry_job(job.name)


def test_failed_resumable_staging_never_passes_the_pre_manifest_fallback(portal, synthetic_ct, monkeypatch):
    # Review round 2: an ordinary move error after six of twelve slices must not make the job retryable.
    client = TestClient(portal.app)
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': [{'name': 'total', 'enabled': True, 'modality': 'CT'}]})
    sid = client.post('/api/v1/upload-sessions').json()['session_id']
    for i, p in enumerate(sorted(synthetic_ct.iterdir())):
        payload = p.read_bytes()
        assert client.put(f'/api/v1/upload-sessions/{sid}/files/{i}',
                          params={'name': p.name, 'offset': 0, 'size': len(payload)}, content=payload).status_code == 200
    starts = pause_dispatcher(portal, monkeypatch)
    moves, real_move = [], portal.shutil.move
    def failing_move(src, dst, *args, **kwargs):
        if len(moves) == 6:
            raise OSError('Synthetic move failure')
        moves.append(src)
        return real_move(src, dst, *args, **kwargs)
    monkeypatch.setattr(portal.shutil, 'move', failing_move)
    result = portal.queue_session_job(sid, {'models': ['total']}, False)
    job = portal.BASE / result['id']
    assert 'input_manifest' in json.loads((job / 'job.json').read_text())
    for start in list(starts):
        start.target(*start.args)
    data = portal.metadata(job)
    assert data['status'] == 'failed' and len(list((job / 'input').iterdir())) == 6
    assert data['input_manifest'] is None and not portal.legacy_input_complete(job, data)
    with pytest.raises(portal.HTTPException) as error:
        portal.retry_job(job.name)
    assert error.value.status_code == 409


def test_retention_keeps_a_job_whose_stop_is_unconfirmed(portal, monkeypatch):
    # Review round 2: deleting a barrier job would leave the dispatcher blocked with nothing to reconcile.
    old = make_job(portal, 'expired-orphan', 1, 'processing')
    later = make_job(portal, 'later', time.time())
    starts = pause_dispatcher(portal, monkeypatch)
    monkeypatch.setattr(portal, 'stop_recorded_task_tree', lambda data: (_ for _ in ()).throw(RuntimeError('stop')))
    with pytest.raises(RuntimeError):
        portal.recover_interrupted_jobs()
    portal.enqueue_jobs([later])
    assert old in portal.recovery_barriers and not starts
    portal.cleanup_once()
    assert old.is_dir()
    monkeypatch.setattr(portal, 'stop_recorded_task_tree', lambda data: None)
    portal.recover_interrupted_jobs()
    assert not portal.recovery_barriers and len(starts) == 1


def test_retention_removes_only_finished_jobs_and_waits_for_recovery_locks(portal, monkeypatch):
    # Review round 3: retention must not race recovery. Active jobs are left to the watchdog and recovery;
    # finished ones are detached under the same locks and deleted outside them.
    import threading
    monkeypatch.setattr(portal, 'RETENTION_HOURS', 1)
    active = make_job(portal, 'expired-active', 1, 'processing')
    finished = make_job(portal, 'expired-finished', 1, 'failed')
    racing = make_job(portal, 'expired-racing', 1, 'failed')
    leftover = portal.BASE / portal.EXPIRED_DIR / 'old-leftover'
    leftover.mkdir(parents=True)
    # Recovery holds its locks and turns a failed job into a queued one (as a retry would) while retention runs.
    portal.submission_lock.acquire()
    cleaner = threading.Thread(target=portal.cleanup_once)
    cleaner.start()
    time.sleep(0.3)
    assert racing.is_dir()  # retention is waiting for the locks
    portal.save_metadata(racing, status='queued')
    portal.submission_lock.release()
    cleaner.join(10)
    assert active.is_dir() and racing.is_dir() and not finished.exists() and not leftover.exists()
    assert {j['id'] for j in portal.list_jobs()['jobs']} >= {'expired-active', 'expired-racing'}
