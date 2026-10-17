"""Synthetic CPU-only progress, read-only log, recovery and watchdog checks."""
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from job_progress import LOG_BYTE_CAP, TaskActivity, log_lines, parse_progress, tail_log
from task_progress import Progress, command_stage


def make_job(portal, models=('total',), **extra):
    job = portal.BASE / 'progress-test'
    (job / 'input').mkdir(parents=True)
    (job / 'input/synthetic.txt').write_text('synthetic input')
    (job / 'output').mkdir()
    (job / 'job.json').write_text(json.dumps({'id': job.name, 'status': 'processing', 'created': 100,
                                            'updated': 100, 'models': list(models), **extra}))
    return job


def test_progress_lines_are_json_flushed_and_rate_limited(capsys, monkeypatch):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr('task_progress.time.monotonic', lambda: clock.now)
    progress = Progress()
    progress.emit('rtstruct_contours', 0, 117)
    progress.emit('rtstruct_contours', 1, 117)
    clock.now = 1
    progress.emit('rtstruct_contours', 43, 117)
    progress.emit('rtstruct_validation', 0, 117)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert all(line.startswith('RTSEG_PROGRESS ') for line in lines)
    assert json.loads(lines[1].split(' ', 1)[1]) == {'stage': 'rtstruct_contours', 'done': 43, 'total': 117}


def test_parse_newest_stage_and_tqdm_noise():
    text = 'Predicting 0%\rPredicting 50%\rPredicting 100%\nRTSEG_PROGRESS {"stage":"rtstruct_validation","done":43,"total":117}\n'
    assert parse_progress(text) == {'stage_key': 'rtstruct_validation', 'stage': 'Validating RTSTRUCT geometry',
                                    'stage_done': 43, 'stage_total': 117}
    assert parse_progress(text + '\rResampling images 12%\r')['stage_key'] == 'resampling'
    assert parse_progress('Saving output')['stage_key'] == 'saving'
    assert log_lines('\x1b[32mPredicting 0%\rPredicting 100%\x1b[0m\r\nSaved\n') == ['Predicting 100%', 'Saved']
    assert parse_progress('RTSEG_PROGRESS {"stage":"rtstruct_validation","done":null,"total":5}') == {
        'stage_key': 'rtstruct_validation', 'stage': 'Validating RTSTRUCT geometry'}
    assert not parse_progress('RTSEG_PROGRESS invalid\nRTSEG_PROGRESS {"stage":[]}')


def test_activity_uses_cpu_and_logs_not_metadata_updates(tmp_path):
    log = tmp_path / 'total.log'; log.write_text('')
    activity = TaskActivity(100)
    activity.sample(log, 120, {(5, 'start'): 100})
    assert not activity.stalled(139, 20)
    activity.sample(log, 140, {(5, 'start'): 101})
    assert not activity.stalled(150, 20)
    assert activity.stalled(160, 20)
    assert activity.stalled(140, 20, remote=True)
    log.write_text('RTSEG_PROGRESS {"stage":"rtstruct_validation","done":1,"total":4}\n')
    fields = activity.sample(log, 161)
    assert fields['last_activity'] == 161 and fields['stage_started'] == 161
    fields = activity.sample(log, 170)
    assert fields['last_activity'] == 161 and fields['stage_started'] == 161
    assert not activity.stalled(170, 20)
    with log.open('a') as stream:
        stream.write('RTSEG_PROGRESS {"stage":"rtstruct_validation","done":2,"total":4}\n')
    fields = activity.sample(log, 172)
    assert fields['stage_done'] == 2 and fields['stage_started'] == 161


def test_log_endpoint_caps_tail_and_rejects_unknown_tasks_and_traversal(portal):
    job = make_job(portal, current_task='total')
    text = 'X' * (LOG_BYTE_CAP + 10) + '\n' + ''.join(f'line {i}\n' for i in range(220)) + 'Predicting 0%\rPredicting 100%\n'
    (job / 'total.log').write_text(text)
    client = TestClient(portal.app)
    result = client.get(f'/api/v1/jobs/{job.name}/log?lines=40').json()
    assert len(result['lines']) == 40 and result['lines'][-1] == 'Predicting 100%'
    assert len(client.get(f'/api/v1/jobs/{job.name}/log?task=total&lines=999999').json()['lines']) == 200
    assert len(client.get(f'/api/v1/jobs/{job.name}/log?lines=-1').json()['lines']) == 1
    assert client.get(f'/api/v1/jobs/{job.name}/log?task=other').status_code == 404
    assert client.get(f'/api/v1/jobs/{job.name}/log?task=../total').status_code == 404
    assert client.get('/api/v1/jobs/bad_name/log?task=total').status_code == 404
    assert client.get('/api/v1/jobs/missing/log?task=total').status_code == 404
    (job / 'total.log').unlink()
    assert client.get(f'/api/v1/jobs/{job.name}/log').json()['lines'] == []
    (job / 'total.log').symlink_to(job / 'job.json')
    assert client.get(f'/api/v1/jobs/{job.name}/log').status_code == 404


def fake_monitor(portal, monkeypatch, *, busy=False):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(portal.time, 'time', lambda: clock.now)
    monkeypatch.setattr(portal.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(portal.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    monkeypatch.setattr(portal, 'RTSEG_STALL_SECONDS', 20)
    monkeypatch.setattr(portal, 'RTSEG_STALL_RETRIES', 1)
    monkeypatch.setattr(portal, 'TASK_MONITOR_SECONDS', 5)
    monkeypatch.setattr(portal, 'proc_stat', lambda pid: ('R', 0, 'start'))
    cpu = SimpleNamespace(ticks=0)
    def sample(root):
        if busy: cpu.ticks += 10
        return {(root, 'start'): cpu.ticks}
    monkeypatch.setattr(portal, 'process_cpu_times', sample)
    launched, killed = [], []
    class FakeProcess:
        pid = 4567
        returncode = None
        def __init__(self, args, **kwargs):
            self.args = args; launched.append(self)
            kwargs['stdout'].write('RTSEG_PROGRESS {"stage":"rtstruct_validation","done":1,"total":5}\n')
            kwargs['stdout'].flush()
        def poll(self):
            if busy and clock.now >= 155: self.returncode = 0
            return self.returncode
    def stop(process):
        killed.append(process); process.returncode = -15
    monkeypatch.setattr(portal.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(portal, 'stop_process_tree', stop)
    return clock, launched, killed


def test_silent_cpu_busy_task_is_not_stalled(portal, monkeypatch):
    job = make_job(portal)
    clock, launched, killed = fake_monitor(portal, monkeypatch, busy=True)
    portal.run_local_task(job, 'total', 1, 1)
    data = portal.metadata(job)
    assert not killed and len(launched) == 1
    assert data['last_cpu_activity'] >= 150 and not data.get('stall_events')
    assert data['task_timings']['total']['status'] == 'done'


def test_idle_task_killed_retried_once_then_failed_without_rerunning_completed(portal, monkeypatch):
    job = make_job(portal, ('already', 'total'), completed_models=['already'])
    retained = job / 'output/already_RTSTRUCT.dcm'; retained.write_text('completed')
    partial = job / 'output/total_RTSTRUCT.dcm'; partial.write_text('partial')
    work = job / 'external/total'; work.mkdir(parents=True); (work / 'partial.txt').write_text('partial')
    clock, launched, killed = fake_monitor(portal, monkeypatch)
    portal.process(job)
    data = portal.metadata(job)
    assert data['status'] == 'failed'
    assert 'total stalled during validating RTSTRUCT geometry for 0.333333 min; retried once' in data['error']
    assert len(launched) == len(killed) == 2
    assert [event['action'] for event in data['stall_events']] == ['retry', 'failed']
    assert all(event['task'] == 'total' for event in data['stall_events'])
    assert retained.read_text() == 'completed' and not partial.exists() and not work.exists()
    assert (job / 'total.log').read_text().count('RTSEG_PROGRESS') == 2
    assert data['completed_models'] == ['already']
    assert data['task_timings']['total']['attempts'] == 2


def test_remote_stall_stops_remote_and_local_process_and_clears_only_task(portal, monkeypatch):
    job = make_job(portal)
    fake_monitor(portal, monkeypatch)
    remote_stops, cleanup = [], []
    monkeypatch.setattr(portal, 'stop_remote_task', lambda remote, task: remote_stops.append((remote, task)))
    monkeypatch.setattr(portal, 'reset_task_output', lambda job, task, remote: cleanup.append(task))
    with pytest.raises(RuntimeError, match='retried once'):
        portal.run_worker_task(job, '/jobs/synthetic', 'total', 1, 1)
    assert remote_stops == [('/jobs/synthetic', 'total')] * 2 and cleanup == ['total']


def test_proc_cpu_sampling_counts_children_and_checks_start_time(portal, monkeypatch):
    monkeypatch.setattr(portal, 'process_tree', lambda root: {11: '123', 12: '456', 13: 'old'})
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path).startswith('/proc/'):
            pid = int(path.parts[2]); fields = ['0']*22
            fields[0] = 'R'; fields[11], fields[12] = '100', str(pid)
            fields[13], fields[14] = '7', '9'
            fields[19] = {11:'123',12:'456',13:'new'}[pid]
            return f'{pid} (name with ) spaces) ' + ' '.join(fields)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    assert portal.process_cpu_times(11) == {(11,'123'):127, (12,'456'):128}


def test_watchdog_requeues_orphan_via_recovery_and_obeys_limit(portal, monkeypatch):
    job = make_job(portal, current_task='total', stage='Running model inference')
    started = []
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda jobs: started.extend(job.name for job in jobs))
    portal.watchdog_once(1400)
    data = portal.metadata(job)
    assert data['status'] == 'queued' and data['recovery_count'] == 1 and started == [job.name]
    assert data['stall_events'][0]['action'] == 'orphan_recovery'
    portal.save_metadata(job, status='processing', recovery_count=3)
    portal.watchdog_once(data['updated'] + 1300)
    assert portal.metadata(job)['status'] == 'failed'
    assert 'automatic recovery limit' in portal.metadata(job)['error']


def test_watchdog_ignores_live_thread_live_process_and_recent_metadata(portal, monkeypatch):
    job = make_job(portal)
    monkeypatch.setattr(portal, 'recover_interrupted_jobs', lambda jobs, **kwargs: pytest.fail('unexpected recovery'))
    portal.watchdog_once(110)
    portal.job_workers[job.name] = SimpleNamespace(is_alive=lambda: True)
    portal.watchdog_once(2000)
    portal.job_workers.clear()
    portal.save_metadata(job, task_pid=42, task_pid_started='start')
    monkeypatch.setattr(portal, 'proc_stat', lambda pid: ('R', 1, 'start'))
    portal.watchdog_once(portal.metadata(job)['updated'] + 1300)


@pytest.mark.parametrize('name,stage', [
    ('dicom_series_to_nifti.py', 'conversion'), ('run_nv_segment.py', 'inference'),
    ('keep_labels.py', 'postprocessing'), ('combine_sct_outputs.py', 'postprocessing'),
    ('convert_validate_rtstruct.py', 'rtstruct_contours'),
    ('convert_masks_validate_rtstruct.py', 'rtstruct_contours'),
    ('pet_bqml_to_suv1000.py', 'staging'), ('resample_image_to_reference.py', 'resampling')])
def test_runner_command_stages(name, stage):
    assert command_stage(['python', '/app/' + name, '--input']) == stage


def test_no_stall_retry_when_disabled(portal, monkeypatch):
    job = make_job(portal)
    _, launched, killed = fake_monitor(portal, monkeypatch)
    monkeypatch.setattr(portal, 'RTSEG_STALL_RETRIES', 0)
    with pytest.raises(RuntimeError, match='stalled during'):
        portal.run_local_task(job, 'total', 1, 1)
    assert len(launched) == len(killed) == 1
    assert [event['action'] for event in portal.metadata(job)['stall_events']] == ['failed']


def test_orphan_recovery_preserves_finished_outputs(portal, monkeypatch):
    job = make_job(portal, ('already', 'total'), completed_models=['already'], current_task='total')
    retained = job / 'output/already_RTSTRUCT.dcm'; retained.write_text('finished')
    partial = job / 'output/total_RTSTRUCT.dcm'; partial.write_text('partial')
    class FakeThread:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def is_alive(self): return True
    monkeypatch.setattr(portal.threading, 'Thread', FakeThread)
    portal.watchdog_once(1400)
    assert retained.read_text() == 'finished' and not partial.exists()
    assert portal.metadata(job)['completed_models'] == ['already']


def test_final_progress_line_updates_detail_when_worker_exits_before_first_poll(portal, monkeypatch):
    job = make_job(portal)
    class Finished:
        returncode = 0
        def __init__(self, command, **kwargs):
            kwargs['stdout'].write('RTSEG_PROGRESS {"stage":"export","done":1,"total":1}\n')
            kwargs['stdout'].flush()
        def poll(self): return 0
    monkeypatch.setattr(portal.subprocess, 'Popen', Finished)
    portal.run_local_task(job, 'total', 1, 1)
    data = portal.metadata(job)
    assert data['stage'] == 'Exporting results'
    assert data['stage_done'] == data['stage_total'] == 1
    assert data['detail'] == 'total (1/1): exporting results'


def test_result_packaging_publishes_export_phase_without_stale_counts(portal, monkeypatch):
    job = make_job(portal, current_task='total', stage='Validating RTSTRUCT geometry',
                   stage_done=43, stage_total=117)
    (job / 'output/total_RTSTRUCT.dcm').write_bytes(b'synthetic packaging fixture')
    observed = []
    original = portal.zipfile.ZipFile
    def capture(*args, **kwargs):
        observed.append(portal.metadata(job))
        return original(*args, **kwargs)
    monkeypatch.setattr(portal.zipfile, 'ZipFile', capture)
    portal.finalize_results(job)
    assert observed[0]['status'] == 'retrieving'
    assert observed[0]['detail'] == 'Preparing results for download'
    assert observed[0]['stage'] == 'Exporting results'
    assert observed[0]['stage_done'] is observed[0]['stage_total'] is None
    assert portal.metadata(job)['status'] == 'complete'


def test_retry_error_does_not_reuse_prior_attempt_oom_text(portal, monkeypatch):
    job = make_job(portal)
    _, launched, killed = fake_monitor(portal, monkeypatch)
    class Attempt:
        pid = 4567
        def __init__(self, command, **kwargs):
            self.args = command
            launched.append(self)
            self.returncode = None if len(launched) == 1 else 1
            if len(launched) == 1:
                kwargs['stdout'].write('CUDA out of memory\nRTSEG_PROGRESS {"stage":"rtstruct_validation"}\n')
                kwargs['stdout'].flush()
        def poll(self): return self.returncode
    monkeypatch.setattr(portal.subprocess, 'Popen', Attempt)
    with pytest.raises(RuntimeError, match='total exited with status 1') as error:
        portal.run_local_task(job, 'total', 1, 1)
    assert not portal.is_gpu_oom(error.value)
    assert len(launched) == 2 and len(killed) == 1
    assert 'CUDA out of memory' in (job / 'total.log').read_text()
