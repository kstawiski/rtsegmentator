"""CPU tests: SSH and model processes are replaced by deterministic fakes."""
import json
import shlex
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nv_reason_ct_report import TASK, write_report


def job_at(portal, models):
    job = portal.BASE / 'remote-test'
    (job / 'input').mkdir(parents=True)
    (job / 'output').mkdir()
    (job / 'input/image.dcm').write_bytes(b'synthetic input')
    (job / 'job.json').write_text(json.dumps({'id':job.name, 'models':models, 'status':'queued'}))
    return job


def draft(job):
    write_report(job / 'output/report', dict(status='not_generated', reason='Synthetic refusal',
                 coverage={}, regions=[], segmentation_summary=[]))


@pytest.mark.parametrize('mode', ['local', 'auto'])
def test_routes_report_last_and_keeps_other_tasks_local(portal, monkeypatch, mode):
    monkeypatch.setattr(portal, 'WORKER_MODE', mode)
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {TASK})
    job = job_at(portal, [TASK, 'total', 'body'])
    calls = []
    def local(job, task, *args):
        calls.append(('local', task))
        (job / 'output' / (task + '_RTSTRUCT.dcm')).write_bytes(task.encode())
    def remote(job, task, *args):
        calls.append(('remote', task))
        assert (job / 'output/total_RTSTRUCT.dcm').exists()
        draft(job)
    monkeypatch.setattr(portal, 'run_local_task', local)
    monkeypatch.setattr(portal, 'run_remote_gpu_task', remote)
    portal.process(job)
    assert calls == [('local','total'), ('local','body'), ('remote',TASK)]
    assert portal.metadata(job)['status'] == 'complete'
    assert portal.metadata(job)['report_status'] == 'not_generated'
    assert (job / 'output/total_RTSTRUCT.dcm').read_bytes() == b'total'


def test_transfer_and_task_scoped_retrieval_cleanup(portal, monkeypatch):
    job = job_at(portal, ['total', TASK])
    (job / 'input_ct').mkdir()
    (job / 'prompt.json').write_text('{}')
    contour = job / 'output/total_RTSTRUCT.dcm'; contour.write_bytes(b'keep me')
    (job / 'output/prostate_sbrt_crop').mkdir()
    (job / 'output/report').mkdir()
    assert portal.worker_archive_members(job, [TASK]) == ['input','input_ct','prompt.json','output/total_RTSTRUCT.dcm']
    assert portal.worker_archive_members(job, ['body']) == ['input','input_ct','prompt.json']
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'scp': draft(job)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(portal.subprocess, 'run', run)
    monkeypatch.setattr(portal, 'transfer_to_worker', lambda *args: calls.append(['transfer', args[2]]))
    monkeypatch.setattr(portal, 'run_worker_task', lambda *args: calls.append(['inference', args[2]]))
    portal.run_remote_gpu_task(job, TASK, 2, 2)
    tar = next(c for c in calls if c[0] == 'tar')
    assert tar[5:] == portal.worker_archive_members(job, [TASK])
    retrieval = next(i for i,c in enumerate(calls) if c[0] == 'scp')
    cleanup = next(i for i,c in enumerate(calls) if c[0] == 'ssh' and 'rm -rf' in c[-1])
    assert calls[retrieval][-2].endswith('/output/report')
    assert retrieval < cleanup
    assert contour.read_bytes() == b'keep me'
    assert not (job / 'selected-series.tar').exists()
    assert not portal.metadata(job)['remote_results_retained']
    calls.clear()
    portal.retrieve_worker_results(job, '/jobs/test', 'body')
    assert calls[0][-2].endswith('/output/body_RTSTRUCT.dcm*')


def test_failed_retrieval_retains_remote_directory(portal, monkeypatch):
    job = job_at(portal, [TASK]); (job / 'output/total_RTSTRUCT.dcm').write_bytes(b'keep')
    monkeypatch.setattr(portal, 'prepare_worker_job', lambda *args: None)
    monkeypatch.setattr(portal, 'run_worker_task', lambda *args: None)
    def fail(*args): raise RuntimeError('worker disconnected')
    monkeypatch.setattr(portal, 'retrieve_worker_results', fail)
    deletions = []
    monkeypatch.setattr(portal, 'remove_worker_job', lambda *args: deletions.append(args))
    with pytest.raises(RuntimeError, match='disconnected'): portal.run_remote_gpu_task(job, TASK, 1, 1)
    assert not deletions
    assert portal.metadata(job)['remote_results_retained']
    assert portal.metadata(job)['remote_task_active'] == TASK
    assert (job / 'output/total_RTSTRUCT.dcm').read_bytes() == b'keep'


def test_ssh_command_quotes_paths_and_times_out_with_remote_stop(portal, monkeypatch):
    job = job_at(portal, [TASK])
    monkeypatch.setattr(portal, 'REMOTE_RUNNER', '/app with spaces/run_task.py')
    monkeypatch.setattr(portal, 'REMOTE_MODEL_ROOT', '/models with spaces')
    monkeypatch.setattr(portal, 'REMOTE_NV_REASON_CT_COVERAGE_MODE', 'full')
    monkeypatch.setattr(portal, 'TASK_TIME_LIMIT_SECONDS', -1)
    calls = []
    class Process:
        def __init__(self, command, **kwargs): self.args = command; calls.append(('popen',command))
        def poll(self): return None
    monkeypatch.setattr(portal.subprocess, 'Popen', Process)
    monkeypatch.setattr(portal, 'stop_remote_task', lambda *args: calls.append(('remote_stop',args)))
    monkeypatch.setattr(portal, 'stop_process_tree', lambda *args: calls.append(('local_stop',args)))
    with pytest.raises(subprocess.TimeoutExpired): portal.run_worker_task(job, '/jobs/example', TASK, 1, 1)
    assert [x[0] for x in calls] == ['popen','remote_stop','local_stop']
    command = calls[0][1]
    assert 'BatchMode=yes' in command
    remote = shlex.split(command[-1])
    assert 'RTSEG_MODEL_ROOT=/models with spaces' in remote
    assert 'RTSEG_NV_REASON_CT_COVERAGE_MODE=full' in remote
    assert remote[-3:] == ['/app with spaces/run_task.py','/jobs/example',TASK]
    assert calls[1][1] == ('/jobs/example', TASK)


def preflight_mock(portal, monkeypatch, remote_value=None, remote_error=None):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'ssh':
            if remote_error: raise remote_error
            return SimpleNamespace(stdout=json.dumps(remote_value), returncode=0)
        if any('model_preflight.py' in x for x in command):
            # Local acceptance must have no effect on the routed report's gate.
            value = {TASK: {'ready':True,'acceptance_ready':True, 'orientation_ready':True}}
        else: value = {'tasks':['total'], 'licensed':[]}
        return SimpleNamespace(stdout=json.dumps(value), returncode=0)
    monkeypatch.setattr(portal.subprocess, 'run', run)
    monkeypatch.setattr(portal, 'ENABLED_EXTERNAL_MODELS', {TASK})
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {TASK})
    return calls


def test_remote_preflight_cached_and_separate_acceptance(portal, monkeypatch):
    calls = preflight_mock(portal, monkeypatch, {TASK:{'ready':True,'acceptance_ready':False, 'orientation_ready':True}})
    for _ in range(3):
        items = {x['name']:x for x in portal.tasks()['tasks']}
        assert not items[TASK]['enabled']
        assert items[TASK]['status'] == 'report_acceptance_pending'
        assert items['total']['enabled']
    assert len([c for c in calls if c[0] == 'ssh']) == 1
    preflight_mock(portal, monkeypatch, {TASK:{'ready':True,'acceptance_ready':True, 'orientation_ready':False}})
    assert next(x for x in portal.refresh_catalog()['tasks'] if x['name'] == TASK)['enabled']


@pytest.mark.parametrize('failure', [OSError('unreachable'), subprocess.TimeoutExpired('ssh', 1), ValueError('bad JSON')])
def test_unreachable_remote_disables_only_routed_task(portal, monkeypatch, failure):
    preflight_mock(portal, monkeypatch, remote_error=failure)
    items = {x['name']:x for x in portal.tasks()['tasks']}
    assert not items[TASK]['enabled']
    assert items[TASK]['status'] == 'remote_worker_unreachable_or_preflight_failed'
    assert items['total']['enabled']


@pytest.mark.parametrize('reachable', [True, False])
def test_health_reports_remote_worker(portal, monkeypatch, reachable):
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {TASK})
    monkeypatch.setattr(portal, 'gpu_memory_mb', lambda: (100,200))
    def run(*args, **kwargs):
        if not reachable: raise subprocess.TimeoutExpired('ssh', 5)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(portal.subprocess, 'run', run)
    response = TestClient(portal.app).get('/api/v1/health')
    assert response.status_code == 200
    value = response.json()
    assert value['remote_worker_reachable'] is reachable
    assert value['remote_gpu_tasks'] == [TASK]
    assert value['status'] == ('ok' if reachable else 'degraded')
    assert value['worker_reachable']  # Local runtime status is independent.


def test_no_remote_configuration_makes_no_ssh_calls(portal, monkeypatch):
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', set())
    monkeypatch.setattr(portal, 'gpu_memory_mb', lambda: (None,None))
    def fail(*args, **kwargs): raise AssertionError('Unexpected SSH call')
    monkeypatch.setattr(portal.subprocess, 'run', fail)
    assert portal.health()['remote_worker_reachable'] is None
    job = job_at(portal, ['total'])
    monkeypatch.setattr(portal, 'run_local_task', lambda *args: (job / 'output/total_RTSTRUCT.dcm').write_bytes(b'result'))
    portal.process(job)
    assert portal.metadata(job)['status'] == 'complete'


@pytest.mark.parametrize('resuming', [False, True])
def test_auto_colab_fallback_finishes_before_remote_report(portal, monkeypatch, resuming):
    monkeypatch.setattr(portal, 'WORKER_MODE', 'auto')
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {TASK})
    monkeypatch.setattr(portal, 'GPU_RELEASE_COMMAND', '')
    job = job_at(portal, ['total',TASK])
    if resuming: portal.save_metadata(job, fallback_reason='gpu_oom')
    calls = []
    def oom(*args): calls.append('local'); raise RuntimeError('CUDA out of memory')
    def colab(job, tasks, finalize):
        calls.append('colab'); assert tasks == ['total'] and not finalize
        (job / 'output/total_RTSTRUCT.dcm').write_bytes(b'colab')
    def remote(job, *args): calls.append('report'); draft(job)
    monkeypatch.setattr(portal, 'run_local_task', oom)
    monkeypatch.setattr(portal, 'run_colab_job_on_nfs_worker', colab)
    monkeypatch.setattr(portal, 'run_remote_gpu_task', remote)
    portal.process(job)
    assert calls == (['colab','report'] if resuming else ['local','colab','report'])
    assert portal.metadata(job)['status'] == 'complete'


def test_colab_does_not_advertise_report_even_if_allowlisted(portal, monkeypatch):
    monkeypatch.setattr(portal, 'COLAB_EXTERNAL_TASKS', {TASK})
    assert not portal.colab_capable(TASK)


def test_restart_stops_remote_before_retrieval_and_preserves_contours(portal, monkeypatch):
    monkeypatch.setattr(portal, 'REMOTE_GPU_TASKS', {TASK})
    job = job_at(portal, ['total',TASK]); contour = job / 'output/total_RTSTRUCT.dcm'; contour.write_bytes(b'kept')
    portal.save_metadata(job, status='processing', remote_task_active=TASK, completed_models=['total'])
    calls = []
    monkeypatch.setattr(portal, 'stop_remote_task', lambda *args: calls.append('stop'))
    def recover(job, *args): calls.append('retrieve'); draft(job)
    monkeypatch.setattr(portal, 'recover_remote_results', recover)
    monkeypatch.setattr(portal.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: calls.append('requeue')))
    portal.recover_interrupted_jobs()
    assert calls == ['stop','retrieve','requeue']
    assert contour.read_bytes() == b'kept'
    assert portal.metadata(job)['report_status'] == 'pending'


@pytest.mark.parametrize('orchestrator,method', [('direct','run_colab_job'),('worker','run_colab_job_via_worker'),('nfs','run_colab_job_on_nfs_worker')])
def test_non_report_colab_routing_preserved(portal, monkeypatch, orchestrator, method):
    job=job_at(portal,['total'])
    monkeypatch.setattr(portal,'WORKER_MODE','colab')
    monkeypatch.setattr(portal,'COLAB_ORCHESTRATOR',orchestrator)
    monkeypatch.setattr(portal,'REMOTE_GPU_TASKS',{TASK})
    calls=[]
    monkeypatch.setattr(portal,method,lambda *args: calls.append(method))
    monkeypatch.setattr(portal,'run_remote_gpu_task',lambda *args: pytest.fail('Unexpected SSH route'))
    portal.process(job)
    assert calls == [method]


def test_non_report_local_empty_prediction_and_crop_note_preserved(portal, monkeypatch):
    job=job_at(portal,['total'])
    portal.save_metadata(job,postprocess=['prostate_sbrt_crop'])
    monkeypatch.setattr(portal,'REMOTE_GPU_TASKS',{TASK})
    def empty(*args): raise RuntimeError('empty prediction: no RTSTRUCT created')
    monkeypatch.setattr(portal,'run_local_task',empty)
    portal.process(job)
    data=portal.metadata(job)
    assert data['status']=='complete' and data['no_structures_found']==['total']
    assert 'no prostate ROI' in data['postprocess_notes'][0]


def test_remote_native_preflight_is_authoritative(portal,monkeypatch):
    calls=preflight_mock(portal,monkeypatch,{'total':{'ready':False,'missing':['TotalSegmentator runtime'],'native':True}})
    monkeypatch.setattr(portal,'REMOTE_GPU_TASKS',{'total'})
    item=next(x for x in portal.tasks()['tasks'] if x['name']=='total')
    assert not item['enabled'] and item['status']=='dependencies_missing'
    command=next(c for c in calls if c[0]=='ssh')
    assert shlex.split(command[-1])[-2:]==['--tasks','total']
