"""CPU-only integration checks for FIFO dispatch, readiness and stall recovery."""
import json
import threading
from types import SimpleNamespace

import pytest


def make_job(portal, name, created, status='queued', **extra):
    job = portal.BASE / name
    (job / 'input').mkdir(parents=True)
    (job / 'input/synthetic.txt').write_text('synthetic input')
    (job / 'output').mkdir()
    (job / 'job.json').write_text(json.dumps(dict(
        id=name, models=['total'], status=status, created=created, updated=created,
        input_manifest=portal.job_input_manifest(job), **extra)))
    return job


def pause_dispatcher(portal, monkeypatch):
    starts = []
    class DeferredThread:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args
        def start(self):
            starts.append(self)
        def is_alive(self):
            return True
    monkeypatch.setattr(portal, 'threading', SimpleNamespace(Thread=DeferredThread))
    return starts


def stale_time(portal, jobs):
    return max(portal.metadata(job).get('updated', 0) for job in jobs) + portal.RTSEG_STALL_SECONDS + 1


def test_watchdog_leaves_fifo_reservations_unchanged(portal, monkeypatch):
    jobs = [make_job(portal, f'queued-{i}', i + 1) for i in range(3)]
    pause_dispatcher(portal, monkeypatch)
    portal.enqueue_jobs(jobs)
    before = [portal.metadata(job) for job in jobs]
    monkeypatch.setattr(portal, 'recover_interrupted_jobs', lambda *a, **k: pytest.fail('FIFO job recovered'))
    portal.watchdog_once(stale_time(portal, jobs))
    assert [portal.metadata(job) for job in jobs] == before
    assert [entry[0] for entry in portal.pending_jobs] == jobs
    assert [data['detail'] for data in before] == [
        'Next in queue', 'Queued behind 1 earlier job', 'Queued behind 2 earlier jobs']


def test_watchdog_leaves_active_readiness_wait_live(portal, monkeypatch):
    job = make_job(portal, 'awaiting-staging', 1)
    ready, entered, drained = (threading.Event() for _ in range(3))
    order = []
    original_drain = portal.drain_jobs
    def drain():
        try:
            original_drain()
        finally:
            drained.set()
    def await_input(job, ready):
        entered.set()
        portal.process_prepared_session(job, ready)
    def run(job):
        order.append(job)
        portal.save_metadata(job, status='complete')
    monkeypatch.setattr(portal, 'drain_jobs', drain)
    monkeypatch.setattr(portal, 'process', run)
    monkeypatch.setattr(portal, 'recover_interrupted_jobs', lambda *a, **k: pytest.fail('Readiness wait recovered'))
    try:
        portal.enqueue_jobs([job], target=await_input, args=(job, ready))
        assert entered.wait(30)
        assert not portal.pending_jobs and not portal.job_workers
        portal.watchdog_once(stale_time(portal, [job]))
        assert not portal.metadata(job).get('stall_events') and not order
    finally:
        ready.set()
        assert drained.wait(30)
    assert order == [job]


def test_watchdog_recovery_reenters_fifo_in_creation_order(portal, monkeypatch):
    # Directory discovery order differs from creation order. A newer job is
    # already waiting when both older workers are found dead.
    late_orphan = make_job(portal, 'orphan-z', 2, 'processing')
    early_orphan = make_job(portal, 'orphan-a', 1, 'processing')
    waiting = make_job(portal, 'waiting', 3)
    starts = pause_dispatcher(portal, monkeypatch)
    order = []
    def run(job):
        order.append(job)
        portal.save_metadata(job, status='complete')
    monkeypatch.setattr(portal, 'process', run)
    portal.enqueue_jobs([waiting])
    portal.watchdog_once(stale_time(portal, [early_orphan, late_orphan, waiting]))
    assert [entry[0] for entry in portal.pending_jobs] == [early_orphan, late_orphan, waiting]
    assert len(starts) == 1 and not order, 'Recovery must use the existing dispatcher'
    assert portal.metadata(early_orphan)['recovery_count'] == portal.metadata(late_orphan)['recovery_count'] == 1
    assert 'recovery_count' not in portal.metadata(waiting)
    starts[0].target(*starts[0].args)
    assert order == [early_orphan, late_orphan, waiting]


def test_watchdog_does_not_preempt_running_job_for_recovered_orphan(portal, monkeypatch):
    running = make_job(portal, 'running', 2, 'processing')
    orphan = make_job(portal, 'older-orphan', 1, 'retrieving')
    waiting = make_job(portal, 'waiting', 3)
    pause_dispatcher(portal, monkeypatch)
    portal.queue_active = True
    portal.queue_current = (running, SimpleNamespace(is_alive=lambda: True))
    portal.queue_running = True
    portal.enqueue_jobs([waiting])
    portal.watchdog_once(stale_time(portal, [running, orphan, waiting]))
    assert portal.queue_current[0] == running
    assert [entry[0] for entry in portal.pending_jobs] == [orphan, waiting]
    assert portal.metadata(orphan)['detail'] == 'Queued behind 1 earlier job'
    assert not portal.metadata(running).get('stall_events')


@pytest.mark.parametrize('ahead,expected', [
    (0, 'Next in queue'), (1, 'Queued behind 1 earlier job'), (4, 'Queued behind 4 earlier jobs')])
def test_queue_wording(portal, ahead, expected):
    assert portal.queue_detail(ahead) == expected


def test_stall_retry_holds_fifo_position_until_second_attempt_finishes(portal, monkeypatch):
    first = make_job(portal, 'first', 1)
    later = make_job(portal, 'later', 2)
    retry_started, release_retry, drained = (threading.Event() for _ in range(3))
    clock = SimpleNamespace(now=100.)
    def advance(seconds):
        clock.now += seconds
    monkeypatch.setattr(portal, 'time', SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=advance))
    monkeypatch.setattr(portal, 'RTSEG_STALL_SECONDS', 20)
    monkeypatch.setattr(portal, 'RTSEG_STALL_RETRIES', 1)
    monkeypatch.setattr(portal, 'proc_stat', lambda pid: ('R', 0, 'synthetic-start'))
    monkeypatch.setattr(portal, 'process_cpu_times', lambda pid: {(pid, 'synthetic-start'): 0})
    launches, kills, order = [], [], []
    class IdleProcess:
        pid = 12345
        returncode = None
        def __init__(self, args, **kwargs):
            launches.append(self)
            kwargs['stdout'].write('RTSEG_PROGRESS {"stage":"rtstruct_validation","done":1,"total":5}\n')
            kwargs['stdout'].flush()
            if len(launches) == 2:
                retry_started.set()
                assert release_retry.wait(30)
        def poll(self):
            return self.returncode
    def stop(process):
        kills.append(process)
        process.returncode = -15
    monkeypatch.setattr(portal.subprocess, 'Popen', IdleProcess)
    monkeypatch.setattr(portal, 'stop_process_tree', stop)
    original_process, original_drain = portal.process, portal.drain_jobs
    def run(job):
        order.append(job)
        if job == first:
            original_process(job)
        else:
            portal.save_metadata(job, status='complete')
    def drain():
        try:
            original_drain()
        finally:
            drained.set()
    monkeypatch.setattr(portal, 'process', run)
    monkeypatch.setattr(portal, 'drain_jobs', drain)
    try:
        portal.enqueue_jobs([first, later])
        assert retry_started.wait(30)
        assert order == [first] and len(launches) == 2 and len(kills) == 1
        assert portal.metadata(later)['detail'] == 'Queued behind 1 earlier job'
        assert portal.queue_current[0] == first
        assert portal.job_workers[first.name].is_alive()
        portal.watchdog_once(stale_time(portal, [first, later]))
        assert not portal.metadata(first).get('recovery_count')
        assert not portal.metadata(later).get('recovery_count')
    finally:
        release_retry.set()
        assert drained.wait(30)
    assert order == [first, later] and len(kills) == 2
    assert portal.metadata(first)['status'] == 'failed'
    assert [event['action'] for event in portal.metadata(first)['stall_events']] == ['retry', 'failed']
    assert portal.metadata(later)['status'] == 'complete'
    assert not portal.job_workers and not portal.queue_running


def test_dead_dispatcher_restarts_with_recovered_job_ahead_of_waiting_job(portal, monkeypatch):
    orphan = make_job(portal, 'orphan', 1, 'processing')
    waiting = make_job(portal, 'waiting', 2)
    starts = pause_dispatcher(portal, monkeypatch)
    portal.enqueue_jobs([waiting])
    dead = SimpleNamespace(is_alive=lambda: False)
    portal.queue_thread = dead
    portal.queue_current = (orphan, dead)
    portal.queue_active = True
    portal.watchdog_once(stale_time(portal, [orphan, waiting]))
    assert len(starts) == 2
    assert not portal.queue_active and portal.queue_current is None
    assert [entry[0] for entry in portal.pending_jobs] == [orphan, waiting]
    assert portal.metadata(orphan)['detail'] == 'Next in queue'
    assert portal.metadata(waiting)['detail'] == 'Queued behind 1 earlier job'


def test_dead_dispatcher_restarts_reservations_without_orphan_recovery(portal, monkeypatch):
    waiting = make_job(portal, 'waiting', 1)
    starts = pause_dispatcher(portal, monkeypatch)
    portal.enqueue_jobs([waiting])
    portal.queue_thread = SimpleNamespace(is_alive=lambda: False)
    portal.watchdog_once(stale_time(portal, [waiting]))
    assert len(starts) == 2 and portal.pending_jobs[0][0] == waiting
    assert not portal.metadata(waiting).get('stall_events')
    assert not portal.metadata(waiting).get('recovery_count')


def test_dead_dispatcher_does_not_strand_later_job_after_recovery_limit(portal, monkeypatch):
    orphan = make_job(portal, 'exhausted-orphan', 1, 'processing', recovery_count=3)
    waiting = make_job(portal, 'waiting', 2)
    starts = pause_dispatcher(portal, monkeypatch)
    portal.enqueue_jobs([waiting])
    dead = SimpleNamespace(is_alive=lambda: False)
    portal.queue_thread = dead
    portal.queue_current = (orphan, dead)
    portal.queue_active = True
    portal.watchdog_once(stale_time(portal, [orphan, waiting]))
    assert portal.metadata(orphan)['status'] == 'failed'
    assert len(starts) == 2 and [entry[0] for entry in portal.pending_jobs] == [waiting]
    assert portal.metadata(waiting)['detail'] == 'Next in queue'
    assert not portal.metadata(waiting).get('recovery_count')


def test_orphan_reset_cannot_let_newer_waiting_job_pass(portal, monkeypatch):
    first = make_job(portal, 'running-first', 1)
    orphan = make_job(portal, 'orphan-second', 2, 'processing')
    later = make_job(portal, 'waiting-third', 3)
    first_started, release_first, first_finished, reset_started, release_reset, drained, recovered, later_started = (
        threading.Event() for _ in range(8))
    original_reset, original_drain = portal.reset_job_output, portal.drain_jobs
    order, errors = [], []
    def run(job):
        order.append(job)
        if job == first:
            first_started.set()
            assert release_first.wait(30)
        if job == later:
            later_started.set()
        portal.save_metadata(job, status='complete')
        if job == first:
            first_finished.set()
    def reset(job, data):
        assert job == orphan
        reset_started.set()
        assert release_reset.wait(30)
        original_reset(job, data)
    def drain():
        try:
            original_drain()
        finally:
            drained.set()
    def recover():
        try:
            portal.watchdog_once(stale_time(portal, [first, orphan, later]))
        except Exception as exc:
            errors.append(exc)
        finally:
            recovered.set()
    monkeypatch.setattr(portal, 'process', run)
    monkeypatch.setattr(portal, 'reset_job_output', reset)
    monkeypatch.setattr(portal, 'drain_jobs', drain)
    worker = threading.Thread(target=recover, daemon=True)
    try:
        portal.enqueue_jobs([first, later])
        assert first_started.wait(30)
        worker.start()
        assert reset_started.wait(30)
        release_first.set()
        assert first_finished.wait(30)
        assert not later_started.wait(.1), 'Newer job passed an older orphan during recovery'
    finally:
        release_first.set()
        release_reset.set()
        if worker.ident is not None:
            assert recovered.wait(30)
            worker.join(30)
        assert drained.wait(30)
    assert not errors and order == [first, orphan, later]


def test_watchdog_survives_recovery_error_and_can_retry_later(portal, monkeypatch):
    job = make_job(portal, 'orphan', 1, 'processing')
    calls = []
    def recover(jobs, **kwargs):
        calls.append((jobs, kwargs))
        if len(calls) == 1:
            raise RuntimeError('Synthetic temporary recovery failure')
    monkeypatch.setattr(portal, 'recover_interrupted_jobs', recover)
    portal.watchdog_once(stale_time(portal, [job]))
    portal.watchdog_once(stale_time(portal, [job]))
    assert calls == [([job], {'orphaned': True})] * 2
    assert len(portal.metadata(job)['stall_events']) == 2


@pytest.mark.parametrize('live_process', [False, True])
def test_dead_dispatcher_waits_for_stale_metadata_then_stops_unmonitored_process(portal, monkeypatch, live_process):
    current = make_job(portal, 'current', 1, 'processing', task_pid=42, task_pid_started='start')
    waiting = make_job(portal, 'waiting', 2)
    starts = pause_dispatcher(portal, monkeypatch)
    portal.enqueue_jobs([waiting])
    dead = SimpleNamespace(is_alive=lambda: False)
    portal.queue_thread = dead
    portal.queue_current = (current, dead)
    portal.queue_active = True
    monkeypatch.setattr(portal, 'proc_stat', lambda pid: ('R', 0, 'start') if live_process else None)
    stopped = []
    monkeypatch.setattr(portal, 'stop_recorded_task_tree', lambda data: stopped.append(data.get('task_pid')))
    now = 1 + portal.RTSEG_STALL_SECONDS - 1
    portal.watchdog_once(now)
    new = make_job(portal, 'new', 3)
    portal.enqueue_jobs([new])
    assert len(starts) == 1 and portal.queue_current[0] == current
    assert [entry[0] for entry in portal.pending_jobs] == [waiting, new]
    assert not portal.metadata(current).get('recovery_count')
    # An unmonitored matching PID must be stopped, then recovery reserves its
    # position before a replacement dispatcher can start later jobs.
    portal.watchdog_once(stale_time(portal, [current, waiting, new]))
    assert len(starts) == 2 and portal.queue_current is None
    assert [entry[0] for entry in portal.pending_jobs] == [current, waiting, new]
    assert portal.metadata(current)['recovery_count'] == 1
    assert stopped == [42]


def test_actual_dispatcher_exit_retains_current_reservation_until_recovery(portal, monkeypatch):
    current = make_job(portal, 'current', 1)
    waiting = make_job(portal, 'waiting', 2)
    lost, drained = threading.Event(), threading.Event()
    original_drain = portal.drain_jobs
    order = []
    class WorkerExited(BaseException):
        pass
    @portal.tracked_job_worker
    def run(job):
        order.append(job)
        if len(order) == 1:
            portal.save_metadata(job, status='processing', task_pid=42, task_pid_started='start')
            raise WorkerExited()
        portal.save_metadata(job, status='complete')
    def drain():
        try:
            original_drain()
        except WorkerExited:
            lost.set()
        else:
            drained.set()
    monkeypatch.setattr(portal, 'process', run)
    monkeypatch.setattr(portal, 'drain_jobs', drain)
    monkeypatch.setattr(portal, 'proc_stat', lambda pid: ('R', 0, 'start'))
    portal.enqueue_jobs([current, waiting])
    assert lost.wait(30)
    portal.queue_thread.join(30)
    assert not portal.queue_thread.is_alive() and not portal.job_workers
    assert portal.queue_active and portal.queue_current[0] == current
    stopped = []
    monkeypatch.setattr(portal, 'stop_recorded_task_tree', lambda data: stopped.append(data.get('task_pid')))
    portal.watchdog_once(stale_time(portal, [current, waiting]))
    assert stopped == [42]
    assert drained.wait(30)
    portal.queue_thread.join(30)
    assert order == [current, current, waiting]
    assert portal.metadata(current)['recovery_count'] == 1
    assert not portal.queue_active and portal.queue_current is None
