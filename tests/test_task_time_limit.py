import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_stop_process_tree_kills_detached_child(portal, tmp_path, monkeypatch):
    """Child starts a new session and ignores TERM, matching a stubborn model."""
    marker = tmp_path / "child.pid"
    child = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(60)"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "],start_new_session=True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", parent])
    child_pid = None
    child_started = None
    monkeypatch.setattr(portal, "TASK_STOP_GRACE_SECONDS", .05)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        child_pid = int(marker.read_text())
        child_started = portal.proc_stat(child_pid)[2]
        portal.stop_process_tree(process)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            stat = portal.proc_stat(child_pid)
            if stat is None or stat[0] == "Z":
                break
            time.sleep(.01)
        assert process.poll() is not None
        assert stat is None or stat[0] == "Z"
    finally:
        if child_pid:
            stat = portal.proc_stat(child_pid)
            if stat and stat[0] != "Z" and stat[2] == child_started:
                try: os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError: pass
        if process.poll() is None:
            process.kill()
        process.wait()


def test_local_timeout_calls_tree_stop(portal, tmp_path, monkeypatch):
    runner = tmp_path / "slow_runner.py"
    runner.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(portal, "LOCAL_RUNNER", str(runner))
    monkeypatch.setattr(portal, "TASK_TIME_LIMIT_SECONDS", 1)
    monkeypatch.setattr(portal, "TASK_STOP_GRACE_SECONDS", .05)
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(portal, 'time', SimpleNamespace(time=time.time, monotonic=lambda: clock.now))
    launches, stops = [], []
    popen, tree_stop = portal.subprocess.Popen, portal.stop_process_tree
    def launch(*args, **kwargs):
        child = popen(*args, **kwargs)
        launches.append(child)
        clock.now = 2.
        return child
    def stop(child):
        stops.append(child)
        tree_stop(child)
    monkeypatch.setattr(portal.subprocess, 'Popen', launch)
    monkeypatch.setattr(portal, 'stop_process_tree', stop)
    # The monitored runner records progress in the job's metadata, so the job needs one.
    (tmp_path / "job.json").write_text('{"id": "synthetic-job", "status": "processing", "models": ["synthetic"]}')
    with pytest.raises(subprocess.TimeoutExpired):
        portal.run_local_task(tmp_path, "synthetic", 1, 1)
    assert len(launches) == 1 and stops == launches
    assert launches[0].poll() is not None
