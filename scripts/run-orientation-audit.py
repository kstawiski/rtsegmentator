#!/usr/bin/env python3
"""Sequential, resumable orientation audit on a managed GPU worker."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nv_segment_logic import MODES, acceptance_from_metrics


def run_command(command, *, env, log, timeout):
    # The dispatcher handles TERM by stopping its inference process group.
    # Keep that handler alive through its grace period before escalating.
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='*')
    parser.add_argument('--force', action='store_true', help='Repeat existing orientation checks')
    parser.add_argument('--nv-segment-acceptance', action='store_true',
                        help='Run fresh NV-Segment checks and write per-task technical acceptance on success')
    args = parser.parse_args(argv)
    if args.nv_segment_acceptance and any(task not in MODES for task in args.tasks):
        parser.error('--nv-segment-acceptance accepts only automatic NV-Segment tasks')
    app = Path(os.environ.get('RTSEG_APP_ROOT', '/app'))
    root = Path(os.environ.get('RTSEG_MODEL_ROOT', '/models'))
    runtime = os.environ.get('RTSEG_RUNTIME_ROOT', '/runtimes')
    python = os.environ.get('RTSEG_WORKER_PYTHON', str(app / '.venv/bin/python'))
    fixtures = json.loads(Path(os.environ.get('RTSEG_ORIENTATION_FIXTURES',
                                              app / 'catalog/orientation-audit-fixtures.json')).read_text())
    prompt_file = Path(os.environ.get('RTSEG_ORIENTATION_PROMPTS', app / 'catalog/orientation-audit-prompts.json'))
    prompts = json.loads(prompt_file.read_text()) if prompt_file.exists() else {}
    tasks = args.tasks or (list(MODES) if args.nv_segment_acceptance else list(fixtures))
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', task) for task in tasks):
        parser.error('Invalid task name')
    if args.nv_segment_acceptance and any(task not in MODES for task in tasks):
        parser.error('--nv-segment-acceptance accepts only automatic NV-Segment tasks')
    validation = root / 'orientation-validation'
    runroot = Path(os.environ.get('RTSEG_ORIENTATION_RUN_ROOT', root / 'results/orientation-audit-20260812'))
    runroot.mkdir(parents=True, exist_ok=True)
    failed = False
    for task in tasks:
        marker = validation / f'{task}.json'
        acceptance = root / 'validation' / f'{task}.json'
        if marker.exists() and not (args.force or args.nv_segment_acceptance):
            print(f'SKIP {task}', flush=True)
            continue
        # A failed revalidation must not leave a previous success selectable.
        marker.unlink(missing_ok=True)
        if args.nv_segment_acceptance:
            acceptance.unlink(missing_ok=True)
        started = time.monotonic()
        status, error = 'failed', None
        try:
            if task not in fixtures:
                raise ValueError(f'NO_FIXTURE {task}')
            source = root / 'testdata' / fixtures[task]
            if not source.is_dir() or not any(source.iterdir()):
                raise ValueError(f'Public DICOM fixture is missing or empty: {fixtures[task]}')
            job = runroot / task
            if job.is_symlink():
                raise ValueError('Audit job directory must not be a symlink')
            shutil.rmtree(job, ignore_errors=True)
            (job / 'output').mkdir(parents=True)
            (job / 'input').symlink_to(source.resolve(), target_is_directory=True)
            if task in prompts:
                shutil.copy2(app / prompts[task], job / 'prompt.json')
            env = os.environ.copy()
            env.update(RTSEG_APP_ROOT=str(app), RTSEG_MODEL_ROOT=str(root), RTSEG_RUNTIME_ROOT=runtime)
            command = [python, str(app / 'run_task.py'), str(job), task]
            if args.nv_segment_acceptance:
                command.append('--nv-segment-acceptance-audit')
            with (runroot / f'{task}.log').open('w') as log:
                run_command(command, env=env, log=log,
                            timeout=int(os.environ.get('TASK_TIME_LIMIT_SECONDS', 12 * 3600)))
                result = job / 'output' / f'{task}_RTSTRUCT.dcm'
                # Keep a provisional marker outside the selectable marker directory.
                provisional = job / 'orientation.json'
                run_command([python, str(app / 'audit_rtstruct_orientation.py'), '--task', task,
                             '--dicom', str(job / 'input'), '--rtstruct', str(result),
                             '--output', str(provisional)], env=env, log=log, timeout=300)
            orientation = json.loads(provisional.read_text())
            if orientation.get('task') != task or orientation.get('result') != 'passed':
                raise ValueError('Orientation audit did not produce a matching success marker')
            if args.nv_segment_acceptance:
                metrics = json.loads((job / 'external' / task / 'prediction.nii.gz.metrics.json').read_text())
                accepted = acceptance_from_metrics(metrics, orientation, task, fixtures[task])
                accepted['accepted_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(acceptance, accepted)
            write_json(marker, orientation)
            status = 'passed'
        except Exception as exc:
            failed = True
            error = repr(exc)
            marker.unlink(missing_ok=True)
            if args.nv_segment_acceptance:
                acceptance.unlink(missing_ok=True)
        row = {'task': task, 'status': status, 'seconds': round(time.monotonic() - started, 1), 'error': error}
        with (runroot / 'summary.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    return int(failed)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
