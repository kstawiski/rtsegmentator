#!/usr/bin/env python3
"""Operator-only LNCTVSeg/HNLNL inference, RTSTRUCT, laterality and VRAM checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import lnctvseg_bundle
from audit_lnctvseg_dice import compare

spec = importlib.util.spec_from_file_location('orientation_audit', APP / 'scripts/run-orientation-audit.py')
orientation_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orientation_audit)
TASKS = (*lnctvseg_bundle.INTEGRATED_TASKS, 'hnlnl_2d3d')
FIXTURE = 'lnctvseg-example-hn-ct/dicom'


def fixture_dice(prediction, model, output):
    reference = model / 'testdata/lnctvseg-example-hn-ct/reference-labels.nii.gz'
    spec = json.loads((APP / 'models/manifests/lnctvseg-fixture.json').read_text())
    with reference.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != spec['reference_labels_sha256']:
        raise ValueError('LNCTVSeg reference-label checksum mismatch')
    result = compare(prediction, reference)
    result['reference_sha256'] = digest
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def descendants(root):
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            tail = path.read_text().rsplit(')', 1)[1].split()
            parents[int(path.parent.name)] = int(tail[1])
        except (OSError, ValueError, IndexError):
            continue
    found = {root}
    while True:
        expanded = found | {pid for pid, parent in parents.items() if parent in found}
        if expanded == found:
            return found
        found = expanded


def measured_run(command, env, log, metrics_path, timeout):
    """Sample task-process VRAM every 0.5 seconds; exclude unrelated GPU jobs."""
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    started, peak, samples = time.monotonic(), 0, []
    try:
        while process.poll() is None:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f'Inference exceeded {timeout} seconds')
            pids = descendants(process.pid)
            try:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                                         '--format=csv,noheader,nounits'], text=True, capture_output=True,
                                        check=True, timeout=30)
            except (subprocess.SubprocessError, OSError):
                # A slow or failed sample on a busy shared host must not stop the task it measures.
                samples.append({'elapsed_seconds': round(time.monotonic() - started, 3), 'task_vram_mib': None})
                time.sleep(0.5)
                continue
            memory = 0
            for line in result.stdout.splitlines():
                try:
                    pid, used, _ = [part.strip() for part in line.split(',')]
                    if int(pid) in pids:
                        memory += int(used)
                except ValueError:
                    continue
            peak = max(peak, memory)
            samples.append({'elapsed_seconds': round(time.monotonic() - started, 3),
                            'task_vram_mib': memory})
            time.sleep(0.5)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        if peak <= 0:
            raise ValueError('No task GPU memory was observed; cannot accept GPU execution')
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    finally:
        metrics = {'seconds': round(time.monotonic() - started, 3), 'peak_vram_mib': peak,
                   'timing_scope': 'complete dispatcher task, including conversion and validation',
                   'vram_measurement': 'sum of active task-process allocations, sampled every 0.5 seconds',
                   'samples': samples}
        metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
    return {key: value for key, value in metrics.items() if key != 'samples'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    args = parser.parse_args(argv)
    model = Path(os.environ['RTSEG_MODEL_ROOT'])
    python = os.environ['RTSEG_WORKER_PYTHON']
    fixture = model / 'testdata' / FIXTURE
    if not fixture.is_dir() or not any(fixture.glob('*.dcm')):
        raise ValueError('Public LNCTVSeg DICOM fixture is missing')
    args.run_root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(RTSEG_APP_ROOT=str(APP), WORKER_MODE='local', REMOTE_GPU_TASKS='')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
                          '--format=csv'], capture_output=True, text=True, check=True).stdout
    (args.run_root / 'gpu.csv').write_text(gpu)
    failed = False
    for task in TASKS:
        acceptance = model / 'validation' / f'{task}.json'
        marker = model / 'orientation-validation' / f'{task}.json'
        acceptance.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        job = args.run_root / task
        (job / 'output').mkdir(parents=True)
        (job / 'input').symlink_to(fixture.resolve(), target_is_directory=True)
        is_lnctvseg = task in lnctvseg_bundle.INTEGRATED_TASKS
        work = job / 'external' / task
        prediction = work / ('prediction' if is_lnctvseg else 'ensemble') / 'case.nii.gz'
        exported = work / 'export/case.nii.gz' if is_lnctvseg else prediction
        dice = None
        try:
            if is_lnctvseg:
                lnctvseg_bundle.validate_assets(model)
            with (job / 'run.log').open('w') as log:
                try:
                    metrics = measured_run([python, str(APP / 'run_task.py'), str(job), task,
                                            '--nodal-acceptance-audit'], env, log, job / 'performance.json',
                                           int(os.environ.get('TASK_TIME_LIMIT_SECONDS', 12 * 3600)))
                finally:
                    # Audit raw IDs, including class 7, even if conversion fails.
                    if is_lnctvseg and prediction.is_file():
                        dice = fixture_dice(prediction, model, job / 'dice.json')
                if is_lnctvseg:
                    if dice is None:
                        raise ValueError('LNCTVSeg prediction/Dice evidence is missing')
                    lnctvseg_bundle.validate_label_contract()
                labels = APP / ('labels/lnctvseg.json' if is_lnctvseg else 'hnlnl-labels.json')
                # run_task already converts and independently checks the contour-to-mask round trip.
                # These further checks use retained predictions and the produced RTSTRUCT.
                orientation_audit.run_command([
                    python, str(APP / 'audit_nodal_laterality.py'), '--ct', str(work / 'input/case_0000.nii.gz'),
                    '--prediction', str(exported), '--labels', str(labels), '--output', str(job / 'laterality.json'),
                ], env=env, log=log, timeout=300)
                orientation_audit.run_command([
                    python, str(APP / 'audit_rtstruct_orientation.py'), '--task', task,
                    '--dicom', str(job / 'input'), '--rtstruct', str(job / 'output' / f'{task}_RTSTRUCT.dcm'),
                    '--output', str(job / 'orientation.json'),
                ], env=env, log=log, timeout=300)
            laterality = json.loads((job / 'laterality.json').read_text())
            orientation = json.loads((job / 'orientation.json').read_text())
            if (laterality.get('result') != 'passed' or orientation.get('result') != 'passed'
                    or orientation.get('task') != task):
                raise ValueError('Missing or inconsistent success evidence')
            if is_lnctvseg and {row['label'] for row in laterality.get('components', [])} != set(range(1, 7)):
                raise ValueError('The LNCTVSeg fixture must produce all six grouped labels at >=1 mL')
            accepted = {'task': task, 'result': 'passed', 'fixture': FIXTURE, 'laterality': 'passed',
                        'accepted_utc': datetime.now(timezone.utc).isoformat(), **metrics}
            if is_lnctvseg:
                accepted.update(source_commit=lnctvseg_bundle.manifest()['source_commit'],
                                manifest_sha256=lnctvseg_bundle.manifest_sha256())
                lnctvseg_bundle.validate_acceptance(accepted)
            orientation.update(laterality='passed', fixture=FIXTURE)
            if is_lnctvseg:
                orientation['manifest_sha256'] = lnctvseg_bundle.manifest_sha256()
            orientation_audit.write_json(acceptance, accepted)
            orientation_audit.write_json(marker, orientation)
            row = accepted
        except BaseException as exc:
            acceptance.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            row = {'task': task, 'result': 'failed', 'error': repr(exc)}
            failed = True
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        if dice is not None:
            row['dice'] = dice
        with (args.run_root / 'summary.jsonl').open('a') as output:
            output.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    return int(failed)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128 + sig))
    raise SystemExit(main())
