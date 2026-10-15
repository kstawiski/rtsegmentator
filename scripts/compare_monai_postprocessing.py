#!/usr/bin/env python3
"""Manager-run GPU comparison of the production MONAI CPU-postprocessing flag.

Uses a fresh output directory and a fresh process per run. No catalog or bundle
files are edited. Run only on an authorized GPU host with a suitable fixture.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))
from monai_bundle import bundle_command, compatible_config


def compare_arrays(off, on, labels):
    import numpy as np
    if off.shape != on.shape or not off.size:
        raise ValueError('Predictions must have the same nonempty voxel grid')
    for data in (off, on):
        if not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all():
            raise ValueError('Predictions must contain finite integer labels')
    label_ids = sorted(set(map(int, labels)) | set(map(int, np.unique(off))) | set(map(int, np.unique(on))))
    results = []
    for label in label_ids:
        a, b = off == label, on == label
        count_a, count_b = int(a.sum()), int(b.sum())
        intersection = int(np.count_nonzero(a & b))
        differing = int(np.count_nonzero(a != b))
        results.append({'label': label, 'name': labels.get(str(label), 'Background' if label == 0 else 'Unmapped'),
                        'voxels_off': count_a, 'voxels_on': count_b, 'intersection_voxels': intersection,
                        'differing_voxels': differing, 'voxel_agreement': 1 - differing / off.size,
                        'dice': 2 * intersection / (count_a + count_b) if count_a + count_b else 1.0})
    return {'voxels': int(off.size), 'exact_voxel_agreement': float(np.mean(off == on)), 'labels': results}


def compare_files(off_path, on_path, labels):
    import nibabel as nib
    import numpy as np
    off, on = nib.load(off_path), nib.load(on_path)
    if off.shape != on.shape or not np.allclose(off.affine, on.affine, atol=1e-4, rtol=0):
        raise ValueError('Prediction shape/affine mismatch; no resampling is allowed for this comparison')
    return compare_arrays(np.asanyarray(off.dataobj), np.asanyarray(on.dataobj), labels)


def measured_worker(request_path):
    """Execute the same monai.bundle CLI in-process to read CUDA allocator peaks."""
    import torch
    from monai.utils import set_determinism
    request = json.loads(Path(request_path).read_text())
    metrics = {'status': 'failed', 'peak_allocated_mib': None, 'peak_reserved_mib': None}
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU comparison requires CUDA; refusing a CPU fallback')
        set_determinism(seed=request['seed'])
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        command = request['command']
        if command[1:3] != ['-m', 'monai.bundle']:
            raise ValueError('Expected the production monai.bundle command')
        sys.argv = ['monai.bundle', *command[3:]]
        runpy.run_module('monai.bundle', run_name='__main__')
        torch.cuda.synchronize(0)
        metrics['status'] = 'ok'
    except BaseException as exc:
        # Preserve failure/OOM evidence and let the parent continue to flag on.
        metrics['error_type'] = type(exc).__name__
        raise
    finally:
        metrics['bundle_wall_seconds'] = time.perf_counter() - started
        if torch.cuda.is_available():
            metrics['peak_allocated_mib'] = torch.cuda.max_memory_allocated(0) / 2**20
            metrics['peak_reserved_mib'] = torch.cuda.max_memory_reserved(0) / 2**20
        Path(request['metrics']).write_text(json.dumps(metrics, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dicom', required=True, type=Path, help='One DICOM series; choose the modality/domain for the selected tasks')
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--runtime-python', required=True, type=Path, help='Prepared MONAI runtime Python with CUDA')
    parser.add_argument('--conversion-python', type=Path, default=Path(sys.executable), help='Python with SimpleITK for DICOM conversion')
    parser.add_argument('--output', required=True, type=Path, help='New directory on authorized project storage')
    parser.add_argument('--task', action='append', help='Repeat for compatible tasks; default: all six flagged CT bundles')
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    catalog = json.loads((APP_ROOT / 'catalog/monai-bundles.json').read_text())
    tasks = args.task or [name for name, spec in catalog.items() if spec.get('postprocess_on_cpu')]
    if len(tasks) != len(set(tasks)):
        parser.error('Do not repeat a task')
    if any(name not in catalog for name in tasks):
        parser.error('Every task must be in catalog/monai-bundles.json')
    if not args.dicom.is_dir() or not args.runtime_python.is_file():
        parser.error('DICOM directory and runtime Python must exist')
    # Refuse reuse rather than mixing stale predictions with the new run.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    nifti = args.output / 'input.nii.gz'
    with (args.output / 'conversion.log').open('w') as log:
        subprocess.run([str(args.conversion_python.resolve()), str(APP_ROOT / 'dicom_series_to_nifti.py'),
                        str(args.dicom.resolve()), str(nifti)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = {'seed': args.seed,
               'memory_measurement': 'PyTorch peak allocated/reserved MiB on cuda:0; excludes non-PyTorch CUDA allocations',
               'dice_empty_both': 1.0, 'tasks': []}
    failed = False
    for task in tasks:
        original = catalog[task]
        bundle = args.model_root.resolve() / 'weights/monai-bundles' / original['bundle']
        entry = {'task': task, 'runs': {}}
        for enabled in (False, True):
            mode = 'on' if enabled else 'off'
            work = args.output / task / mode
            inference = work / 'inference'
            inference.mkdir(parents=True)
            spec = {**original, 'postprocess_on_cpu': enabled}
            started = time.perf_counter()
            try:
                config = compatible_config(bundle, spec, work)
                command = bundle_command(args.runtime_python.resolve(), bundle, spec, config, nifti, inference)
                metrics_path = work / 'metrics.json'
                request_path = work / 'request.json'
                request_path.write_text(json.dumps({'command': command, 'metrics': str(metrics_path), 'seed': args.seed}, indent=2) + '\n')
                with (work / 'run.log').open('w') as log:
                    result = subprocess.run([str(args.runtime_python.resolve()), str(Path(__file__).resolve()), '--worker', str(request_path)],
                                            env=env, stdout=log, stderr=subprocess.STDOUT)
                metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {'status': 'failed', 'error_type': 'NoWorkerMetrics'}
                metrics['returncode'] = result.returncode
                prediction = inference / spec.get('output_relative', 'input/input_trans.nii.gz')
                if result.returncode or not prediction.is_file():
                    metrics['status'] = 'failed'
                    metrics['prediction_exists'] = prediction.is_file()
            except Exception as exc:
                metrics = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
            metrics['process_wall_seconds'] = time.perf_counter() - started
            entry['runs'][mode] = metrics
            print(json.dumps({'task': task, 'flag': mode, **metrics}), flush=True)
        if all(run['status'] == 'ok' for run in entry['runs'].values()):
            try:
                if original.get('labels_from_metadata'):
                    metadata = json.loads((bundle / 'configs/metadata.json').read_text())
                    labels = metadata['network_data_format']['outputs']['pred']['channel_def']
                else:
                    labels = original['labels']
                relative = original.get('output_relative', 'input/input_trans.nii.gz')
                entry['comparison'] = compare_files(args.output / task / 'off/inference' / relative,
                                                    args.output / task / 'on/inference' / relative, labels)
                print(json.dumps({'task': task, 'comparison': entry['comparison']}), flush=True)
            except Exception as exc:
                entry['comparison'] = {'status': 'unavailable', 'error': str(exc)}
                failed = True
        else:
            entry['comparison'] = {'status': 'unavailable', 'reason': 'Both runs must succeed; inspect run.log (including possible OOM)'}
            failed = True
        summary['tasks'].append(entry)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        measured_worker(sys.argv[2])
    else:
        raise SystemExit(main())
