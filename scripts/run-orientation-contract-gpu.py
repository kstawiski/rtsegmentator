#!/usr/bin/env python3
"""Manager check: public CT storage variants, inference and physical comparisons.

No acceptance markers are readied or published. All bulk files stay in run-root.
Run with the worker Python and configured RTSEG_MODEL_ROOT/RTSEG_RUNTIME_ROOT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import cv2
import nibabel as nib
import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from audit_lnctvseg_dice import compare as author_dice
from audit_nodal_laterality import check_laterality
from orientation_contract import restore_image
from rtstruct_geometry import dicom_grid
import viewer

# Select tasks on fixtures that contain the labelled anatomy. All four storage
# variants plus the original DICOM are run for each assigned task.
TASKS = {
    'hnlnl_2d3d': ('lnctvseg-example-hn-ct', 'nnU-Net v1 2-D/3-D ensemble; known sided failure'),
    'lnctvseg_nnunet_v1': ('lnctvseg-example-hn-ct', 'nnU-Net v1; six sided groups and author reference'),
    'cads_open_558_headneck_oars': ('lnctvseg-example-hn-ct', 'CADS nnU-Net OAR pipeline; bilateral glands and nerves'),
    'lungmask_ltrc_lobes': ('msd-lung', '2-D lung-lobe model; five explicitly sided labels'),
}
VARIANTS = {'HFS': 'LPS', 'HFP': 'RAS', 'FFS': 'RPI', 'FFP': 'LAI'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, env, log, timeout=3600):
    started = time.monotonic()
    process = subprocess.Popen(list(map(str, command)), env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
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
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    return time.monotonic() - started


def prepare_variants(source, directory, python, env):
    """Re-export one physical volume and verify every CT voxel and affine."""
    directory.mkdir(parents=True)
    original = directory / 'original'
    original.symlink_to(source.resolve(), target_is_directory=True)
    reference_path = directory / 'reference.nii.gz'
    evidence = {}
    with (directory / 'prepare.log').open('w') as log:
        run([python, APP / 'dicom_series_to_nifti.py', original, reference_path], env, log)
        reference = nib.load(reference_path)
        if ''.join(nib.aff2axcodes(reference.affine)) != 'LPS':
            raise ValueError('Manager fixtures must be axial CT after the plane-preserving contract')
        values = np.asanyarray(reference.dataobj)
        # The reference writer preserves integer CT HU exactly. Refuse silent
        # clipping/rounding of a fixture with other intensity representation.
        if (not np.isfinite(values).all() or not np.array_equal(values, np.rint(values))
                or values.min() < -32768 or values.max() > 32767):
            raise ValueError('Fixture HU cannot be re-exported losslessly as signed 16-bit CT')
        for position, codes in VARIANTS.items():
            transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(reference.affine),
                                                       nib.orientations.axcodes2ornt(codes))
            temporary = directory / 'variant.nii.gz'
            nib.save(reference.as_reoriented(transform), temporary)
            target = directory / position
            try:
                run([python, APP / 'nifti_to_reference_dicom.py', temporary, target, '--modality', 'CT',
                     '--preserve-storage', '--patient-position', position], env, log)
            finally:
                temporary.unlink(missing_ok=True)
            check = directory / 'check.nii.gz'
            try:
                run([python, APP / 'dicom_series_to_nifti.py', target, check], env, log)
                actual = nib.load(check)
                if (actual.shape != reference.shape
                        or not np.allclose(actual.affine, reference.affine, atol=1e-4, rtol=0)
                        or not np.array_equal(np.asanyarray(actual.dataobj), values)):
                    raise ValueError(f'{position}: re-export changed physical grid or CT values')
                evidence[position] = {'storage_axcodes': codes, 'exact_ct_voxels': True,
                                      'max_affine_delta_mm': float(np.max(abs(actual.affine - reference.affine)))}
            finally:
                check.unlink(missing_ok=True)
                Path(str(check) + '.orientation.json').unlink(missing_ok=True)
    write_json(directory / 'variants.json', evidence)
    return reference_path


def task_labels(task):
    if task == 'cads_open_558_headneck_oars':
        return json.loads((APP / 'catalog/cads-open.json').read_text())[task]['labels']
    filename = {'hnlnl_2d3d': 'hnlnl-labels.json',
                'lnctvseg_nnunet_v1': 'labels/lnctvseg.json',
                'lungmask_ltrc_lobes': 'labels/lungmask-lobes.json'}[task]
    return json.loads((APP / filename).read_text())


def exported_prediction(source, rtstruct, reference, labels):
    """Independently rasterize exported contours; report Dice on one physical grid."""
    shape, affine, _ = dicom_grid(source, reference)
    volume = np.zeros(shape, dtype=np.uint16)
    series = viewer.index_series(sorted(source.iterdir()))
    structure, mapped = viewer.map_structure(viewer.read_structure(rtstruct), rtstruct,
                                            rtstruct.parent, series)
    numbers = {name: int(number) for number, name in labels.items()}
    by_key = {roi['key']: numbers[roi['name']] for roi in structure['rois']}
    for k, rois in mapped.items():
        for roi in rois:
            plane = np.zeros(shape[:2][::-1], dtype=np.uint8)
            cv2.fillPoly(plane, [np.rint(p).astype(np.int32) for p in roi['paths']], 1)
            mask = plane.T.astype(bool)
            if np.any(volume[:, :, k][mask]):
                raise ValueError('Selected multilabel task exported overlapping ROIs')
            volume[:, :, k][mask] = by_key[roi['key']]
    return restore_image(nib.Nifti1Image(volume, affine), reference.shape, reference.affine)[0]


def physical_dice(first, second, labels):
    second, _ = restore_image(second, first.shape, first.affine)
    aa, bb = np.asanyarray(first.dataobj), np.asanyarray(second.dataobj)
    rows = []
    for value, name in labels.items():
        a, b = aa == int(value), bb == int(value)
        na, nb = int(a.sum()), int(b.sum())
        intersection = int(np.count_nonzero(a & b))
        rows.append({'label': int(value), 'name': name, 'hfs_voxels': na, 'variant_voxels': nb,
                     'intersection_voxels': intersection,
                     'dice': 2 * intersection / (na + nb) if na + nb else None,
                     'status': 'both_empty' if not na + nb else 'measured'})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=12 * 3600)
    args = parser.parse_args(argv)
    model = Path(os.environ['RTSEG_MODEL_ROOT']).resolve()
    python = os.environ.get('RTSEG_WORKER_PYTHON', sys.executable)
    root = args.run_root.resolve()
    if any(root.is_relative_to(path) for path in (model, Path('/tmp'), Path('/dev/shm'), Path.home())):
        raise ValueError('run-root must be project storage outside the immutable model bundle and local SSD')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'plan.json', {'tasks': TASKS, 'variants': ['HFS', 'HFP', 'FFS', 'FFP', 'original'],
                                   'dice_scope': 'Exported RTSTRUCT labels, independently rasterized by viewer',
                                   'author_dice_scope': 'Raw LNCTVSeg IDs 1–6; development fixture, not independent accuracy',
                                   'acceptance_markers_written': False})
    failed = False
    with tempfile.TemporaryDirectory(prefix='oc-', dir='/tmp') as short_tmp:
        env = os.environ.copy()
        env.update(RTSEG_APP_ROOT=str(APP), RTSEG_SCRATCH_ROOT=str(root / 'scratch'),
                   TMPDIR=short_tmp, TEMP=short_tmp, TMP=short_tmp,
                   PYTHONDONTWRITEBYTECODE='1', WORKER_MODE='local', REMOTE_GPU_TASKS='')
        references = {}
        for fixture in sorted({value[0] for value in TASKS.values()}):
            references[fixture] = prepare_variants(model / 'testdata' / fixture / 'dicom',
                                                   root / 'fixtures' / fixture, python, env)
        if args.prepare_only:
            return 0
        for task, (fixture, reason) in TASKS.items():
            labels = task_labels(task)
            reference = nib.load(references[fixture])
            baseline = None
            for variant in [*VARIANTS, 'original']:
                job = root / 'jobs' / task / variant
                (job / 'output').mkdir(parents=True)
                source = root / 'fixtures' / fixture / variant
                (job / 'input').symlink_to(source, target_is_directory=True)
                write_json(job / 'job.json', {'id': f'{task}-{variant}', 'models': [task],
                                              'status': 'orientation_audit'})
                started = time.monotonic()
                row = {'task': task, 'variant': variant, 'fixture': fixture, 'reason': reason}
                print(f'START {task} {variant}', flush=True)
                try:
                    command = [python, APP / 'run_task.py', job, task]
                    if task in {'hnlnl_2d3d', 'lnctvseg_nnunet_v1'}:
                        command.append('--nodal-acceptance-audit')
                    with (job / 'run.log').open('w') as log:
                        run(command, env, log, args.timeout)
                    rtstruct = job / 'output' / f'{task}_RTSTRUCT.dcm'
                    prediction = exported_prediction(source, rtstruct, reference, labels)
                    nib.save(prediction, job / 'exported-labels.nii.gz')
                    # HaN-Seg splits eyes into anterior/posterior left/right
                    # labels. Supply explicit side tokens to the shared audit.
                    audit_labels = {key: (name + '_L' if name.endswith(('_AL', '_PL')) else
                                          name + '_R' if name.endswith(('_AR', '_PR')) else name)
                                    for key, name in labels.items()}
                    laterality = check_laterality(reference, prediction, audit_labels)
                    write_json(job / 'laterality.json', laterality)
                    row['laterality'] = laterality['result']
                    if variant == 'HFS':
                        baseline = prediction
                    if baseline is None:
                        raise ValueError('HFS run failed: physical variant Dice has no baseline')
                    row['physical_dice'] = physical_dice(baseline, prediction, labels)
                    write_json(job / 'physical-dice.json', row['physical_dice'])
                    # Storage should not change exported predictions. Record any
                    # difference, including GPU nondeterminism, for investigation.
                    row['result'] = 'passed' if (laterality['result'] == 'passed' and
                        all(item['dice'] in (None, 1.0) for item in row['physical_dice'])) else 'failed'
                except Exception as exc:
                    row.update(result='failed', error=repr(exc))
                finally:
                    # Preserve raw author-reference evidence even if contour
                    # conversion or the laterality audit failed.
                    if task == 'lnctvseg_nnunet_v1':
                        raw = job / 'external' / task / 'prediction/case.nii.gz'
                        if raw.is_file():
                            try:
                                author = model / 'testdata' / fixture / 'reference-labels.nii.gz'
                                expected = json.loads((APP / 'models/manifests/lnctvseg-fixture.json').read_text())
                                with author.open('rb') as stream:
                                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                                if digest != expected['reference_labels_sha256']:
                                    raise ValueError('Author reference checksum differs from the pinned fixture')
                                row['author_dice'] = author_dice(raw, author)
                                write_json(job / 'author-dice.json', row['author_dice'])
                            except Exception as exc:
                                row.update(result='failed', author_dice_error=repr(exc))
                        else:
                            row.update(result='failed', author_dice_error='Raw prediction unavailable after inference')
                    row['wall_seconds'] = round(time.monotonic() - started, 3)
                failed |= row['result'] != 'passed'
                write_json(job / 'result.json', row)
                with (root / 'summary.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(f"END {task} {variant}: {row['result']}; wall_seconds={row['wall_seconds']}", flush=True)
    return int(failed)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128 + sig))
    raise SystemExit(main())
