#!/usr/bin/env python3
"""Pinned U-MambaMTL wrapper with native-T2 lesion RTSTRUCT artifacts."""
from __future__ import annotations
import os
import sys
# Reject unsafe direct invocation before importing any application/package code.
if __name__ == '__main__':
    if (not sys.flags.ignore_environment or not sys.flags.no_user_site
            or not sys.dont_write_bytecode or not sys.pycache_prefix
            or sys.flags.optimize or sys.flags.isolated or sys.flags.safe_path
            or sys._xoptions != {'pycache_prefix': sys.pycache_prefix}
            or not os.path.isdir(sys.pycache_prefix) or os.path.islink(sys.pycache_prefix)
            or os.stat(sys.pycache_prefix).st_uid != os.getuid()
            or os.stat(sys.pycache_prefix).st_mode & 0o077 or any(os.scandir(sys.pycache_prefix))):
        raise SystemExit('Held MRI requires -E -s -B -X pycache_prefix=<empty-private-directory>')
import argparse
import json
from pathlib import Path
import numpy as np
from held_mri import UMAMBA, export_masks, extract_umamba, runtime_gpu
from held_mri_upstream import umamba_pipeline
from prostate_models import runtime_provenance
from prostate_postprocess import assert_same_grid, write_array


def main():
    import nibabel as nib
    import SimpleITK as sitk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=[UMAMBA], required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--held-mri-acceptance-audit', action='store_true')
    args = parser.parse_args()
    from held_mri import verify_release
    verify_release(UMAMBA, args.model_root)
    from held_mri import runtime_start
    gpu = runtime_start(UMAMBA, args.model_root, worker_python=args.worker_python, acceptance_audit=args.held_mri_acceptance_audit)
    work, output = args.work.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = runtime_provenance(UMAMBA, args.model_root)
    provenance.update(json.loads((work / 'context.json').read_text()))
    provenance.update({'gpu': gpu, 'weight_licence': 'Not stated',
                       'owner_acceptance': 'Owner accepted these terms on 2026-10-22',
                       'upstream_role_key_mismatch_preserved': True})
    result = umamba_pipeline(args.model_root.resolve(), work / 'native', work / 'upstream')
    reference = nib.load(work / 'native/t2w.nii.gz')
    reference_sitk = sitk.ReadImage(str(work / 'native/t2w.nii.gz'))
    assert_same_grid(result['detection_image'], reference_sitk)
    detection, masks, report = extract_umamba(result['probability_xyz'], reference.header.get_zooms(), provenance)
    if not np.array_equal(detection, result['detection_xyz']):
        raise ValueError('Candidate adapter changed upstream detection map')
    if not np.array_equal(detection.T, sitk.GetArrayFromImage(result['detection_image'])):
        raise ValueError('U-Mamba writer changed detection axes or values')
    if float(detection.max()) != result['case_likelihood']:
        raise ValueError('Case likelihood differs from upstream')
    for kind, array in (('probability', result['probability_xyz']), ('detection', detection)):
        if array.shape != reference.shape:
            raise ValueError('U-Mamba inverse transforms did not restore native T2 shape')
        write_array(array.T.astype(np.float32), reference_sitk, str(output) + f'.{kind}.nii.gz')
    report['artifacts'] = {kind + '_map': output.name + f'.{kind}.nii.gz' for kind in ('probability', 'detection')}
    # Private work paths are not part of the exported provenance.
    report['provenance']['io_adapter'] = 'Exact upstream string constants relocated into the job; no numerical AST edits'
    export_masks(work, output, reference, masks, report)


if __name__ == '__main__':
    main()
