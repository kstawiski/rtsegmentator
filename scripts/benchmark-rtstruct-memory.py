#!/usr/bin/env python3
"""Synthetic CPU-only before/after RSS benchmark, each mode in a fresh process.

Run with --compare (default shape: 256 columns x 256 rows x 400 slices,
60 structures). The full-volume reference is frozen in tests/roundtrip_reference.py.
No images are written to disk.
"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


def benchmark(mode, shape, rois):
    import nibabel as nib
    import numpy as np
    from pydicom.dataset import Dataset
    from rtstruct_roundtrip import CroppedMask, data_box, validate_roi_roundtrip
    from roundtrip_reference import full_volume_reference
    started = time.perf_counter()
    data = np.zeros(shape, np.uint16)
    boxes = {}
    # Disjoint compact structures with a different slice offset in each cell.
    for i in range(rois):
        x, y, z = 2 + (i % 8) * (shape[0] // 8), 2 + ((i // 8) % 8) * (shape[1] // 8), 2 + (i % 5)
        box = (slice(x, min(x + max(2, shape[0] // 12), shape[0])),
               slice(y, min(y + max(2, shape[1] // 12), shape[1])),
               slice(z, min(z + 12, shape[2])))
        data[box] = i + 1
        boxes[i + 1] = box
    image = nib.Nifti1Image(data, np.diag([-1., -1., 1., 1.]))
    expected = {i: data == i for i in boxes} if mode == 'before' else None
    reports = []
    for label, box in boxes.items():
        # Build exact source-pixel rectangles, no contour-writer dependency.
        x, y, z = box
        roi = Dataset()
        roi.ContourSequence = []
        for k in range(z.start, z.stop):
            contour = Dataset()
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            contour.NumberOfContourPoints = 4
            contour.ContourData = [x.start, y.start, k, x.stop-1, y.start, k,
                                   x.stop-1, y.stop-1, k, x.start, y.stop-1, k]
            roi.ContourSequence.append(contour)
        if mode == 'before':
            report = full_volume_reference(roi, image, expected[label])
        else:
            # Match the converter: scan bounded slabs, then recreate just
            # this ROI box. No volume-sized ROI masks are allocated.
            actual_box = data_box(data, label=label)
            cropped = CroppedMask(data[actual_box] == label, tuple(s.start for s in actual_box), data.shape)
            report = validate_roi_roundtrip(roi, image, cropped)
            del cropped
        reports.append(report)
    return {'mode': mode, 'shape': list(shape), 'rois': rois,
            'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            'seconds': round(time.perf_counter() - started, 3), 'reports': reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', nargs=3, type=int, default=[256, 256, 400])
    p.add_argument('--rois', type=int, default=60)
    p.add_argument('--mode', choices=['before', 'after'])
    p.add_argument('--compare', action='store_true')
    a = p.parse_args()
    if a.mode:
        print(json.dumps(benchmark(a.mode, tuple(a.shape), a.rois)))
        return
    results = []
    for mode in ('before', 'after'):
        output = subprocess.check_output([sys.executable, __file__, '--mode', mode,
                   '--shape', *map(str, a.shape), '--rois', str(a.rois)], text=True)
        results.append(json.loads(output))
    same = results[0]['reports'] == results[1]['reports']
    for result in results: result.pop('reports')
    print(json.dumps({'results': results, 'reports_equal': same}, indent=2))
    if not same: raise SystemExit(1)


if __name__ == '__main__': main()
