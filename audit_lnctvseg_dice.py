#!/usr/bin/env python3
"""Compare raw LNCTVSeg IDs with the author's development-cohort reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def compare(prediction, reference):
    """Align storage axes only; refuse a different physical voxel grid."""
    pred_img = nib.as_closest_canonical(nib.load(prediction))
    ref_img = nib.as_closest_canonical(nib.load(reference))
    if (pred_img.shape != ref_img.shape or len(ref_img.shape) != 3
            or not np.allclose(pred_img.affine, ref_img.affine, rtol=0, atol=1e-4)):
        raise ValueError('Prediction and reference must have the same physical voxel grid')
    pred, ref = np.asarray(pred_img.dataobj), np.asarray(ref_img.dataobj)
    for name, data, allowed in [('Prediction', pred, set(range(8))),
                                ('Reference', ref, set(range(7)))]:
        if not set(np.unique(data)).issubset(allowed):
            raise ValueError(f'{name} contains noninteger or unexpected label IDs')
    if not set(range(1, 7)).issubset(set(np.unique(ref))):
        raise ValueError('The reference must contain all six published foreground IDs')
    voxel_ml = abs(float(np.linalg.det(ref_img.affine[:3, :3]))) / 1000
    rows = []
    for label in range(1, 7):
        p, r = pred == label, ref == label
        pc, rc = int(p.sum()), int(r.sum())
        intersection = int(np.count_nonzero(p & r))
        rows.append({'label': label, 'prediction_voxels': pc, 'reference_voxels': rc,
                     'intersection_voxels': intersection, 'dice': 2 * intersection / (pc + rc),
                     'prediction_ml': pc * voxel_ml, 'reference_ml': rc * voxel_ml})
    return {'comparison': 'Published numeric IDs 1-6 unchanged; undocumented class 7 omitted from export',
            'scope': 'Author development-cohort fixture; not independent accuracy or acceptance',
            'labels': rows, 'mean_dice': float(np.mean([r['dice'] for r in rows])),
            'prediction_label_ids': [int(x) for x in np.unique(pred)],
            'unmapped_label_7_voxels': int(np.count_nonzero(pred == 7)),
            'unmapped_label_7_ml': int(np.count_nonzero(pred == 7)) * voxel_ml}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.prediction, args.reference)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
