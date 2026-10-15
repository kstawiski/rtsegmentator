#!/usr/bin/env python3
"""Check nodal-label laterality in patient space against the CT body midline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def label_side(name):
    words = name.replace('_', ' ').split()
    if 'L' in words or 'Left' in words:
        return 'L'
    if 'R' in words or 'Right' in words:
        return 'R'
    return None


def check_laterality(ct, prediction, labels, minimum_ml=1.0):
    """Require every component >=1 mL to have its centroid on the named side.

    The slice-wise body midline is the midpoint of the largest tissue component's
    patient-left extent. CT threshold is -300 HU; detached table/mask components
    are excluded. Distances use NIfTI RAS converted to DICOM LPS (positive X left).
    This checks gross laterality, not target delineation accuracy.
    """
    if ct.shape != prediction.shape or not np.allclose(ct.affine, prediction.affine, atol=1e-3):
        raise ValueError('Prediction and CT grids differ')
    if len(ct.shape) != 3 or minimum_ml <= 0:
        raise ValueError('Expected 3-D images and a positive component threshold')
    image = np.asanyarray(ct.dataobj)
    data = np.asanyarray(prediction.dataobj)
    if not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all():
        raise ValueError('Prediction contains noninteger or nonfinite values')
    if set(np.unique(data)) - {0} - {int(k) for k in labels}:
        raise ValueError('Prediction contains unmapped labels')
    # Canonical storage makes slice-wise body measurements independent of source
    # axis order or sign. Oblique affine terms remain in the physical coordinates.
    ct = nib.as_closest_canonical(ct)
    prediction = nib.as_closest_canonical(prediction)
    image, data = np.asanyarray(ct.dataobj), np.asanyarray(prediction.dataobj)
    affine = ct.affine
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    midlines = {}

    def midline(k):
        if k not in midlines:
            components, count = ndimage.label(image[:, :, k] > -300, structure=np.ones((3, 3)))
            if not count:
                raise ValueError(f'No body tissue on slice {k}')
            body = components == (1 + np.argmax(np.bincount(components.ravel())[1:]))
            ij = np.argwhere(body)
            if len(ij) < 100:
                raise ValueError(f'Insufficient body tissue on slice {k}')
            left = -(ij[:, 0] * affine[0, 0] + ij[:, 1] * affine[0, 1]
                     + k * affine[0, 2] + affine[0, 3])
            midlines[k] = float((left.min() + left.max()) / 2)
        return midlines[k]

    rows, failures, checked_sides, ignored = [], [], set(), 0
    for label, name in labels.items():
        side = label_side(name)
        if side is None:
            continue
        components, count = ndimage.label(data == int(label), structure=np.ones((3, 3, 3)))
        sizes = np.bincount(components.ravel())
        for index, box in enumerate(ndimage.find_objects(components), 1):
            volume_ml = float(sizes[index] * voxel_ml)
            if volume_ml < minimum_ml:
                ignored += 1
                continue
            ijk = np.argwhere(components[box] == index) + np.array([s.start for s in box])
            left = -nib.affines.apply_affine(affine, ijk)[:, 0]
            distance = left - np.array([midline(int(k)) for k in ijk[:, 2]])
            signed = distance if side == 'L' else -distance
            passed = bool(float(signed.mean()) > 0)
            row = dict(label=int(label), name=name, component=index, side=side,
                       volume_ml=round(volume_ml, 4),
                       centroid_distance_to_named_side_mm=round(float(signed.mean()), 4),
                       fraction_on_named_side=round(float(np.mean(signed > 0)), 6), passed=passed)
            rows.append(row)
            checked_sides.add(side)
            if not passed:
                failures.append(f'{name} component {index} is on the opposite side of the body midline')
    if checked_sides != {'L', 'R'}:
        failures.append('No components >=1 mL on both named sides; laterality evidence is insufficient')
    return dict(result='failed' if failures else 'passed', method='slice_body_midline_lps_x',
                body_threshold_hu=-300, minimum_component_ml=minimum_ml,
                ignored_components=ignored, components=rows, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct', required=True)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = check_laterality(nib.load(args.ct), nib.load(args.prediction),
                             json.loads(Path(args.labels).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return int(result['result'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
