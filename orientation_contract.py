"""Lossless, plane-preserving HFS storage and exact source-grid restoration.

NIfTI axcodes: axial LPS, coronal LIP, sagittal PIR. The acquisition plane
is the closest anatomical axis of the *source slice axis* (nibabel's polar
orientation assignment, including its deterministic axis-order tie break).
Oblique direction cosines are retained; this is never a resampling operation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np

VERSION = 1
TARGETS = {2: ('axial', 'LPS'), 1: ('coronal', 'LIP'), 0: ('sagittal', 'PIR')}
IDENTITY = np.array([[0, 1], [1, 1], [2, 1]])
AFFINE_ATOL = 1e-3  # Existing DICOM/GDCM grid tolerance, in millimetres.


def sidecar_path(path):
    return Path(str(path) + '.orientation.json')


def grid_record(shape, affine):
    return {'shape': list(map(int, shape)), 'affine': np.asarray(affine).tolist()}


def model_orientation(image):
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise ValueError('Orientation contract requires a finite 3-D image grid')
    source = nib.orientations.io_orientation(image.affine)
    if not np.isfinite(source).all():
        raise ValueError('Orientation contract requires a nonsingular affine')
    plane, codes = TARGETS[int(source[2, 0])]
    transform = nib.orientations.ornt_transform(source, nib.orientations.axcodes2ornt(codes))
    return plane, codes, transform


def normalize_image(image):
    plane, codes, transform = model_orientation(image)
    result = image if np.array_equal(transform, IDENTITY) else image.as_reoriented(transform)
    model_to_source = nib.orientations.inv_ornt_aff(transform, image.shape)
    record = {'version': VERSION, 'plane': plane, 'target_axcodes': codes,
              'source_axcodes': ''.join(nib.aff2axcodes(image.affine)),
              'source': grid_record(image.shape, image.affine),
              'model': grid_record(result.shape, result.affine),
              'source_to_model_orientation': transform.astype(int).tolist(),
              'model_to_source_indices': model_to_source.astype(int).tolist(),
              'source_to_model_indices': np.linalg.inv(model_to_source).astype(int).tolist(),
              'changed': not np.array_equal(transform, IDENTITY)}
    return result, record


def normalize_file(path):
    """Leave the original SITK file completely untouched for identity storage."""
    image = nib.load(path)
    result, record = normalize_image(image)
    if record['changed']:
        nib.save(result, path)
    # Record the serialized affine, including NIfTI float32 header precision.
    saved = nib.load(path)
    record['model'] = grid_record(saved.shape, saved.affine)
    sidecar_path(path).write_text(json.dumps(record, indent=2) + '\n')
    return record


def exact_orientation(shape, affine, target_shape, target_affine):
    """Return source->target orientation; reject shifts, crops, scale and shear.

    Only the 48 full-grid signed permutations are allowed. Translation must be
    exactly the endpoint offset implied by each flip, within the existing
    0.001 mm affine serialization tolerance (rtol=0).
    """
    if len(shape) != 3 or len(target_shape) != 3:
        raise ValueError('Exact orientation requires two 3-D grids')
    affine, target_affine = np.asarray(affine), np.asarray(target_affine)
    if not np.isfinite(affine).all() or not np.isfinite(target_affine).all():
        raise ValueError('Nonfinite prediction/source affine')
    try:
        relative = np.linalg.solve(target_affine, affine)
        orientation = nib.orientations.io_orientation(relative)
        if not np.isfinite(orientation).all():
            raise ValueError('Singular prediction grid')
        restored_shape = tuple(np.asarray(shape)[np.argsort(orientation[:, 0])])
        restored_affine = affine @ nib.orientations.inv_ornt_aff(orientation, shape)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Singular prediction/source affine') from exc
    if (restored_shape != tuple(target_shape)
            or not np.allclose(restored_affine, target_affine, atol=AFFINE_ATOL, rtol=0)):
        raise ValueError('prediction affine/grid is not an exact permutation/flip of the DICOM grid '
                         '(shift, crop, resampling or shear is not allowed)')
    return orientation


def restore_image(image, target_shape, target_affine):
    orientation = exact_orientation(image.shape, image.affine, target_shape, target_affine)
    data = nib.orientations.apply_orientation(np.asanyarray(image.dataobj), orientation)
    # Use the original target affine exactly, not a floating-point inverse.
    header = image.header.copy()
    result = nib.Nifti1Image(data, target_affine, header)
    return result, {'prediction': grid_record(image.shape, image.affine),
                    'dicom': grid_record(target_shape, target_affine),
                    'prediction_to_dicom_orientation': orientation.astype(int).tolist()}


def map_prompt(prompt, record):
    """Map viewer [column,row,acquired slice] points into the model array."""
    result = copy.deepcopy(prompt)
    matrix = np.asarray(record['source_to_model_indices'])

    def point(value):
        xyz = np.asarray(value, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Prompt coordinates must be finite [column,row,slice] triples')
        return nib.affines.apply_affine(matrix, xyz).tolist()

    for key in ('positive_points', 'negative_points'):
        if key in result:
            result[key] = [point(value) for value in result[key]]
    if result.get('box'):
        corners = np.asarray([point(value) for value in result['box']])
        result['box'] = [corners.min(0).tolist(), corners.max(0).tolist()]
    return result


def write_export_record(output, model_input, inverse):
    record = {'version': VERSION, 'inverse': inverse}
    sidecar = sidecar_path(model_input)
    if sidecar.is_file():
        record['input_contract'] = json.loads(sidecar.read_text())
    sidecar_path(output).write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True, type=Path)
    parser.add_argument('--model-input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(sidecar_path(args.model_input).read_text())
    args.output.write_text(json.dumps(map_prompt(json.loads(args.prompt.read_text()), record), indent=2) + '\n')


if __name__ == '__main__':
    main()
