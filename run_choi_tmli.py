"""Choi voxel-space inference; numeric output, no confirmed axis default."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MODEL_RELATIVE = Path('nnUNet/3d_fullres/Task001_LN/nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1')
AXIS_VARIANTS = tuple(''.join(bits) for bits in itertools.product('01', repeat=3))
TARGET_SPACING = (1.113, 1.113, 5.0)


def axis_variant(value=None):
    value = os.environ.get('CHOI_AXIS_VARIANT', 'unconfirmed') if value is None else value
    if value not in AXIS_VARIANTS:
        raise ValueError('Choi axis variant is unconfirmed or invalid. Specify --axis-variant '
                         '000–111 (x/y/z flip bits) for manager evaluation; task stays disabled.')
    return value


def source_image(source):
    import nibabel as nib
    import numpy as np
    image = nib.load(source)
    if len(image.shape) != 3 or min(image.shape) < 1:
        raise ValueError('Choi requires one 3D CT in HU')
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Choi input must declare millimetre spatial units')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError('Invalid source affine')
    return image


def target_geometry(image, variant):
    """Axis-aligned LPS grid covering all eight source voxel-face corners.

    Variant bits are storage x/y/z reversals relative to LPS identity, not
    reversals relative to the incoming file (which may be prone or oblique).
    Centre the grid on the physical bounding box. Preserve spacing and enlarge
    each in-plane dimension beyond 512 when necessary; never crop a large FOV.
    """
    import nibabel as nib
    import numpy as np
    variant = axis_variant(variant)
    corners = np.array(list(itertools.product(*[(-.5, n-.5) for n in image.shape])))
    physical = nib.affines.apply_affine(image.affine, corners)
    low, high = physical.min(axis=0), physical.max(axis=0)
    spacing = np.array(TARGET_SPACING)
    # NIfTI sforms store float32 values; avoid an extra voxel from header rounding.
    size = np.maximum(1, np.ceil((high-low)/spacing - 1e-4).astype(int))
    size[:2] = np.maximum(512, size[:2])
    signs = np.array([1 if bit == '0' else -1 for bit in variant])
    affine = np.eye(4)
    affine[:3, :3] = np.diag(spacing * signs * [-1, -1, 1])
    affine[:3, 3] = (low+high)/2 - affine[:3, :3] @ ((size-1)/2)
    return {'schema_version': 1, 'axis_variant': variant, 'flip_axes_xyz': (signs < 0).tolist(),
            'target_shape_xyz': size.tolist(), 'target_spacing_mm': list(TARGET_SPACING),
            'target_affine_ras_mm': affine.tolist(),
            'source_shape_xyz': list(image.shape), 'source_affine_ras_mm': image.affine.tolist(),
            'source_bounds_voxel_faces_ras_mm': [low.tolist(), high.tolist()],
            'inplane_policy': 'Centre-pad to at least 512; enlarge each dimension if needed; no crop',
            'outside_hu': -1024, 'intensity_offset': 1024,
            'size_rounding_tolerance_voxels': 1e-4,
            'network_affine_ras_mm': network_affine().tolist(),
            'network_header': 'LPS identity direction, zero origin, 1 mm spacing; array order unchanged'}


def network_affine():
    # NIfTI stores RAS. SimpleITK reads this as identity LPS, zero origin.
    import numpy as np
    return np.diag([-1., -1., 1., 1.])


def resample_hu(values, transform, shape):
    """Linear interpolation with an air value outside source voxel support.

    Medical-image support extends half a voxel beyond outermost centres.
    Clamp interpolation at those centres; set points outside that support to
    air. Work one z plane at a time to bound coordinate-array memory.
    """
    import numpy as np
    from scipy.ndimage import affine_transform
    output = affine_transform(values, transform[:3, :3], transform[:3, 3],
                              output_shape=shape, order=1, mode='nearest', prefilter=False)
    x, y = np.arange(shape[0])[:, None], np.arange(shape[1])[None, :]
    bases = [transform[axis, 0]*x + transform[axis, 1]*y + transform[axis, 3] for axis in range(3)]
    for z in range(shape[2]):
        valid = np.ones(shape[:2], dtype=bool)
        for axis in range(3):
            coordinate = bases[axis] + transform[axis, 2]*z
            valid &= (coordinate >= -.5) & (coordinate < values.shape[axis]-.5)
        output[:, :, z][~valid] = -1024.
    return output


def prepare_input(source, destination, variant=None):
    import nibabel as nib
    import numpy as np
    variant = axis_variant(variant)
    image = source_image(source)
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('CT contains non-finite intensities')
    geometry = target_geometry(image, variant)
    transform = np.linalg.inv(image.affine) @ np.array(geometry['target_affine_ras_mm'])
    values = resample_hu(values, transform, geometry['target_shape_xyz'])
    values += 1024.0
    shifted = nib.Nifti1Image(values, network_affine())
    shifted.set_qform(network_affine(), code=1)
    shifted.set_sform(network_affine(), code=1)
    shifted.header.set_xyzt_units('mm')
    shifted.set_data_dtype(np.float32)
    nib.save(shifted, destination)
    return geometry


def restore_prediction(prediction, source, geometry, destination):
    """Restore the target's physical affine, then NN-sample on the source grid."""
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import affine_transform
    original = source_image(source)
    if (list(original.shape) != geometry['source_shape_xyz'] or
            not np.allclose(original.affine, geometry['source_affine_ras_mm'], atol=1e-5, rtol=0)):
        raise ValueError('Geometry manifest does not match source')
    pred = nib.load(prediction)
    if (list(pred.shape) != geometry['target_shape_xyz'] or
            not np.allclose(pred.affine, network_affine(), atol=1e-5, rtol=0)):
        raise ValueError('Prediction does not match the voxel-space network grid')
    labels = np.asanyarray(pred.dataobj)
    if (not np.isfinite(labels).all() or np.any(labels != np.rint(labels)) or
            np.any(labels < 0) or np.any(labels > 18)):
        raise ValueError('Prediction must contain integer labels 0–18')
    transform = np.linalg.inv(geometry['target_affine_ras_mm']) @ original.affine
    restored = affine_transform(labels.astype(np.uint8), transform[:3, :3], transform[:3, 3],
                                output_shape=original.shape, order=0, mode='grid-constant',
                                cval=0, prefilter=False)
    output = nib.Nifti1Image(restored, original.affine, original.header.copy())
    output.set_data_dtype(np.uint8)
    nib.save(output, destination)


def validate_plans(source):
    """Fail if a different release would resample the deliberately dummy header."""
    import pickle
    import numpy as np
    with (Path(source) / 'plans.pkl').open('rb') as stream:
        plans = pickle.load(stream)
    stage = plans['plans_per_stage'][max(plans['plans_per_stage'])]
    if (not np.allclose(stage['current_spacing'], [1, 1, 1]) or
            list(plans['transpose_forward']) != [0, 1, 2]):
        raise ValueError('Expected Choi full-resolution 1 mm plans with identity transpose')


def prepare_model(source, results, folds):
    target = Path(results) / MODEL_RELATIVE
    target.mkdir(parents=True, exist_ok=False)
    # Only checkpoints are linked. Metadata that inference might copy is private.
    shutil.copy2(Path(source) / 'plans.pkl', target / 'plans.pkl')
    for fold in folds:
        folder = target / f'fold_{fold}'
        folder.mkdir()
        for name in ('model_final_checkpoint.model', 'model_final_checkpoint.model.pkl'):
            original = (Path(source) / f'fold_{fold}' / name).resolve(strict=True)
            if name.endswith('.pkl'):
                shutil.copy2(original, folder / name)
            else:
                (folder / name).symlink_to(original)
    return target


def prediction_command(executable, inputs, outputs, folds):
    # nnU-Net v1's normal predictor averages fold softmax probabilities before
    # argmax/export. Never ensemble already discretised restored label maps.
    return [str(executable), '-i', str(inputs), '-o', str(outputs), '-t', 'Task001_LN',
            '-m', '3d_fullres', '-tr', 'nnUNetTrainerV2_noMirroring', '-p', 'nnUNetPlansv2.1',
            '-chk', 'model_final_checkpoint', '-f', *map(str, folds), '--disable_tta',
            '--mode', 'normal', '--num_threads_preprocessing', '1', '--num_threads_nifti_save', '1']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Original CT NIfTI in HU')
    p.add_argument('--model', type=Path, required=True, help='Released trainer folder containing plans.pkl')
    p.add_argument('--work', type=Path, required=True, help='New private work directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predict', type=Path, default=Path(sys.executable).parent / 'nnUNet_predict')
    p.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    p.add_argument('--axis-variant', default=os.environ.get('CHOI_AXIS_VARIANT', 'unconfirmed'),
                   help='x/y/z flip bits relative to LPS identity: 000–111; default unconfirmed')
    args = p.parse_args(argv)
    try:
        axis_variant(args.axis_variant)
    except ValueError as exc:
        p.error(str(exc))
    if len(set(args.folds)) != len(args.folds):
        p.error('Duplicate folds')
    validate_plans(args.model)
    args.work.mkdir(parents=True, exist_ok=False)
    inputs, outputs, results = (args.work / name for name in ('input', 'prediction', 'results'))
    inputs.mkdir(); outputs.mkdir()
    geometry = prepare_input(args.input, inputs / 'case_0000.nii.gz', args.axis_variant)
    (args.work / 'geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')
    print(json.dumps(geometry), flush=True)
    prepare_model(args.model, results, args.folds)
    env = os.environ.copy()
    env.update(RESULTS_FOLDER=str(results), nnUNet_raw_data_base=str(args.work / 'raw'),
               nnUNet_preprocessed=str(args.work / 'preprocessed'), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(prediction_command(args.predict, inputs, outputs, args.folds), env=env, check=True)
    prediction = outputs / 'case.nii.gz'
    if not prediction.is_file():
        raise RuntimeError('Choi inference produced no prediction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    restore_prediction(prediction, args.input, geometry, args.output)


if __name__ == '__main__':
    main()
