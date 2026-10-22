#!/usr/bin/env python3
"""NCI/NVIDIA lesion and PI-RADS pipeline using pinned upstream operators."""
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
from held_mri import NCI, NCI_THRESHOLD, export_masks, roi_name, runtime_gpu
from held_mri_upstream import nci_pipeline
from prostate_models import runtime_provenance


def nci_measurements(reference, result):
    """Use classifier-grid IDs, class promotion and actual native-grid volumes."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from skimage.measure import label, regionprops
    from scipy.special import softmax
    from monai.data import MetaTensor
    from monai.transforms import Spacing
    gland = nib.load(result['gland_path'])
    if gland.shape != reference.shape or not np.allclose(gland.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError('NCI gland does not match native T2 grid')
    gland_array = gland.get_fdata()
    if not np.all(np.isin(gland_array, (0, 1))):
        raise ValueError('NCI gland labels are not binary; published multiplication would change probability scaling')
    masks = {'NCI_Prostate': gland_array > 0}
    native_voxel_ml = float(abs(np.linalg.det(reference.affine[:3, :3])) / 1000)
    report = {'schema_version': 1, 'task': NCI, 'status': 'empty', 'reference_role': 't2w',
              'channel_order': ['t2w', 'adc', 'hbv'], 'lesions': [],
              'gland': {'roi_name': 'NCI_Prostate', 'voxel_count': int(masks['NCI_Prostate'].sum()),
                        'volume_ml': float(masks['NCI_Prostate'].sum() * native_voxel_ml),
                        'measurement_grid': 'native_t2',
                        'upstream_volume_mm3_assuming_0_5mm': float(gland_array.sum() * 0.125)},
              'extraction': {'threshold': NCI_THRESHOLD, 'probability_channel': 1,
                             'components': 'skimage.measure.label default full 3-D connectivity on classifier 0.5-mm grid',
                             'promotion': 'Argmax class 2 (PI-RADS 4) and major axis >40 voxels becomes PI-RADS 5'}}
    # Retain the upstream background measurement under a name that states the defect.
    regions = label(gland_array)
    first = np.unique(regions)[0]
    props = regionprops((regions == first).astype(np.uint8))
    if props:
        report['gland']['upstream_first_region_major_axis_mm_assuming_0_5mm'] = float(props[0].major_axis_length * 0.5)
        report['gland']['upstream_first_region_is_background'] = bool(first == 0)
    if result.get('empty_lesions'):
        return masks, report
    data = result['classifier_data']
    components = label(np.asarray(data['nda_pred_resize']))
    logits = np.atleast_2d(result['classifier_logits'])
    if not np.isfinite(logits).all() or logits.shape[1] != 4:
        raise ValueError('Invalid NCI classifier outputs')
    probabilities = softmax(logits, axis=1)
    properties = regionprops(components)
    if len(properties) != len(logits):
        raise ValueError('NCI classifier/component count mismatch')
    # Recreate only the metadata of the SAME upstream Spacing transform, using its
    # nearest-neighbour lesion tensor. No guessed 0.5-mm affine or native relabeling.
    original = nib.load(result['lesion_path'])
    affine = np.squeeze(np.asarray(data['affine'])) if 'affine' in data else original.affine
    tensor = MetaTensor(original.get_fdata()[None], affine=affine)
    spaced = Spacing(pixdim=(0.5, 0.5, 0.5), mode='nearest')(tensor)
    if not np.array_equal(np.asarray(spaced)[0], np.asarray(data['nda_pred_resize'])):
        raise ValueError('Classifier spacing does not match export geometry')
    for index, prop in enumerate(properties):
        cls = int(logits[index].argmax())
        pirads = 5 if cls == 2 and prop.major_axis_length > 40 else cls + 2
        name = roi_name(NCI, int(prop.label), pirads)
        iso_mask = components == prop.label
        native = resample_from_to(nib.Nifti1Image(iso_mask.astype(np.uint8), np.asarray(spaced.affine)),
                                  reference, order=0).get_fdata() > 0
        if not native.any():
            raise ValueError('An NCI classifier component vanishes on native T2; no lesion may be silently discarded')
        masks[name] = native
        report['lesions'].append({'candidate_id': int(prop.label), 'roi_name': name, 'pi_rads': pirads,
            'classifier_argmax_pi_rads': cls + 2, 'classifier_probabilities': probabilities[index].tolist(),
            'classifier_probability_classes': [2, 3, 4, 5], 'classifier_logits': logits[index].tolist(),
            'major_axis_length_mm': float(prop.major_axis_length * 0.5),
            'major_axis_measurement_grid': 'classifier_0.5mm',
            'classifier_volume_ml': float(iso_mask.sum() * 0.125 / 1000),
            'upstream_lesion_volume_mm3': float(iso_mask.sum() * 0.125),
            'volume_ml': float(native.sum() * native_voxel_ml),
            'voxel_count': int(native.sum()), 'volume_measurement_grid': 'native_t2'})
    report['status'] = 'nonempty'
    return masks, report


def main():
    import nibabel as nib
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=[NCI], required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--held-mri-acceptance-audit', action='store_true')
    args = parser.parse_args()
    from held_mri import verify_release
    verify_release(NCI, args.model_root)
    from held_mri import runtime_start
    gpu = runtime_start(NCI, args.model_root, worker_python=args.worker_python, acceptance_audit=args.held_mri_acceptance_audit)
    work, output = args.work.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    context = json.loads((work / 'context.json').read_text())
    if context.get('dicom_roles'):
        from held_mri_upstream import nci_dicom_images, save_nci_native
        save_nci_native(nci_dicom_images(context['dicom_roles']), work / 'native')
    result = nci_pipeline(args.model_root.resolve(), work / 'native', work / 'upstream')
    reference = nib.load(work / 'native/t2w.nii.gz')
    masks, report = nci_measurements(reference, result)
    provenance = runtime_provenance(NCI, args.model_root)
    context.pop('dicom_roles', None)
    provenance.update(context)
    provenance.update({'gpu': gpu, 'weight_licence': 'NIH/NVIDIA non-commercial research; excludes diagnosis and treatment',
                       'owner_acceptance': 'Owner accepted these terms on 2026-10-22',
                       'classifier_probabilities': 'Softmax of observed upstream four-class logits; argmax unchanged',
                       'empty_classifier_skipped': bool(result.get('empty_lesions'))})
    report['provenance'] = provenance
    report['artifacts'] = {}
    for kind, key in (('probability', 'probability_path'), ('detection', 'lesion_path'), ('gland', 'gland_path')):
        image = nib.load(result[key])
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
            raise ValueError(f'NCI {kind} geometry is not native T2; refusing upstream orientation defect')
        path = Path(str(output) + f'.{kind}.nii.gz')
        shutil.copyfile(result[key], path)
        report['artifacts'][kind + '_map'] = path.name
    export_masks(work, output, reference, masks, report)


if __name__ == '__main__':
    main()
