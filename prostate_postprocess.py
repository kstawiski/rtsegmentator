"""Published prostate postprocessing, isolated from model execution."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

RGA_PARAMETERS = {
    'threshold': 'dynamic', 'min_voxels_detection': 10, 'num_lesions_to_extract': 5,
    'dynamic_threshold_factor': 2.5, 'max_prob_round_decimals': None,
    'remove_adjacent_lesion_candidates': True,
}


def roi_name(candidate_id, likelihood):
    if not isinstance(candidate_id, int) or not 1 <= candidate_id <= 5:
        raise ValueError('PI-CAI candidate identifier must be 1 to 5')
    if not math.isfinite(likelihood) or not 0 < likelihood <= 1:
        raise ValueError('Candidate likelihood must be in (0, 1]')
    return f'csPCa_L{candidate_id:02d}_p{likelihood:.2f}'


def extract_picai(probability, spacing_xyz, task, provenance=None):
    """Return detection map, distinct indexed masks and a full-precision report.

    RGA 0.2.4 can report a zero-score component when only undersized peaks
    remain. Such entries contain no detection-map voxels and are not ROIs.
    """
    from importlib.metadata import version
    from picai_prep.preprocessing import crop_or_pad
    from report_guided_annotation import extract_lesion_candidates
    if version('report-guided-annotation') != '0.2.4' or version('picai-prep') != '2.0.0':
        raise RuntimeError('PI-CAI postprocessing requires RGA 0.2.4 and picai-prep 2.0.0')
    probability = np.asarray(probability)
    if (probability.ndim != 3 or not probability.size or not np.isfinite(probability).all()
            or np.any(probability < 0) or np.any(probability > 1)):
        raise ValueError('Expected a finite 3-D probability map in [0, 1]')
    spacing = np.asarray(spacing_xyz, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Expected positive x,y,z voxel spacing')
    cropped = crop_or_pad(crop_or_pad(probability, (20, 384, 384)), probability.shape)
    detection, candidates, indexed = extract_lesion_candidates(cropped, **RGA_PARAMETERS)
    masks, lesions = {}, []
    for candidate_id, score in candidates:
        mask = (indexed == candidate_id) & (detection > 0)
        if score <= 0 or not mask.any():
            continue
        candidate_id, score = int(candidate_id), float(score)
        name = roi_name(candidate_id, score)
        count = int(mask.sum())
        masks[name] = mask
        lesions.append({'candidate_id': candidate_id, 'roi_name': name, 'likelihood': score,
                        'voxel_count': count, 'volume_mm3': count * float(np.prod(spacing)),
                        'volume_ml': count * float(np.prod(spacing)) / 1000})
    report = {'schema_version': 1, 'task': task, 'status': 'nonempty' if lesions else 'empty',
              'reference_role': 't2w', 'channel_order': ['t2w', 'adc', 'hbv'],
              'case_likelihood': float(np.max(detection)), 'lesions': lesions,
              'extraction': {'package': 'report-guided-annotation', 'version': '0.2.4',
                             **RGA_PARAMETERS, 'max_prob_failsafe_stopping_threshold': 0.01,
                             'connectivity': 26, 'adjacency_growth_voxels': 1,
                             'crop_pad_shape_zyx': [20, 384, 384], 'crop_units': 'voxels',
                             'zero_score_components_exported': False},
              'provenance': provenance or {}}
    return detection, masks, report


def largest_component(mask, connectivity=3):
    """MONAI default is full 3-D connectivity; Umeå explicitly uses 18."""
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, ndimage.generate_binary_structure(3, connectivity))
    if count == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def fill_umu_zones(prostate, labels, urethra, spacing_xyz):
    zones = [largest_component((labels == label) & prostate, connectivity=2) & ~urethra
             for label in range(1, 5)]
    holes = prostate & ~urethra & ~np.any(zones, axis=0)
    if holes.any():
        if not any(zone.any() for zone in zones):
            raise ValueError('Umeå postprocessing has no zone seed to fill the prostate')
        # The upstream EDT on an empty zone measures the array boundary. Exclude
        # absent seeds, so missing zones cannot acquire invented components.
        distances = np.stack([ndimage.distance_transform_edt(~zone, sampling=spacing_xyz)
                              if zone.any() else np.full(zone.shape, np.inf) for zone in zones])
        nearest = distances.argmin(axis=0)
        for index, zone in enumerate(zones):
            zone[holes & (nearest == index)] = True
    result = np.zeros(labels.shape, dtype=np.uint8)
    for label, zone in enumerate(zones, 1):
        result[zone] = label
    result[urethra] = 5
    return result


def umu_postprocess(probabilities_czyx, spacing_xyz):
    """Umeå weighted quadratic centreline and 3-mm-radius slice circles.

    Upstream arrays are x,y,z,c. Enforce its equal in-plane spacing assumption;
    slice spacing can be anisotropic. No arbitrary RAS reorientation is applied.
    """
    spacing = np.asarray(spacing_xyz, float)
    probabilities = np.asarray(probabilities_czyx)
    if (probabilities.ndim != 4 or probabilities.shape[0] != 6
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1)):
        raise ValueError('Umeå requires six finite probability channels in c,z,y,x order')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Expected positive x,y,z voxel spacing')
    if not np.isclose(spacing[0], spacing[1], rtol=1e-5, atol=1e-6):
        raise ValueError('Published Umeå urethra construction requires equal in-plane spacing')
    probs = probabilities.transpose(3, 2, 1, 0)
    labels = probs.argmax(axis=3)
    background = largest_component(labels == 0, connectivity=2)
    prostate = largest_component(~background, connectivity=2)
    if not prostate.any():
        return np.zeros(probabilities.shape[1:], dtype=np.uint8)
    urethra_probability = probs[..., 5]
    peaks = urethra_probability.max(axis=(0, 1))
    z = np.arange(probs.shape[2])
    if np.count_nonzero(peaks) < 3:
        raise ValueError('Umeå quadratic centreline requires three slices with nonzero urethra probability')
    xy = np.array([np.round(np.argwhere(urethra_probability[..., k] == peaks[k]).mean(axis=0)) for k in z])
    centers = np.stack([np.round(np.polyval(np.polyfit(z, xy[:, axis], deg=2, w=peaks), z)) for axis in range(2)])
    x, y = np.ogrid[:probs.shape[0], :probs.shape[1]]
    urethra = np.stack([((x - centers[0, k]) ** 2 + (y - centers[1, k]) ** 2)
                        <= (3.0 / spacing[0]) ** 2 for k in z], axis=2) & prostate
    result = fill_umu_zones(prostate, labels, urethra, spacing)
    return result.transpose(2, 1, 0)


def restore_to_reference(image, reference, *, label=True):
    import SimpleITK as sitk
    if image.GetDimension() != 3 or reference.GetDimension() != 3:
        raise ValueError('Restoration requires 3-D images')
    return sitk.Resample(image, reference, sitk.Transform(),
                         sitk.sitkNearestNeighbor if label else sitk.sitkLinear, 0,
                         sitk.sitkUInt8 if label else sitk.sitkFloat32)


def assert_same_grid(image, reference):
    if image.GetSize() != reference.GetSize() or any(
        not np.allclose(a, b, rtol=0, atol=1e-4) for a, b in (
            (image.GetSpacing(), reference.GetSpacing()), (image.GetOrigin(), reference.GetOrigin()),
            (image.GetDirection(), reference.GetDirection()))):
        raise ValueError('Prediction does not match the T2 reference grid')


def write_array(array_zyx, reference, path):
    import SimpleITK as sitk
    if tuple(np.asarray(array_zyx).shape) != tuple(reversed(reference.GetSize())):
        raise ValueError('Array shape does not match reference; copying a header is unsafe')
    image = sitk.GetImageFromArray(np.asarray(array_zyx))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), True)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
