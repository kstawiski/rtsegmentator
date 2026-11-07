"""CT reconstruction-field padding detection and model input preparation."""
from __future__ import annotations

import numpy as np


# Scanners fill the area outside the circular reconstruction field with a constant far below air
# (for example -2000, -3024 or -8192 HU). The shared detector finds border-connected low values
# that fit a large, roughly centred circle. Listed-task preparation replaces the most frequent
# exterior value per acquired slice; NV-Segment replaces every recognised head-foot exterior voxel.
# NV-Reason-CT can restrict detector candidates to declared DICOM padding values and applies
# its separate HU refusal gate. Segmentation preparation has no mostly-padded-volume refusal.

# CT tasks whose image-derived intensity statistics can retain reconstruction-field fill.
# NV-Segment uses ct_model_input with its base head-foot-axis, all-exterior replacement rule.
CT_PADDING_TASKS = frozenset({
    "bamf_liver_lesions_ct",
    "lnctvseg_nnunet_v1",
    "medical_sam3_3d_prompted",
    "medsam_vit_b_box_2d",
    "monai_whole_body_ct",
    "myeloma_spine_verse20_early_ct",
    "myeloma_spine_verse20_mm_ct",
    "myeloma_spine_verse20_stable_ct",
    "nninteractive_v1_prompted",
    "sam_med2d_prompted",
    "segvol_prompted_ct",
    "voxtell_v1_1_text_prompted",
})

PADDING_BELOW_HU = -1500
MAX_PADDING_FRACTION = 0.5
EXTERIOR_RING_COVERAGE = 0.98


def circle_exterior(exterior, yy, xx):
    """True when a border-connected mask is exactly the outside of one large, roughly centred circle: its inner
    boundary fits a circle (RMS <= 1.5 px), everything 2 px beyond that circle is in the mask, and nothing of the
    mask reaches 2 px inside it. Reconstruction fields can sit a few pixels off the image centre."""
    from scipy import ndimage
    rows, cols = exterior.shape
    edge = exterior & ~ndimage.binary_erosion(exterior, border_value=1)  # image edges are not boundaries
    ey, ex = np.nonzero(edge)
    if ey.size < 20:
        return False
    a = np.c_[2 * ex, 2 * ey, np.ones(ex.size)]
    cx, cy, c = np.linalg.lstsq(a, ex.astype(float) ** 2 + ey.astype(float) ** 2, rcond=None)[0]
    radius = np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))
    size = min(rows, cols)
    if (radius < 0.3 * size or np.hypot(cx - (cols - 1) / 2, cy - (rows - 1) / 2) > 0.1 * size
            or np.sqrt(np.mean((np.hypot(ex - cx, ey - cy) - radius) ** 2)) > 1.5):
        return False
    distance = np.hypot(xx - cx, yy - cy)
    beyond = distance >= radius + 2
    if beyond.any() and exterior[beyond].mean() < EXTERIOR_RING_COVERAGE:
        return False
    return not exterior[distance < radius - 2].any()


def exterior_padding(candidates, slice_axis=2):
    """Per slice, candidate voxels that form the exterior of a reconstruction circle, the fraction of each slice
    they cover, and the fraction of each slice that is a candidate at all. Other candidates are not padding."""
    from scipy import ndimage
    padding = np.zeros(candidates.shape, dtype=bool)
    slices, out = np.moveaxis(candidates, slice_axis, 0), np.moveaxis(padding, slice_axis, 0)
    yy, xx = np.mgrid[:slices.shape[1], :slices.shape[2]]
    per_slice, low_per_slice = np.zeros(slices.shape[0]), slices.mean(axis=(1, 2))
    previous, previous_ok, previous_exterior = None, False, None
    for k in range(slices.shape[0]):
        if not slices[k].any():
            continue
        if previous is not None and np.array_equal(slices[k], previous):
            ok, exterior = previous_ok, previous_exterior  # identical consecutive masks are the common case
        else:
            labels, _ = ndimage.label(slices[k])
            border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            border = border[border > 0]
            exterior = np.isin(labels, border) if border.size else None
            ok = exterior is not None and circle_exterior(exterior, yy, xx)
            previous, previous_ok, previous_exterior = slices[k], ok, exterior
        if ok:
            out[k] = exterior
            per_slice[k] = exterior.mean()
    return padding, per_slice, low_per_slice


CT_AIR_HU = -1024


def axial_axis(affine) -> int:
    """Array axis whose direction is closest to the patient's head-foot axis (axial slices are taken across it)."""
    import numpy as np
    m = np.asarray(affine, dtype=float)[:3, :3]
    return int(np.argmax(np.abs(m[2]) / np.linalg.norm(m, axis=0)))


def ct_model_input(image, path):
    """NV-Segment's base rule: replace every recognised exterior voxel on the head-foot axis."""
    data = np.asanyarray(image.dataobj)
    low = data < PADDING_BELOW_HU
    if not low.any():
        return 0
    padding = exterior_padding(low, slice_axis=axial_axis(image.affine))[0]
    del low
    return _write_ct_model_input(image, path, data, padding)


def acquired_ct_model_input(image, path, slice_axis=2):
    """Replace each acquired slice's most frequent exterior value; return replaced and residual counts.

    Only this axis is inspected. Other exterior values and values outside the detected exterior are retained.
    The orientation contract keeps the acquired slice axis at array axis 2, including oblique series.
    """
    data = np.asanyarray(image.dataobj)
    low = data < PADDING_BELOW_HU
    if not low.any():
        return 0, 0
    padding = exterior_padding(low, slice_axis=slice_axis)[0]
    del low
    exterior_count = int(np.count_nonzero(padding))
    slices = np.moveaxis(data, slice_axis, 0)
    masks = np.moveaxis(padding, slice_axis, 0)
    for plane, mask in zip(slices, masks):
        if mask.any():
            values, counts = np.unique(plane[mask], return_counts=True)
            mask &= plane == values[np.argmax(counts)]
    count = _write_ct_model_input(image, path, data, padding)
    return count, exterior_count - count


def _write_ct_model_input(image, path, data, padding):
    """Share lossless writing between preparation rules; write nothing for an empty replacement mask."""
    import nibabel as nib
    count = int(np.count_nonzero(padding))
    if not count:
        return 0
    if data.dtype.kind in 'iu':  # the narrowest signed integer type that holds every value (the fill is negative)
        lo, hi = int(data.min()), int(data.max())
        dtype = next((t for t in (np.int16, np.int32, np.int64) if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max), None)
        if dtype is None:
            raise ValueError(f'CT values {lo}..{hi} do not fit a signed integer NIfTI')
    else:
        dtype = np.float64 if data.dtype == np.float64 or not np.array_equal(data, data.astype(np.float32)) else np.float32
    # a private array: proxy-backed data was read from disk already, array-backed data belongs to the caller
    shared = not nib.is_proxy(image.dataobj) or isinstance(data, np.memmap) or not data.flags.writeable
    data = data.astype(dtype, copy=shared or data.dtype != dtype)
    data[padding] = CT_AIR_HU
    del padding
    # one file for the bundle: a .hdr/.img pair is written as its single-file NIfTI counterpart
    single = nib.Nifti2Image if isinstance(image, (nib.Nifti2Image, nib.Nifti2Pair)) else nib.Nifti1Image
    header = image.header.copy(); header.set_data_dtype(dtype); header.set_slope_inter(1, 0)
    nib.save(single(data, image.affine, header), path)
    saved = nib.load(path)
    if (type(saved) is not single or saved.shape != image.shape
            or not np.array_equal(saved.affine, image.affine)):
        raise ValueError('CT model input lost the native grid')
    return count
