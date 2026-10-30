"""CT reconstruction-field padding shared by the NV-Reason-CT HU check and the NV-Segment CT input (numpy/scipy only)."""
from __future__ import annotations

import numpy as np


# Scanners fill the area outside the circular reconstruction field with a constant far below air (for example
# -2000, -3024 or, for unsigned storage with intercept -8192, -8192 HU). The model input is clipped to
# [-1000, 1000] HU, so that fill reaches the model as air, but it distorts the intensity gate. Only voxels that
# (1) are below PADDING_BELOW_HU and, when the DICOM header identifies padding values, equal one of them, and
# (2) on their slice form the complete exterior of a circle centred on the image are excluded. Anything else that
# is low (interior corruption, even when connected to the border) stays in the distribution and fails the rule.
# A slice or volume that is mostly padding is refused.
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
