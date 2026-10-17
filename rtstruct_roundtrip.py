"""Independent physical-space RTSTRUCT contour round-trip validation."""

from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataelem import DataElement, RawDataElement
from scipy.ndimage import find_objects


def label_boxes(data):
    """Find all positive-label boxes in one traversal of the source volume."""
    maximum = int(data.max(initial=0))
    if maximum == 0:
        return {}
    return {label: box for label, box in enumerate(find_objects(data, max_label=maximum), 1)
            if box is not None}


# Match complete DS numbers, rather than only allowed characters: fromstring
# alone can silently truncate malformed values such as "1.2.3" or "1e+".
_DS_NUMBER = rb" *[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)? *"
_DS_VALUES = re.compile(_DS_NUMBER + rb"(?:\\" + _DS_NUMBER + rb")*")


def has_coordinates(values):
    """Original truth semantics, also supporting bulk-decoded NumPy arrays."""
    return bool(values.size) if isinstance(values, np.ndarray) else bool(values)


def read_rtstruct(path):
    """Read the saved file, strictly parse ordinary contour DS bytes in bulk.

    Exceptional values use pydicom's conventional decoder, retaining existing
    empty/nonfinite geometry errors and rejecting malformed numeric fields.
    The common path does not change any process-global pydicom settings.
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    for roi in getattr(ds, "ROIContourSequence", []):
        for contour in getattr(roi, "ContourSequence", []):
            raw = contour.get_item(0x30060050)
            if (isinstance(raw, RawDataElement) and raw.VR in (None, "DS") and raw.value
                    and _DS_VALUES.fullmatch(raw.value)):
                values = np.fromstring(raw.value.decode("ascii"), dtype=np.float64, sep="\\")
                if values.size == raw.value.count(b"\\") + 1:
                    value = values[0] if values.size == 1 else values
                    contour[raw.tag] = DataElement(raw.tag, "DS", value, already_converted=True)
                    continue
            if raw is not None:
                previous = pydicom.config.use_DS_numpy
                try:
                    pydicom.config.use_DS_numpy = False
                    contour.ContourData
                finally:
                    pydicom.config.use_DS_numpy = previous
    return ds


@dataclass
class CroppedMask:
    """Owned voxel box in the original [column, row, slice] grid."""
    data: np.ndarray
    offset: tuple[int, int, int]
    shape: tuple[int, int, int]


def mask_box(mask):
    # Axis reductions avoid an Nx3 coordinate array for a large solid ROI.
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
        bounds.append(slice(int(occupied[0]), int(occupied[-1]) + 1) if len(occupied) else slice(0, 0))
    return tuple(bounds)


def data_box(data, *, label=None, threshold=0):
    """Find an ROI box with at most one 2-D comparison array in memory.

    Scan along the largest stride so both C and Fortran NIfTI storage are
    visited in contiguous slabs, without a volume-sized temporary mask.
    """
    axis = int(np.argmax(np.abs(data.strides)))
    other = [i for i in range(3) if i != axis]
    slabs = np.moveaxis(data, axis, 0)
    lower = list(data.shape)
    upper = [0, 0, 0]
    for index, slab in enumerate(slabs):
        plane = slab == label if label is not None else slab > threshold
        first = np.flatnonzero(np.any(plane, axis=1))
        if not len(first):
            continue
        second = np.flatnonzero(np.any(plane, axis=0))
        lower[axis] = min(lower[axis], index)
        upper[axis] = index + 1
        for dim, occupied in zip(other, (first, second)):
            lower[dim] = min(lower[dim], int(occupied[0]))
            upper[dim] = max(upper[dim], int(occupied[-1]) + 1)
    if not any(upper):
        return (slice(0, 0),) * 3
    return tuple(slice(lo, hi) for lo, hi in zip(lower, upper))


def crop_mask(mask, box=None):
    mask = np.asarray(mask, dtype=bool)
    box = mask_box(mask) if box is None else box
    return CroppedMask(np.array(mask[box], dtype=bool, copy=True),
                       tuple(s.start for s in box), tuple(mask.shape))


def _source_plane(k, ds, source_slices):
    column, row = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
    normal = np.cross(column, row)
    normal /= np.linalg.norm(normal)
    origin = np.asarray(ds.ImagePositionPatient, dtype=float)
    neighbor_step = np.inf
    for neighbor in (k - 1, k + 1):
        if 0 <= neighbor < len(source_slices):
            distance = abs((np.asarray(source_slices[neighbor].ImagePositionPatient) - origin) @ normal)
            if distance > 0:
                neighbor_step = min(neighbor_step, float(distance))
    return (k, origin, column, row, normal, float(ds.PixelSpacing[1]),
            float(ds.PixelSpacing[0]), neighbor_step)


def source_plane_geometry(source_slices):
    """Cache acquired-plane transforms once across the ROIs of a conversion."""
    return {str(ds.SOPInstanceUID): _source_plane(k, ds, source_slices)
            for k, ds in enumerate(source_slices)}


def validate_roi_roundtrip(roi_contour, mask_image, expected_mask,
                           minimum_dice: float = 0.95, *, source_slices=None,
                           source_geometry=None) -> dict:
    """Rasterize patient-space contours on the source grid and compare masks.

    This intentionally does not use rt-utils' RTSTRUCT reader. Its returned
    array follows an internal column/row convention that can hide or suggest
    transpositions. With DICOM sources, each SOP reference selects its acquired
    plane, including irregular stacks that a NIfTI affine cannot represent.
    Without them, retain the inverse-NIfTI-affine check for existing callers.
    Dice and the 0.05-voxel plane tolerance are unchanged.
    """
    cropped = expected_mask if isinstance(expected_mask, CroppedMask) else crop_mask(expected_mask)
    expected = cropped.data
    shape = cropped.shape
    ox, oy, oz = cropped.offset
    inverse_affine = np.linalg.inv(mask_image.affine)
    polygons_by_slice: dict[int, list[np.ndarray]] = {}
    by_uid = (source_geometry if source_geometry is not None else
              {str(ds.SOPInstanceUID): (k, ds) for k, ds in enumerate(source_slices)}
              if source_slices is not None else None)
    used_planes = {}

    for contour in getattr(roi_contour, "ContourSequence", []):
        values = getattr(contour, "ContourData", None)
        if not has_coordinates(values):
            raise ValueError("empty contour coordinates")
        lps = np.asarray(values, dtype=float).reshape(-1, 3)
        if not np.isfinite(lps).all():
            raise ValueError("invalid contour coordinates")
        if int(getattr(contour, "NumberOfContourPoints", 0)) != len(lps):
            raise ValueError("contour point count does not match coordinates")
        if contour.ContourGeometricType != "CLOSED_PLANAR":
            raise ValueError("expected CLOSED_PLANAR contour")
        if by_uid is None:
            ras = lps * np.array([-1.0, -1.0, 1.0])
            ijk = nib.affines.apply_affine(inverse_affine, ras)
            slice_positions = ijk[:, 2]
            k = int(round(float(np.median(slice_positions))))
            xy = ijk[:, :2]
        else:
            refs = [str(im.ReferencedSOPInstanceUID)
                    for im in getattr(contour, "ContourImageSequence", [])]
            if len(refs) != 1 or refs[0] not in by_uid:
                raise ValueError("contour must reference exactly one source SOP Instance UID")
            if source_geometry is None:
                if refs[0] not in used_planes:
                    index, ds = by_uid[refs[0]]
                    used_planes[refs[0]] = _source_plane(index, ds, source_slices)
                plane = used_planes[refs[0]]
            else:
                plane = by_uid[refs[0]]
            k, origin, column, row, normal, dx, dy, neighbor_step = plane
            delta = lps - origin
            xy = np.column_stack((delta @ column / dx, delta @ row / dy))
            # Large gaps must not relax the original plane tolerance.
            step = min(float(mask_image.header.get_zooms()[2]), neighbor_step)
            slice_positions = k + (delta @ normal) / step
        if not 0 <= k < shape[2]:
            raise ValueError(f"contour maps outside NIfTI slice range: {k}")
        if np.max(np.abs(slice_positions - k)) > 0.05:
            grid = "DICOM source" if by_uid is not None else "NIfTI"
            raise ValueError(f"contour is not planar on the {grid} voxel grid")
        polygons_by_slice.setdefault(k, []).append(
            np.rint(xy).astype(np.int32)
        )

    expected_count = int(expected.sum())
    reconstructed_count = intersection = 0
    for k, polygons in polygons_by_slice.items():
        # Include all reconstructed pixels inside the source grid, even when
        # contours extend beyond the expected box. Cropping to the expected
        # box alone would inflate Dice and change acceptance semantics.
        points = np.concatenate(polygons)
        lo = np.maximum(points.min(0), 0)
        hi = np.minimum(points.max(0) + 1, shape[:2])
        if np.any(hi <= lo):
            continue
        plane = np.zeros((int(hi[1] - lo[1]), int(hi[0] - lo[0])), dtype=np.uint8)
        cv2.fillPoly(plane, [poly - lo for poly in polygons], 1)
        reconstructed_count += int(np.count_nonzero(plane))
        if oz <= k < oz + expected.shape[2]:
            start = np.maximum(lo, (ox, oy))
            stop = np.minimum(hi, (ox + expected.shape[0], oy + expected.shape[1]))
            if np.all(stop > start):
                x, y = start
                xx, yy = stop
                intersection += int(np.count_nonzero(
                    plane[y-lo[1]:yy-lo[1], x-lo[0]:xx-lo[0]].T &
                    expected[x-ox:xx-ox, y-oy:yy-oy, k-oz]))
        del plane
    denominator = expected_count + reconstructed_count
    dice = float(2 * intersection / denominator) if denominator else 1.0
    if dice < minimum_dice:
        raise ValueError(
            f"physical contour round-trip Dice {dice:.6f} < {minimum_dice:.6f}"
        )
    return {
        "expected_voxels": expected_count,
        "roundtrip_voxels": reconstructed_count,
        "intersection_voxels": intersection,
        "roundtrip_dice": round(dice, 6),
    }
