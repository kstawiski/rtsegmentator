"""Frozen full-volume validator from the pre-crop implementation, for CPU equivalence evidence."""

from __future__ import annotations

import cv2
import nibabel as nib
import numpy as np


def full_volume_reference(roi_contour, mask_image, expected_mask,
                           minimum_dice: float = 0.95, *, source_slices=None) -> dict:
    """Rasterize patient-space contours on the source grid and compare masks.

    This intentionally does not use rt-utils' RTSTRUCT reader. Its returned
    array follows an internal column/row convention that can hide or suggest
    transpositions. With DICOM sources, each SOP reference selects its acquired
    plane, including irregular stacks that a NIfTI affine cannot represent.
    Without them, retain the inverse-NIfTI-affine check for existing callers.
    Dice and the 0.05-voxel plane tolerance are unchanged.
    """
    expected = np.asarray(expected_mask, dtype=bool)
    reconstructed = np.zeros(expected.shape, dtype=bool)
    inverse_affine = np.linalg.inv(mask_image.affine)
    polygons_by_slice: dict[int, list[np.ndarray]] = {}
    by_uid = ({str(ds.SOPInstanceUID): (k, ds) for k, ds in enumerate(source_slices)}
              if source_slices is not None else None)

    for contour in getattr(roi_contour, "ContourSequence", []):
        values = getattr(contour, "ContourData", None)
        if not values:
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
            k, ds = by_uid[refs[0]]
            column, row = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
            normal = np.cross(column, row)
            normal /= np.linalg.norm(normal)
            delta = lps - np.asarray(ds.ImagePositionPatient, dtype=float)
            xy = np.column_stack((delta @ column / float(ds.PixelSpacing[1]),
                                  delta @ row / float(ds.PixelSpacing[0])))
            # Large gaps must not relax the original plane tolerance.
            step = float(mask_image.header.get_zooms()[2])
            for neighbor in (k - 1, k + 1):
                if 0 <= neighbor < len(source_slices):
                    distance = abs((np.asarray(source_slices[neighbor].ImagePositionPatient)
                                    - np.asarray(ds.ImagePositionPatient)) @ normal)
                    if distance > 0:
                        step = min(step, float(distance))
            slice_positions = k + (delta @ normal) / step
        if not 0 <= k < expected.shape[2]:
            raise ValueError(f"contour maps outside NIfTI slice range: {k}")
        if np.max(np.abs(slice_positions - k)) > 0.05:
            grid = "DICOM source" if by_uid is not None else "NIfTI"
            raise ValueError(f"contour is not planar on the {grid} voxel grid")
        polygons_by_slice.setdefault(k, []).append(
            np.rint(xy).astype(np.int32)
        )

    for k, polygons in polygons_by_slice.items():
        # OpenCV points are (column, row), while its image is (row, column).
        # fillPoly also reconstructs rt-utils' single points and zero-area
        # lines; they undergo the same plane, reference and Dice checks.
        plane = np.zeros((expected.shape[1], expected.shape[0]), dtype=np.uint8)
        cv2.fillPoly(plane, polygons, 1)
        reconstructed[:, :, k] = plane.T.astype(bool)

    expected_count = int(expected.sum())
    reconstructed_count = int(reconstructed.sum())
    intersection = int(np.logical_and(expected, reconstructed).sum())
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
