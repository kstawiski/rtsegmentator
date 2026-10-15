"""Independent physical-space RTSTRUCT contour round-trip validation."""

from __future__ import annotations

import cv2
import nibabel as nib
import numpy as np


def validate_roi_roundtrip(roi_contour, mask_image, expected_mask,
                           minimum_dice: float = 0.95) -> dict:
    """Rasterize patient-space contours on the NIfTI grid and compare masks.

    This intentionally does not use rt-utils' RTSTRUCT reader. Its returned
    array follows an internal column/row convention that can hide or suggest
    transpositions. Contour LPS coordinates are transformed directly through
    the inverse NIfTI affine instead.
    """
    expected = np.asarray(expected_mask, dtype=bool)
    reconstructed = np.zeros(expected.shape, dtype=bool)
    inverse_affine = np.linalg.inv(mask_image.affine)
    polygons_by_slice: dict[int, list[np.ndarray]] = {}

    for contour in getattr(roi_contour, "ContourSequence", []):
        values = getattr(contour, "ContourData", None)
        if not values:
            continue
        lps = np.asarray(values, dtype=float).reshape(-1, 3)
        ras = lps * np.array([-1.0, -1.0, 1.0])
        ijk = nib.affines.apply_affine(inverse_affine, ras)
        slice_positions = ijk[:, 2]
        k = int(round(float(np.median(slice_positions))))
        if not 0 <= k < expected.shape[2]:
            raise ValueError(f"contour maps outside NIfTI slice range: {k}")
        if np.max(np.abs(slice_positions - k)) > 0.05:
            raise ValueError("contour is not planar on the NIfTI voxel grid")
        polygons_by_slice.setdefault(k, []).append(
            np.rint(ijk[:, :2]).astype(np.int32)
        )

    for k, polygons in polygons_by_slice.items():
        # OpenCV points are (column, row), while its image is (row, column).
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
