"""Retain pixel-centre points and lines, including both acquired volume edges."""
import copy
import json

import cv2
import nibabel as nib
import numpy as np
import pydicom
import pytest

from rtstruct_geometry import add_roi, create_rtstruct
from rtstruct_roundtrip import validate_roi_roundtrip
from test_rtstruct_geometry import assert_viewer_exact, convert, inputs
from test_viewer import GEOMETRIES, make_mask, make_series


def edge_mask():
    mask = make_mask().transpose(1, 0, 2)
    mask[0, 0, 0] = True                # Isolated scan-edge pixel.
    mask[3:5, 1, 0] = True             # Two-pixel piece.
    mask[8:11, 2, 0] = True            # Three-pixel line.
    mask[16, :6, 0] = True             # Six-pixel line.
    mask[[25, 26, 27], [0, 1, 0], 0] = True  # Zero-area, four-point contour.
    mask[35:40, 5:9, 0] = True         # Ordinary polygon on the same plane.
    mask[-1, -1, -1] = True
    mask[2:8, -1, -1] = True
    return mask


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("iop,spacing,reverse,gaps,orientation", GEOMETRIES)
def test_edge_components_survive_both_converters(synthetic_ct, tmp_path, independent,
                                                iop, spacing, reverse, gaps, orientation):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    image_path, prediction, mask = inputs(source, tmp_path, edge_mask())
    result, output = convert(source, tmp_path, image_path, prediction, independent)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["omitted_rois"] == []
    assert "RTSTRUCT_CONVERSION_NOTE" not in result.stderr
    assert report["geometry_bbox_max_delta_mm"] == 0
    assert report["physical_roundtrip"]["Asymmetric L"] == {
        "expected_voxels": int(mask.sum()), "roundtrip_voxels": int(mask.sum()),
        "intersection_voxels": int(mask.sum()), "roundtrip_dice": 1.0}
    counts = [int(c.NumberOfContourPoints)
              for c in pydicom.dcmread(output).ROIContourSequence[0].ContourSequence]
    assert 1 in counts and 2 in counts
    polygons, _ = cv2.findContours(mask[:, :, 0].T.astype(np.uint8),
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert any(len(p) >= 3 and cv2.contourArea(p) == 0 for p in polygons)
    assert_viewer_exact(source, output, mask.transpose(1, 0, 2))
    print(f"EDGE_COMPONENTS iop={iop} spacing={spacing} gaps={gaps} independent={independent}: "
          f"{int(mask.sum())} voxels; Dice=1.0; differing voxels=0")


@pytest.mark.parametrize("piece", ["point", "line", "zero-area"])
@pytest.mark.parametrize("source_grid,corruption", [
    (source_grid, corruption)
    for source_grid in (False, True)
    for corruption in ("shift", "off-plane", "missing", "wrong-uid", "count", "nonfinite", "empty")
    if source_grid or corruption != "wrong-uid"  # NIfTI-only mode has no SOP map.
])
def test_degenerate_pieces_do_not_bypass_gate(synthetic_ct, tmp_path, piece, source_grid, corruption):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, iop=[1, 0, 0, 0, -1, 0])
    mask = np.zeros((48, 32, 6), dtype=bool)
    if piece == "point":
        mask[0, 0, 0] = True
    elif piece == "line":
        mask[:6, 0, 0] = True
    else:
        mask[[0, 1, 2], [0, 1, 0], 0] = True
    _, prediction, _ = inputs(source, tmp_path, mask)
    image = nib.load(prediction)
    rt = create_rtstruct(source, image)
    assert add_roi(rt, mask, piece, [1, 2, 3])
    roi = copy.deepcopy(rt.ds.ROIContourSequence[0])
    kwargs = {"source_slices": rt.series_data} if source_grid else {}
    assert validate_roi_roundtrip(roi, image, mask, **kwargs)["roundtrip_dice"] == 1
    contour = roi.ContourSequence[0]
    points = np.asarray(contour.ContourData).reshape(-1, 3)
    if corruption == "shift":
        points[:, 0] += 1
    elif corruption == "off-plane":
        points[:, 2] += .101  # Above the unchanged 0.05-voxel threshold.
    elif corruption == "missing":
        del roi.ContourSequence[0]
    elif corruption == "wrong-uid":
        contour.ContourImageSequence[0].ReferencedSOPInstanceUID = rt.series_data[1].SOPInstanceUID
    elif corruption == "count":
        contour.NumberOfContourPoints += 1
    elif corruption == "nonfinite":
        points[0, 0] = np.nan
    elif corruption == "empty":
        points = np.empty((0, 3))
    contour.ContourData = points.ravel().tolist()
    with pytest.raises(ValueError):
        validate_roi_roundtrip(roi, image, mask, **kwargs)


def test_only_empty_mask_has_no_contours(synthetic_ct, tmp_path):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    mask = np.zeros((48, 32, 6), dtype=bool)
    _, prediction, _ = inputs(source, tmp_path, mask)
    rt = create_rtstruct(source, nib.load(prediction))
    assert not add_roi(rt, mask, "Empty", [1, 2, 3])
    assert len(rt.ds.ROIContourSequence) == 0
    mask[0, 0, 0] = True
    assert add_roi(rt, mask, "Point", [1, 2, 3])
    assert len(rt.ds.ROIContourSequence[0].ContourSequence) == 1
