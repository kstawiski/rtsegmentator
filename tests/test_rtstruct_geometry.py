"""CPU evidence for both production writers; synthetic DICOM and masks only."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pydicom
import pytest
from rt_utils import RTStructBuilder

from crop_prostate_sbrt import crop_ct_rtstruct
from rtstruct_geometry import add_roi, create_rtstruct, dicom_grid
from orientation_contract import restore_image
from rtstruct_roundtrip import validate_roi_roundtrip
from test_viewer import GEOMETRIES, make_mask, make_series, rasterize
import viewer

ROOT = Path(__file__).parents[1]


def inputs(source, root, mask=None):
    image_path = root / "input.nii.gz"
    result = subprocess.run([sys.executable, str(ROOT / "dicom_series_to_nifti.py"),
                             str(source), str(image_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    image = nib.load(image_path)
    mask = make_mask().transpose(1, 0, 2) if mask is None else mask
    prediction = root / "prediction.nii.gz"
    # Existing tests specify masks on acquired DICOM pixels. Feed the same
    # physical mask in the new model storage, retaining source-mask assertions.
    _, source_affine, _ = dicom_grid(source, image)
    model_mask, _ = restore_image(nib.Nifti1Image(mask.astype(np.uint8), source_affine),
                                  image.shape, image.affine)
    nib.save(model_mask, prediction)
    return image_path, prediction, mask


def convert(source, root, image_path, prediction, independent=False, labels=None):
    labels = labels or {"1": "Asymmetric L"}
    mapping = root / "labels.json"
    output = root / "model_RTSTRUCT.dcm"
    if independent:
        mapping.write_text(json.dumps({name: str(prediction) for name in labels.values()}))
        script, args = "convert_masks_validate_rtstruct.py", ["--masks-json", str(mapping)]
    else:
        mapping.write_text(json.dumps(labels))
        script, args = "convert_validate_rtstruct.py", ["--prediction", str(prediction), "--labels", str(mapping)]
    result = subprocess.run([sys.executable, str(ROOT / script), *args,
                             "--model-input", str(image_path), "--dicom", str(source),
                             "--output", str(output)], capture_output=True, text=True)
    return result, output


def assert_viewer_exact(source, output, mask):
    series = viewer.index_series(sorted(source.iterdir()))
    _, mapped = viewer.map_structure(viewer.read_structure(output), output, output.parent, series)
    for k in range(mask.shape[2]):
        for roi in mapped.get(k, []):
            assert np.array_equal(rasterize(roi["paths"], mask.shape[:2]), mask[:, :, k])
        if k not in mapped:
            assert not mask[:, :, k].any()


@pytest.mark.parametrize("iop,spacing,reverse,gaps,orientation", GEOMETRIES)
def test_independent_masks_full_geometry_matrix(synthetic_ct, tmp_path, iop, spacing, reverse, gaps, orientation):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    image_path, prediction, mask = inputs(source, tmp_path)
    # Two identical masks prove that overlap is preserved rather than flattened.
    result, output = convert(source, tmp_path, image_path, prediction, independent=True,
                             labels={"1": "Asymmetric L", "2": "Overlapping L"})
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["rois"] == 2
    assert all(item["roundtrip_dice"] == 1 for item in report["physical_roundtrip"].values())
    assert_viewer_exact(source, output, mask.transpose(1, 0, 2))
    print(f"MASK_WRITER iop={iop} spacing={spacing} reverse={reverse} gaps={gaps}: {json.dumps(report)}")


def contour_equivalence(before, after):
    """Compare every point, contour type, point count and referenced source plane."""
    assert len(before) == len(after)
    maximum, points = 0.0, 0
    for a, b in zip(before, after):
        assert a.ContourImageSequence == b.ContourImageSequence
        assert a.ContourGeometricType == b.ContourGeometricType
        assert a.NumberOfContourPoints == b.NumberOfContourPoints
        aa, bb = np.asarray(a.ContourData).reshape(-1, 3), np.asarray(b.ContourData).reshape(-1, 3)
        assert aa.shape == bb.shape
        assert np.isfinite(aa).all() and np.isfinite(bb).all()
        maximum = max(maximum, float(np.linalg.norm(aa - bb, axis=1).max()))
        points += len(aa)
    return maximum, points


@pytest.mark.parametrize("with_edges", [False, True])
@pytest.mark.parametrize("iop,spacing,reverse,gaps,orientation",
                         [g for g in GEOMETRIES if g.values[1][0] == g.values[1][1] and not g.values[3]])
def test_regular_stack_equivalence_to_rt_utils(synthetic_ct, tmp_path, iop, spacing, reverse, gaps,
                                              orientation, with_edges):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    from test_rtstruct_degenerate import edge_mask
    image_path, prediction, mask = inputs(source, tmp_path, edge_mask() if with_edges else None)
    result, output = convert(source, tmp_path, image_path, prediction)
    assert result.returncode == 0, result.stdout + result.stderr
    old = RTStructBuilder.create_new(dicom_series_path=str(source))
    old.add_roi(mask=mask.transpose(1, 0, 2), name="Asymmetric L")
    new = pydicom.dcmread(output)
    before, after = old.ds.ROIContourSequence[0].ContourSequence, new.ROIContourSequence[0].ContourSequence
    maximum, points = contour_equivalence(before, after)
    assert maximum <= 1e-3
    print(f"EQUIVALENCE iop={iop} reverse={reverse} edges={with_edges}: "
          f"{points} points, max distance={maximum:.9f} mm")


def test_per_slice_origin_orientation_and_spacing(synthetic_ct, tmp_path):
    source = tmp_path / "source"
    paths = make_series(synthetic_ct, source)
    # Modify a middle slice: the NIfTI header still describes the first slice.
    ds = pydicom.dcmread(paths[3])
    ds.ImagePositionPatient = [34.25, -61.5, 18.0]
    ds.ImageOrientationPatient = [0.8, 0.6, 0, -0.6, 0.8, 0]
    ds.PixelSpacing = [2.3, 0.7]
    ds.save_as(paths[3], enforce_file_format=True)
    image_path, prediction, mask = inputs(source, tmp_path)
    result, output = convert(source, tmp_path, image_path, prediction)
    assert result.returncode == 0, result.stdout + result.stderr
    contour = next(c for c in pydicom.dcmread(output).ROIContourSequence[0].ContourSequence
                   if c.ContourImageSequence[0].ReferencedSOPInstanceUID == ds.SOPInstanceUID)
    points = np.asarray(contour.ContourData).reshape(-1, 3) - np.asarray(ds.ImagePositionPatient)
    xy = np.column_stack((points @ np.array([0.8, 0.6, 0]) / .7,
                          points @ np.array([-.6, .8, 0]) / 2.3))
    assert np.array_equal(rasterize([xy], (32, 48)), mask[:, :, 3].T)


@pytest.mark.parametrize("corruption", ["spacing", "off-plane", "wrong-uid", "reflection", "missing-polygon"])
def test_roundtrip_gate_rejects_corrupt_contours(synthetic_ct, tmp_path, corruption):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, spacing=[2.3, .7], gaps=True)
    image_path, prediction, mask = inputs(source, tmp_path)
    image = nib.load(prediction)
    rt = create_rtstruct(source, image)
    assert add_roi(rt, mask, "L", [1, 2, 3])
    roi = copy.deepcopy(rt.ds.ROIContourSequence[0])
    contour = roi.ContourSequence[0]
    points = np.asarray(contour.ContourData).reshape(-1, 3)
    if corruption == "spacing": points[:, 0] *= 2.3 / .7
    if corruption == "off-plane": points[:, 2] += .101  # 0.0505 voxel; above unchanged limit.
    if corruption == "reflection": points[:, 0] = 2 * points[:, 0].mean() - points[:, 0]
    if corruption == "wrong-uid":
        contour.ContourImageSequence[0].ReferencedSOPInstanceUID = rt.series_data[2].SOPInstanceUID
    if corruption == "missing-polygon": del roi.ContourSequence[0]
    contour.ContourData = points.ravel().tolist()
    with pytest.raises(ValueError):
        validate_roi_roundtrip(roi, image, mask, source_slices=rt.series_data)


def test_model_grid_mismatch_still_fails(synthetic_ct, tmp_path):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    image_path, prediction, mask = inputs(source, tmp_path)
    image = nib.load(prediction)
    affine = image.affine.copy()
    affine[0, 3] += 3
    with pytest.raises(ValueError, match="affine"):
        create_rtstruct(source, nib.Nifti1Image(mask.astype(np.uint8), affine))


def test_square_pixel_gaps_use_acquired_planes(synthetic_ct, tmp_path):
    source = tmp_path / "source"
    paths = make_series(synthetic_ct, source, gaps=True)
    image_path, prediction, mask = inputs(source, tmp_path)
    result, output = convert(source, tmp_path, image_path, prediction)
    assert result.returncode == 0, result.stdout + result.stderr
    positions = {str(ds.SOPInstanceUID): float(ds.ImagePositionPatient[2])
                 for ds in (pydicom.dcmread(path) for path in paths)}
    for contour in pydicom.dcmread(output).ROIContourSequence[0].ContourSequence:
        uid = str(contour.ContourImageSequence[0].ReferencedSOPInstanceUID)
        assert np.allclose(np.asarray(contour.ContourData)[2::3], positions[uid], atol=1e-6)
    assert_viewer_exact(source, output, mask.transpose(1, 0, 2))


def test_crop_preserves_corrected_non_square_gapped_geometry(synthetic_ct, tmp_path):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, iop=[-1, 0, 0, 0, -1, 0], spacing=[2.3, .7], gaps=True)
    image_path, prediction, mask = inputs(source, tmp_path)
    result, output = convert(source, tmp_path, image_path, prediction, labels={"1": "PTV_test"})
    assert result.returncode == 0, result.stdout + result.stderr
    cropped = tmp_path / "cropped"
    report = crop_ct_rtstruct(source.iterdir(), [output], cropped, margin_mm=1)
    assert report["slices_kept"] == 4
    after = cropped / "model_RTSTRUCT_cropped.dcm"
    before_contours = pydicom.dcmread(output).ROIContourSequence[0].ContourSequence
    after_contours = pydicom.dcmread(after).ROIContourSequence[0].ContourSequence
    assert [c.ContourData for c in before_contours] == [c.ContourData for c in after_contours]
    assert_viewer_exact(cropped / "CT", after, mask.transpose(1, 0, 2)[:, :, 1:5])
