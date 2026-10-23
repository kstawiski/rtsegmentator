"""Synthetic CPU adaptation of the supplied crop deployment acceptance checks."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from rt_utils import RTStructBuilder

from crop_prostate_sbrt import NoSelectedROIError, crop_ct_rtstruct


def make_struct(ct, dest, selected=True):
    rt = RTStructBuilder.create_new(dicom_series_path=str(ct))
    mask = np.zeros((32, 32, 12), dtype=bool)
    mask[3:9, 18:25, 4:7] = True
    rt.add_roi(mask=mask, name="PTV_test" if selected else "Bowel - PTV")
    body = np.zeros_like(mask)
    body[1:30, 1:30, :] = True
    rt.add_roi(mask=body, name="BODY")
    rt.save(str(dest))
    ds = pydicom.dcmread(dest)
    ds.FrameOfReferenceUID = pydicom.dcmread(next(ct.iterdir())).FrameOfReferenceUID
    ds.save_as(dest, enforce_file_format=True)


@pytest.mark.parametrize("keep_uids", [False, True])
def test_crop_preserves_pixels_and_remaps_all_references(synthetic_ct, tmp_path, keep_uids):
    rt_path = tmp_path / "struct.dcm"
    make_struct(synthetic_ct, rt_path)
    files = sorted(synthetic_ct.iterdir())
    before = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in [*files, rt_path]}
    out = tmp_path / "cropped"
    report = crop_ct_rtstruct(files, [rt_path], out, margin_mm=2, keep_uids=keep_uids)
    assert report["slices_kept"] == 5
    assert report["kept_z_range"] == [6.0, 14.0]
    assert report["selected_roi_names"] == ["PTV_test"]
    originals = {float(pydicom.dcmread(f).ImagePositionPatient[2]): pydicom.dcmread(f) for f in files}
    cropped = [pydicom.dcmread(f) for f in (out / "CT").glob("*.dcm")]
    uids = {str(ds.SOPInstanceUID) for ds in cropped}
    for ds in cropped:
        original = originals[float(ds.ImagePositionPatient[2])]
        assert ds.PixelData == original.PixelData
        assert ds.SpecificCharacterSet == original.SpecificCharacterSet
        for key in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID", "SOPInstanceUID"):
            assert (ds.get(key) == original.get(key)) == keep_uids
    rt = pydicom.dcmread(out / "struct_cropped.dcm")
    assert rt.StudyInstanceUID == cropped[0].StudyInstanceUID
    assert rt.FrameOfReferenceUID == cropped[0].FrameOfReferenceUID
    assert all(roi.ReferencedFrameOfReferenceUID == cropped[0].FrameOfReferenceUID
               for roi in rt.StructureSetROISequence)
    refs = {str(i.ReferencedSOPInstanceUID) for roi in rt.ROIContourSequence
            for c in getattr(roi, "ContourSequence", []) for i in c.ContourImageSequence}
    assert refs and refs <= uids
    body = next(r for r, n in zip(rt.ROIContourSequence, rt.StructureSetROISequence) if n.ROIName == "BODY")
    assert len(body.ContourSequence) == 5
    assert all(hashlib.sha256(f.read_bytes()).hexdigest() == digest for f, digest in before.items())


def test_crop_rejects_subtraction_only_target(synthetic_ct, tmp_path):
    path = tmp_path / "struct.dcm"
    make_struct(synthetic_ct, path, selected=False)
    with pytest.raises(NoSelectedROIError):
        crop_ct_rtstruct(synthetic_ct.iterdir(), [path], tmp_path / "output")
    assert not (tmp_path / "output").exists()



def make_targets(ct, dest, slices):
    """RTSTRUCT whose ROI names map to the CT slice indices they cover (empty: no contours)."""
    rt = RTStructBuilder.create_new(dicom_series_path=str(ct))
    for name, covered in slices.items():
        mask = np.zeros((32, 32, 12), dtype=bool)
        for k in covered:
            mask[5:10, 5:10, k] = True
        rt.add_roi(mask=mask, name=name)
    rt.save(str(dest))
    ds = pydicom.dcmread(dest)
    ds.FrameOfReferenceUID = pydicom.dcmread(next(ct.iterdir())).FrameOfReferenceUID
    ds.save_as(dest, enforce_file_format=True)


def test_crop_range_covers_every_filled_target(synthetic_ct, tmp_path):
    first, second = tmp_path / "plan_a.dcm", tmp_path / "plan_b.dcm"
    make_targets(synthetic_ct, first, {"PTV1": [2, 3], "PTV 2": [8, 9], "Bowel - PTV": range(12), "PTV3": []})
    make_targets(synthetic_ct, second, {"ptv_4": [5], "CTV": [6]})
    report = crop_ct_rtstruct(sorted(synthetic_ct.iterdir()), [first, second], tmp_path / "out", margin_mm=0)
    assert report["selected_roi_names"] == ["CTV", "PTV 2", "PTV1", "ptv_4"]
    extents = {t["roi_name"]: (t["rtstruct"], t["zmin"], t["zmax"], t["contours"]) for t in report["selected_rois"]}
    assert extents == {"PTV1": ("plan_a_cropped.dcm", 4.0, 6.0, 2), "PTV 2": ("plan_a_cropped.dcm", 16.0, 18.0, 2),
                       "ptv_4": ("plan_b_cropped.dcm", 10.0, 10.0, 1), "CTV": ("plan_b_cropped.dcm", 12.0, 12.0, 1)}
    assert [e["roi_name"] for e in report["ignored_empty_rois"]] == ["PTV3"]
    assert report["kept_z_range"] == [4.0, 18.0] and report["slices_kept"] == 8
    assert {m["roi"] for m in report["validation"]["masks_checked"]} == set(extents)
    text = (tmp_path / "out" / "crop_report.txt").read_text()
    assert "plan_a_cropped.dcm / 2 / PTV 2: z 16 to 18 mm, 2 contours" in text
    assert "Selected ROIs without contours (ignored):" in text and "PTV3" in text


def test_crop_reports_unfilled_targets(synthetic_ct, tmp_path):
    path = tmp_path / "struct.dcm"
    make_targets(synthetic_ct, path, {"PTV1": [], "PTV 2": [], "BODY": range(12)})
    with pytest.raises(NoSelectedROIError, match=r"without contours: PTV 2, PTV1"):
        crop_ct_rtstruct(sorted(synthetic_ct.iterdir()), [path], tmp_path / "output")
    assert not (tmp_path / "output").exists()

def test_direct_mapping_roundtrip_for_asymmetric_mask(synthetic_ct, tmp_path):
    data = np.zeros((32, 32, 12), dtype=np.int16)
    data[18:25, 3:9, 4:7] = 1
    prediction = tmp_path / "prediction.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([-1., -1., 2., 1.])), prediction)
    labels = tmp_path / "labels.json"
    labels.write_text('{"1":"asymmetric_target"}')
    result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "convert_validate_rtstruct.py"),
                             "--prediction", str(prediction), "--model-input", str(prediction),
                             "--dicom", str(synthetic_ct), "--output", str(tmp_path / "output.dcm"),
                             "--labels", str(labels)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["valid"]
    assert report["physical_roundtrip"]["asymmetric_target"]["roundtrip_dice"] == 1.0
    assert report["geometry_bbox_max_delta_mm"] == 0
