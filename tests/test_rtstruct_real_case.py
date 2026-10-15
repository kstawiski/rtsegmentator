"""Optional public-CT regression; external data are never copied into the repo.

Set RTSTRUCT_REAL_CASE_DIR to a directory with prediction.nii.gz and labels.json,
and RTSTRUCT_REAL_DICOM to its source DICOM directory. The default suite uses the
equivalent generated edge-component fixtures in test_rtstruct_degenerate.py.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pydicom
import pytest
from rt_utils import RTStructBuilder

from test_rtstruct_geometry import ROOT, contour_equivalence, convert
from test_viewer import rasterize
import viewer


def test_public_ct_all_labels_roundtrip_and_equivalence(tmp_path):
    case_arg, dicom_arg = os.getenv("RTSTRUCT_REAL_CASE_DIR"), os.getenv("RTSTRUCT_REAL_DICOM")
    if not case_arg or not dicom_arg:
        pytest.skip("Set RTSTRUCT_REAL_CASE_DIR and RTSTRUCT_REAL_DICOM for the external public CT")
    case, source = Path(case_arg), Path(dicom_arg)
    prediction = case / "prediction.nii.gz"
    labels = json.loads((case / "labels.json").read_text())
    assert len(labels) == 117
    image_path = tmp_path / "input.nii.gz"
    read = subprocess.run([sys.executable, str(ROOT / "dicom_series_to_nifti.py"),
                           str(source), str(image_path)], capture_output=True, text=True)
    assert read.returncode == 0, read.stdout + read.stderr
    image = nib.load(prediction)
    data = np.asanyarray(image.dataobj)
    result, output = convert(source, tmp_path, image_path, prediction, labels=labels)
    print("REAL_CONVERTER " + result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["omitted_rois"] == []
    new = pydicom.dcmread(output)
    new_names = {int(roi.ROINumber): str(roi.ROIName) for roi in new.StructureSetROISequence}
    new_by_name = {new_names[int(roi.ReferencedROINumber)]: roi for roi in new.ROIContourSequence}
    series = viewer.index_series(sorted(source.iterdir()))
    structure, mapped = viewer.map_structure(viewer.read_structure(output), output, tmp_path, series)
    keys = {roi["key"]: roi["name"] for roi in structure["rois"]}
    paths_by_name = {}
    for k, rois in mapped.items():
        for roi in rois:
            paths_by_name.setdefault(keys[roi["key"]], {})[k] = roi["paths"]
    old = RTStructBuilder.create_new(dicom_series_path=str(source))
    rows = []
    for value, name in labels.items():
        mask = data == int(value)
        count = int(mask.sum())
        row = {"label": int(value), "roi": name, "source_voxels": count}
        # Include empty definitions: old rt-utils creates an empty contour set;
        # production correctly writes no ROI when the source has no voxels.
        old.add_roi(mask=mask.transpose(1, 0, 2), name=name)
        before = old.ds.ROIContourSequence[-1].ContourSequence
        if not count:
            row.update(status="PASS_EMPTY", contours=0, points=0,
                       max_point_delta_mm=0.0, viewer_voxels=0,
                       differing_voxels=0, viewer_dice=1.0)
            assert not before and name not in new_by_name and name not in paths_by_name
        else:
            after = new_by_name[name].ContourSequence
            maximum, points = contour_equivalence(before, after)
            reconstructed = np.zeros(mask.shape, dtype=bool)
            for k, paths in paths_by_name.get(name, {}).items():
                reconstructed[:, :, k] = rasterize(paths, mask.shape[:2][::-1]).T
            difference = int(np.count_nonzero(mask != reconstructed))
            rebuilt = int(reconstructed.sum())
            intersection = int(np.count_nonzero(mask & reconstructed))
            dice = 2 * intersection / (count + rebuilt)
            row.update(status="PASS" if difference == 0 and maximum <= 1e-3 else "FAIL",
                       contours=len(after), points=points, max_point_delta_mm=maximum,
                       viewer_voxels=rebuilt, differing_voxels=difference, viewer_dice=dice,
                       gate_dice=report["physical_roundtrip"][name]["roundtrip_dice"])
        rows.append(row)
        print("REAL_ROI " + json.dumps(row), flush=True)
    summary = {"labels": len(rows), "nonempty": sum(r["source_voxels"] > 0 for r in rows),
               "empty": sum(r["source_voxels"] == 0 for r in rows),
               "failures": sum(r["status"] == "FAIL" for r in rows),
               "contours": sum(r["contours"] for r in rows),
               "points": sum(r["points"] for r in rows),
               "max_point_delta_mm": max(r["max_point_delta_mm"] for r in rows),
               "differing_voxels": sum(r["differing_voxels"] for r in rows)}
    print("REAL_SUMMARY " + json.dumps(summary), flush=True)
    (tmp_path / "real-results.json").write_text(json.dumps({"summary": summary, "rois": rows,
                                                           "converter": report}, indent=2))
    assert summary["nonempty"] == report["rois"] == len(new_by_name)
    assert summary["failures"] == 0
