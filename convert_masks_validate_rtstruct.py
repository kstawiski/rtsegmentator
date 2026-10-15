#!/usr/bin/env python3
"""Convert independent NIfTI masks to an overlap-preserving RTSTRUCT."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import RTStructureSetStorage
from rtstruct_geometry import add_roi, create_rtstruct, mask_bounds, report_omissions
from rtstruct_roundtrip import validate_roi_roundtrip


def deterministic_color(name: str) -> list[int]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return [64 + value % 192 for value in digest[:3]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-json", required=True,
                        help="JSON object mapping ROI names to NIfTI mask paths")
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--dicom", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Include mask voxels strictly above this value")
    args = parser.parse_args()

    model_input = nib.load(args.model_input)
    mappings = json.loads(Path(args.masks_json).read_text())
    if not isinstance(mappings, dict) or not mappings:
        raise SystemExit("masks JSON must be a non-empty ROI-name to path object")

    rtstruct = create_rtstruct(args.dicom, model_input)
    expected_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_masks: dict[str, tuple[nib.Nifti1Image, np.ndarray]] = {}
    omitted = []
    for name, filename in mappings.items():
        mask_image = nib.load(filename)
        if mask_image.shape != model_input.shape:
            raise SystemExit(f"shape mismatch for {name}: {mask_image.shape} != {model_input.shape}")
        if not np.allclose(mask_image.affine, model_input.affine, atol=1e-3):
            raise SystemExit(f"affine mismatch for {name}")
        mask = np.asanyarray(mask_image.dataobj) > args.threshold
        if not np.any(mask):
            continue
        if not add_roi(rtstruct, mask, str(name), deterministic_color(str(name))):
            omitted.append({"name": str(name), "voxels": int(mask.sum()),
                            "reason": "no non-degenerate contour polygon"})
            continue
        expected_bounds[str(name)] = mask_bounds(mask, rtstruct.series_data)
        expected_masks[str(name)] = (mask_image, mask)

    report_omissions(args.output, omitted)
    if not expected_bounds:
        raise SystemExit("no contourable structures: no RTSTRUCT created" if omitted else
                         "all supplied masks are empty: no RTSTRUCT created")
    rtstruct.save(args.output)

    ds = pydicom.dcmread(args.output, stop_before_pixels=True)
    if ds.SOPClassUID != RTStructureSetStorage or ds.Modality != "RTSTRUCT":
        raise SystemExit("output is not DICOM RT Structure Set Storage")
    rois = getattr(ds, "StructureSetROISequence", [])
    contours = getattr(ds, "ROIContourSequence", [])
    if len(rois) != len(expected_bounds) or len(contours) != len(expected_bounds):
        raise SystemExit("RTSTRUCT ROI count does not match non-empty input masks")

    source = set()
    for filename in Path(args.dicom).iterdir():
        try:
            source.add(str(pydicom.dcmread(filename, stop_before_pixels=True,
                                           specific_tags=["SOPInstanceUID"]).SOPInstanceUID))
        except Exception:
            pass
    refs = set()
    for roi in contours:
        for contour in getattr(roi, "ContourSequence", []):
            for image in getattr(contour, "ContourImageSequence", []):
                refs.add(str(image.ReferencedSOPInstanceUID))
    if not refs or not refs.issubset(source):
        raise SystemExit("RTSTRUCT references missing or foreign source images")

    roi_names = {int(roi.ROINumber): str(roi.ROIName) for roi in rois}
    spacing = np.asarray(model_input.header.get_zooms()[:3], dtype=float)
    tolerance = max(2.1, float(spacing.max()) * 2.1)
    max_delta = 0.0
    roundtrip = {}
    for roi in contours:
        name = roi_names[int(roi.ReferencedROINumber)]
        pieces = [np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
                  for c in getattr(roi, "ContourSequence", [])
                  if getattr(c, "ContourData", None)]
        if not pieces or name not in expected_bounds:
            raise SystemExit(f"missing contour geometry for {name}")
        points = np.concatenate(pieces)
        expected_min, expected_max = expected_bounds[name]
        delta = float(np.max(np.maximum(np.abs(points.min(0) - expected_min),
                                        np.abs(points.max(0) - expected_max))))
        max_delta = max(max_delta, delta)
        if delta > tolerance:
            raise SystemExit(f"RTSTRUCT geometry mismatch for {name}: "
                             f"{delta:.3f} mm > {tolerance:.3f} mm")
        image, mask = expected_masks[name]
        roundtrip[name] = validate_roi_roundtrip(roi, image, mask, source_slices=rtstruct.series_data)

    print(json.dumps({"valid": True, "rois": len(rois),
                      "referenced_images": len(refs),
                      "geometry_bbox_max_delta_mm": round(max_delta, 4),
                      "geometry_tolerance_mm": round(tolerance, 4),
                      "physical_roundtrip": roundtrip,
                      "omitted_rois": omitted, "bytes": Path(args.output).stat().st_size}))


if __name__ == "__main__":
    main()
