#!/usr/bin/env python3
"""Convert independent NIfTI masks to an overlap-preserving RTSTRUCT."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import RTStructureSetStorage
from rtstruct_geometry import add_roi, slice_geometry, create_rtstruct, report_omissions, dicom_grid, unique_roi_names, validate_reference_identity, validate_roi_completeness, staged_rtstruct
from orientation_contract import exact_orientation, restore_image, write_export_record, sidecar_path
from rtstruct_roundtrip import CroppedMask, has_coordinates, data_box, read_rtstruct, source_plane_geometry, validate_roi_roundtrip, validate_rtstruct_extent
from task_progress import Progress


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
    Path(args.output).unlink(missing_ok=True)
    sidecar_path(args.output).unlink(missing_ok=True)

    model_input = nib.load(args.model_input)
    mappings = json.loads(Path(args.masks_json).read_text())
    if not isinstance(mappings, dict) or not mappings:
        raise SystemExit("masks JSON must be a non-empty ROI-name to path object")

    shape, affine, _ = dicom_grid(args.dicom, model_input)
    source_input, inverse = restore_image(model_input, shape, affine)
    inverse["masks"] = {}
    rtstruct = create_rtstruct(args.dicom, source_input)
    plane_geometry = [slice_geometry(ds) for ds in rtstruct.series_data]
    expected_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_sources = {}
    nonempty_names = set()
    progress = Progress()
    progress.emit("rtstruct_contours", 0, len(mappings))
    omitted = []
    for done, (name, filename) in enumerate(zip(unique_roi_names(mappings), mappings.values()), 1):
        mask_image = nib.load(filename)
        exact_orientation(mask_image.shape, mask_image.affine, model_input.shape, model_input.affine)
        mask_image, mask_inverse = restore_image(mask_image, shape, affine)
        inverse["masks"][str(name)] = mask_inverse
        mask_data = np.asanyarray(mask_image.dataobj)
        box = data_box(mask_data, threshold=args.threshold)
        offset = tuple(s.start for s in box)
        mask = mask_data[box] > args.threshold
        del mask_data
        if not np.any(mask):
            omitted.append({"name": str(name), "voxels": 0, "reason": "empty input mask"})
            del mask, mask_image
            continue
        bounds = []
        nonempty_names.add(str(name))
        if not add_roi(rtstruct, mask, str(name), deterministic_color(str(name)), offset=offset, bounds_out=bounds,
                       plane_geometry=plane_geometry):
            omitted.append({"name": str(name), "voxels": int(mask.sum()),
                            "reason": "no contour returned for nonempty mask"})
            del mask, mask_image
            continue
        expected_bounds[str(name)] = bounds
        expected_sources[str(name)] = (filename, box)
        del mask, mask_image
        progress.emit("rtstruct_contours", done, len(mappings))

    report_omissions(args.output, omitted)
    if not expected_bounds:
        raise SystemExit("no contourable structures: no RTSTRUCT created" if any(roi["voxels"] for roi in omitted) else
                         "all supplied masks are empty: no RTSTRUCT created")
    progress.emit("rtstruct_writing")
    try:
        with staged_rtstruct(args.output) as partial:
            rtstruct.save(str(partial))
            progress.emit("rtstruct_validation", 0, len(expected_bounds))

            ds = read_rtstruct(partial)
            if ds.SOPClassUID != RTStructureSetStorage or ds.Modality != "RTSTRUCT":
                raise SystemExit("output is not DICOM RT Structure Set Storage")
            rois = getattr(ds, "StructureSetROISequence", [])
            contours = getattr(ds, "ROIContourSequence", [])
            if len(rois) != len(expected_bounds) or len(contours) != len(expected_bounds):
                raise SystemExit("RTSTRUCT ROI count does not match non-empty input masks")
            validate_reference_identity(ds, rtstruct.series_data)
            validate_roi_completeness(rois, contours, nonempty_names)

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
            validate_rtstruct_extent(ds, rtstruct.series_data)

            roi_names = {int(roi.ROINumber): str(roi.ROIName) for roi in rois}
            spacing = np.asarray(model_input.header.get_zooms()[:3], dtype=float)
            tolerance = max(2.1, float(spacing.max()) * 2.1)
            source_geometry = source_plane_geometry(rtstruct.series_data)
            max_delta = 0.0
            roundtrip = {}
            for done, roi in enumerate(contours, 1):
                name = roi_names[int(roi.ReferencedROINumber)]
                pieces = [np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
                          for c in getattr(roi, "ContourSequence", [])
                          if has_coordinates(getattr(c, "ContourData", None))]
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
                filename, box = expected_sources[name]
                image, _ = restore_image(nib.load(filename), shape, affine)
                # One independent mask is loaded at a time; retain only its voxel box.
                mask = CroppedMask(np.asanyarray(image.dataobj)[box] > args.threshold,
                                   tuple(s.start for s in box), tuple(shape))
                roundtrip[name] = validate_roi_roundtrip(roi, image, mask, source_geometry=source_geometry)
                del mask, image
                progress.emit("rtstruct_validation", done, len(contours))

            progress.emit("export")
            write_export_record(args.output, args.model_input, inverse)
            summary = json.dumps({"valid": True, "rois": len(rois),
                              "referenced_images": len(refs),
                              "geometry_bbox_max_delta_mm": round(max_delta, 4),
                              "geometry_tolerance_mm": round(tolerance, 4),
                              "physical_roundtrip": roundtrip,
                              "omitted_rois": omitted, "bytes": partial.stat().st_size})
            os.replace(partial, args.output)
            print(summary)
    except (Exception, SystemExit):
        Path(args.output).unlink(missing_ok=True)
        sidecar_path(args.output).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
