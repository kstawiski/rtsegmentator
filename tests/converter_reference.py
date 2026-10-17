"""Frozen converter mains from 3600f83 for exact JSON/byte comparison."""
import argparse, hashlib, json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import RTStructureSetStorage
from rtstruct_geometry import create_rtstruct, report_omissions, dicom_grid, unique_roi_names
from orientation_contract import exact_orientation, restore_image, write_export_record
from writer_reference import add_roi, mask_bounds
from roundtrip_reference import full_volume_reference as validate_roi_roundtrip

def deterministic_color(name: str) -> list[int]:
    """Return a stable, visible RGB color without relying on global RNG state."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return [64 + value % 192 for value in digest[:3]]

def labels_main():
    p=argparse.ArgumentParser(); p.add_argument("--prediction",required=True); p.add_argument("--model-input",required=True)
    p.add_argument("--dicom",required=True); p.add_argument("--output",required=True); p.add_argument("--labels",required=True)
    a=p.parse_args(); pred=nib.load(a.prediction); inp=nib.load(a.model_input)
    exact_orientation(pred.shape, pred.affine, inp.shape, inp.affine)
    shape, affine, _ = dicom_grid(a.dicom, inp)
    inp, _ = restore_image(inp, shape, affine)
    pred, inverse = restore_image(pred, shape, affine)
    data=np.rint(np.asanyarray(pred.dataobj)).astype(np.int16)
    raw=json.loads(Path(a.labels).read_text())
    items=[(int(k),v) for k,v in raw.items() if int(k)>0]
    labels=dict(zip((k for k,_ in items),unique_roi_names(v for _,v in items))); present=set(np.unique(data))-set([0])
    unknown=present-set(labels)
    if unknown: raise SystemExit(f"unmapped prediction labels: {sorted(unknown)}")
    if not present: raise SystemExit("empty prediction: no RTSTRUCT created")
    # Restore [DICOM column, DICOM row, acquired slice] array indices. The
    # shared writer transposes each plane for OpenCV without reversing rows.
    rtstruct = create_rtstruct(a.dicom, inp)
    expected_bounds = {}
    expected_masks = {}
    omitted = []
    for class_idx, class_name in labels.items():
        binary = data == class_idx
        if not np.any(binary):
            continue
        if not add_roi(rtstruct, binary, class_name, deterministic_color(class_name)):
            omitted.append({"name": class_name, "voxels": int(binary.sum()),
                            "reason": "no contour returned for nonempty mask"})
            continue
        expected_bounds[class_name] = mask_bounds(binary, rtstruct.series_data)
        expected_masks[class_name] = binary
    report_omissions(a.output, omitted)
    if not expected_masks:
        raise SystemExit("no contourable structures: no RTSTRUCT created")
    rtstruct.save(str(a.output))
    ds=pydicom.dcmread(a.output,stop_before_pixels=True)
    if ds.SOPClassUID != RTStructureSetStorage or ds.Modality != "RTSTRUCT": raise SystemExit("output is not DICOM RT Structure Set Storage")
    rois=getattr(ds,"StructureSetROISequence",[]); contours=getattr(ds,"ROIContourSequence",[])
    if not rois or not contours: raise SystemExit("RTSTRUCT contains no ROIs/contours")
    source=set()
    for f in Path(a.dicom).iterdir():
        try: source.add(str(pydicom.dcmread(f,stop_before_pixels=True,specific_tags=["SOPInstanceUID"]).SOPInstanceUID))
        except Exception: pass
    refs=set()
    for roi in contours:
        for c in getattr(roi,"ContourSequence",[]):
            for im in getattr(c,"ContourImageSequence",[]): refs.add(str(im.ReferencedSOPInstanceUID))
    if not refs or not refs.issubset(source): raise SystemExit("RTSTRUCT references missing or foreign source images")
    roi_names={int(roi.ROINumber):str(roi.ROIName) for roi in rois}
    # Compare each contour's physical LPS extent with the source-grid label
    # extent. This catches LR/AP/SI reflection and permutation errors that can
    # remain inside a broad patient-space envelope.
    spacing=np.asarray(inp.header.get_zooms()[:3],dtype=float)
    tolerance=max(2.1,float(spacing.max())*2.1)
    max_delta=0.0; roundtrip={}
    for roi in contours:
        name=roi_names[int(roi.ReferencedROINumber)]
        pieces=[np.asarray(c.ContourData,dtype=float).reshape(-1,3)
                for c in getattr(roi,"ContourSequence",[]) if getattr(c,"ContourData",None)]
        if not pieces or name not in expected_bounds: raise SystemExit(f"missing contour geometry for {name}")
        points=np.concatenate(pieces); expected_min,expected_max=expected_bounds[name]
        delta=float(np.max(np.maximum(np.abs(points.min(0)-expected_min),
                                      np.abs(points.max(0)-expected_max))))
        max_delta=max(max_delta,delta)
        if delta>tolerance:
            raise SystemExit(f"RTSTRUCT geometry mismatch for {name}: {delta:.3f} mm > {tolerance:.3f} mm")
        try:
            roundtrip[name]=validate_roi_roundtrip(roi,pred,expected_masks[name],
                                                  source_slices=rtstruct.series_data)
        except ValueError as exc:
            raise SystemExit(f"RTSTRUCT physical round-trip mismatch for {name}: {exc}") from exc
    write_export_record(a.output, a.model_input, inverse)
    print(json.dumps({"valid":True,"rois":len(rois),"contour_sets":len(contours),
        "referenced_images":len(refs),"geometry_bbox_max_delta_mm":round(max_delta,4),
        "geometry_tolerance_mm":round(tolerance,4),"physical_roundtrip":roundtrip,
        "omitted_rois":omitted,"bytes":Path(a.output).stat().st_size}))

def masks_main() -> None:
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

    shape, affine, _ = dicom_grid(args.dicom, model_input)
    source_input, inverse = restore_image(model_input, shape, affine)
    inverse["masks"] = {}
    rtstruct = create_rtstruct(args.dicom, source_input)
    expected_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_masks: dict[str, tuple[nib.Nifti1Image, np.ndarray]] = {}
    omitted = []
    for name, filename in zip(unique_roi_names(mappings), mappings.values()):
        mask_image = nib.load(filename)
        exact_orientation(mask_image.shape, mask_image.affine, model_input.shape, model_input.affine)
        mask_image, mask_inverse = restore_image(mask_image, shape, affine)
        inverse["masks"][str(name)] = mask_inverse
        mask = np.asanyarray(mask_image.dataobj) > args.threshold
        if not np.any(mask):
            continue
        if not add_roi(rtstruct, mask, str(name), deterministic_color(str(name))):
            omitted.append({"name": str(name), "voxels": int(mask.sum()),
                            "reason": "no contour returned for nonempty mask"})
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

    write_export_record(args.output, args.model_input, inverse)
    print(json.dumps({"valid": True, "rois": len(rois),
                      "referenced_images": len(refs),
                      "geometry_bbox_max_delta_mm": round(max_delta, 4),
                      "geometry_tolerance_mm": round(tolerance, 4),
                      "physical_roundtrip": roundtrip,
                      "omitted_rois": omitted, "bytes": Path(args.output).stat().st_size}))
