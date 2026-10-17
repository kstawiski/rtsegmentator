#!/usr/bin/env python3
import argparse, hashlib, json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import RTStructureSetStorage
from rtstruct_geometry import add_roi, create_rtstruct, mask_bounds, report_omissions, dicom_grid, unique_roi_names
from orientation_contract import exact_orientation, restore_image, write_export_record
from rtstruct_roundtrip import CroppedMask, data_box, validate_roi_roundtrip
from task_progress import Progress


def deterministic_color(name: str) -> list[int]:
    """Return a stable, visible RGB color without relying on global RNG state."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return [64 + value % 192 for value in digest[:3]]

def main():
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
    expected_boxes = {}
    progress = Progress()
    progress.emit("rtstruct_contours", 0, len(present))
    done = 0
    omitted = []
    for class_idx, class_name in labels.items():
        if class_idx not in present:
            continue
        box = data_box(data, label=class_idx)
        offset = tuple(s.start for s in box)
        binary = data[box] == class_idx
        if not add_roi(rtstruct, binary, class_name, deterministic_color(class_name), offset=offset):
            omitted.append({"name": class_name, "voxels": int(binary.sum()),
                            "reason": "no contour returned for nonempty mask"})
            del binary
            continue
        expected_bounds[class_name] = mask_bounds(binary, rtstruct.series_data, offset=offset)
        expected_boxes[class_name] = (class_idx, box)
        del binary
        done += 1
        progress.emit("rtstruct_contours", done, len(present))
    report_omissions(a.output, omitted)
    if not expected_boxes:
        raise SystemExit("no contourable structures: no RTSTRUCT created")
    progress.emit("rtstruct_writing")
    rtstruct.save(str(a.output))
    progress.emit("rtstruct_validation", 0, len(expected_boxes))
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
    for done, roi in enumerate(contours, 1):
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
            class_idx, box = expected_boxes[name]
            expected = CroppedMask(data[box] == class_idx, tuple(s.start for s in box), data.shape)
            roundtrip[name]=validate_roi_roundtrip(roi,pred,expected,
                                                  source_slices=rtstruct.series_data)
        except ValueError as exc:
            raise SystemExit(f"RTSTRUCT physical round-trip mismatch for {name}: {exc}") from exc
        del expected
        progress.emit("rtstruct_validation", done, len(contours))
    progress.emit("export")
    write_export_record(a.output, a.model_input, inverse)
    print(json.dumps({"valid":True,"rois":len(rois),"contour_sets":len(contours),
        "referenced_images":len(refs),"geometry_bbox_max_delta_mm":round(max_delta,4),
        "geometry_tolerance_mm":round(tolerance,4),"physical_roundtrip":roundtrip,
        "omitted_rois":omitted,"bytes":Path(a.output).stat().st_size}))
if __name__=="__main__": main()
