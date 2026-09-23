#!/usr/bin/env python3
"""Record strict post-fix RTSTRUCT geometry and identity evidence."""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pydicom

p=argparse.ArgumentParser()
p.add_argument("--task",required=True); p.add_argument("--dicom",required=True)
p.add_argument("--rtstruct",required=True); p.add_argument("--output",required=True)
a=p.parse_args(); source=Path(a.dicom); result=Path(a.rtstruct)

images={}; positions=[]
for f in source.iterdir():
    try:
        ds=pydicom.dcmread(f,stop_before_pixels=True)
        uid=str(ds.SOPInstanceUID); images[uid]=f
        if hasattr(ds,"ImagePositionPatient"): positions.append([float(x) for x in ds.ImagePositionPatient])
    except Exception: pass
if not images or not positions: raise SystemExit("source DICOM geometry is unavailable")
ds=pydicom.dcmread(result)
expected=f"RTSEG: {a.task}"[:64]
if str(getattr(ds,"StructureSetLabel","")) != a.task[:16]: raise SystemExit("wrong StructureSetLabel")
if str(getattr(ds,"StructureSetName","")) != expected: raise SystemExit("wrong StructureSetName")
if str(getattr(ds,"SeriesDescription","")) != expected: raise SystemExit("wrong SeriesDescription")

refs=set(); points=[]
for roi in getattr(ds,"ROIContourSequence",[]):
    for contour in getattr(roi,"ContourSequence",[]):
        values=np.asarray(contour.ContourData,dtype=float).reshape(-1,3); points.append(values)
        for image in getattr(contour,"ContourImageSequence",[]): refs.add(str(image.ReferencedSOPInstanceUID))
if not points or not refs or not refs.issubset(images): raise SystemExit("empty or foreign contour references")
points=np.concatenate(points); origins=np.asarray(positions)
# A generous patient-space envelope catches axis/sign catastrophes without
# assuming anatomy or requiring contours to touch the DICOM volume boundary.
lo=origins.min(0)-1000; hi=origins.max(0)+1000
if np.any(points.min(0)<lo) or np.any(points.max(0)>hi): raise SystemExit("contours lie outside source patient-space envelope")
payload={"task":a.task,"result":"passed","orientation_fix":"dynamic_nifti_to_LAS",
         "structure_set_label":str(ds.StructureSetLabel),"structure_set_name":str(ds.StructureSetName),
         "series_description":str(ds.SeriesDescription),"rois":len(getattr(ds,"StructureSetROISequence",[])),
         "referenced_images":len(refs),"contour_min_lps":points.min(0).tolist(),
         "contour_max_lps":points.max(0).tolist(),"source_origin_min_lps":origins.min(0).tolist(),
         "source_origin_max_lps":origins.max(0).tolist(),"artifact_bytes":result.stat().st_size,
         "artifact_sha256":hashlib.sha256(result.read_bytes()).hexdigest()}
target=Path(a.output); target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload))
