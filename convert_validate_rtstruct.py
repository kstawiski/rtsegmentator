#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import RTStructureSetStorage
from totalsegmentator.dicom_io import save_mask_as_rtstruct

def main():
    p=argparse.ArgumentParser(); p.add_argument("--prediction",required=True); p.add_argument("--model-input",required=True)
    p.add_argument("--dicom",required=True); p.add_argument("--output",required=True); p.add_argument("--labels",required=True)
    a=p.parse_args(); pred=nib.load(a.prediction); inp=nib.load(a.model_input)
    if pred.shape != inp.shape: raise SystemExit(f"shape mismatch: prediction {pred.shape}, input {inp.shape}")
    if not np.allclose(pred.affine,inp.affine,atol=1e-3): raise SystemExit("prediction affine does not match model input")
    data=np.rint(np.asanyarray(pred.dataobj)).astype(np.int16)
    # TotalSegmentator's RTSTRUCT writer expects radiological LAS voxel axes.
    # SimpleITK writes our geometry-faithful DICOM conversion as LPS, so giving
    # its raw array directly to that writer reflects contours in the AP axis.
    # Reorient the label array (nearest-neighbour by construction) without
    # resampling before applying the writer's DICOM plane transform.
    current=nib.orientations.io_orientation(pred.affine)
    expected=nib.orientations.axcodes2ornt(("L","A","S"))
    data=nib.orientations.apply_orientation(data,nib.orientations.ornt_transform(current,expected))
    raw=json.loads(Path(a.labels).read_text())
    labels={int(k):str(v) for k,v in raw.items() if int(k)>0}; present=set(np.unique(data))-set([0])
    unknown=present-set(labels)
    if unknown: raise SystemExit(f"unmapped prediction labels: {sorted(unknown)}")
    if not present: raise SystemExit("empty prediction: no RTSTRUCT created")
    save_mask_as_rtstruct(data,labels,a.dicom,a.output)
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
    print(json.dumps({"valid":True,"rois":len(rois),"contour_sets":len(contours),"referenced_images":len(refs),"bytes":Path(a.output).stat().st_size}))
if __name__=="__main__": main()
