#!/usr/bin/env python3
"""Revalidate retained pre-fix artifacts after reversing their legacy row reflection."""
import json, os, shutil, subprocess
from pathlib import Path

import numpy as np
import pydicom

app=Path(os.environ.get("RTSEG_APP_ROOT","/app"))
root=Path(os.environ.get("RTSEG_MODEL_ROOT","/models"))
tasks=set(json.loads((app/"catalog/orientation-audit-tasks.json").read_text()))
markers=root/"orientation-validation"; markers.mkdir(exist_ok=True)
destination=root/"results/orientation-audit-20260812/revalidated"; destination.mkdir(parents=True,exist_ok=True)

print("Indexing fixture SOP Instance UIDs...",flush=True); uid_files={}
for f in (root/"testdata").rglob("*"):
    if not f.is_file(): continue
    try:
        d=pydicom.dcmread(f,stop_before_pixels=True,specific_tags=["SOPInstanceUID"])
        if hasattr(d,"SOPInstanceUID"): uid_files.setdefault(str(d.SOPInstanceUID),[]).append(f)
    except Exception: pass
print(f"Indexed {len(uid_files)} images",flush=True)

artifacts={}
aliases={
 "UWLAIR-preRT-RTSTRUCT.dcm":"uwlair_hntsmrg_pre_t2",
 "LN-Seg-FM-original-RTSTRUCT.dcm":"lnsegfm_nnunet",
 "LN-Seg-FM-resencm-RTSTRUCT.dcm":"lnsegfm_resenc_m",
 "LN-Seg-FM-resencl-RTSTRUCT.dcm":"lnsegfm_resenc_l",
 "LN-Seg-FM-swin-RTSTRUCT.dcm":"lnsegfm_swinunetr",
 "ISRT-PET1-early-deform-RTSTRUCT.dcm":"isrt_pet1_early_deform",
 "ISRT-PET1-early-rigid-RTSTRUCT.dcm":"isrt_pet1_early_rigid",
 "ISRT-PET1-late-deform-RTSTRUCT.dcm":"isrt_pet1_late_deform",
 "ISRT-PET1-late-rigid-RTSTRUCT.dcm":"isrt_pet1_late_rigid",
 "ensemble-RTSTRUCT.dcm":None,
 "RTSTRUCT.dcm":None,
}
for f in (root/"results").rglob("*RTSTRUCT.dcm"):
    if destination in f.parents: continue
    task=aliases.get(f.name,f.name.removesuffix("_RTSTRUCT.dcm"))
    if f.name=="ensemble-RTSTRUCT.dcm" and "isrt-ct" in f.parts:
        task={"resunet":"isrt_ct_resunet","segresnet":"isrt_ct_segresnet","swinunetr":"isrt_ct_swinunetr"}.get(f.parent.name)
    if f.name=="RTSTRUCT.dcm":
        task="raidionics_mediastinal_lymphnodes" if "raidionics-miednice" in f.parts else "pediatric_thoracic_lymphoma" if "lymphoma" in f.parts else None
    if task in tasks and (task not in artifacts or f.stat().st_mtime>artifacts[task].stat().st_mtime): artifacts[task]=f

for task, source_rt in sorted(artifacts.items()):
    marker=markers/f"{task}.json"
    if marker.exists(): continue
    try:
        ds=pydicom.dcmread(source_rt)
        referenced=[]
        for roi in getattr(ds,"ROIContourSequence",[]):
            for contour in getattr(roi,"ContourSequence",[]):
                seq=getattr(contour,"ContourImageSequence",[])
                if seq: referenced.append(str(seq[0].ReferencedSOPInstanceUID))
        if not referenced or any(x not in uid_files for x in referenced): raise RuntimeError("not all references map to retained fixtures")
        candidate_dirs={f.parent for f in uid_files[referenced[0]]}
        fixture_dir=next((d for d in candidate_dirs if all(any(f.parent==d for f in uid_files[x]) for x in referenced)),None)
        if fixture_dir is None: raise RuntimeError("no single retained fixture contains all references")
        fixture_map={x:next(f for f in uid_files[x] if f.parent==fixture_dir) for x in referenced}
        # Legacy TotalSegmentator conversion reflected the DICOM row coordinate.
        for roi in getattr(ds,"ROIContourSequence",[]):
            for contour in getattr(roi,"ContourSequence",[]):
                seq=getattr(contour,"ContourImageSequence",[])
                if not seq: continue
                image=pydicom.dcmread(fixture_map[str(seq[0].ReferencedSOPInstanceUID)],stop_before_pixels=True)
                ipp=np.asarray(image.ImagePositionPatient,float); iop=np.asarray(image.ImageOrientationPatient,float)
                row_direction=iop[3:6]; row_spacing=float(image.PixelSpacing[0]); rows=int(image.Rows)
                points=np.asarray(contour.ContourData,float).reshape(-1,3)
                row_coordinate=((points-ipp)@row_direction)/row_spacing
                points += (((rows-1)-2*row_coordinate)*row_spacing)[:,None]*row_direction
                contour.ContourData=points.reshape(-1).tolist()
        display=f"RTSEG: {task}"[:64]
        ds.StructureSetLabel=task[:16]; ds.StructureSetName=display
        ds.StructureSetDescription=f"RTsegmentator model {task}"[:64]; ds.SeriesDescription=display
        output=destination/f"{task}_RTSTRUCT.dcm"; ds.save_as(output,enforce_file_format=True)
        subprocess.run([str(app/".venv/bin/python"),str(app/"audit_rtstruct_orientation.py"),
                        "--task",task,"--dicom",str(fixture_dir),"--rtstruct",str(output),
                        "--output",str(marker)],check=True,capture_output=True,text=True)
        data=json.loads(marker.read_text()); data.update(audit_method="legacy_artifact_row_reflection_reversed",
            legacy_artifact=str(source_rt),fixture=str(fixture_dir)); marker.write_text(json.dumps(data,indent=2)+"\n")
        print(json.dumps({"task":task,"status":"passed","fixture":str(fixture_dir)}),flush=True)
    except Exception as exc:
        print(json.dumps({"task":task,"status":"failed","error":repr(exc)}),flush=True)

missing=sorted(tasks-{p.stem for p in markers.glob("*.json")})
print(json.dumps({"passed":len(tasks)-len(missing),"missing":missing}),flush=True)
