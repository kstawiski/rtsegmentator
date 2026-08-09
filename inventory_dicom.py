from collections import defaultdict
from pathlib import Path
import json, sys, pydicom
root=Path(sys.argv[1]); groups=defaultdict(lambda:{"count":0,"bytes":0,"path":""})
for f in root.rglob("*"):
    if not f.is_file(): continue
    try:
        d=pydicom.dcmread(f,stop_before_pixels=True,specific_tags=["Modality","StudyInstanceUID","SeriesInstanceUID"])
        m=str(getattr(d,"Modality","")); study=str(getattr(d,"StudyInstanceUID","")); series=str(getattr(d,"SeriesInstanceUID",""))
        if m and series:
            g=groups[(m,study,series)]; g["count"]+=1; g["bytes"]+=f.stat().st_size; g["path"]=str(f.parent)
    except Exception: pass
summary=defaultdict(lambda:{"series":0,"files":0})
for (m,_,_),g in groups.items(): summary[m]["series"]+=1; summary[m]["files"]+=g["count"]
print(json.dumps({"summary":summary,"smallest":{m:sorted([{**g,"study":s,"series":se} for (mm,s,se),g in groups.items() if mm==m],key=lambda x:x["bytes"])[:5] for m in summary}},default=dict))
