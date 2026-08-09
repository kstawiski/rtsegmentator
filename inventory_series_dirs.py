from pathlib import Path
import json,sys,pydicom
root=Path(sys.argv[1]); out=[]
for patient in root.iterdir():
    if not patient.is_dir(): continue
    for series in patient.iterdir():
        if not series.is_dir(): continue
        try: f=next(x for x in series.iterdir() if x.is_file())
        except StopIteration: continue
        try:
            d=pydicom.dcmread(f,stop_before_pixels=True)
            out.append({"modality":str(getattr(d,"Modality","")),"body":str(getattr(d,"BodyPartExamined","")),"description":str(getattr(d,"SeriesDescription","")),"path":str(series)})
        except Exception: pass
print(json.dumps(out))
