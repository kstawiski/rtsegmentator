from pathlib import Path
import json,sys,pydicom
out=[]
for folder in Path(sys.argv[1]).iterdir():
    if not folder.is_dir(): continue
    try: f=next(x for x in folder.iterdir() if x.is_file() and x.suffix.lower()==".dcm")
    except StopIteration: continue
    try:
        d=pydicom.dcmread(f,stop_before_pixels=True)
        out.append({"modality":str(getattr(d,"Modality","")),"body":str(getattr(d,"BodyPartExamined","")),"description":str(getattr(d,"SeriesDescription","")),"path":str(folder)})
    except Exception: pass
print(json.dumps(out))
