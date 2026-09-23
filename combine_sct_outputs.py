"""Combine SCT's one-or-many output masks into one RTSTRUCT label volume."""
import argparse, json, re
from pathlib import Path
import nibabel as nib
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--glob", required=True)
p.add_argument("--output", required=True)
p.add_argument("--labels", required=True)
p.add_argument("--output-names", default="{}")
p.add_argument("--multiclass-labels", default="{}")
a=p.parse_args()
files=sorted(Path().glob(a.glob) if not Path(a.glob).is_absolute() else Path(a.glob).parent.glob(Path(a.glob).name))
if not files: raise SystemExit(f"SCT produced no files matching {a.glob}")
names=json.loads(a.output_names); multiclass=json.loads(a.multiclass_labels)
imgs=[nib.load(str(f)) for f in files]
shape=imgs[0].shape
if any(i.shape != shape for i in imgs): raise SystemExit("SCT outputs have inconsistent shapes")
if len(imgs)==1 and multiclass:
    data=np.rint(np.asanyarray(imgs[0].dataobj)).astype(np.uint16)
    labels=multiclass
else:
    data=np.zeros(shape, dtype=np.uint16); labels={}
    for idx,(f,img) in enumerate(zip(files,imgs),1):
        key=re.sub(r"^prediction_?|\.nii(?:\.gz)?$", "", f.name).strip("_")
        label=next((v for k,v in names.items() if k.lower() in key.lower()), None)
        if label is None and len(files)==1 and len(set(names.values()))==1:
            label=next(iter(names.values()))
        label=label or key or f"SCT_Label_{idx}"
        mask=np.asanyarray(img.dataobj)>0
        if mask.any(): data[mask]=idx
        labels[str(idx)]=label
if not np.any(data): raise SystemExit("SCT prediction is empty")
header=imgs[0].header.copy(); header.set_data_dtype(np.uint16)
nib.save(nib.Nifti1Image(data, imgs[0].affine, header), a.output)
Path(a.labels).write_text(json.dumps(labels,indent=2,sort_keys=True)+"\n")
