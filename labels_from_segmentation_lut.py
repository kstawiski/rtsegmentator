"""Create label JSON for only integer labels present in a segmentation."""
import argparse
import json
from structure_labels import lut_labels
from pathlib import Path

import nibabel as nib
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("segmentation")
p.add_argument("lut")
p.add_argument("output")
a = p.parse_args()

names = {int(k): v for k, v in lut_labels(Path(a.lut), freesurfer=True, allow_empty=True).items()}
present = sorted(int(v) for v in np.unique(np.asanyarray(nib.load(a.segmentation).dataobj)) if v)
missing = [value for value in present if value not in names]
if missing:
    raise SystemExit(f"LUT does not define prediction labels: {missing}")
labels = {str(value): names[value].replace(" ", "_") for value in present}
Path(a.output).write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
