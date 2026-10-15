#!/usr/bin/env python3
"""Zero every label that is not listed in a label map, rewriting the multilabel NIfTI in place.

TotalSegmentator removes a task's auxiliary training labels only when its class_map has "<task>_auxiliary".
appendicular_bones_mr (TotalSegmentator 2.17.0, Dataset855) has none, so its humerus/femur/liver/spleen helper labels
reach the prediction and the RTSTRUCT converter rightly rejects them as unmapped.
Usage: keep_labels.py <prediction.nii.gz> <labels.json>
"""
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main(prediction, labels):
    keep = {int(key) for key in json.loads(Path(labels).read_text())}
    image = nib.load(prediction)
    data = np.asarray(image.dataobj)
    present = {int(value) for value in np.unique(data)} - {0}
    dropped = sorted(present - keep)
    if dropped:
        cleaned = np.where(np.isin(data, dropped), 0, data).astype(data.dtype)
        partial = Path(prediction).with_name(".keep-labels-" + Path(prediction).name)
        nib.save(nib.Nifti1Image(cleaned, image.affine, image.header), partial)
        os.replace(partial, prediction)
    print(f"labels kept {sorted(present & keep)}; dropped outside the label map {dropped}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
