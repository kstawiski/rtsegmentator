#!/usr/bin/env python3
"""Apply MATTO's percentile normalization to a brain-extracted NIfTI."""

import sys
import nibabel as nib
import numpy as np

source = nib.load(sys.argv[1])
data = np.asarray(source.dataobj, dtype=np.float32)
foreground = data != 0
if not foreground.any():
    raise SystemExit("Brain-extracted input is empty")
low, high = np.percentile(data[foreground], (0.1, 99.9))
data = np.clip(data, low, high)
data = (data - low) / max(float(high - low), 1e-8)
data[~foreground] = 0
nib.save(nib.Nifti1Image(data.astype(np.float32), source.affine, source.header), sys.argv[2])
