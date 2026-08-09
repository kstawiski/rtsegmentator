#!/usr/bin/env python3
"""Crop an image to the nonzero physical bounding box of a geometry-matched mask."""
import sys
from pathlib import Path

import SimpleITK as sitk
import numpy as np

image = sitk.ReadImage(sys.argv[1])
mask = sitk.ReadImage(sys.argv[2])
if image.GetSize() != mask.GetSize():
    raise SystemExit(f"Image/mask size mismatch: {image.GetSize()} != {mask.GetSize()}")
arr = sitk.GetArrayViewFromImage(mask)
where = np.argwhere(arr > 0)
if not where.size:
    raise SystemExit("Mask is empty")
z0, y0, x0 = where.min(axis=0)
z1, y1, x1 = where.max(axis=0) + 1
index = [int(x0), int(y0), int(z0)]
size = [int(x1 - x0), int(y1 - y0), int(z1 - z0)]
cropped = sitk.RegionOfInterest(image, size=size, index=index)
target = Path(sys.argv[3])
target.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(cropped, str(target))
print({"index": index, "size": size, "output": str(target)})
