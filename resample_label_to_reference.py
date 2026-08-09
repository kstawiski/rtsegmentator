"""Resample a cropped discrete label map into a full reference image geometry."""
from pathlib import Path
import sys

import SimpleITK as sitk

label_path, reference_path, output_path = map(Path, sys.argv[1:4])
label, reference = sitk.ReadImage(str(label_path)), sitk.ReadImage(str(reference_path))
result = sitk.Resample(label, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
output_path.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(result, str(output_path), True)
print({"input": label.GetSize(), "output": result.GetSize()})
