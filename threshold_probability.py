"""Threshold a scalar probability NIfTI into a binary label map."""
from pathlib import Path
import sys

import SimpleITK as sitk

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
image = sitk.ReadImage(str(source))
label = sitk.Cast(image >= threshold, sitk.sitkUInt8)
destination.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(label, str(destination), True)
stats = sitk.StatisticsImageFilter()
stats.Execute(label)
print({"threshold": threshold, "foreground_voxels": int(stats.GetSum())})
