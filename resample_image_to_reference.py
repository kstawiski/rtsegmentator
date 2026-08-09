#!/usr/bin/env python3
"""Resample a scalar image into a reference volume's physical geometry."""
import sys
from pathlib import Path
import SimpleITK as sitk

moving = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
reference = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)
result = sitk.Resample(
    moving,
    reference,
    sitk.Transform(),
    sitk.sitkLinear,
    0.0,
    sitk.sitkFloat32,
)
target = Path(sys.argv[3])
target.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(result, str(target))
print({"size": result.GetSize(), "spacing": result.GetSpacing(), "output": str(target)})
