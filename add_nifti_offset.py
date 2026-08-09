"""Add a fixed intensity offset while preserving complete NIfTI geometry."""

import sys
import SimpleITK as sitk

source, destination, offset = sys.argv[1], sys.argv[2], float(sys.argv[3])
image = sitk.ReadImage(source)
shifted = sitk.ShiftScale(image, shift=offset, scale=1.0,
                          outputPixelType=sitk.sitkFloat32)
sitk.WriteImage(shifted, destination)
