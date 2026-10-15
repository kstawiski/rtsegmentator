"""Convert a single staged DICOM series to geometry-preserving NIfTI."""
from pathlib import Path
import sys

import SimpleITK as sitk
from dicom_series import ordered_dicom_files

source, output = Path(sys.argv[1]), Path(sys.argv[2])
files = ordered_dicom_files(source)
reader = sitk.ImageSeriesReader()
reader.SetFileNames(files)
image = reader.Execute()
output.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(image, str(output), True)
print(f"{len(files)} slices -> {image.GetSize()} {image.GetSpacing()}")
