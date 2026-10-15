"""Convert a single staged DICOM series to geometry-preserving NIfTI."""
from pathlib import Path
import sys

import SimpleITK as sitk
from dicom_series import ordered_dicom_files
from orientation_contract import normalize_file

source, output = Path(sys.argv[1]), Path(sys.argv[2])
files = ordered_dicom_files(source)
reader = sitk.ImageSeriesReader()
reader.SetFileNames(files)
image = reader.Execute()
output.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(image, str(output), True)
contract = normalize_file(output)
print(f"Orientation contract: {contract['source_axcodes']} -> {contract['target_axcodes']}; changed={contract['changed']}")
print(f"{len(files)} slices -> {image.GetSize()} {image.GetSpacing()}")
