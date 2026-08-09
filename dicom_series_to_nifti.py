"""Convert a single staged DICOM series to geometry-preserving NIfTI."""
from pathlib import Path
import sys

import SimpleITK as sitk

source, output = Path(sys.argv[1]), Path(sys.argv[2])
series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(source)) or []
if len(series_ids) != 1:
    raise SystemExit(f"expected one DICOM series, found {len(series_ids)}")
files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(source), series_ids[0])
reader = sitk.ImageSeriesReader()
reader.SetFileNames(files)
image = reader.Execute()
output.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(image, str(output), True)
print(f"{len(files)} slices -> {image.GetSize()} {image.GetSpacing()}")
