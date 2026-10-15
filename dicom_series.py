"""Source-plane ordering shared by NIfTI conversion and RTSTRUCT writing."""
from pathlib import Path

import SimpleITK as sitk


def ordered_dicom_files(source):
    source = str(Path(source))
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(source) or []
    if len(series_ids) != 1:
        raise ValueError(f"expected one DICOM series, found {len(series_ids)}")
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(source, series_ids[0])
    if not files:
        raise ValueError("DICOM series contains no source slices")
    return files
