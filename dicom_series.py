"""Source-plane ordering shared by NIfTI conversion and RTSTRUCT writing."""
from pathlib import Path
from functools import lru_cache

import SimpleITK as sitk


def ordered_dicom_files(source):
    directory = Path(source).resolve()
    stat = directory.stat()
    count = sum(1 for _ in directory.iterdir())
    # Callers may mutate their result without changing the cached ordering.
    return list(_ordered_dicom_files(str(directory), stat.st_mtime_ns, count))


@lru_cache(maxsize=32)
def _ordered_dicom_files(source, mtime_ns, entry_count):
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(source) or []
    if len(series_ids) != 1:
        raise ValueError(f"expected one DICOM series, found {len(series_ids)}")
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(source, series_ids[0])
    if not files:
        raise ValueError("DICOM series contains no source slices")
    return tuple(files)
