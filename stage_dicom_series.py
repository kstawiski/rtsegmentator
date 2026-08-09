"""Stage one DICOM image series without copying unrelated modalities/series."""
from pathlib import Path
import os
import sys

import pydicom

source, destination, wanted_uid = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
destination.mkdir(parents=True, exist_ok=True)
count = 0
for candidate in source.rglob("*"):
    if not candidate.is_file():
        continue
    try:
        ds = pydicom.dcmread(
            candidate,
            stop_before_pixels=True,
            specific_tags=["SeriesInstanceUID", "Rows", "Columns"],
        )
    except Exception:
        continue
    if str(getattr(ds, "SeriesInstanceUID", "")) != wanted_uid:
        continue
    if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
        continue
    target = destination / candidate.name
    if not target.exists():
        os.symlink(candidate.resolve(), target)
    count += 1
print(count)
if count == 0:
    raise SystemExit("series not found")
