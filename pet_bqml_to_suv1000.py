#!/usr/bin/env python3
"""Convert a BQML PET NIfTI to the SUVbw*1000 convention used by ISRT models."""
import argparse
import datetime as dt
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

p = argparse.ArgumentParser()
p.add_argument("input_nifti")
p.add_argument("dicom_dir")
p.add_argument("output_nifti")
a = p.parse_args()

files = sorted(x for x in Path(a.dicom_dir).iterdir() if x.is_file())
ds = pydicom.dcmread(files[0], stop_before_pixels=True)
if str(getattr(ds, "Units", "")).upper() != "BQML":
    raise SystemExit(f"Expected PET Units=BQML, got {getattr(ds, 'Units', None)!r}")
weight_kg = float(ds.PatientWeight)
radio = ds.RadiopharmaceuticalInformationSequence[0]
dose_bq = float(radio.RadionuclideTotalDose)
half_life_s = float(radio.RadionuclideHalfLife)

def parse_datetime(date: str, time: str) -> dt.datetime:
    time = time.split(".")[0].ljust(6, "0")[:6]
    return dt.datetime.strptime(date + time, "%Y%m%d%H%M%S")

start_dt_text = str(getattr(radio, "RadiopharmaceuticalStartDateTime", ""))
if start_dt_text:
    injection = parse_datetime(start_dt_text[:8], start_dt_text[8:])
else:
    injection = parse_datetime(str(ds.SeriesDate), str(radio.RadiopharmaceuticalStartTime))
# DECY/START pixel values are referenced to series start, not each later bed frame.
scan = parse_datetime(str(ds.SeriesDate), str(ds.SeriesTime))
if scan < injection:
    scan += dt.timedelta(days=1)
elapsed = (scan - injection).total_seconds()
decayed_dose = dose_bq * math.exp(-math.log(2.0) * elapsed / half_life_s)
factor = weight_kg * 1000.0 / decayed_dose

img = nib.load(a.input_nifti)
bqml = np.asarray(img.dataobj, dtype=np.float32)
suv1000 = bqml * factor * 1000.0
target = Path(a.output_nifti)
target.parent.mkdir(parents=True, exist_ok=True)
nib.save(nib.Nifti1Image(suv1000.astype(np.float32), img.affine, img.header), target)
print({"elapsed_s": elapsed, "decayed_dose_bq": decayed_dose,
       "suvbw_factor": factor, "suv_max": float(suv1000.max() / 1000.0),
       "output": str(target)})
