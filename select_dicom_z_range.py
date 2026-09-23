#!/usr/bin/env python3
"""Create a linked DICOM fixture containing slices in a physical z range."""

import argparse
import os
from pathlib import Path

import pydicom


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("z_min", type=float)
parser.add_argument("z_max", type=float)
args = parser.parse_args()
args.destination.mkdir(parents=True, exist_ok=True)
selected = 0
for source in args.source.iterdir():
    try:
        dataset = pydicom.dcmread(source, stop_before_pixels=True)
        z = float(dataset.ImagePositionPatient[2])
    except Exception:
        continue
    if args.z_min <= z <= args.z_max:
        target = args.destination / source.name
        if not target.exists():
            os.symlink(source.resolve(), target)
        selected += 1
if not selected:
    raise SystemExit("No DICOM slices fall inside the requested z range")
print(f"selected {selected} slices")
