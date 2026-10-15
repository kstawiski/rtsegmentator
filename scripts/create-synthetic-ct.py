#!/usr/bin/env python3
"""Create a small synthetic CT series for transport/integration testing only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
study, series, frame = generate_uid(), generate_uid(), generate_uid()
x, y = np.ogrid[-1:1:128j, -1:1:128j]
for k in range(96):
    z = -1 + 2 * k / 95
    body = x * x / 0.72**2 + y * y / 0.48**2 + z * z / 0.88**2 < 1
    image = np.full((128, 128), -1000, dtype=np.int16)
    image[body] = 40
    sop = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(a.output / f"IM{k + 1:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, sop
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
    ds.Modality, ds.PatientName, ds.PatientID = "CT", "RESEARCH^SYNTHETIC", "COLAB-E2E"
    ds.StudyID, ds.StudyDate, ds.StudyTime = "COLAB-E2E", "20260823", "065000"
    ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
    ds.SeriesDescription, ds.ImageType = "Synthetic Colab integration test", ["DERIVED", "SECONDARY"]
    ds.Rows, ds.Columns = image.shape
    ds.ImageOrientationPatient, ds.ImagePositionPatient = [1, 0, 0, 0, 1, 0], [-192, -192, -144 + 3 * k]
    ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [3, 3], 3, 3
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleIntercept, ds.RescaleSlope, ds.PixelData = 0, 1, image.tobytes()
    ds.save_as(ds.filename, enforce_file_format=True)
