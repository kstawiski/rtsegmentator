#!/usr/bin/env python3
"""Create a geometry-faithful research DICOM image series from a NIfTI volume."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, PositronEmissionTomographyImageStorage, generate_uid

p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output_dir")
p.add_argument("--modality", default="MR", choices=["MR", "CT", "PT"])
p.add_argument("--description", default="Synthetic geometry reference; research validation only")
a = p.parse_args()

# Normalize storage axes to RAS before treating axis 2 as the DICOM slice
# direction. Some public challenge files (notably KiTS19) store arrays as
# (slice, row, column); their affine still describes the correct anatomy, so
# indexing the raw third axis would create the wrong number of DICOM slices.
img = nib.as_closest_canonical(nib.load(a.input))
values = np.asarray(img.dataobj, dtype=np.float32)
if values.ndim != 3:
    raise SystemExit(f"Expected 3-D NIfTI, got {values.shape}")

# NIfTI world coordinates are RAS; DICOM patient coordinates are LPS.
ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
affine = ras_to_lps @ img.affine
axes = affine[:3, :3]
spacing = np.linalg.norm(axes, axis=0)
directions = axes / spacing
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)

finite = values[np.isfinite(values)]
lo, hi = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
if a.modality == "CT":
    # Preserve Hounsfield units directly whenever they fit the normal signed
    # 16-bit CT representation. This avoids quantization in validation data.
    slope, intercept = 1.0, 0.0
    stored = np.clip(np.rint(values), -32768, 32767).astype(np.int16)
    sop_class = CTImageStorage
    pixel_representation = 1
else:
    slope = max((hi - lo) / 65534.0, 1e-6)
    intercept = lo
    stored = np.clip(np.rint((values - intercept) / slope), 0, 65534).astype(np.uint16)
    sop_class = MRImageStorage if a.modality == "MR" else PositronEmissionTomographyImageStorage
    pixel_representation = 0

study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
for k in range(values.shape[2]):
    sop_uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(out / f"IM{k + 1:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.Modality = a.modality
    ds.PatientName = "RESEARCH^SYNTHETIC"
    ds.PatientID = "HNTSMRG24-101-SYNTHETIC"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.StudyID = "SYNTH-HNTS-101"
    ds.StudyDate = "20240801"
    ds.SeriesDate = "20240801"
    ds.StudyTime = "120000"
    ds.SeriesTime = "120000"
    ds.SeriesNumber = 1
    ds.InstanceNumber = k + 1
    ds.SeriesDescription = a.description
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.Rows = values.shape[1]
    ds.Columns = values.shape[0]
    ds.ImageOrientationPatient = [*directions[:, 0], *directions[:, 1]]
    ds.ImagePositionPatient = (affine @ [0, 0, k, 1])[:3].tolist()
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
    ds.SliceThickness = float(spacing[2])
    ds.SpacingBetweenSlices = float(spacing[2])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = pixel_representation
    ds.RescaleIntercept = intercept
    ds.RescaleSlope = slope
    ds.PixelData = stored[:, :, k].T.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(ds.filename, enforce_file_format=True)

print(f"Wrote {values.shape[2]} synthetic {a.modality} reference slices to {out}")
