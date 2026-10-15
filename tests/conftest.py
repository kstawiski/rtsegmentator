from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


@pytest.fixture
def portal(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTAL_DATA", str(tmp_path / "jobs"))
    monkeypatch.setenv("WORKER_MODE", "local")
    monkeypatch.setenv("LOCAL_PYTHON", sys.executable)
    monkeypatch.setenv("LOCAL_RUNNER", str(Path(__file__).parents[1] / "run_task.py"))
    spec = importlib.util.spec_from_file_location("test_portal", Path(__file__).parents[1] / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_ct(tmp_path):
    """Fully synthetic images: no copied headers, pixels or patient identifiers."""
    root = tmp_path / "ct"
    root.mkdir()
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for k in range(12):
        path = root / f"slice-{k:02d}.dcm"
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.Modality, ds.PatientName, ds.PatientID = "CT", "SYNTHETIC^TEST", "SYNTHETIC"
        ds.StudyDate, ds.StudyTime, ds.StudyID = "20000101", "120000", "SYNTHETIC"
        ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
        ds.SeriesDescription = "Synthetic geometry test"
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.Rows = ds.Columns = 32
        ds.ImagePositionPatient = [0, 0, float(k * 2)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing, ds.SliceThickness = [1, 1], 2
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.PixelData = np.full((32, 32), k, dtype=np.int16).tobytes()
        ds.save_as(path, enforce_file_format=True)
    return root
