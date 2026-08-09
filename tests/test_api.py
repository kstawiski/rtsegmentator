from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pydicom
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def load_app(tmp_path: Path):
    os.environ["PORTAL_DATA"] = str(tmp_path / "jobs")
    os.environ["WORKER_MODE"] = "local"
    os.environ["LOCAL_PYTHON"] = os.sys.executable
    os.environ["LOCAL_RUNNER"] = str(Path(__file__).parents[1] / "run_task.py")
    os.environ["RTSEG_API_KEY"] = "test-key"
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    return module.app


def make_slice(path: Path, series_uid: str, instance: int) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = generate_uid()
    ds.Modality = "CT"
    ds.SeriesDescription = "API synthetic test"
    ds.PatientName = "SYNTHETIC^TEST"
    ds.PatientID = "SYNTHETIC"
    ds.Rows = ds.Columns = 8
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [0, 0, float(instance)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = np.zeros((8, 8), dtype=np.int16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_versioned_api_catalog_and_auth(tmp_path):
    with TestClient(load_app(tmp_path)) as client:
        assert client.get("/api/v1/health").status_code == 401
        headers = {"X-API-Key": "test-key"}
        health = client.get("/api/v1/health", headers=headers)
        assert health.status_code == 200
        assert health.json()["worker_mode"] == "local"
        catalog = client.get("/api/v1/models", headers=headers).json()["tasks"]
        assert catalog
        assert all(item.get("description") and item.get("structures") and item.get("reference") for item in catalog)
        assert client.get("/api/openapi.json").status_code == 200


def test_assessment_api_groups_dicom_series(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    series_uid = generate_uid()
    paths = []
    for index in range(3):
        path = source / f"slice-{index}.dcm"
        make_slice(path, series_uid, index)
        paths.append(path)
    headers = {"X-API-Key": "test-key"}
    with TestClient(load_app(tmp_path)) as client:
        handles = [path.open("rb") for path in paths]
        try:
            files = [("files", (path.name, handle, "application/dicom")) for path, handle in zip(paths, handles)]
            response = client.post("/api/v1/uploads", headers=headers, files=files)
        finally:
            for handle in handles:
                handle.close()
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["series"]) == 1
    assert payload["series"][0]["modality"] == "CT"
    assert payload["series"][0]["slices"] == 3
