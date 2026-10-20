from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def load_app(tmp_path: Path):
    os.environ["PORTAL_DATA"] = str(tmp_path / "jobs")
    os.environ["WORKER_MODE"] = "local"
    os.environ["LOCAL_PYTHON"] = sys.executable
    os.environ["LOCAL_RUNNER"] = str(Path(__file__).parents[1] / "run_task.py")
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


def test_versioned_api_catalog(tmp_path):
    with TestClient(load_app(tmp_path)) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["worker_mode"] == "local"
        catalog = client.get("/api/v1/models").json()["tasks"]
        assert catalog
        assert all(item.get("description") and item.get("structures") and item.get("reference") for item in catalog)
        import json
        contracts = json.loads((Path(__file__).parents[1] / "catalog/structures.json").read_text())
        for item in catalog:
            contract = contracts[item["name"]]
            assert item["structure_kind"] == contract["kind"]
            assert item["structure_names"] == contract["names"]
            assert item["structure_count"] == contract["count"]
            assert item["structure_note"] == contract["note"]
            assert not item["enabled"] or item["structure_kind"] != "unresolved"

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
    with TestClient(load_app(tmp_path)) as client:
        handles = [path.open("rb") for path in paths]
        try:
            files = [("files", (path.name, handle, "application/dicom")) for path, handle in zip(paths, handles)]
            response = client.post("/api/uploads", files=files)
        finally:
            for handle in handles:
                handle.close()
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["series"]) == 1
    assert payload["series"][0]["modality"] == "CT"
    assert payload["series"][0]["slices"] == 3


@pytest.mark.parametrize('positive,negative,box,accepted', [
    (True, False, False, True),
    (False, False, True, True),
    (True, False, True, True),
    (True, True, True, True),
    (True, True, False, True),
    (False, True, True, False),
    (False, True, False, False),
])
def test_sam_med3d_prompt_combinations(portal, positive, negative, box, accepted):
    name = 'sam_med3d_turbo_prompted'
    prompt = {'positive_points': [[1, 1, 1]] if positive else [],
              'negative_points': [[2, 2, 1]] if negative else [],
              'box': [[4, 4, 1], [12, 12, 1]] if box else None}
    items = {name: portal.model_details(name)}
    selected = {'columns': 32, 'rows': 32, 'slices': 12}
    if accepted:
        assert portal.validate_assessed_prompt([name], items, selected, json.dumps(prompt)) == prompt
    else:
        with pytest.raises(portal.HTTPException) as error:
            portal.validate_assessed_prompt([name], items, selected, json.dumps(prompt))
        assert error.value.status_code == 400
        assert 'positive point' in error.value.detail
        if box:
            assert 'ignores the box when points are present' in error.value.detail


@pytest.mark.parametrize('route,series_field', [
    ('/api/uploads/upload-synthetic/start', 'series_key'),
    ('/api/v1/uploads/upload-synthetic/jobs', 'series_keys'),
])
def test_sam_med3d_negative_points_and_box_rejected_before_submission(portal, monkeypatch, route, series_field):
    name = 'sam_med3d_turbo_prompted'
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks': [{'name': name, **portal.model_details(name)}]})
    queued = []
    monkeypatch.setattr(portal, 'enqueue_jobs', lambda jobs: queued.extend(jobs))
    stage = portal.BASE / 'upload-synthetic'
    stage.mkdir()
    (stage / 'upload.json').write_text(json.dumps({'series': [
        {'key': 'synthetic-ct', 'modality': 'CT', 'columns': 32, 'rows': 32, 'slices': 12, 'files': []}
    ]}))
    # No lifecycle or worker is needed: exercise the real HTTP validation route.
    client = TestClient(portal.app)
    try:
        response = client.post(route, data={series_field: 'synthetic-ct', 'models': name,
            'prompt_json': json.dumps({'negative_points': [[2, 2, 1]], 'box': [[4, 4, 1], [12, 12, 1]]})})
    finally:
        client.close()
    assert response.status_code == 400
    assert 'SAM-Med3D ignores the box when points are present' in response.json()['detail']
    assert 'Add a positive point' in response.json()['detail']
    assert queued == []
    assert list(portal.BASE.iterdir()) == [stage]
    assert (stage / 'upload.json').is_file()
