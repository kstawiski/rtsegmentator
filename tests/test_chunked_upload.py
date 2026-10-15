"""CPU coverage for the upload contracts named in deployment validation notes."""
import json
import pytest
import time

from fastapi.testclient import TestClient


def test_resume_offsets_limits_and_path_sanitization(portal, monkeypatch):
    client = TestClient(portal.app)
    created = client.post("/api/v1/upload-sessions").json()
    assert created["max_request_bytes"] == 32 * 1024**2
    sid = created["session_id"]
    url = f"/api/v1/upload-sessions/{sid}/files/0"
    params = {"name": "../../CT.dcm", "offset": 0, "size": 6}
    assert client.put(url, params=params, content=b"abc").json()["received"] == 3
    conflict = client.put(url, params=params, content=b"abc")
    assert conflict.status_code == 409
    assert conflict.json()["expected_offset"] == 3
    params["offset"] = 3
    assert client.put(url, params=params, content=b"def").json()["complete"]
    assert (portal.BASE / sid / "files/0").read_bytes() == b"abcdef"
    record = json.loads((portal.BASE / sid / "session.json").read_text())
    assert record["files"]["0"]["name"] == "CT.dcm"
    assert record["received_bytes"] == 6
    monkeypatch.setattr(portal, "MAX_REQUEST_BYTES", 2)
    params.update(offset=0, size=3)
    assert client.put(url.replace("/0", "/1"), params=params, content=b"abc").status_code == 413


def test_completed_synthetic_upload_assesses_once(portal, synthetic_ct):
    client = TestClient(portal.app)
    sid = client.post("/api/v1/upload-sessions").json()["session_id"]
    for index, path in enumerate(sorted(synthetic_ct.iterdir())):
        payload = path.read_bytes()
        response = client.put(f"/api/v1/upload-sessions/{sid}/files/{index}",
                              params={"name": path.name, "offset": 0, "size": len(payload)}, content=payload)
        assert response.status_code == 200
    assess = f"/api/v1/upload-sessions/{sid}/assess"
    assert client.post(assess).status_code == 202
    assert client.post(assess).status_code == 202
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/upload-sessions/{sid}").json()
        if status["status"] != "assessing":
            break
        time.sleep(.01)
    assert status["status"] == "ready", status
    assert status["result"]["series"][0]["slices"] == 12
    assert len(list(portal.BASE.glob("upload-*"))) == 1


def test_incomplete_session_cannot_finalize(portal):
    client = TestClient(portal.app)
    sid = client.post("/api/v1/upload-sessions").json()["session_id"]
    assert client.post(f"/api/v1/upload-sessions/{sid}/assess").status_code == 409


def test_report_session_cbct_rejected_after_staging(portal, synthetic_ct, monkeypatch):
    import pydicom
    from types import SimpleNamespace
    from nv_reason_ct_report import TASK
    source = next(synthetic_ct.iterdir())
    ds=pydicom.dcmread(source); ds.ManufacturerModelName='Patient Verification'; ds.save_as(source,enforce_file_format=True)
    monkeypatch.setattr(portal, 'tasks', lambda: {'tasks':[{'name':TASK,'enabled':True}]})
    monkeypatch.setattr(portal.threading, 'Thread', lambda target,args,**kw: SimpleNamespace(start=lambda: target(*args)))
    monkeypatch.setattr(portal, 'process', lambda *args: pytest.fail('CBCT reached inference'))
    client=TestClient(portal.app)
    sid=client.post('/api/v1/upload-sessions').json()['session_id']; payload=source.read_bytes()
    assert client.put(f'/api/v1/upload-sessions/{sid}/files/0',params={'name':'slice.dcm','offset':0,'size':len(payload)},content=payload).status_code == 200
    response=client.post(f'/api/v1/upload-sessions/{sid}/jobs',data={'models':TASK})
    assert response.status_code == 202
    job=client.get('/api/v1/jobs/'+response.json()['id']).json()
    assert job['status'] == 'failed' and 'CBCT' in job['error']
