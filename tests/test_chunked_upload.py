"""CPU coverage for the upload contracts named in deployment validation notes."""
import json
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
