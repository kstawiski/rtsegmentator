"""Result manifest, its API route, and the standard-library client, end to end over HTTP.

Synthetic images and a synthetic RTSTRUCT only; no model is executed.
"""
from __future__ import annotations

import io
import json
import socket
import threading
import time
import zipfile
from pathlib import Path

import jsonschema
import numpy as np
import pydicom
import pytest
import uvicorn
from fastapi.testclient import TestClient
from rt_utils import RTStructBuilder

import result_manifest
import rtseg_client

SCHEMA = json.loads((Path(__file__).parents[1] / "schemas/result-manifest-1.0.schema.json").read_text())


def write_rtstruct(series_dir: Path, output: Path, names=("liver", "spleen")) -> Path:
    rt = RTStructBuilder.create_new(dicom_series_path=str(series_dir))
    for index, name in enumerate(names):
        mask = np.zeros((32, 32, 12), dtype=bool)
        mask[4 + 8 * index:10 + 8 * index, 6:20, 3:9] = True
        rt.add_roi(mask=mask, name=name)
    rt.save(str(output))
    return output


def series_uid(series_dir: Path) -> str:
    first = sorted(series_dir.iterdir())[0]
    return str(pydicom.dcmread(first, stop_before_pixels=True).SeriesInstanceUID)


def finished_job(portal, synthetic_ct, *, models=("total", "lung_nodules"), empty=("lung_nodules",)):
    jid = "20260101-000000-00000000000000000001-abcd1234"
    job = portal.BASE / jid
    (job / "input").mkdir(parents=True)
    (job / "output").mkdir()
    rtstruct = write_rtstruct(synthetic_ct, job / "output/total_RTSTRUCT.dcm")
    (job / "output/total_RTSTRUCT.dcm.orientation.json").write_text('{"orientation": "checked"}')
    (job / "job.json").write_text(json.dumps({
        "id": jid, "status": "processing", "created": 1767225600.0, "models": list(models),
        "batch_id": "batch-synthetic", "batch_index": 1, "batch_size": 1, "postprocess": [],
        "series": {"key": "k", "series_uid": series_uid(synthetic_ct), "modality": "CT",
                   "description": "Synthetic", "slices": 12, "rows": 32, "columns": 32,
                   "study_date": "20000101", "cbct": False}}))
    portal.finalize_results(job, list(empty))
    return job, rtstruct


def test_finalized_package_carries_valid_manifest(portal, synthetic_ct):
    job, rtstruct = finished_job(portal, synthetic_ct)
    package = job / "rtstruct-results.zip"
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read(result_manifest.MANIFEST_NAME))
    jsonschema.validate(manifest, SCHEMA)
    assert names == {"total_RTSTRUCT.dcm", "total_RTSTRUCT.dcm.orientation.json", "NO_STRUCTURES_FOUND.txt",
                     result_manifest.MANIFEST_NAME}
    assert manifest["generator"] == {"name": "RTsegmentator", "version": portal.__version__}
    assert manifest["source_series"]["series_instance_uid"] == series_uid(synthetic_ct)
    assert "study_date" not in manifest["source_series"] and "description" not in manifest["source_series"]
    total, nodules = manifest["models"]
    assert total["outcome"] == "rtstruct" and nodules["outcome"] == "no_structures_found"
    assert total["rtstruct"]["roi_names"] == ["liver", "spleen"]
    assert total["rtstruct"]["referenced_series_instance_uids"] == [series_uid(synthetic_ct)]
    assert total["structure_kind"] == "fixed" and "liver" in total["catalog_structures"]
    assert total["sidecars"] == ["total_RTSTRUCT.dcm.orientation.json"]
    by_path = {item["path"]: item for item in manifest["files"]}
    assert by_path["total_RTSTRUCT.dcm"]["sha256"] == result_manifest.sha256_file(rtstruct)
    assert rtseg_client.verify_result_zip(package) == manifest


def test_manifest_route_and_result_route(portal, synthetic_ct):
    job, _ = finished_job(portal, synthetic_ct)
    portal.save_metadata(job, status="complete")
    client = TestClient(portal.app)
    response = client.get(f"/api/v1/jobs/{job.name}/manifest")
    assert response.status_code == 200
    assert response.json()["job"]["id"] == job.name
    assert client.get("/api/v1/jobs/..%2F..%2Fetc/manifest").status_code == 404
    assert client.get("/api/v1/jobs/bad.id/result").status_code == 404
    portal.save_metadata(job, status="processing")
    assert client.get(f"/api/v1/jobs/{job.name}/manifest").status_code == 409


def tampered(package: Path, target: Path, *, replace=None, add=None):
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(target, "w") as output:
        for info in source.infolist():
            data = source.read(info.filename)
            if replace and info.filename == replace:
                data = data[:-1] + bytes([data[-1] ^ 1])
            output.writestr(info.filename, data)
        if add:
            output.writestr(add, b"extra")
    return target


def test_client_rejects_tampered_or_unlisted_members(portal, synthetic_ct, tmp_path):
    job, _ = finished_job(portal, synthetic_ct)
    package = job / "rtstruct-results.zip"
    with pytest.raises(rtseg_client.ManifestError, match="checksum mismatch"):
        rtseg_client.verify_result_zip(tampered(package, tmp_path / "a.zip", replace="total_RTSTRUCT.dcm"))
    with pytest.raises(rtseg_client.ManifestError, match="missing from the manifest"):
        rtseg_client.verify_result_zip(tampered(package, tmp_path / "b.zip", add="unexpected.dcm"))
    bare = tmp_path / "c.zip"
    with zipfile.ZipFile(bare, "w") as archive:
        archive.writestr("total_RTSTRUCT.dcm", b"x")
    with pytest.raises(rtseg_client.ManifestError, match="no rtsegmentator_manifest.json"):
        rtseg_client.verify_result_zip(bare)


def test_manifest_rejects_unsafe_and_describes_unrequested_members(tmp_path, synthetic_ct):
    rtstruct = write_rtstruct(synthetic_ct, tmp_path / "other_RTSTRUCT.dcm")
    job = {"id": "x", "models": ["total"]}
    with pytest.raises(ValueError, match="escapes"):
        result_manifest.build_manifest(job, [("../x", b"1")], version="1", catalog={}, structure_catalog={})
    manifest = result_manifest.build_manifest(job, [("other_RTSTRUCT.dcm", rtstruct)], version="1",
                                              catalog={}, structure_catalog={})
    assert [(m["name"], m["outcome"], m.get("requested")) for m in manifest["models"]] == [
        ("total", "no_rtstruct", None), ("other", "rtstruct", False)]
    jsonschema.validate(manifest, SCHEMA)


def test_extract_refuses_escaping_members(tmp_path):
    payload = {"schema": rtseg_client.MANIFEST_SCHEMA, "schema_version": "1.0",
               "files": [{"path": "../evil", "size": 1, "sha256": result_manifest.sha256_bytes(b"x")}]}
    package = tmp_path / "evil.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../evil", b"x")
        archive.writestr(rtseg_client.MANIFEST_NAME, json.dumps(payload))
    with pytest.raises(rtseg_client.ManifestError, match="unsafe member name"):
        rtseg_client.extract_result_zip(package, tmp_path / "out")
    assert not (tmp_path / "evil").exists()


@pytest.fixture
def live_portal(portal, monkeypatch):
    """Serve the real app over HTTP; the queue writes a synthetic RTSTRUCT instead of running a model."""
    catalog = {"tasks": [{"name": "total", "modality": "CT", "enabled": True, "structure_kind": "fixed"}]}
    monkeypatch.setattr(portal, "tasks", lambda: catalog)

    def run_immediately(jobs, **kwargs):
        for job in jobs:
            data = portal.metadata(job)
            write_rtstruct(job / "input", job / "output/total_RTSTRUCT.dcm")
            portal.save_metadata(job, status="processing", models=data["models"])
            portal.finalize_results(job)
    monkeypatch.setattr(portal, "enqueue_jobs", run_immediately)
    monkeypatch.setattr(portal, "MAX_REQUEST_BYTES", 4096)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(portal.app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.02)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=10)


def test_client_submit_wait_download_verify(live_portal, synthetic_ct, tmp_path):
    client = rtseg_client.RTsegClient(live_portal, sleep=lambda _: time.sleep(0.05))
    files = rtseg_client.collect_files([synthetic_ct])
    session = client.upload_files(files)  # 4 KiB request limit forces several chunks per file
    upload = client.assess(session, poll=0.05)
    chosen = rtseg_client.select_series(upload["series"], series_uids=[series_uid(synthetic_ct)])
    batch = client.submit(upload["upload_id"], [chosen[0]["key"]], ["total"])
    job_id = batch["jobs"][0]["id"]
    record = client.wait(job_id, poll=0.05, timeout=30)
    assert record["status"] == "complete"
    package = client.download(job_id, tmp_path / "result.zip")
    manifest = rtseg_client.extract_result_zip(package, tmp_path / "extracted")
    assert manifest == client.manifest(job_id)
    assert manifest["models"][0]["rtstruct"]["roi_names"] == ["liver", "spleen"]
    assert (tmp_path / "extracted/total_RTSTRUCT.dcm").is_file()


def test_client_cli_verify_and_errors(portal, synthetic_ct, capsys, tmp_path):
    job, _ = finished_job(portal, synthetic_ct)
    capsys.readouterr()  # rt-utils prints while the fixture RTSTRUCT is written
    assert rtseg_client.main(["verify", str(job / "rtstruct-results.zip")]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    assert rtseg_client.main(["verify", str(broken)]) == 1
    with pytest.raises(ValueError):
        rtseg_client.RTsegClient("ftp://example")
    with pytest.raises(ValueError):
        rtseg_client.parse_headers(["no separator"])
    assert rtseg_client.parse_headers(["Authorization: Bearer x\nX-Proxy: y"]) == {
        "Authorization": "Bearer x", "X-Proxy": "y"}


def test_select_series_never_chooses_unavailable(tmp_path):
    rows = [{"key": "a", "series_uid": "1", "modality": "CT"},
            {"key": "b", "series_uid": "2", "modality": "MR", "unavailable_reason": "rejected"}]
    assert [row["key"] for row in rtseg_client.select_series(rows, modalities=["ct"])] == ["a"]
    with pytest.raises(rtseg_client.ClientError, match="2"):
        rtseg_client.select_series(rows, series_uids=["2"])


def package_with(tmp_path, members, name="pkg.zip"):
    files = [{"path": path, "size": len(data), "sha256": result_manifest.sha256_bytes(data)}
             for path, data in members]
    manifest = {"schema": rtseg_client.MANIFEST_SCHEMA, "schema_version": "1.0", "files": files}
    package = tmp_path / name
    with zipfile.ZipFile(package, "w") as archive:
        for path, data in members:
            archive.writestr(path, data)
        archive.writestr(rtseg_client.MANIFEST_NAME, json.dumps(manifest))
    return package


@pytest.mark.parametrize("members", [
    [("report.json", b"a"), ("./report.json", b"b")],       # same target after normalisation
    [("Report.json", b"a"), ("report.json", b"b")],         # same target on case-insensitive filesystems
    [("a//b.json", b"a")],
])
def test_noncanonical_or_colliding_names_are_rejected(tmp_path, members):
    with pytest.raises(rtseg_client.ManifestError, match="unsafe member name|duplicate member names"):
        rtseg_client.extract_result_zip(package_with(tmp_path, members), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_extract_never_overwrites(tmp_path):
    package = package_with(tmp_path, [("total_RTSTRUCT.dcm", b"rt")])
    rtseg_client.extract_result_zip(package, tmp_path / "out")
    with pytest.raises(rtseg_client.ManifestError, match="overwrite"):
        rtseg_client.extract_result_zip(package, tmp_path / "out")


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_job_creation_is_not_replayed_after_an_ambiguous_failure(monkeypatch):
    calls = []

    def urlopen(request):
        calls.append(request.get_method())
        raise rtseg_client.urllib.error.HTTPError(request.full_url, 504, "gateway", {}, io.BytesIO(b""))
    client = rtseg_client.RTsegClient("http://portal.test", sleep=lambda _: None)
    monkeypatch.setattr(client, "_open", urlopen)
    with pytest.raises(rtseg_client.ClientError, match="HTTP 504"):
        client.submit("upload-1", ["key"], ["total"])
    assert calls == ["POST"]
    calls.clear()
    with pytest.raises(rtseg_client.ClientError, match="HTTP 504"):
        client.job("job-1")
    assert calls == ["GET"] * (client.retries + 1)

    def refused(request):
        calls.append(request.get_method())
        raise rtseg_client.urllib.error.URLError("connection reset")
    monkeypatch.setattr(client, "_open", refused)
    calls.clear()
    with pytest.raises(rtseg_client.ClientError, match="may have taken effect"):
        client.submit("upload-1", ["key"], ["total"])
    assert calls == ["POST"]


def test_download_streams_to_disk_atomically(monkeypatch, tmp_path):
    payload = b"z" * (3 * rtseg_client.STREAM_BLOCK + 17)
    client = rtseg_client.RTsegClient("http://portal.test")
    monkeypatch.setattr(client, "_open", lambda request: FakeResponse(payload))
    target = client.download("job-1", tmp_path / "nested" / "job-1.zip")
    assert target.read_bytes() == payload
    assert not list(target.parent.glob("*.partial"))


def test_cross_origin_redirects_are_refused():
    handler = rtseg_client._SameOriginRedirects()
    request = rtseg_client.urllib.request.Request("https://portal.test/api/v1/jobs/x",
                                                   headers={"Authorization": "Bearer secret"})
    for target in ("https://elsewhere.test/api", "http://portal.test/api", "https://portal.test:8443/api"):
        with pytest.raises(rtseg_client.ClientError, match="another origin"):
            handler.redirect_request(request, None, 302, "Found", {}, target)
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://portal.test/api/v1/jobs/y")


@pytest.mark.parametrize("job_id", ["../../etc/passwd", "a/b", "", None, "x.zip"])
def test_job_ids_from_the_portal_are_validated(job_id, tmp_path):
    client = rtseg_client.RTsegClient("http://portal.test")
    with pytest.raises(rtseg_client.ClientError, match="Invalid job ID"):
        client.download(job_id, tmp_path / "x.zip")


def test_upload_refuses_a_file_that_shrinks(tmp_path):
    chunk = 1 << 16  # larger than the read buffer, so each chunk is read from the file
    path = tmp_path / "slice.dcm"
    path.write_bytes(b"x" * (4 * chunk))
    client = rtseg_client.RTsegClient("http://portal.test")

    def request(method, route, **kwargs):
        if method == "POST":
            return {"session_id": "session-1", "max_request_bytes": chunk}
        with path.open("r+b") as handle:  # the file shrinks after the first chunk
            handle.truncate(chunk)
        return {"received": kwargs["query"]["offset"] + len(kwargs["body"])}
    client._request = request
    with pytest.raises(rtseg_client.ClientError, match="changed size"):
        client.upload_files([path])


def test_non_json_reply_is_a_client_error(monkeypatch):
    client = rtseg_client.RTsegClient("http://portal.test")
    monkeypatch.setattr(client, "_open", lambda request: FakeResponse(b"<html>Sign in</html>"))
    with pytest.raises(rtseg_client.ClientError, match="non-JSON"):
        client.health()


def test_default_series_selection_follows_model_modalities():
    catalog = [{"name": "total", "modality": "CT"}, {"name": "pet_pair", "modalities": ["PT"]},
               {"name": "total_mr", "modality": "MR"}]
    assert rtseg_client.model_modalities(catalog, ["total", "pet_pair"]) == {"CT", "PT"}


def test_manifest_keeps_requested_models_notes_and_unreadable_headers(portal, synthetic_ct, monkeypatch):
    jid = "20260101-000000-00000000000000000002-abcd1234"
    job = portal.BASE / jid
    (job / "input").mkdir(parents=True)
    (job / "output/lnctvseg_nnunet_v1").mkdir(parents=True)
    write_rtstruct(synthetic_ct, job / "output/total_RTSTRUCT.dcm")
    (job / "output/lnctvseg_nnunet_v1/OMITTED_LABELS.txt").write_text("Omitted class 7 (12 voxels).\n")
    # An auto-mode Colab fallback narrows job.json models to the subset it sends to Colab.
    (job / "job.json").write_text(json.dumps({
        "id": jid, "status": "processing", "created": 1767225600.0, "models": ["lung_nodules"],
        "requested_models": ["total", "lung_nodules"]}))

    def unreadable(path, arcname):
        raise OSError("truncated header")
    monkeypatch.setattr(portal.result_manifest, "rtstruct_record", unreadable)
    portal.finalize_results(job, ["lung_nodules"])
    with zipfile.ZipFile(job / "rtstruct-results.zip") as archive:
        manifest = json.loads(archive.read(result_manifest.MANIFEST_NAME))
    jsonschema.validate(manifest, SCHEMA)
    assert manifest["job"]["models"] == ["total", "lung_nodules"]
    total, nodules = manifest["models"]
    assert (total["outcome"], "requested" in total) == ("rtstruct", False)
    assert total["rtstruct"]["roi_names"] == [] and "truncated header" in total["rtstruct"]["error"]
    assert nodules["outcome"] == "no_structures_found"
    assert "Omitted class 7 (12 voxels)." in manifest["notes"]


def test_directory_entries_with_content_and_unlisted_rtstructs_are_rejected(tmp_path):
    files = [{"path": "a.txt", "size": 1, "sha256": result_manifest.sha256_bytes(b"a")}]
    manifest = {"schema": rtseg_client.MANIFEST_SCHEMA, "schema_version": "1.0", "files": files,
                "models": [{"name": "total", "outcome": "rtstruct", "rtstruct": {"path": "hidden.dcm/"}}]}
    package = tmp_path / "dir.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("a.txt", b"a")
        archive.writestr(zipfile.ZipInfo("hidden.dcm/"), b"unchecked RTSTRUCT bytes")
        archive.writestr(rtseg_client.MANIFEST_NAME, json.dumps(manifest))
    with pytest.raises(rtseg_client.ManifestError) as raised:
        rtseg_client.verify_result_zip(package)
    assert "directory entry with content" in str(raised.value)
    assert "not a listed member" in str(raised.value)
