"""Synthetic source pixels -> RTSTRUCT -> viewer pixels, with no inference or patient data."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
from urllib.parse import quote
import zipfile

import cv2
from fastapi.testclient import TestClient
import nibabel as nib
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
import pytest
from rt_utils import RTStructBuilder

import viewer
from rtstruct_geometry import dicom_grid
from orientation_contract import restore_image


GEOMETRIES = [
    pytest.param([1, 0, 0, 0, 1, 0], [1, 1], False, False, "axial", id="axial-hfs"),
    pytest.param([-1, 0, 0, 0, -1, 0], [1, 1], False, False, "axial", id="flipped-both-axes"),
    pytest.param([-1, 0, 0, 0, 1, 0], [1, 1], True, False, "axial", id="feet-first-negative-normal"),
    pytest.param([1, 0, 0, 0, 1, 0], [2.3, 0.7], False, False, "axial", id="non-square-spacing"),
    pytest.param([1, 0, 0, 0, 1, 0], [1, 1], True, False, "axial", id="reversed-filenames"),
    pytest.param([-1, 0, 0, 0, -1, 0], [2.3, 0.7], True, True, "axial", id="flipped-gaps-reversed-anisotropic"),
    pytest.param([1, 0, 0, 0, 0, -1], [1.7, 0.6], True, False, "coronal", id="coronal-constant-z"),
    pytest.param([0, 1, 0, 0, 0, 1], [1.7, 0.6], True, False, "sagittal", id="sagittal-constant-z"),
    pytest.param([0.8, 0, 0.6, 0, 1, 0], [2.3, 0.7], True, False, "oblique", id="oblique"),
    pytest.param([1, 0, 0, 0, -1, 0], [.9375, .9375], False, False, "axial", id="flipped-column"),
    pytest.param([1, 0, 0, 0, -1, 0], [2.3, .7], True, True, "axial", id="flipped-column-gaps-anisotropic"),
    pytest.param([0, 1, 0, 0, 0, -1], [1, 1], True, False, "sagittal", id="sagittal-negative-normal"),
    pytest.param([0, 1, 0, 0, 0, -1], [1.7, .6], True, True, "sagittal", id="sagittal-negative-normal-gaps-anisotropic"),
]


def ready(client, url):
    """Wait only in tests that need a finished asynchronous overlay index."""
    deadline = time.monotonic() + 20
    while True:
        response = client.get(url)
        if response.status_code != 202 and (response.status_code != 200 or response.json().get("contours_state") != "building"):
            return response
        assert time.monotonic() < deadline, "Contour build did not finish"
        time.sleep(.005)


def make_series(template, root, iop=(1, 0, 0, 0, 1, 0), spacing=(1, 1), reverse=False,
                gaps=False, shape=(32, 48), count=6):
    root.mkdir(parents=True)
    base = pydicom.dcmread(next(template.iterdir()))
    base.SeriesInstanceUID = generate_uid()
    normal = np.cross(iop[:3], iop[3:])
    origin = np.array([31.25, -57.5, 12.0])
    positions = np.arange(count) * 2.0
    if gaps:
        positions[3:] += 4
    paths = []
    for k, position in enumerate(positions):
        ds = copy.deepcopy(base)
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta = base.file_meta.copy()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.ImageOrientationPatient = list(iop)
        ds.ImagePositionPatient = (origin + normal * position).tolist()
        ds.PixelSpacing = list(spacing)
        ds.SpacingBetweenSlices, ds.SliceThickness = 2, 2
        ds.Rows, ds.Columns = shape
        ds.InstanceNumber = count - k  # Deliberately contradict geometric order.
        ds.PatientPosition = "FFS" if normal[2] < 0 else "HFS"
        pixels = np.full(shape, k * 100, dtype=np.int16)
        pixels[3:19, 27:31] += 50
        pixels[15:19, 27:41] += 50
        ds.PixelData = pixels.tobytes()
        path = root / f"slice-{count-k if reverse else k:04d}.dcm"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def make_job(portal, source, name="viewer-test", status="complete"):
    job = portal.BASE / name
    job.mkdir(parents=True)
    shutil.copytree(source, job / "input")
    (job / "output").mkdir()
    (job / "job.json").write_text(json.dumps({"id": name, "created": time.time(), "status": status, "models": ["total"]}))
    return job


def make_upload(portal, source):
    upload = portal.BASE / "upload-viewer-test"
    shutil.copytree(source, upload / "input")
    data = portal.inspect_upload(upload / "input")
    (upload / "upload.json").write_text(json.dumps({"created": time.time(), **data}))
    key = data["series"][0]["key"]
    return upload, key, f"/api/v1/uploads/{upload.name}/series/{quote(key, safe='')}/viewer"


def make_mask(count=6, shape=(32, 48)):
    mask = np.zeros((*shape, count), dtype=bool)
    for k in (1, 3, 4):
        mask[3:19, 27:31, k] = True
        mask[15:19, 27:41, k] = True
    return mask


def make_rtstruct(source, output, mask=None):
    mask = make_mask() if mask is None else mask
    rt = RTStructBuilder.create_new(dicom_series_path=str(source))
    rt.add_roi(mask=mask, name="Asymmetric L", color=[17, 34, 51], approximate_contours=False)
    # rt-utils 1.2.7 swaps unequal spacings and assumes a uniform stack. For those
    # fixtures, retain its RTSTRUCT container but independently specify DICOM
    # contours from mask pixel centres and each source image's actual geometry.
    positions = [np.dot(ds.ImagePositionPatient, np.cross(ds.ImageOrientationPatient[:3],
                                                         ds.ImageOrientationPatient[3:])) for ds in rt.series_data]
    unequal_spacing = not np.isclose(*rt.series_data[0].PixelSpacing)
    irregular = len(positions) > 2 and not np.allclose(np.diff(positions), np.diff(positions)[0])
    if unequal_spacing or irregular:
        contours = []
        for k, ds in enumerate(rt.series_data):
            polygons, _ = cv2.findContours(mask[:, :, k].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for polygon in polygons:
                xy = polygon[:, 0, :]
                basis = np.array([np.array(ds.ImageOrientationPatient[:3]) * float(ds.PixelSpacing[1]),
                                  np.array(ds.ImageOrientationPatient[3:]) * float(ds.PixelSpacing[0])])
                contour = Dataset()
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour.NumberOfContourPoints = len(xy)
                contour.ContourData = (xy @ basis + np.array(ds.ImagePositionPatient)).ravel().tolist()
                reference = Dataset()
                reference.ReferencedSOPClassUID = ds.SOPClassUID
                reference.ReferencedSOPInstanceUID = ds.SOPInstanceUID
                contour.ContourImageSequence = [reference]
                contours.append(contour)
        rt.ds.ROIContourSequence[0].ContourSequence = contours
    rt.save(str(output))


def rasterize(paths, shape):
    result = np.zeros(shape, dtype=np.uint8)
    if paths:
        cv2.fillPoly(result, [np.rint(path).astype(np.int32) for path in paths], 1)
    return result.astype(bool)


@pytest.mark.parametrize("iop,spacing,reverse,gaps,orientation", GEOMETRIES)
def test_orientation_exact_mask_pixels(portal, synthetic_ct, tmp_path, iop, spacing, reverse, gaps, orientation):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    job = make_job(portal, source)
    mask = make_mask()
    output = job / "output/total_RTSTRUCT.dcm"
    # Use the actual inference reader, including its acquired-plane order and
    # average NIfTI spacing for the gapped stack. No synthetic RTSTRUCT fallback.
    model_input = tmp_path / "input.nii.gz"
    reader = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "dicom_series_to_nifti.py"),
                             str(source), str(model_input)], capture_output=True, text=True)
    assert reader.returncode == 0, reader.stdout + reader.stderr
    input_image = nib.load(model_input)
    prediction = tmp_path / "prediction.nii.gz"
    _, source_affine, _ = dicom_grid(source, input_image)
    model_mask, _ = restore_image(
        nib.Nifti1Image(mask.transpose(1, 0, 2).astype(np.uint8), source_affine),
        input_image.shape, input_image.affine)
    nib.save(model_mask, prediction)
    labels = tmp_path / "labels.json"
    labels.write_text('{"1": "Asymmetric L"}')
    converted = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "convert_validate_rtstruct.py"),
                                "--prediction", str(prediction), "--model-input", str(model_input),
                                "--dicom", str(source), "--output", str(output), "--labels", str(labels)],
                               capture_output=True, text=True)
    assert converted.returncode == 0, converted.stdout + converted.stderr
    validation = json.loads(converted.stdout.splitlines()[-1])
    assert validation["physical_roundtrip"]["Asymmetric L"]["roundtrip_dice"] == 1
    print("PRODUCTION_CONVERTER " + json.dumps(validation))
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    response = ready(client, base)
    assert response.status_code == 200, response.text
    info = response.json()
    assert info["series"]["orientation"] == orientation
    assert info["series"]["plane"] == orientation
    assert info["series"]["column_direction"] == iop[:3]
    assert info["series"]["row_direction"] == iop[3:]
    assert info["series"]["pixel_spacing"] == spacing
    roi = info["structure_sets"][0]["rois"][0]
    assert roi["slices"] == [1, 3, 4]
    if iop == [1, 0, 0, 0, 1, 0] and spacing == [1, 1] and not reverse:
        print("VIEWER_EXAMPLE " + json.dumps(info))
        print("CONTOUR_EXAMPLE " + json.dumps(client.get(base + "/contours/1").json()))
    for k in range(6):
        contours = client.get(f"{base}/contours/{k}").json()["rois"]
        assert np.array_equal(rasterize(contours[0]["paths"] if contours else [], mask.shape[:2]), mask[:, :, k])
        png = client.get(f"{base}/slices/{k}.png?center=300&width=1201")
        assert png.status_code == 200
        pixels = np.array(Image.open(io.BytesIO(png.content)))
        assert pixels[0, 0] == int(np.clip(((k * 100 - 299.5) / 1200 + 0.5) * 255, 0, 255))
    # Legacy prompt preview and upload viewer must share identical index and pixel coordinates.
    upload, key, upload_base = make_upload(portal, source)
    for k in range(6):
        old = client.get(f"/api/uploads/{upload.name}/preview", params={"series_key": key, "index": k})
        new = client.get(f"{upload_base}/slices/{k}.png")
        assert old.content == new.content
        assert old.headers["X-Slice-Index"] == str(k)
    upload_info = client.get(upload_base).json()
    assert upload_info["structure_sets"] == []
    assert upload_info["series"] == info["series"]
    print(f"orientation={orientation} iop={iop} spacing={spacing} reversed={reverse} gaps={gaps}: Dice=1.0, exact pixels, preview order identical")


@pytest.mark.skipif(importlib.util.find_spec("SimpleITK") is None, reason="Optional worker dependency; install SimpleITK to verify GDCM prompt ordering")
@pytest.mark.parametrize("iop,spacing,reverse,gaps,orientation", GEOMETRIES)
def test_prompt_nifti_order(portal, synthetic_ct, tmp_path, iop, spacing, reverse, gaps, orientation):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, iop, spacing, reverse, gaps)
    nifti = tmp_path / "prompt-input.nii.gz"
    converted = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "dicom_series_to_nifti.py"),
                                str(source), str(nifti)], capture_output=True, text=True)
    assert converted.returncode == 0, converted.stdout + converted.stderr
    image = nib.load(nifti)
    shape, affine, _ = dicom_grid(source, image)
    restored, _ = restore_image(image, shape, affine)
    data = restored.get_fdata()
    _, series = portal.viewer.series(source)
    for k, item in enumerate(series.slices):
        assert np.array_equal(data[:, :, k].T, pydicom.dcmread(item.path).pixel_array)


def test_uid_priority_distance_fallback_gaps_and_per_slice_origin(portal, synthetic_ct, tmp_path):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, spacing=[2.3, 0.7], reverse=True, gaps=True)
    job = make_job(portal, source)
    output = job / "output/model_RTSTRUCT.dcm"
    make_rtstruct(source, output)
    rt = pydicom.dcmread(output)
    roi = rt.ROIContourSequence[0]
    original = roi.ContourSequence[0]
    # No UID -> nearest actual plane; a missing plane must not acquire an overlay.
    del original.ContourImageSequence
    missing = copy.deepcopy(original)
    points = np.asarray(original.ContourData).reshape(-1, 3).copy()
    points[:, 2] = 18  # origin 12 + missing offset 6; nearest is 4, tolerance is 1.
    missing.ContourData = points.ravel().tolist()
    foreign = copy.deepcopy(original)
    ref = Dataset()
    ref.ReferencedSOPInstanceUID = generate_uid()
    foreign.ContourImageSequence = [ref]
    roi.ContourSequence.extend([missing, foreign])
    rt.save_as(output, enforce_file_format=True)
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    assert ready(client, base).json()["structure_sets"][0]["rois"][0]["slices"] == [1, 3, 4]
    assert len(client.get(base + "/contours/1").json()["rois"][0]["paths"]) == 1
    assert client.get(base + "/contours/2").json()["rois"] == []
    # Move one source slice within its plane and move its referenced contour with it.
    _, series = portal.viewer.series(job / "input")
    item = series.slices[3]
    ds = pydicom.dcmread(item.path)
    shift = np.array([7, -4.6, 0])
    ds.ImagePositionPatient = (np.asarray(ds.ImagePositionPatient) + shift).tolist()
    ds.save_as(item.path, enforce_file_format=True)
    rt = pydicom.dcmread(output)
    contour = rt.ROIContourSequence[0].ContourSequence[1]
    contour.ContourData = (np.asarray(contour.ContourData).reshape(-1, 3) + shift).ravel().tolist()
    rt.save_as(output, enforce_file_format=True)
    paths = ready(client, base + "/contours/3").json()["rois"][0]["paths"]
    assert np.array_equal(rasterize(paths, (32, 48)), make_mask()[:, :, 3])


def test_windows_rescale_monochrome_and_validation(portal, synthetic_ct, tmp_path):
    source = tmp_path / "source"
    paths = make_series(synthetic_ct, source)
    ds = pydicom.dcmread(paths[0])
    ds.RescaleSlope, ds.RescaleIntercept = 2, -100
    ds.WindowCenter, ds.WindowWidth = [0, 100], [201, 500]
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.save_as(paths[0], enforce_file_format=True)
    job = make_job(portal, source)
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    info = client.get(base).json()["series"]
    assert info["default_window"] == "dicom"
    assert {p["id"] for p in info["window_presets"]} == {"dicom", "soft-tissue", "lung", "bone", "brain"}
    png = client.get(base + "/slices/0.png")
    pixels = np.array(Image.open(io.BytesIO(png.content)))
    expected = 255 - np.clip((((ds.pixel_array * 2 - 100) + 0.5) / 200 + 0.5) * 255, 0, 255).astype(np.uint8)
    assert np.array_equal(pixels, expected)
    assert Image.open(io.BytesIO(png.content)).mode == "L"
    assert png.headers["Cache-Control"] == "no-store"
    for query in ("window=missing", "center=1", "width=1", "center=1&width=0", "center=nan&width=100", "center=1&width=inf", "window=lung&center=1&width=100"):
        assert client.get(base + "/slices/0.png?" + query).status_code == 422
    assert client.get(base + "/slices/0.png?center=0&width=1").status_code == 200
    for index in (-1, 6):
        assert client.get(f"{base}/slices/{index}.png").status_code == 404
        assert client.get(f"{base}/contours/{index}").status_code == 404
    assert client.get(base + "/slices/nan.png").status_code == 422


@pytest.mark.parametrize("modality", ["MR", "PT"])
def test_non_ct_window_is_stable_across_slices(portal, synthetic_ct, tmp_path, modality):
    source = tmp_path / "source"
    paths = make_series(synthetic_ct, source)
    for path in paths:
        ds = pydicom.dcmread(path)
        ds.Modality = modality
        ds.save_as(path, enforce_file_format=True)
    job = make_job(portal, source)
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    info = client.get(base).json()["series"]
    assert info["modality"] == modality
    assert info["default_window"] == "auto"
    before = info["window_presets"]
    assert client.get(base + "/slices/0.png").status_code == 200
    assert client.get(base + "/slices/5.png").status_code == 200
    assert client.get(base).json()["series"]["window_presets"] == before


@pytest.mark.parametrize("mutation,expected", [("off-plane", 422), ("nonplanar", 422), ("open", 422), ("foreign-frame", 200)])
def test_invalid_contours_and_foreign_frame(portal, synthetic_ct, tmp_path, mutation, expected):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    job = make_job(portal, source)
    output = job / "output/total_RTSTRUCT.dcm"
    make_rtstruct(source, output)
    rt = pydicom.dcmread(output)
    contour = rt.ROIContourSequence[0].ContourSequence[0]
    if mutation == "off-plane":
        points = np.asarray(contour.ContourData).reshape(-1, 3)
        points[:, 2] += 3
        contour.ContourData = points.ravel().tolist()
    elif mutation == "nonplanar":
        contour.ContourData[2] += 0.5
    elif mutation == "open":
        contour.ContourGeometricType = "OPEN_PLANAR"
    else:
        rt.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = generate_uid()
    rt.save_as(output, enforce_file_format=True)
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    assert client.get(base).status_code == 200  # Invalid overlays do not block images.
    response = ready(client, base + "/contours/1")
    assert response.status_code == expected, response.text
    if mutation == "foreign-frame":
        assert ready(client, base).json()["structure_sets"][0]["rois"][0]["slices"] == []


def test_multiple_structures_cache_invalidation_and_report(portal, synthetic_ct, tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    job = make_job(portal, source)
    output = job / "output/total_RTSTRUCT.dcm"
    make_rtstruct(source, output)
    nested = job / "output/nested"
    nested.mkdir()
    other = nested / "second.bin"  # Discover RTSTRUCT by SOP class, not extension.
    shutil.copyfile(output, other)
    rt = pydicom.dcmread(other)
    del rt.ROIContourSequence[0].ROIDisplayColor
    rt.save_as(other, enforce_file_format=True)
    calls = []
    read = pydicom.dcmread
    def counted(path, *args, **kwargs):
        calls.append(Path(path))
        return read(path, *args, **kwargs)
    monkeypatch.setattr(viewer.pydicom, "dcmread", counted)
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    info = ready(client, base).json()
    assert len(info["structure_sets"]) == 2
    rois = [s["rois"][0] for s in info["structure_sets"]]
    assert len({r["key"] for r in rois}) == 2
    assert {r["color"] for r in rois} == {"#112233", viewer.PALETTE[0]}
    assert not info["report"]
    for _ in range(4):
        assert len(client.get(base + "/contours/1").json()["rois"]) == 2
        assert client.get(base).status_code == 200
    assert len(calls) == 8  # Six headers once and each RTSTRUCT once.
    rt.StructureSetROISequence[0].ROIName = "Changed ROI"
    rt.save_as(other, enforce_file_format=True)
    changed = ready(client, base).json()
    assert any(s["rois"][0]["name"] == "Changed ROI" for s in changed["structure_sets"])
    assert calls.count(output) == 1 and calls.count(other) == 2
    report = job / "output/report"
    report.mkdir()
    (report / (portal.REPORT_TASK + ".json")).write_text("{}")
    assert client.get(base).json()["report"]


@pytest.mark.parametrize("mutation,message", [("multiframe", "Multi-frame"), ("spacing", "spacing"), ("orientation", "orientation"), ("duplicate", "coincident")])
def test_unsupported_geometry_is_explicit(portal, synthetic_ct, tmp_path, mutation, message):
    source = tmp_path / "source"
    paths = make_series(synthetic_ct, source)
    ds = pydicom.dcmread(paths[1])
    if mutation == "multiframe":
        ds.NumberOfFrames = 2
    elif mutation == "spacing":
        ds.PixelSpacing = [0, 1]
    elif mutation == "orientation":
        ds.ImageOrientationPatient = [-1, 0, 0, 0, -1, 0]
    else:
        ds.ImagePositionPatient = pydicom.dcmread(paths[0]).ImagePositionPatient
    ds.save_as(paths[1], enforce_file_format=True)
    job = make_job(portal, source)
    response = TestClient(portal.app).get(f"/api/v1/jobs/{job.name}/viewer")
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_retention_deletion_and_path_boundaries(portal, synthetic_ct, tmp_path):
    job = make_job(portal, synthetic_ct, status="processing")
    # Report-only completion must also preserve local images, without adding them to the ZIP.
    report = job / "output/report"
    report.mkdir()
    (report / (portal.REPORT_TASK + ".json")).write_text('{"status": "generated"}')
    (job / "input_ct").mkdir()
    portal.finalize_results(job)
    assert (job / "input").is_dir()
    assert not (job / "input_ct").exists()
    with zipfile.ZipFile(job / "rtstruct-results.zip") as archive:
        assert not any(name.startswith("input/") for name in archive.namelist())
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}"
    assert client.get(base + "/viewer").status_code == 200
    assert client.delete(base).json() == {"id": job.name, "deleted": True}
    assert not portal.viewer.series_cache.items and not portal.viewer.overlay_cache.items
    assert not job.exists()
    assert client.get(base + "/viewer").status_code == 404
    assert client.delete(base).status_code == 404
    expired = make_job(portal, synthetic_ct, name="expired")
    portal.save_metadata(expired, created=time.time() - (portal.RETENTION_HOURS + 1) * 3600)
    assert client.get("/api/v1/jobs/expired/viewer").status_code == 404
    portal.cleanup_once()
    assert not expired.exists()
    for invalid in ("bad_name", "bad.name", "..%2Foutside", "upload-viewer-test", "session-test", "a" * 256):
        assert client.get(f"/api/v1/jobs/{invalid}/viewer").status_code == 404
        assert client.delete(f"/api/v1/jobs/{invalid}").status_code == 404
    outside = tmp_path / "outside"
    outside.mkdir()
    (portal.BASE / "linked").symlink_to(outside, target_is_directory=True)
    assert client.get("/api/v1/jobs/linked/viewer").status_code == 404
    assert client.delete("/api/v1/jobs/linked").status_code == 404
    assert outside.exists()
    upload, key, upload_base = make_upload(portal, synthetic_ct)
    metadata = json.loads((upload / "upload.json").read_text())
    metadata["series"][0]["files"] = ["../../outside/secret.dcm"]
    (upload / "upload.json").write_text(json.dumps(metadata))
    assert client.get(upload_base).status_code == 404


@pytest.mark.parametrize("status", ["queued", "transferring", "processing", "retrieving", "unknown"])
def test_delete_rejects_active_jobs(portal, synthetic_ct, status):
    job = make_job(portal, synthetic_ct, status=status)
    assert TestClient(portal.app).delete(f"/api/v1/jobs/{job.name}").status_code == 409
    assert job.exists()


@pytest.mark.parametrize("mode,routed", [("local", False), ("ssh", False), ("auto", True), ("colab", False)])
def test_modes_and_remote_delete(portal, synthetic_ct, monkeypatch, mode, routed):
    job = make_job(portal, synthetic_ct, status="failed")
    monkeypatch.setattr(portal, "WORKER_MODE", mode)
    monkeypatch.setattr(portal, "COLAB_ORCHESTRATOR", "worker")
    monkeypatch.setattr(portal, "REMOTE_GPU_TASKS", {"total"} if routed else set())
    calls = []
    monkeypatch.setattr(portal, "remove_worker_job", lambda *args: calls.append(args))
    client = TestClient(portal.app)
    assert client.get(f"/api/v1/jobs/{job.name}/viewer").status_code == 200
    assert client.delete(f"/api/v1/jobs/{job.name}").status_code == 200
    assert calls == ([(job, f"{portal.REMOTE_BASE}/{job.name}")] if mode in {"ssh", "colab"} or routed else [])


def test_remote_delete_failure_keeps_local_job(portal, synthetic_ct, monkeypatch):
    job = make_job(portal, synthetic_ct, status="failed")
    portal.save_metadata(job, remote_results_retained=True)
    def fail(*args):
        raise subprocess.TimeoutExpired("ssh", 30)
    monkeypatch.setattr(portal, "remove_worker_job", fail)
    assert TestClient(portal.app).delete(f"/api/v1/jobs/{job.name}").status_code == 502
    assert job.exists()


def test_512_performance(portal, synthetic_ct, tmp_path):
    source = tmp_path / "source"
    make_series(synthetic_ct, source, shape=(512, 512), count=96)
    job = make_job(portal, source)
    make_rtstruct(source, job / "output/total_RTSTRUCT.dcm", make_mask(96, (512, 512)))
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    start = time.perf_counter()
    assert client.get(base).status_code == 200
    timings = {"cold_metadata_seconds": time.perf_counter() - start}
    ready(client, base + "/contours/3")
    for name, endpoint in (("png", "/slices/3.png"), ("contours", "/contours/3"), ("range32", "/contours?start=0&count=32")):
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            assert client.get(base + endpoint).status_code == 200
            samples.append(time.perf_counter() - start)
        timings[name + "_median_seconds"] = float(np.median(samples))
        timings[name + "_max_seconds"] = max(samples)
        assert np.median(samples) < 1
    print("VIEWER_PERFORMANCE " + json.dumps(timings, sort_keys=True))


def test_background_single_flight_range_and_replacement(portal, synthetic_ct, tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    job = make_job(portal, source)
    output = job / "output/model_RTSTRUCT.dcm"
    make_rtstruct(source, output)
    entered, release = threading.Event(), threading.Event()
    calls, read = [], viewer.read_structure

    def slow(path):
        calls.append(path)
        entered.set()
        assert release.wait(10)
        return read(path)

    monkeypatch.setattr(viewer, "read_structure", slow)
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}/viewer"
    try:
        start = time.perf_counter()
        first = client.get(base)
        elapsed = time.perf_counter() - start
        # The parser is held for up to 10 s; well under that proves metadata did not wait for it,
        # with headroom for a heavily loaded test host.
        assert elapsed < 5
        assert first.status_code == 200 and first.json()["contours_ready"] is False
        assert entered.wait(2)
        for _ in range(4):
            assert client.get(base).json()["contours_state"] == "building"
            for endpoint in ("/contours/1", "/contours?start=0&count=6"):
                pending = client.get(base + endpoint)
                assert pending.status_code == 202
                assert pending.headers["Retry-After"] == "1"
                assert pending.headers["Cache-Control"] == "no-store"
                assert pending.json()["contours_state"] == "building"
        assert client.get(base + "/slices/1.png").status_code == 200
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as pool:
            assert all(r.json()["contours_state"] == "building" for r in pool.map(lambda _: client.get(base), range(8)))
        assert len(calls) == 1
        print("BACKGROUND_START " + json.dumps({"metadata_seconds_with_parser_stalled": elapsed, "parallel_metadata_requests": 8, "parser_calls": len(calls)}))
    finally:
        release.set()
    info = ready(client, base).json()
    assert info["contours_ready"] is True
    batch = client.get(base + "/contours?start=0&count=6")
    assert batch.status_code == 200 and batch.headers["Cache-Control"] == "no-store"
    assert batch.json()["start"] == 0
    for k in range(6):
        assert batch.json()["slices"][str(k)] == client.get(base + f"/contours/{k}").json()["rois"]
    for query, status in (("start=-1&count=2", 422), ("start=0&count=0", 422), ("start=0&count=33", 422),
                          ("start=5&count=2", 404), ("start=6&count=1", 404), ("start=0", 422)):
        assert client.get(base + "/contours?" + query).status_code == status
    rt = pydicom.dcmread(output)
    rt.StructureSetROISequence[0].ROIName = "Replaced"
    rt.save_as(output, enforce_file_format=True)
    assert ready(client, base).json()["structure_sets"][0]["rois"][0]["name"] == "Replaced"
    assert len(calls) == 2


def test_deletion_during_background_build_does_not_repopulate(portal, synthetic_ct, tmp_path, monkeypatch):
    job = make_job(portal, synthetic_ct)
    make_rtstruct(job / "input", job / "output/model_RTSTRUCT.dcm", np.ones((32, 32, 12), dtype=bool))
    entered, release = threading.Event(), threading.Event()
    read = viewer.read_structure

    def slow(path):
        result = read(path)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(viewer, "read_structure", slow)
    client = TestClient(portal.app)
    base = f"/api/v1/jobs/{job.name}"
    client.get(base + "/viewer")
    assert entered.wait(3)
    from concurrent.futures import ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(1) as pool:
            deleted = pool.submit(client.delete, base)
            deadline = time.monotonic() + 3
            while job.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert not job.exists()
            release.set()
            assert deleted.result(3).status_code == 200
    finally:
        release.set()
    assert not portal.viewer.overlay_cache.items
    assert not portal.viewer.rt_cache.items
    assert not portal.viewer.series_cache.items
    assert client.get(base + "/viewer").status_code == 404


@pytest.mark.parametrize("implicit", [False, True])
def test_raw_contour_coordinates_equal_pydicom(synthetic_ct, tmp_path, implicit):
    output = tmp_path / "structure.dcm"
    make_rtstruct(synthetic_ct, output, np.ones((32, 32, 12), dtype=bool))
    ds = pydicom.dcmread(output)
    strings = ["1.25E+1", "-2.5e-2", "0", "+3.0000", "-.125", "2E+1", "1", "2", "3.25"]
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    contour.ContourData = strings
    contour.NumberOfContourPoints = 3
    if implicit:
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.save_as(output, enforce_file_format=True)
    reference = np.asarray(pydicom.dcmread(output).ROIContourSequence[0].ContourSequence[0].ContourData).reshape(-1, 3)
    raw_ds = pydicom.dcmread(output)
    raw_contour = raw_ds.ROIContourSequence[0].ContourSequence[0]
    element = raw_contour.get_item(0x30060050)
    assert isinstance(element, pydicom.dataelem.RawDataElement)
    assert element.value.endswith(b" ")  # Even-length DICOM DS space padding.
    assert np.array_equal(viewer.contour_points(raw_contour), reference)
    assert isinstance(raw_contour.get_item(0x30060050), pydicom.dataelem.RawDataElement)
    # Also support a dataset whose element was already converted by a caller.
    converted = pydicom.dcmread(output).ROIContourSequence[0].ContourSequence[0]
    _ = converted.ContourData
    assert np.array_equal(viewer.contour_points(converted), reference)


@pytest.mark.parametrize("raw", [b"1\\2\\3garbage", b"1\\\\3", b"NaN\\2\\3", b"1\\2", b"1\\2\\3\\"])
def test_raw_contour_rejects_malformed_values(raw):
    from pydicom.dataelem import RawDataElement
    ds = Dataset()
    ds._dict[0x30060050] = RawDataElement(0x30060050, "DS", len(raw), raw, 0, False, True)
    with pytest.raises(ValueError):
        viewer.contour_points(ds)


def test_float32_window_pixels_equal_original_float64():
    rng = np.random.default_rng(765)
    source = np.arange(-32768, 32768, dtype=np.int16).reshape(256, 256)
    windows = [(p["center"], p["width"]) for p in viewer.CT_WINDOWS]
    windows += [(300, 1201), (0, 1), (40, 1.001), (1e12, 1e8)]
    windows += list(zip(rng.uniform(-1000, 1000, 24), rng.uniform(1, 3000, 24)))
    for slope, intercept in ((1, 0), (2, -100), (.7139, -1024.7), (-1.2345, 3.14)):
        exact = source.astype(np.float64) * slope + intercept
        for center, width in windows:
            expected = (np.where(exact > center - .5, 255, 0) if width == 1 else
                        np.clip(((exact - (center - .5)) / (width - 1) + .5) * 255, 0, 255)).astype(np.uint8)
            assert np.array_equal(viewer.window_pixels(source, center, width, slope, intercept), expected), (center, width, slope, intercept)


@pytest.mark.parametrize("dtype", ["<i2", ">i2", "<u2", ">u2", "i1", "u1"])
def test_window_lookup_signed_unsigned_and_endian(dtype):
    kind = np.dtype(dtype)
    limits = np.iinfo(kind)
    source = np.resize(np.array([limits.min, 0, 1, 30, 127, limits.max], dtype=kind), (512, 512)).astype(kind)
    assert source.dtype == kind
    exact = source.astype(np.float64) * .7139 - 1000.7
    expected = np.clip(((exact - (-1000.5)) / 399 + .5) * 255, 0, 255).astype(np.uint8)
    assert np.array_equal(viewer.window_pixels(source, -1000, 400, .7139, -1000.7), expected)
