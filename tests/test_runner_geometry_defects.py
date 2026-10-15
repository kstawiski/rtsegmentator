"""Production runner regressions, exercised without model runtimes or GPUs."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_backend_defects import catalog_with, expose, submit
from test_readonly_bundle import mirror_function
from test_remote_gpu_tasks import job_at
from test_rtstruct_geometry import ROOT, convert, inputs
from test_viewer import make_mask, make_series


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("only_tiny", [False, True])
def test_single_voxel_roi_reaches_result_archive(portal, monkeypatch, synthetic_ct, tmp_path, independent, only_tiny):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    mask = make_mask().transpose(1, 0, 2).astype(np.uint8)
    if only_tiny:
        mask[:] = 0
    mask[1, 2, 2] = 2
    image_path, prediction, _ = inputs(source, tmp_path, mask)
    if independent and not only_tiny:
        # Keep the single-voxel ROI separate from the larger ROI.
        tiny = tmp_path / "tiny.nii.gz"
        image = nib.load(prediction)
        nib.save(nib.Nifti1Image((mask == 2).astype(np.uint8), image.affine), tiny)
        nib.save(nib.Nifti1Image((mask == 1).astype(np.uint8), image.affine), prediction)
        mapping = tmp_path / "masks.json"
        mapping.write_text(json.dumps({"CTV": str(prediction), "CTVT": str(tiny)}))
        output = tmp_path / "model_RTSTRUCT.dcm"
        result = subprocess.run([sys.executable, str(ROOT / "convert_masks_validate_rtstruct.py"),
                                 "--masks-json", str(mapping), "--model-input", str(image_path),
                                 "--dicom", str(source), "--output", str(output)], capture_output=True, text=True)
    else:
        labels = {"2": "CTVT"} if only_tiny else {"1": "CTV", "2": "CTVT"}
        result, output = convert(source, tmp_path, image_path, prediction, independent, labels)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RTSTRUCT_CONVERSION_NOTE" not in result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["rois"] == (1 if only_tiny else 2)
    assert report["omitted_rois"] == []
    assert report["physical_roundtrip"]["CTVT"] == {
        "expected_voxels": 1, "roundtrip_voxels": 1,
        "intersection_voxels": 1, "roundtrip_dice": 1.0}
    if not only_tiny:
        assert report["physical_roundtrip"]["CTV"]["roundtrip_dice"] == 1

    job = job_at(portal, ["cervix_cbct"])
    def run(*args):
        (job / "cervix_cbct.log").write_text(result.stderr)
        shutil.copyfile(output, job / "output/cervix_cbct_RTSTRUCT.dcm")
    monkeypatch.setattr(portal, "run_local_task", run)
    portal.process(job)
    metadata = portal.metadata(job)
    assert metadata["status"] == "complete"
    assert not metadata.get("conversion_notes")
    assert not metadata.get("no_structures_found")
    with zipfile.ZipFile(job / "rtstruct-results.zip") as archive:
        assert "CONVERSION_NOTES.txt" not in archive.namelist()
        assert "NO_STRUCTURES_FOUND.txt" not in archive.namelist()
        assert any(name.endswith("_RTSTRUCT.dcm") for name in archive.namelist())


def test_polygon_and_isolated_voxel_both_pass_unchanged_gate(synthetic_ct, tmp_path):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    mask = np.zeros((48, 32, 6), dtype=np.uint8)
    mask[2:4, 2:4, 1] = 1  # A valid four-pixel polygon.
    mask[5, 2, 3] = 1      # 20% of the ROI uses a single-point contour.
    image_path, prediction, _ = inputs(source, tmp_path, mask)
    result, _ = convert(source, tmp_path, image_path, prediction)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["physical_roundtrip"]["Asymmetric L"] == {
        "expected_voxels": 5, "roundtrip_voxels": 5,
        "intersection_voxels": 5, "roundtrip_dice": 1.0}
    assert "RTSTRUCT_CONVERSION_NOTE" not in result.stderr


def test_empty_sct_prediction_completes_with_note(portal, monkeypatch, synthetic_ct, tmp_path):
    source = tmp_path / "source"
    make_series(synthetic_ct, source)
    image_path, prediction, _ = inputs(source, tmp_path, np.zeros((48, 32, 6), dtype=np.uint8))
    combined, labels = tmp_path / "combined.nii.gz", tmp_path / "sct-labels.json"
    result = subprocess.run([sys.executable, str(ROOT / "combine_sct_outputs.py"),
                             "--glob", str(prediction), "--output", str(combined),
                             "--labels", str(labels), "--output-names", '{"prediction":"Lesion"}'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not nib.load(combined).get_fdata().any()
    converted, _ = convert(source, tmp_path, image_path, combined, labels=json.loads(labels.read_text()))
    assert converted.returncode != 0 and "empty prediction: no RTSTRUCT created" in converted.stderr
    job = job_at(portal, ["sct_lesion_ms"])
    def run(*args):
        raise RuntimeError(converted.stderr)
    monkeypatch.setattr(portal, "run_local_task", run)
    portal.process(job)
    metadata = portal.metadata(job)
    assert metadata["status"] == "complete"
    assert metadata["no_structures_found"] == ["sct_lesion_ms"]
    with zipfile.ZipFile(job / "rtstruct-results.zip") as archive:
        assert "sct_lesion_ms" in archive.read("NO_STRUCTURES_FOUND.txt").decode()
    # Missing SCT output remains a model error, not a successful empty result.
    missing = subprocess.run([sys.executable, str(ROOT / "combine_sct_outputs.py"),
                             "--glob", str(tmp_path / "missing*.nii.gz"), "--output", str(combined),
                             "--labels", str(labels)], capture_output=True, text=True)
    assert missing.returncode != 0 and not portal.is_empty_prediction(RuntimeError(missing.stderr))


def test_hd_ctbet_dispatch_writes_training_log_in_job_mirror(tmp_path):
    # Execute the actual dispatcher branch with only inference replaced.
    tree = ast.parse((ROOT / "run_task.py").read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "task == 'hd_ctbet_ct'")
    bundle = tmp_path / "bundle"
    fold = bundle / "runtime-homes/hd-ctbet/hd-bet_params/fold_0"
    fold.mkdir(parents=True)
    checkpoint = fold / "model.model"
    checkpoint.write_bytes(b"synthetic immutable weights")
    checkpoint.chmod(0o444)
    fold.chmod(0o555)
    job = tmp_path / "job"
    stale_home = job / "external/hd_ctbet_ct/hd-ctbet-home"
    stale_home.mkdir(parents=True)
    (stale_home / "previous-attempt.txt").write_text("left by a failed attempt")
    output = job / "output.dcm"
    calls = []
    def run(command, env=None):
        calls.append(command)
        if command[0].endswith("/hd-ctbet"):
            writable = Path(env["HOME"]) / "hd-bet_params/fold_0"
            assert writable.is_relative_to(job)
            assert not writable.is_symlink()
            assert not (Path(env["HOME"]) / "previous-attempt.txt").exists()
            assert (writable / "model.model").read_bytes() == checkpoint.read_bytes()
            log = writable / "training_log_synthetic.txt"
            log.write_text("synthetic log")
            assert log.read_text() == "synthetic log"
            assert not (fold / log.name).exists()
            (job / "external/hd_ctbet_ct/brain_mask.nii.gz").touch()
    namespace = dict(job=job, task="hd_ctbet_ct", output=output, APP_ROOT=ROOT, MODEL_ROOT=bundle,
                     RUNTIME_ROOT=tmp_path / "runtime", os=os, sys=sys, shutil=shutil,
                     run=run, writable_mirror=mirror_function())
    try:
        exec(compile(ast.Module(body=branch.body, type_ignores=[]), "run_task.py", "exec"), namespace)
        assert len(calls) == 3
        assert checkpoint.read_bytes() == b"synthetic immutable weights"
        assert not (job / "external/hd_ctbet_ct").exists()
    finally:
        fold.chmod(0o755)


@pytest.mark.parametrize("points", ["positive_points", "negative_points"])
def test_segvol_box_and_points_rejected_by_shared_contract(portal, monkeypatch, synthetic_ct, points):
    expose(portal, monkeypatch, ["segvol_prompted_ct"])
    contract = portal.model_details("segvol_prompted_ct")["prompt_contract"]
    assert contract["allow_box_with_points"] is False
    prompt = {"text": "spleen", "box": [[2, 3, 0], [8, 9, 0]], points: [[4, 5, 0]]}
    response = submit(TestClient(portal.app), "assessed", next(synthetic_ct.iterdir()),
                      "segvol_prompted_ct", prompt)
    assert response.status_code == 400
    assert "cannot combine box and point prompts" in response.json()["detail"]
    assert not list(portal.BASE.glob("*/job.json"))


def test_segvol_prompt_contract_survives_catalog_gates(portal, monkeypatch):
    details = portal.model_details("segvol_prompted_ct")
    item = catalog_with(portal, monkeypatch, details, {"ready": True, "orientation_ready": True})
    assert item["prompt_contract"] == {"allow_box_with_points": False}


@pytest.mark.parametrize("prompt", [{"text": "spleen"}, {"positive_points": [[4, 5, 0]]},
    {"text": "spleen", "positive_points": [[4, 5, 0]], "negative_points": [[8, 9, 0]]},
    {"text": "spleen", "box": [[2, 3, 0], [8, 9, 0]]}])
def test_segvol_separate_prompt_modes_stay_valid(portal, monkeypatch, synthetic_ct, prompt):
    expose(portal, monkeypatch, ["segvol_prompted_ct"])
    response = submit(TestClient(portal.app), "assessed", next(synthetic_ct.iterdir()),
                      "segvol_prompted_ct", prompt)
    assert response.status_code == 200, response.text
