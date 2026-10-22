"""Offline checks for the nine CADS research-weight tasks. No GPU and no network."""
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).parents[1]


def catalog():
    return json.loads((ROOT / "catalog/cads-research.json").read_text())


def test_research_catalogue_licence_and_dataset_spellings():
    import app
    tasks = catalog()
    assert len(tasks) == 9
    assert set(tasks) == set(app.CADS_RESEARCH_TASKS) <= set(app.EXTERNAL_TASKS)
    spellings = {
        "cads_research_556_rt_oars": "right psoas major",
        "cads_research_557_brain_tissues": "white matter",
    }
    for name, spec in tasks.items():
        assert spec["enabled"] is True and spec["requires_preflight"] is True
        assert spec["status"] == "acceptance_pending"
        assert spec["noncommercial"] is True
        assert spec["license"] == "CC BY-NC-SA 4.0"
        assert spec["weights_license"] == "CC BY-NC-SA 4.0"
        assert spec["code_license"] == "Apache-2.0"
        assert "Owner accepted these terms on 2026-10-22" in spec["license_note"]
        assert "CC BY-NC-SA 4.0" in spec["license_note"]
        assert "nearly identical" in spec["description"]
        assert spec["reference"] == "https://github.com/murong-xu/CADS"
        assert "licensed" not in spec
        assert name in app.EXTERNAL_TASKS
        assert app.EXTERNAL_TASKS[name]["noncommercial"] is True
    assert spellings["cads_research_556_rt_oars"] in tasks["cads_research_556_rt_oars"]["labels"].values()
    assert "right_psoas_major" not in tasks["cads_research_556_rt_oars"]["labels"].values()
    assert spellings["cads_research_557_brain_tissues"] in tasks["cads_research_557_brain_tissues"]["labels"].values()
    structures = json.loads((ROOT / "catalog/structures.json").read_text())
    assert structures["cads_research_556_rt_oars"]["names"].count("right psoas major") == 1
    assert structures["cads_research_557_brain_tissues"]["names"].count("eye balls") == 1
    assert structures["cads_research_551_organs"]["kind"] == "fixed"
    assert structures["cads_research_551_organs"]["count"] == 17


def _plant(root, license_name, task_ids):
    from cads_variants import METADATA_FILES, checkpoint_path, metadata_path
    for task_id in task_ids:
        path = checkpoint_path(root, license_name, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        if license_name == "research":
            for name in METADATA_FILES:
                meta = metadata_path(root, license_name, task_id, name)
                meta.parent.mkdir(parents=True, exist_ok=True)
                meta.write_bytes(b"metadata")
    package = root / "sources/CADS/cads"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")


def _allow_synthetic_metadata(monkeypatch):
    def synthetic(model_root, task_id, app_root):
        from cads_variants import DATASETS, METADATA_FILES, metadata_path, required_task_ids
        found = {}
        for tid in required_task_ids(task_id):
            dataset = DATASETS[tid]
            found[dataset] = {}
            for name in METADATA_FILES:
                path = metadata_path(model_root, "research", tid, name)
                if not path.is_file():
                    raise SystemExit(f"CADS research metadata missing: {dataset}/{name}; refusing to launch")
                found[dataset][name] = "synthetic"
        return found
    monkeypatch.setattr("cads_variants.research_metadata_hashes", synthetic)


def _dispatch(monkeypatch, tmp_path, synthetic_ct, task, calls, envs):
    job = tmp_path / "job"
    (job / "output").mkdir(parents=True)
    (job / "input").symlink_to(synthetic_ct)
    monkeypatch.setenv("RTSEG_APP_ROOT", str(ROOT))
    monkeypatch.setenv("RTSEG_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RTSEG_RUNTIME_ROOT", str(tmp_path / "runtimes"))
    _allow_synthetic_metadata(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_task.py", str(job), task])
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            envs.append(kwargs.get("env") or {})
            if "-license" in command:
                out = Path(command[command.index("-out") + 1])
                task_id = command[command.index("-task") + 1]
                dest = out / "input" / f"input_part_{task_id}.nii.gz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"nii")
            if command[1].endswith("convert_validate_rtstruct.py"):
                shutil.copyfile(next(synthetic_ct.iterdir()), command[command.index("--output") + 1])

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, "tempdir", f"/proc/self/fd/{fd}")
        runpy.run_path(str(ROOT / "run_task.py"), run_name="__main__")
    finally:
        os.close(fd)


@pytest.mark.parametrize("task,task_id,extra", [
    ("cads_research_551_organs", "551", ()),
    ("cads_research_558_headneck_oars", "558", (552, 553)),
    ("cads_open_551_organs", "551", ()),
])
def test_dispatch_selects_the_requested_weight_variant(tmp_path, monkeypatch, synthetic_ct, task, task_id, extra):
    license_name = "research" if task.startswith("cads_research_") else "open"
    _plant(tmp_path / "models", license_name, (int(task_id), *extra))
    calls, envs = [], []
    _dispatch(monkeypatch, tmp_path, synthetic_ct, task, calls, envs)
    predict = next(command for command in calls if "-license" in command)
    assert predict[predict.index("-license") + 1] == license_name
    assert predict[predict.index("-task") + 1] == task_id
    assert predict[1:3] == ["-m", "cads.scripts.predict_images"]
    assert envs[calls.index(predict)]["CADS_WEIGHTS_PATH"].endswith("/weights/cads")
    assert not (tmp_path / "job/external" / task).exists()


def test_research_dispatch_keeps_dataset_json_spacing(tmp_path, monkeypatch, synthetic_ct):
    _plant(tmp_path / "models", "research", (556,))
    seen = {}

    class Process:
        def __init__(self, command, **kwargs):
            if "-license" in command:
                work = Path(command[command.index("-out") + 1]).parent
                seen["labels"] = json.loads((work / "labels.json").read_text())
                task_id = command[command.index("-task") + 1]
                dest = Path(command[command.index("-out") + 1]) / "input" / f"input_part_{task_id}.nii.gz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"nii")
            if command[1].endswith("convert_validate_rtstruct.py"):
                shutil.copyfile(next(synthetic_ct.iterdir()), command[command.index("--output") + 1])

        def wait(self):
            return 0

        def poll(self):
            return 0

    job = tmp_path / "job"
    (job / "output").mkdir(parents=True)
    (job / "input").symlink_to(synthetic_ct)
    monkeypatch.setenv("RTSEG_APP_ROOT", str(ROOT))
    monkeypatch.setenv("RTSEG_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RTSEG_RUNTIME_ROOT", str(tmp_path / "runtimes"))
    _allow_synthetic_metadata(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_task.py", str(job), "cads_research_556_rt_oars"])
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(subprocess, "Popen", Process)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, "tempdir", f"/proc/self/fd/{fd}")
        runpy.run_path(str(ROOT / "run_task.py"), run_name="__main__")
    finally:
        os.close(fd)
    assert seen["labels"]["12"] == "right psoas major"


def test_missing_checkpoint_refuses_to_download(tmp_path, monkeypatch, synthetic_ct):
    _plant(tmp_path / "models", "research", (558,))
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append(args) or (_ for _ in ()).throw(AssertionError("spawned")))
    job = tmp_path / "job"
    (job / "output").mkdir(parents=True)
    (job / "input").symlink_to(synthetic_ct)
    monkeypatch.setenv("RTSEG_APP_ROOT", str(ROOT))
    monkeypatch.setenv("RTSEG_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RTSEG_RUNTIME_ROOT", str(tmp_path / "runtimes"))
    monkeypatch.setattr(sys, "argv", ["run_task.py", str(job), "cads_research_558_headneck_oars"])
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, "tempdir", f"/proc/self/fd/{fd}")
        with pytest.raises(SystemExit, match="refusing to download"):
            runpy.run_path(str(ROOT / "run_task.py"), run_name="__main__")
    finally:
        os.close(fd)
    assert calls == []


def test_research_marker_rejects_waiver_without_hashing(monkeypatch, tmp_path):
    import cads_variants
    monkeypatch.setattr(cads_variants, "file_sha256", lambda *_: (_ for _ in ()).throw(AssertionError("hashed")))
    record = {"task": "cads_research_551_organs", "result": "passed", "waiver": "no"}
    with pytest.raises(ValueError, match="waiver"):
        cads_variants.validate_research_acceptance(record, "cads_research_551_organs", tmp_path, ROOT)
    record = {"task": "cads_research_557_brain_tissues", "result": "passed", "license": "research",
              "task_id": 557, "fixture": "lnctvseg-example-hn-ct/dicom", "nonempty_rois": 0}
    with pytest.raises(ValueError, match="Empty prediction"):
        cads_variants.validate_research_acceptance(record, "cads_research_557_brain_tissues", tmp_path, ROOT)


def test_bad_research_marker_does_not_pass_preflight(monkeypatch, tmp_path):
    import model_preflight
    monkeypatch.setattr(model_preflight, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(model_preflight, "APP_ROOT", ROOT)
    monkeypatch.setattr(model_preflight, "COMMON", [])
    monkeypatch.setattr(model_preflight, "REQUIREMENTS", {
        "cads_research_551_organs": ["validation/cads_research_551_organs.json"]})
    monkeypatch.setattr("cads_variants.file_sha256", lambda *_: (_ for _ in ()).throw(AssertionError("hashed")))
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation/cads_research_551_organs.json").write_text(json.dumps({
        "task": "cads_research_551_organs", "result": "passed", "deviation": "ignored"}))
    result = model_preflight.audit(["cads_research_551_organs"])["cads_research_551_organs"]
    assert result["acceptance_ready"] is False and result["ready"] is False
    assert any("inconsistent acceptance" in item for item in result["missing"])


def test_substituted_open_plan_is_refused_before_launch(tmp_path):
    import cads_variants
    root = tmp_path / "models"
    _plant(root, "research", (551,))
    plans = cads_variants.metadata_path(root, "research", 551, "plans.json")
    plans.write_bytes(b"open-plan-substituted")
    with pytest.raises(SystemExit, match="Dataset551_Totalseg251/plans.json"):
        cads_variants.require_installed(root, "research", 551, ROOT)


def test_missing_dependency_plan_is_refused_before_launch(tmp_path, monkeypatch):
    import cads_variants
    root = tmp_path / "models"
    _plant(root, "research", (557, 553))
    pins = cads_variants._manifest_metadata(ROOT)

    def pinned(path):
        text = str(path)
        name = Path(path).name
        for dataset, entry in pins.items():
            if dataset in text and name in entry:
                return entry[name]
        raise AssertionError(path)

    monkeypatch.setattr(cads_variants, "file_sha256", pinned)
    cads_variants.metadata_path(root, "research", 553, "plans.json").unlink()
    with pytest.raises(SystemExit, match="Dataset553_Totalseg253/plans.json"):
        cads_variants.require_installed(root, "research", 557, ROOT)


def test_acceptance_rejects_a_marker_whose_metadata_hash_differs(monkeypatch, tmp_path):
    import cads_variants
    expected = {"Dataset551_Totalseg251": {"plans.json": "a" * 64, "dataset.json": "b" * 64}}
    monkeypatch.setattr(cads_variants, "research_metadata_hashes", lambda *_: expected)
    monkeypatch.setattr(cads_variants, "file_sha256", lambda *_: (_ for _ in ()).throw(AssertionError("hashed checkpoint")))
    record = {
        "task": "cads_research_551_organs", "result": "passed", "license": "research", "task_id": 551,
        "fixture": "kits19-case00000/dicom", "nonempty_rois": 1, "referenced_images_with_contours": 1,
        "rtstruct_bytes": 10, "validator": "convert_validate_rtstruct.py",
        "checks": sorted(cads_variants.RESEARCH_CHECKS), "dependency_tasks": [],
        "expected_output_name": "input_part_551.nii.gz",
        "metadata_hashes": {"Dataset551_Totalseg251": {"plans.json": "c" * 64, "dataset.json": "b" * 64}},
    }
    with pytest.raises(ValueError, match="Metadata hashes"):
        cads_variants.validate_research_acceptance(record, "cads_research_551_organs", tmp_path, ROOT)
