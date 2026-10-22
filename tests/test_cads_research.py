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
H100_GPU = {
    "name": "NVIDIA H100L-47C",
    "compute_capability": [9, 0],
    "torch": "2.6.0+cu124",
    "cuda": "12.4",
    "compiled_architectures": ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
}
P6000_GPU = {
    "name": "Quadro P6000",
    "compute_capability": [6, 1],
    "torch": "2.6.0+cu124",
    "cuda": "12.4",
    "compiled_architectures": ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
}


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
        assert spec["status"] == "technical_checks_recorded"
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
    import cads_variants
    pins = cads_variants._manifest_checkpoint_hashes(ROOT)
    meta_pins = cads_variants._manifest_metadata(ROOT)

    def synthetic_meta(model_root, task_id, app_root):
        from cads_variants import DATASETS, METADATA_FILES, metadata_path, required_task_ids
        found = {}
        for tid in required_task_ids(task_id):
            dataset = DATASETS[tid]
            found[dataset] = {}
            for name in METADATA_FILES:
                path = metadata_path(model_root, "research", tid, name)
                if not path.is_file():
                    raise SystemExit(f"CADS research metadata missing: {dataset}/{name}; refusing to launch")
                found[dataset][name] = meta_pins[dataset][name]
        return found

    def synthetic_checkpoints(model_root, task_id, app_root):
        from cads_variants import DATASETS, checkpoint_path, required_task_ids
        found = {}
        for tid in required_task_ids(task_id):
            dataset = DATASETS[tid]
            if not checkpoint_path(model_root, "research", tid).is_file():
                raise ValueError(f"Missing checkpoint: {dataset}")
            found[dataset] = pins[dataset]
        return found

    monkeypatch.setattr("cads_variants.research_metadata_hashes", synthetic_meta)
    monkeypatch.setattr("cads_variants.research_checkpoint_hashes", synthetic_checkpoints)
    monkeypatch.setattr("cads_variants.installed_source_commit", lambda *_: cads_variants.SOURCE_COMMIT)
    monkeypatch.setattr("cads_variants.gpu_preflight", lambda *a, **kw: dict(P6000_GPU))


def _make_device_records(task, gpu=P6000_GPU):
    import cads_variants
    task_id = cads_variants.TASK_IDS[task]
    pins = cads_variants._manifest_checkpoint_hashes(ROOT)
    meta_pins = cads_variants._manifest_metadata(ROOT)
    tids = cads_variants.required_task_ids(task_id)
    checkpoints = {cads_variants.DATASETS[tid]: pins[cads_variants.DATASETS[tid]] for tid in tids}
    metadata = {cads_variants.DATASETS[tid]: dict(meta_pins[cads_variants.DATASETS[tid]]) for tid in tids}
    binding = {
        "source_commit": cads_variants.SOURCE_COMMIT,
        "checkpoint_hashes": checkpoints,
        "metadata_hashes": metadata,
    }
    acceptance = {
        "task": task,
        "result": "passed",
        "license": "research",
        "task_id": task_id,
        "fixture": cads_variants.PUBLIC_FIXTURES[task],
        "gpu": dict(gpu),
        **binding,
        "expected_output_name": f"input_part_{task_id}.nii.gz",
        "dependency_tasks": list(cads_variants.DEPENDENCIES.get(task_id, ())),
        "validator": "convert_validate_rtstruct.py",
        "checks": sorted(cads_variants.RESEARCH_CHECKS),
        "nonempty_rois": 3,
        "roi_names": ["organ"],
        "referenced_images_with_contours": 4,
        "rtstruct_bytes": 512,
    }
    display = f"RTSEG: {task}"[:64]
    orientation = {
        "task": task,
        "result": "passed",
        "orientation_fix": cads_variants.ORIENTATION_FIX,
        "structure_set_label": task[:16],
        "structure_set_name": display,
        "series_description": display,
        "rois": 3,
        "referenced_images": 4,
        "contour_min_lps": [-10.0, -10.0, -10.0],
        "contour_max_lps": [10.0, 10.0, 10.0],
        "source_origin_min_lps": [-20.0, -20.0, -20.0],
        "source_origin_max_lps": [20.0, 20.0, 20.0],
        "artifact_bytes": 512,
        "artifact_sha256": "a" * 64,
        "gpu": dict(gpu),
        **binding,
    }
    return acceptance, orientation


def _plant_device_markers(root, task, gpu=P6000_GPU):
    import cads_variants
    key = cads_variants.device_key(gpu)
    acceptance, orientation = _make_device_records(task, gpu=gpu)
    cads_variants.write_device_records(root / "validation" / f"{task}.json", {key: acceptance})
    cads_variants.write_device_records(root / "orientation-validation" / f"{task}.json", {key: orientation})


def _dispatch(monkeypatch, tmp_path, synthetic_ct, task, calls, envs):
    job = tmp_path / "job"
    (job / "output").mkdir(parents=True)
    (job / "input").symlink_to(synthetic_ct)
    monkeypatch.setenv("RTSEG_APP_ROOT", str(ROOT))
    monkeypatch.setenv("RTSEG_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RTSEG_RUNTIME_ROOT", str(tmp_path / "runtimes"))
    _allow_synthetic_metadata(monkeypatch)
    if task.startswith("cads_research_"):
        _plant_device_markers(tmp_path / "models", task, gpu=P6000_GPU)
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
    _plant_device_markers(tmp_path / "models", "cads_research_556_rt_oars", gpu=P6000_GPU)
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
        cads_variants.validate_research_acceptance(record, "cads_research_551_organs", tmp_path, ROOT, gpu=P6000_GPU)
    record = {
        "schema_version": 2,
        "devices": {
            "P6000": {
                "task": "cads_research_557_brain_tissues",
                "result": "passed",
                "license": "research",
                "task_id": 557,
                "fixture": "lnctvseg-example-hn-ct/dicom",
                "gpu": dict(P6000_GPU),
                "nonempty_rois": 0,
            }
        },
    }
    with pytest.raises(ValueError, match="Empty prediction"):
        cads_variants.validate_research_acceptance(record, "cads_research_557_brain_tissues", tmp_path, ROOT, gpu=P6000_GPU)


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
        "schema_version": 2,
        "devices": {
            "P6000": {
                "task": "cads_research_551_organs", "result": "passed", "license": "research", "task_id": 551,
                "fixture": "kits19-case00000/dicom", "gpu": dict(P6000_GPU), "nonempty_rois": 1,
                "referenced_images_with_contours": 1, "rtstruct_bytes": 10,
                "validator": "convert_validate_rtstruct.py",
                "checks": sorted(cads_variants.RESEARCH_CHECKS), "dependency_tasks": [],
                "expected_output_name": "input_part_551.nii.gz",
                "metadata_hashes": {"Dataset551_Totalseg251": {"plans.json": "c" * 64, "dataset.json": "b" * 64}},
            }
        },
    }
    with pytest.raises(ValueError, match="Metadata hashes"):
        cads_variants.validate_research_acceptance(record, "cads_research_551_organs", tmp_path, ROOT, gpu=P6000_GPU)


def test_h100_record_against_p6000_probe_is_rejected(monkeypatch, tmp_path, synthetic_ct):
    import cads_variants
    import model_preflight

    task = "cads_research_551_organs"
    root = tmp_path / "models"
    _plant(root, "research", (551,))
    _allow_synthetic_metadata(monkeypatch)
    _plant_device_markers(root, task, gpu=H100_GPU)

    marker = json.loads((root / "validation" / f"{task}.json").read_text())
    orientation = json.loads((root / "orientation-validation" / f"{task}.json").read_text())
    with pytest.raises(ValueError, match="Missing acceptance for executing device P6000"):
        cads_variants.validate_research_acceptance(marker, task, root, ROOT, gpu=P6000_GPU)
    with pytest.raises(ValueError, match="Missing orientation for executing device P6000"):
        cads_variants.validate_orientation(orientation, task, root, ROOT, gpu=P6000_GPU)

    # Relabelling an H100 record under P6000 is also rejected.
    forged = {"schema_version": 2, "devices": {"P6000": marker["devices"]["H100"]}}
    with pytest.raises(ValueError, match="executing GPU/runtime mismatch"):
        cads_variants.validate_research_acceptance(forged, task, root, ROOT, gpu=P6000_GPU)

    # Preflight on a P6000 probe rejects an H100-only marker.
    monkeypatch.setattr(model_preflight, "MODEL_ROOT", root)
    monkeypatch.setattr(model_preflight, "RUNTIME_ROOT", root)
    monkeypatch.setattr(model_preflight, "APP_ROOT", ROOT)
    monkeypatch.setattr(model_preflight, "COMMON", [])
    monkeypatch.setattr(model_preflight, "REQUIREMENTS", {task: [f"validation/{task}.json"]})
    monkeypatch.setattr(model_preflight, "cads_gpu_preflight", lambda *a, **kw: dict(P6000_GPU))
    audit = model_preflight.audit([task])[task]
    assert audit["ready"] is False
    assert audit["acceptance_ready"] is False
    assert audit["orientation_ready"] is False
    assert f"validation/{task}.json (inconsistent acceptance contents)" in audit["missing"]
    assert f"orientation-validation/{task}.json (inconsistent orientation contents)" in audit["missing"]

    # Runner path on a P6000 probe rejects an H100-only marker before spawning inference.
    job = tmp_path / "job_h100_on_p6000"
    (job / "output").mkdir(parents=True)
    (job / "input").symlink_to(synthetic_ct)
    monkeypatch.setenv("RTSEG_APP_ROOT", str(ROOT))
    monkeypatch.setenv("RTSEG_MODEL_ROOT", str(root))
    monkeypatch.setenv("RTSEG_RUNTIME_ROOT", str(root))
    monkeypatch.setattr(sys, "argv", ["run_task.py", str(job), task])
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference spawned")),
    )
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, "tempdir", f"/proc/self/fd/{fd}")
        with pytest.raises(ValueError, match="Missing acceptance for executing device P6000"):
            runpy.run_path(str(ROOT / "run_task.py"), run_name="__main__")
    finally:
        os.close(fd)


def test_legacy_deviceless_record_is_rejected(monkeypatch, tmp_path, synthetic_ct):
    import cads_variants
    import model_preflight

    task = "cads_research_551_organs"
    root = tmp_path / "models"
    _plant(root, "research", (551,))
    _allow_synthetic_metadata(monkeypatch)
    acceptance, orientation = _make_device_records(task, gpu=H100_GPU)
    acceptance.pop("gpu")
    orientation.pop("gpu")
    (root / "validation").mkdir(parents=True, exist_ok=True)
    (root / "orientation-validation").mkdir(parents=True, exist_ok=True)
    (root / "validation" / f"{task}.json").write_text(json.dumps(acceptance))
    (root / "orientation-validation" / f"{task}.json").write_text(json.dumps(orientation))

    for gpu in (H100_GPU, P6000_GPU):
        with pytest.raises(ValueError, match="schema_version 2"):
            cads_variants.validate_research_acceptance(acceptance, task, root, ROOT, gpu=gpu)
        with pytest.raises(ValueError, match="schema_version 2"):
            cads_variants.validate_orientation(orientation, task, root, ROOT, gpu=gpu)

    monkeypatch.setattr(model_preflight, "MODEL_ROOT", root)
    monkeypatch.setattr(model_preflight, "RUNTIME_ROOT", root)
    monkeypatch.setattr(model_preflight, "APP_ROOT", ROOT)
    monkeypatch.setattr(model_preflight, "COMMON", [])
    monkeypatch.setattr(model_preflight, "REQUIREMENTS", {task: [f"validation/{task}.json"]})
    monkeypatch.setattr(model_preflight, "cads_gpu_preflight", lambda *a, **kw: dict(P6000_GPU))
    audit = model_preflight.audit([task])[task]
    assert audit["ready"] is False
    assert audit["acceptance_ready"] is False
    assert audit["orientation_ready"] is False

    job = tmp_path / "job_legacy"
    (job / "output").mkdir(parents=True)
    (job / "input").symlink_to(synthetic_ct)
    monkeypatch.setenv("RTSEG_APP_ROOT", str(ROOT))
    monkeypatch.setenv("RTSEG_MODEL_ROOT", str(root))
    monkeypatch.setenv("RTSEG_RUNTIME_ROOT", str(root))
    monkeypatch.setattr(sys, "argv", ["run_task.py", str(job), task])
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile, "tempdir", f"/proc/self/fd/{fd}")
        with pytest.raises(ValueError, match="schema_version 2"):
            runpy.run_path(str(ROOT / "run_task.py"), run_name="__main__")
    finally:
        os.close(fd)


def test_p6000_record_accepted_on_p6000_probe(monkeypatch, tmp_path):
    import cads_variants
    import model_preflight

    task = "cads_research_551_organs"
    root = tmp_path / "models"
    _plant(root, "research", (551,))
    _allow_synthetic_metadata(monkeypatch)
    _plant_device_markers(root, task, gpu=P6000_GPU)

    marker = json.loads((root / "validation" / f"{task}.json").read_text())
    orientation = json.loads((root / "orientation-validation" / f"{task}.json").read_text())
    cads_variants.validate_research_acceptance(marker, task, root, ROOT, gpu=P6000_GPU)
    cads_variants.validate_orientation(orientation, task, root, ROOT, gpu=P6000_GPU)

    monkeypatch.setattr(model_preflight, "MODEL_ROOT", root)
    monkeypatch.setattr(model_preflight, "RUNTIME_ROOT", root)
    monkeypatch.setattr(model_preflight, "APP_ROOT", ROOT)
    monkeypatch.setattr(model_preflight, "COMMON", [])
    monkeypatch.setattr(model_preflight, "REQUIREMENTS", {task: [f"validation/{task}.json"]})
    monkeypatch.setattr(model_preflight, "cads_gpu_preflight", lambda *a, **kw: dict(P6000_GPU))
    audit = model_preflight.audit([task])[task]
    assert audit["ready"] is True
    assert audit["acceptance_ready"] is True
    assert audit["orientation_ready"] is True
    assert audit["missing"] == []


def test_record_update_preserves_other_device_and_failure_invalidates_only_attempted_device(monkeypatch, tmp_path):
    import cads_variants

    task = "cads_research_551_organs"
    root = tmp_path / "models"
    _plant(root, "research", (551,))
    _allow_synthetic_metadata(monkeypatch)

    h100_acc, h100_ori = _make_device_records(task, gpu=H100_GPU)
    p6000_acc, p6000_ori = _make_device_records(task, gpu=P6000_GPU)

    cads_variants.store_device_records(root, ROOT, task, h100_acc, h100_ori)
    cads_variants.store_device_records(root, ROOT, task, p6000_acc, p6000_ori)

    val_payload = json.loads((root / "validation" / f"{task}.json").read_text())
    ori_payload = json.loads((root / "orientation-validation" / f"{task}.json").read_text())
    assert set(val_payload["devices"]) == {"H100", "P6000"}
    assert set(ori_payload["devices"]) == {"H100", "P6000"}
    cads_variants.validate_research_acceptance(val_payload, task, root, ROOT, gpu=H100_GPU)
    cads_variants.validate_research_acceptance(val_payload, task, root, ROOT, gpu=P6000_GPU)
    cads_variants.validate_orientation(ori_payload, task, root, ROOT, gpu=H100_GPU)
    cads_variants.validate_orientation(ori_payload, task, root, ROOT, gpu=P6000_GPU)

    # Invalidate only P6000; H100 must remain intact and valid.
    cads_variants.invalidate_device(root, task, "P6000")
    val_after = json.loads((root / "validation" / f"{task}.json").read_text())
    ori_after = json.loads((root / "orientation-validation" / f"{task}.json").read_text())
    assert set(val_after["devices"]) == {"H100"}
    assert set(ori_after["devices"]) == {"H100"}
    cads_variants.validate_research_acceptance(val_after, task, root, ROOT, gpu=H100_GPU)
    cads_variants.validate_orientation(ori_after, task, root, ROOT, gpu=H100_GPU)
    with pytest.raises(ValueError, match="Missing acceptance for executing device P6000"):
        cads_variants.validate_research_acceptance(val_after, task, root, ROOT, gpu=P6000_GPU)
