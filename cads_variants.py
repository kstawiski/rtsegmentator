"""CADS open and research weight layout, with strict research acceptance checks.

No machine-learning imports. The portal accepts only the open and research
licences. A missing checkpoint is an error: CADS would otherwise download it.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

DATASETS = {
    551: "Dataset551_Totalseg251",
    552: "Dataset552_Totalseg252",
    553: "Dataset553_Totalseg253",
    554: "Dataset554_Totalseg254",
    555: "Dataset555_Totalseg255",
    556: "Dataset556_GC256",
    557: "Dataset557_Brain257",
    558: "Dataset558_OAR258",
    559: "Dataset559_Saros259",
}
# predict_images.py downloads these extra tasks for post-processing.
DEPENDENCIES = {557: (553,), 558: (552, 553)}
TRAINER = "nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
CHECKPOINT_RELATIVE = f"{TRAINER}/fold_all/checkpoint_final.pth"
SOURCE_COMMIT = "7480324808ab490b5e27035c8bfd28486e8c004a"
PORTAL_LICENSES = ("open", "research")
RESEARCH_CHECKS = frozenset({
    "dicom_to_nifti", "gpu_inference", "research_license_flag",
    "dependency_tasks", "expected_output_name", "rtstruct_validator", "public_fixture",
    "metadata_pins",
})
METADATA_FILES = ("plans.json", "dataset.json")
# Confirmed public fixtures already in the bundle. KiTS19 case 00000 is an
# abdominal contrast CT. The LNCTVSeg example is a public head-and-neck CT.
PUBLIC_FIXTURES = {
    "cads_research_551_organs": "kits19-case00000/dicom",
    "cads_research_552_vertebrae": "kits19-case00000/dicom",
    "cads_research_553_organs_vessels": "kits19-case00000/dicom",
    "cads_research_554_bones_muscles": "kits19-case00000/dicom",
    "cads_research_555_ribs": "lnctvseg-example-hn-ct/dicom",
    "cads_research_556_rt_oars": "kits19-case00000/dicom",
    "cads_research_557_brain_tissues": "lnctvseg-example-hn-ct/dicom",
    "cads_research_558_headneck_oars": "lnctvseg-example-hn-ct/dicom",
    "cads_research_559_body_regions": "kits19-case00000/dicom",
}
TASK_IDS = {
    "cads_research_551_organs": 551,
    "cads_research_552_vertebrae": 552,
    "cads_research_553_organs_vessels": 553,
    "cads_research_554_bones_muscles": 554,
    "cads_research_555_ribs": 555,
    "cads_research_556_rt_oars": 556,
    "cads_research_557_brain_tissues": 557,
    "cads_research_558_headneck_oars": 558,
    "cads_research_559_body_regions": 559,
}
FAILURE_KEYS = frozenset({
    "error", "errors", "fatal_error", "failure", "failures", "failed_checks",
    "unresolved_failures", "deviation", "deviations", "waiver", "waived", "waivers",
})
REJECTED_STRINGS = frozenset({"failed", "failure", "waived", "waiver", "deviation"})
ORIENTATION_FIX = "direct_column_row_slice_mapping"
SCHEMA_VERSION = 2
DEVICES = ("H100", "P6000")
_GPU_PROBE_SCRIPT = (
    "import json, torch\n"
    "if not torch.cuda.is_available():\n"
    "    raise RuntimeError('GPU inference is required; CUDA is unavailable in envs/cads')\n"
    "cap = torch.cuda.get_device_capability(0)\n"
    "archs = list(torch.cuda.get_arch_list())\n"
    "if not archs and hasattr(torch._C, '_cuda_getArchFlags'):\n"
    "    flags = torch._C._cuda_getArchFlags()\n"
    "    archs = flags.split() if flags else []\n"
    "print(json.dumps({\n"
    "    'name': torch.cuda.get_device_name(0),\n"
    "    'compute_capability': [int(cap[0]), int(cap[1])],\n"
    "    'torch': str(torch.__version__),\n"
    "    'cuda': str(torch.version.cuda or ''),\n"
    "    'compiled_architectures': archs,\n"
"}))\n"
)


def cads_python(runtime_root, model_root=None):
    candidates = [Path(runtime_root) / "envs/cads/bin/python"]
    if model_root is not None:
        candidates.append(Path(model_root) / "envs/cads/bin/python")
    return next((path for path in candidates if path.is_file()), candidates[0])


def device_supported(major, minor, arch_list):
    """Check whether compiled_architectures supports (major, minor) via cubin or PTX."""
    if type(major) is not int or type(minor) is not int or not isinstance(arch_list, list):
        return False
    device = (major, minor)
    for item in arch_list:
        if not isinstance(item, str):
            return False
        match = re.fullmatch(r"(sm|compute)_(\d)(\d+)([a-z]*)", item.strip())
        if not match:
            continue
        kind, target_major, target_minor, suffix = match.groups()
        target = (int(target_major), int(target_minor))
        if suffix:
            continue
        if kind == "sm" and target[0] == device[0] and target[1] <= device[1]:
            return True
        if kind == "compute" and target <= device:
            return True
    return False


def device_key(gpu):
    if not isinstance(gpu, dict):
        raise ValueError("CADS research acceptance requires a GPU identity mapping")
    capability = gpu.get("compute_capability")
    name = gpu.get("name", "")
    if (isinstance(capability, list) and len(capability) == 2
            and all(type(item) is int for item in capability) and isinstance(name, str)):
        if capability == [9, 0] and "H100" in name:
            return "H100"
        if capability == [6, 1] and "P6000" in name:
            return "P6000"
    raise ValueError("CADS research acceptance requires an identified H100 or P6000 GPU")


def validate_gpu(gpu):
    key = device_key(gpu)
    torch_version = gpu.get("torch")
    cuda_version = gpu.get("cuda")
    archs = gpu.get("compiled_architectures")
    if not isinstance(torch_version, str) or not torch_version.strip():
        raise ValueError("CADS GPU record is missing torch version")
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        raise ValueError("CADS GPU record is missing CUDA version")
    if (not isinstance(archs, list) or not archs
            or not all(isinstance(item, str) and item.strip() for item in archs)):
        raise ValueError("CADS GPU record is missing compiled_architectures")
    major, minor = gpu["compute_capability"]
    if not device_supported(major, minor, archs):
        raise ValueError(
            f"envs/cads torch {torch_version} ({archs}) does not support {gpu.get('name')} ({major}.{minor})")
    return key


def gpu_preflight(runtime_root, model_root=None):
    """Query the live CUDA device and torch/CUDA build through envs/cads."""
    python = cads_python(runtime_root, model_root=model_root)
    if not python.is_file():
        raise ValueError(f"CADS runtime interpreter missing: {python}")
    try:
        result = subprocess.run(
            [str(python), "-c", _GPU_PROBE_SCRIPT],
            capture_output=True, text=True, timeout=60, check=True,
        )
        gpu = json.loads(result.stdout)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ValueError(f"Cannot verify CADS research executing GPU/runtime: {detail.strip()}") from exc
    validate_gpu(gpu)
    return gpu


@contextmanager
def marker_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_device_records(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        records = payload["devices"] if payload.get("schema_version") == SCHEMA_VERSION else {}
        return dict(records) if isinstance(records, dict) else {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def write_device_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "devices": records}, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def invalidate_device(model_root, task, label=None):
    """Retain other GPU evidence, remove the attempted device's record and legacy unscoped records."""
    for folder in ("validation", "orientation-validation"):
        path = Path(model_root) / folder / f"{task}.json"
        with marker_lock(path):
            records = read_device_records(path)
            if label is not None:
                records.pop(label, None)
            if label is not None and records:
                write_device_records(path, records)
            else:
                path.unlink(missing_ok=True)


def marker_has_failure(value):
    """Reject a failed check, a waiver, or a boolean false hidden in the marker."""
    if value is False:
        return True
    if isinstance(value, str) and value.lower() in REJECTED_STRINGS:
        return True
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FAILURE_KEYS:
                return True
            if key in {"passed", "predeclared_passed"} and item is not True:
                return True
            if key in {"result", "status"} and item not in {"passed", "nonempty"}:
                return True
            if marker_has_failure(item):
                return True
        return False
    if isinstance(value, list):
        return any(marker_has_failure(item) for item in value)
    return False


def weight_directory(license_name, task_id):
    if license_name not in PORTAL_LICENSES:
        raise ValueError("Portal CADS licence must be open or research")
    return f"weights/cads/{license_name}/{DATASETS[int(task_id)]}"


def checkpoint_path(model_root, license_name, task_id):
    return Path(model_root) / weight_directory(license_name, task_id) / CHECKPOINT_RELATIVE


def required_task_ids(task_id):
    task_id = int(task_id)
    return (task_id, *DEPENDENCIES.get(task_id, ()))


def metadata_path(model_root, license_name, task_id, name):
    """plans.json and dataset.json sit beside the trainer folder, not inside fold_all."""
    if name not in METADATA_FILES:
        raise ValueError(f"Unexpected CADS metadata file: {name}")
    return checkpoint_path(model_root, license_name, task_id).parents[1] / name


def require_installed(model_root, license_name, task_id, app_root=None):
    """Stop before predict_images.py, which downloads a missing dataset folder."""
    missing = [tid for tid in required_task_ids(task_id)
               if not checkpoint_path(model_root, license_name, tid).is_file()]
    if missing:
        raise SystemExit(
            f"CADS {license_name} checkpoint missing for task {missing}; refusing to download")
    if license_name != "research":
        return
    if app_root is None:
        app_root = Path(__file__).resolve().parent
    try:
        research_metadata_hashes(model_root, task_id, app_root)
    except ValueError as exc:
        raise SystemExit(f"{exc}; refusing to launch") from exc


def research_requirements(task, task_id):
    requirements = ["envs/cads/bin/python", "sources/CADS/cads/scripts/predict_images.py",
                    "@models/manifests/cads-research.json"]
    for tid in required_task_ids(task_id):
        directory = weight_directory("research", tid)
        requirements.append(directory)
        requirements.append(f"{directory}/{CHECKPOINT_RELATIVE}")
        for name in METADATA_FILES:
            requirements.append(f"{directory}/{TRAINER}/{name}")
    requirements.append(f"validation/{task}.json")
    return requirements


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_source_commit(source):
    """Read the pinned CADS checkout from .git. Do not run git."""
    git = Path(source) / ".git"
    if not git.is_dir():
        raise ValueError("Cannot verify installed CADS source commit")
    head = (git / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:].strip()
        loose = git / ref
        if loose.is_file():
            head = loose.read_text().strip()
        else:
            packed = (git / "packed-refs").read_text().splitlines()
            head = next(line.split()[0] for line in packed
                        if line and not line.startswith(("#", "^")) and line.split()[-1] == ref)
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("Cannot verify installed CADS source commit")
    return head


def _manifest_checkpoint_hashes(app_root):
    manifest = json.loads((Path(app_root) / "models/manifests/cads-research.json").read_text())
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("Research checkpoint manifest is missing")
    for dataset in DATASETS.values():
        value = checkpoints.get(dataset)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Research checkpoint pin is missing: {dataset}")
    return checkpoints


def _manifest_metadata(app_root):
    manifest = json.loads((Path(app_root) / "models/manifests/cads-research.json").read_text())
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Research metadata manifest is missing")
    for dataset in DATASETS.values():
        entry = metadata.get(dataset)
        if not isinstance(entry, dict):
            raise ValueError(f"Research metadata pin is missing: {dataset}")
        for name in METADATA_FILES:
            value = entry.get(name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Research metadata pin is missing: {dataset}/{name}")
    return metadata


def research_metadata_hashes(model_root, task_id, app_root):
    """Hash plans.json and dataset.json for this task and its dependency tasks."""
    pins = _manifest_metadata(app_root)
    expected = {}
    for tid in required_task_ids(task_id):
        dataset = DATASETS[tid]
        expected[dataset] = {}
        for name in METADATA_FILES:
            path = metadata_path(model_root, "research", tid, name)
            if not path.is_file():
                raise ValueError(f"CADS research metadata missing: {dataset}/{name}")
            digest = file_sha256(path)
            if digest != pins[dataset][name]:
                raise ValueError(
                    f"CADS research metadata does not match the pin: {dataset}/{name}")
            expected[dataset][name] = digest
    return expected


def research_checkpoint_hashes(model_root, task_id, app_root):
    pins = _manifest_checkpoint_hashes(app_root)
    expected = {}
    for tid in required_task_ids(task_id):
        dataset = DATASETS[tid]
        digest = file_sha256(checkpoint_path(model_root, "research", tid))
        if digest != pins[dataset]:
            raise ValueError(f"Installed research checkpoint does not match the pin: {dataset}")
        expected[dataset] = digest
    return expected


def research_binding(model_root, task_id, app_root):
    installed = installed_source_commit(Path(model_root) / "sources/CADS")
    if installed != SOURCE_COMMIT:
        raise ValueError("CADS source commit mismatch")
    return {
        "source_commit": installed,
        "checkpoint_hashes": research_checkpoint_hashes(model_root, task_id, app_root),
        "metadata_hashes": research_metadata_hashes(model_root, task_id, app_root),
    }


def _validate_research_bindings(record, task_id, model_root, app_root):
    expected_metadata = research_metadata_hashes(model_root, task_id, app_root)
    if record.get("metadata_hashes") != expected_metadata:
        raise ValueError("Metadata hashes mismatch")
    expected_checkpoints = research_checkpoint_hashes(model_root, task_id, app_root)
    if record.get("checkpoint_hashes") != expected_checkpoints:
        raise ValueError("Checkpoint hashes mismatch")
    installed = installed_source_commit(Path(model_root) / "sources/CADS")
    if not (record.get("source_commit") == installed == SOURCE_COMMIT):
        raise ValueError("CADS source commit mismatch")


def _extract_device_record(record, gpu, kind="Acceptance"):
    if not isinstance(record, dict):
        raise ValueError(f"{kind} record must be a JSON object")
    if marker_has_failure({k: v for k, v in record.items() if k != "devices"}):
        raise ValueError(f"{kind} contains a failure, deviation, or waiver")
    if record.get("schema_version") != SCHEMA_VERSION or not isinstance(record.get("devices"), dict):
        raise ValueError(f"{kind} requires schema_version 2 per-device records")
    devices = record["devices"]
    if not devices or not (set(devices) <= set(DEVICES)):
        raise ValueError(f"{kind} devices must be H100 and/or P6000")
    key = validate_gpu(gpu)
    device_record = devices.get(key)
    if not isinstance(device_record, dict):
        raise ValueError(f"Missing {kind.lower()} for executing device {key}")
    if marker_has_failure(device_record):
        raise ValueError(f"{kind} contains a failure, deviation, or waiver")
    record_gpu = device_record.get("gpu")
    if validate_gpu(record_gpu) != key or record_gpu != gpu:
        raise ValueError(f"{kind} executing GPU/runtime mismatch")
    return device_record


def validate_research_acceptance(record, task, model_root, app_root, *, gpu=None, runtime_root=None):
    """Accept a research marker only when the executing device record and every declared check passed."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    task_id = TASK_IDS[task]
    require(isinstance(record, dict), "Acceptance record must be a JSON object")
    require(not marker_has_failure({k: v for k, v in record.items() if k != "devices"}),
            "Acceptance contains a failure, deviation, or waiver")
    require(record.get("schema_version") == SCHEMA_VERSION and isinstance(record.get("devices"), dict),
            "Acceptance requires schema_version 2 per-device records")
    gpu = gpu if gpu is not None else gpu_preflight(runtime_root or model_root, model_root=model_root)
    record = _extract_device_record(record, gpu, kind="Acceptance")
    require(record.get("task") == task and record.get("result") == "passed",
            "Acceptance task/result mismatch")
    require(record.get("license") == "research", "Acceptance licence is not research")
    require(record.get("task_id") == task_id, "Acceptance task id mismatch")
    require(record.get("fixture") == PUBLIC_FIXTURES[task],
            "Acceptance fixture is not the confirmed public fixture")
    require(type(record.get("nonempty_rois")) is int and record["nonempty_rois"] > 0,
            "Empty prediction cannot pass")
    require(type(record.get("referenced_images_with_contours")) is int
            and record["referenced_images_with_contours"] > 0,
            "RTSTRUCT does not reference source images")
    require(type(record.get("rtstruct_bytes")) is int and record["rtstruct_bytes"] > 0,
            "RTSTRUCT is empty")
    require(record.get("validator") == "convert_validate_rtstruct.py", "Unexpected RTSTRUCT validator")
    checks = record.get("checks")
    require(isinstance(checks, list) and all(isinstance(item, str) for item in checks)
            and RESEARCH_CHECKS <= set(checks), "Incomplete acceptance checks")
    require(record.get("dependency_tasks") == list(DEPENDENCIES.get(task_id, ())),
            "Dependency task list mismatch")
    require(record.get("expected_output_name") == f"input_part_{task_id}.nii.gz",
            "Unexpected CADS output name")
    _validate_research_bindings(record, task_id, model_root, app_root)


def validate_orientation_geometry(record, task):
    """Require the geometry payload written by audit_rtstruct_orientation.py."""
    if not isinstance(record, dict) or record.get("task") != task or record.get("result") != "passed":
        raise ValueError("Orientation task/result mismatch")
    if marker_has_failure(record):
        raise ValueError("Orientation marker contains a failure, deviation, or waiver")
    if record.get("orientation_fix") != ORIENTATION_FIX:
        raise ValueError("Orientation fix is not the reviewed mapping")
    expected_name = f"RTSEG: {task}"[:64]
    if record.get("structure_set_label") != task[:16]:
        raise ValueError("Structure set label mismatch")
    if record.get("structure_set_name") != expected_name or record.get("series_description") != expected_name:
        raise ValueError("Structure set name mismatch")
    if type(record.get("rois")) is not int or record["rois"] < 1:
        raise ValueError("Orientation marker has no ROI")
    if type(record.get("referenced_images")) is not int or record["referenced_images"] < 1:
        raise ValueError("Orientation marker has no referenced images")
    for key in ("contour_min_lps", "contour_max_lps", "source_origin_min_lps", "source_origin_max_lps"):
        value = record.get(key)
        if (not isinstance(value, list) or len(value) != 3
                or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)):
            raise ValueError("Orientation bounds are incomplete")
    if type(record.get("artifact_bytes")) is not int or record["artifact_bytes"] < 1:
        raise ValueError("Orientation artifact is empty")
    if not isinstance(record.get("artifact_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["artifact_sha256"]):
        raise ValueError("Orientation artifact hash is missing")


def validate_orientation(record, task, model_root=None, app_root=None, *, gpu=None, runtime_root=None):
    """Validate orientation geometry, plus per-device and artifact bindings for CADS research tasks."""
    if task not in TASK_IDS:
        return validate_orientation_geometry(record, task)
    if not isinstance(record, dict):
        raise ValueError("Orientation record must be a JSON object")
    if marker_has_failure({k: v for k, v in record.items() if k != "devices"}):
        raise ValueError("Orientation marker contains a failure, deviation, or waiver")
    if record.get("schema_version") != SCHEMA_VERSION or not isinstance(record.get("devices"), dict):
        raise ValueError("Orientation requires schema_version 2 per-device records")
    if model_root is None:
        raise ValueError("Model root is required for CADS research orientation binding")
    if app_root is None:
        app_root = Path(__file__).resolve().parent
    gpu = gpu if gpu is not None else gpu_preflight(runtime_root or model_root, model_root=model_root)
    device_record = _extract_device_record(record, gpu, kind="Orientation")
    validate_orientation_geometry(device_record, task)
    _validate_research_bindings(device_record, TASK_IDS[task], model_root, app_root)


def store_device_records(model_root, app_root, task, acceptance_record, orientation_record, *, runtime_root=None):
    gpu = acceptance_record.get("gpu")
    key = validate_gpu(gpu)
    if orientation_record.get("gpu") != gpu:
        raise ValueError("Orientation GPU record does not match acceptance GPU record")
    validate_research_acceptance(
        {"schema_version": SCHEMA_VERSION, "devices": {key: acceptance_record}},
        task, model_root, app_root, gpu=gpu, runtime_root=runtime_root,
    )
    validate_orientation(
        {"schema_version": SCHEMA_VERSION, "devices": {key: orientation_record}},
        task, model_root, app_root, gpu=gpu, runtime_root=runtime_root,
    )
    for folder, record, validator in (
        ("validation", acceptance_record, validate_research_acceptance),
        ("orientation-validation", orientation_record, validate_orientation),
    ):
        path = Path(model_root) / folder / f"{task}.json"
        with marker_lock(path):
            records = read_device_records(path)
            records[key] = record
            write_device_records(path, records)
            validator(
                json.loads(path.read_text()),
                task, model_root, app_root, gpu=gpu, runtime_root=runtime_root,
            )
