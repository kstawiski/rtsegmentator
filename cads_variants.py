"""CADS open and research weight layout, with strict research acceptance checks.

No machine-learning imports. The portal accepts only the open and research
licences. A missing checkpoint is an error: CADS would otherwise download it.
"""
from __future__ import annotations

import hashlib
import json
import re
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


def validate_research_acceptance(record, task, model_root, app_root):
    """Accept a research marker only when every declared check passed."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    task_id = TASK_IDS[task]
    require(isinstance(record, dict) and record.get("task") == task and record.get("result") == "passed",
            "Acceptance task/result mismatch")
    require(not marker_has_failure(record), "Acceptance contains a failure, deviation, or waiver")
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
    expected_metadata = research_metadata_hashes(model_root, task_id, app_root)
    require(record.get("metadata_hashes") == expected_metadata, "Metadata hashes mismatch")
    pins = _manifest_checkpoint_hashes(app_root)
    expected = {}
    for tid in required_task_ids(task_id):
        dataset = DATASETS[tid]
        digest = file_sha256(checkpoint_path(model_root, "research", tid))
        if digest != pins[dataset]:
            raise ValueError(f"Installed research checkpoint does not match the pin: {dataset}")
        expected[dataset] = digest
    require(record.get("checkpoint_hashes") == expected, "Checkpoint hashes mismatch")
    installed = installed_source_commit(Path(model_root) / "sources/CADS")
    require(record.get("source_commit") == installed == SOURCE_COMMIT, "CADS source commit mismatch")


def validate_orientation(record, task):
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
