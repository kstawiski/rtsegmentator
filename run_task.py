import re
import atexit
import logging
import tempfile
import os
import json
import signal
import shutil
import subprocess
import sys
from pathlib import Path

import pydicom
from nv_segment_logic import MODES as NV_SEGMENT_MODES

job=Path(sys.argv[1]); task=sys.argv[2]
if not re.fullmatch(r"[A-Za-z0-9_]+",task): raise SystemExit("Invalid task")
output=job/"output"/f"{task}_RTSTRUCT.dcm"
child = None
APP_ROOT = Path(os.environ.get("RTSEG_APP_ROOT", Path(__file__).resolve().parent))
MODEL_ROOT = Path(os.environ.get("RTSEG_MODEL_ROOT", "/models"))
RUNTIME_ROOT = Path(os.environ.get("RTSEG_RUNTIME_ROOT", MODEL_ROOT))
_venv_totalsegmentator = Path(sys.executable).parent / "TotalSegmentator"
TOTAL_SEGMENTATOR = (str(_venv_totalsegmentator) if _venv_totalsegmentator.is_file()
                     else shutil.which("TotalSegmentator") or str(_venv_totalsegmentator))
logger = logging.getLogger(__name__)
_owned_scratch = []


def cleanup_scratch():
    for path in _owned_scratch:
        shutil.rmtree(path, ignore_errors=True)
    _owned_scratch.clear()


def short_job_temp():
    # Leave room for multiprocessing's pymp-*/listener-* Unix socket names.
    for base in dict.fromkeys([tempfile.gettempdir(), "/tmp", "/var/tmp"]):
        if len(os.fsencode(str(Path(base).absolute()))) + len("/rtseg-") + 8 > 60:
            continue
        try:
            path = Path(tempfile.mkdtemp(prefix="rtseg-", dir=base)).absolute()
        except OSError:
            continue
        _owned_scratch.append(path)
        return path
    raise RuntimeError("No writable system temporary directory with a short enough path for inference")


def scratch_directories():
    configured = os.environ.get("RTSEG_SCRATCH_ROOT")
    if not configured:
        path = short_job_temp()
        return path, path
    root = Path(configured).absolute()
    try:
        root.mkdir(parents=True, exist_ok=True)
        # Check real write access, including read-only mounts and ACLs.
        with tempfile.TemporaryFile(dir=root):
            pass
    except OSError:
        logger.warning("RTSEG_SCRATCH_ROOT is not writable; using a short per-job temporary directory")
        path = short_job_temp()
        return path, path
    if len(os.fsencode(str(root))) > 60:
        logger.warning("RTSEG_SCRATCH_ROOT exceeds 60 bytes; using a short per-job TMPDIR/TEMP/TMP")
        return root, short_job_temp()
    return root, root


atexit.register(cleanup_scratch)
SCRATCH_ROOT, TEMP_ROOT = scratch_directories()
NNUNET_V2_TASKS = json.loads((APP_ROOT / "catalog/nnunet-v2.json").read_text())
MULTITALENT_V1_TASKS = json.loads((APP_ROOT / "catalog/multitalent-v1.json").read_text())
NNUNET_V1_TASKS = json.loads((APP_ROOT / "catalog/nnunet-v1.json").read_text())
MONAI_BUNDLE_TASKS = json.loads((APP_ROOT / "catalog/monai-bundles.json").read_text())
CADS_OPEN_TASKS = json.loads((APP_ROOT / "catalog/cads-open.json").read_text())
UNIVERSAL_MODEL_TASKS = json.loads((APP_ROOT / "catalog/universal-model.json").read_text())
MRANNOTATOR_TASKS = json.loads((APP_ROOT / "catalog/mrannotator.json").read_text())
PYCAD_TASKS = json.loads((APP_ROOT / "catalog/pycad-model-zoo.json").read_text())
ANTSPYNET_TASKS = json.loads((APP_ROOT / "catalog/antspynet.json").read_text())
SCT_TASKS = json.loads((APP_ROOT / "catalog/sct.json").read_text())
GOUHFI_TASKS = json.loads((APP_ROOT / "catalog/gouhfi.json").read_text())
FASTSURFER_TASKS = json.loads((APP_ROOT / "catalog/fastsurfer.json").read_text())
CERR_TASKS = json.loads((APP_ROOT / "catalog/cerr.json").read_text())

def run(cmd, env=None, cwd=None):
    global child
    env = os.environ.copy() if env is None else env.copy()
    for name in ("TMPDIR", "TEMP", "TMP"):
        env[name] = str(TEMP_ROOT)
    child = subprocess.Popen(cmd, start_new_session=True, env=env, cwd=cwd)
    code = child.wait()
    child = None
    if code:
        raise SystemExit(code)

def stop_child(*_):
    if child is not None and child.poll() is None:
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
    raise SystemExit(143)

def writable_mirror(source: Path, target: Path) -> Path:
    """Mirror immutable files with symlinks, but materialize every directory.

    The target must be fresh. Expand directory aliases without linking writable
    paths back to the bundle, and reject cycles, overlaps and special files.
    Existing files remain read-only through the bundle mount, not this helper.
    """
    source = source.resolve(strict=True)
    target = target.absolute()
    if target.resolve() != target:
        raise ValueError(f"Mirror target has a symlink ancestor: {target}")

    def populate(src, dst, ancestors):
        src = src.resolve(strict=True)
        if src in ancestors or target == src or target.is_relative_to(src):
            raise ValueError(f"Cyclic or overlapping mirror source: {src}")
        dst.mkdir(parents=True)
        for entry in src.iterdir():
            original = entry.resolve(strict=True)
            if original.is_dir():
                populate(original, dst / entry.name, ancestors | {src})
            elif original.is_file():
                (dst / entry.name).symlink_to(original)
            else:
                raise ValueError(f"Not a regular mirror file: {entry}")

    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    try:
        populate(source, target, set())
    except Exception:
        if target.is_dir(): shutil.rmtree(target)
        raise
    return target


def identify_rtstruct(path: Path, model_name: str):
    """Give every result a model-specific identity in common TPS displays."""
    ds = pydicom.dcmread(path)
    display = f"RTSEG: {model_name}"[:64]
    ds.StructureSetLabel = model_name[:16]
    ds.StructureSetName = display
    ds.StructureSetDescription = f"RTsegmentator model {model_name}"[:64]
    ds.SeriesDescription = display
    ds.save_as(path, enforce_file_format=True)

signal.signal(signal.SIGTERM, stop_child)
signal.signal(signal.SIGINT, stop_child)
try:
    isrt_tasks = {
        "isrt_ct_resunet": ("resunet", "ResUNet"),
        "isrt_ct_segresnet": ("segresnet", "SegResNet"),
        "isrt_ct_swinunetr": ("swinunetr", "SwinUNTER"),
    }
    pet_tasks = {
        "isrt_pet1_early_deform": ("Early_fusion", "PET_1_deform", "early-deform"),
        "isrt_pet1_early_rigid": ("Early_fusion", "PET_1_rigid", "early-rigid"),
        "isrt_pet1_late_deform": ("Late_fusion", "PET_1_deform", "late-deform"),
        "isrt_pet1_late_rigid": ("Late_fusion", "PET_1_rigid", "late-rigid"),
    }
    lnsegfm_tasks = {
        "lnsegfm_nnunet": "nnUNetTrainer__nnUNetPlans__3d_fullres",
        "lnsegfm_resenc_m": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
        "lnsegfm_resenc_l": "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
        "lnsegfm_swinunetr": "nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_swinunetr",
    }
    lnq_tasks = {
        "lnq_inguinal_v1": ("inguinal-v1", "inguinal", ["0"]),
        "lnq_abdominopelvic_v1": ("abdominopelvic-v1", "abdominopelvic", ["0"]),
        "lnq_axillary_v1": ("axillary-v1", "axillary", ["0"]),
        "lnq_mediastinal_v1": ("mediastinal-v1", "mediastinal", None),
    }
    lungmask_tasks = {
        "lungmask_r231": ("R231", "labels/lungmask-lungs.json"),
        "lungmask_ltrc_lobes": ("LTRCLobes", "labels/lungmask-lobes.json"),
        "lungmask_ltrc_lobes_r231": ("LTRCLobes_R231", "labels/lungmask-lobes.json"),
    }
    hippunfold_tasks = {
        "hippunfold_adni_t1_subfields_mr": "ADNI_T1w_v1",
        "hippunfold_maguire_t2_subfields_mr": "maguire_T2w",
    }
    if task == "nv_reason_ct_report":
        from model_preflight import nv_reason_python, NV_REASON_REVISION
        env = os.environ.copy()
        cache = Path(os.environ.get("RTSEG_NV_REASON_CT_CACHE", SCRATCH_ROOT / "nv-reason-ct-cache"))
        cache.mkdir(parents=True, exist_ok=True)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HOME=str(cache / "huggingface"), TRITON_CACHE_DIR=str(cache / "triton"),
                   TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"))
        run([str(nv_reason_python()), str(APP_ROOT / "run_nv_reason_ct.py"),
             "--job", str(job), "--model-dir", str(MODEL_ROOT / "weights/nv-reason-ct" / NV_REASON_REVISION),
             "--worker-python", sys.executable], env=env)
        if not (job / "output/report/nv_reason_ct_report.json").is_file():
            raise SystemExit("Report worker did not create its expected JSON")
    elif task in NV_SEGMENT_MODES:
        from nv_segment_logic import check_modality, validate_acceptance
        # Only the explicit operator audit CLI can bootstrap acceptance. The
        # portal and generic SSH route never supply this argument.
        if sys.argv[3:] != ["--nv-segment-acceptance-audit"]:
            if sys.argv[3:]:
                raise SystemExit("Unexpected NV-Segment dispatcher arguments")
            validate_acceptance(json.loads((MODEL_ROOT / "validation" / f"{task}.json").read_text()), task)
        modalities = set()
        for source_file in (job / "input").iterdir():
            if not source_file.is_file():
                continue
            try:
                ds = pydicom.dcmread(source_file, stop_before_pixels=True,
                                    specific_tags=["Modality", "SOPInstanceUID"])
            except pydicom.errors.InvalidDicomError:
                continue
            if getattr(ds, "SOPInstanceUID", None):
                modalities.add(str(getattr(ds, "Modality", "")))
        if len(modalities) != 1:
            raise SystemExit(f"Expected one input modality, found {sorted(modalities)}")
        modality = modalities.pop()
        check_modality(task, modality)
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction, labels = work / "input.nii.gz", work / "prediction.nii.gz", work / "labels.json"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/lnq/bin/python"), str(APP_ROOT / "run_nv_segment.py"),
             "--mode", task, "--input", str(nifti), "--output", str(prediction),
             "--labels-out", str(labels), "--input-modality", modality,
             "--model-root", str(MODEL_ROOT),
             "--runtime-python", str(RUNTIME_ROOT / "envs/lnq/bin/python"),
             "--brain-python", str(MODEL_ROOT / "runtime-envs/nv-segment/bin/python"),
             "--work", str(work / "pipeline"), "--device", "cuda"])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
    elif task in CERR_TASKS:
        spec = CERR_TASKS[task]
        if spec.get("runner") not in {"smit", "legacy_mrrn", "legacy_prostate"}:
            raise SystemExit(f"{task} has no accepted CERR runner: {spec.get('blocked_status', spec.get('runner'))}")
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        model_input, prediction, labels = work / "input.nii.gz", work / "prediction.nii.gz", work / "labels.json"
        labels.write_text(json.dumps(spec["labels"], indent=2, sort_keys=True) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(model_input)])
        if spec["runner"] == "smit":
            environment = os.environ.copy()
            overlay = MODEL_ROOT / "runtime-homes/cerr/monai08-overlay"
            environment["PYTHONPATH"] = f"{overlay}:{environment.get('PYTHONPATH', '')}"
            run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/python"),
                 str(APP_ROOT / "run_cerr_smit.py"), "--model", spec["upstream_model"],
                 "--source", str(MODEL_ROOT / "sources/cerr" / spec["upstream_model"]),
                 "--checkpoint", str(MODEL_ROOT / spec["checkpoint"]),
                 "--input", str(model_input), "--output", str(prediction),
                 "--work", str(work / "pipeline")], env=environment)
        else:
            legacy_input, legacy_output = work / "legacy-input", work / "legacy-output"
            legacy_input.mkdir(); legacy_output.mkdir()
            shutil.copy2(model_input, legacy_input / "case.nii.gz")
            legacy_env = os.environ.copy()
            if spec["runner"] == "legacy_prostate":
                legacy_env["CUDA_VISIBLE_DEVICES"] = ""
            run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/python"),
                 str(APP_ROOT / "run_cerr_legacy.py"), "--model", spec["upstream_model"],
                 "--source", str(MODEL_ROOT / "sources/cerr" / spec["upstream_model"]),
                 "--weights", str(MODEL_ROOT / spec["checkpoint"]),
                 "--input", str(legacy_input), "--output", str(legacy_output),
                 "--work", str(work / "pipeline")], env=legacy_env)
            generated = list(legacy_output.glob("*.nii.gz"))
            if len(generated) != 1:
                raise SystemExit(f"Expected one CERR legacy prediction, found {len(generated)}")
            shutil.copy2(generated[0], prediction)
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(model_input),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in hippunfold_tasks:
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction = work / "input.nii.gz", work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        hippunfold_image = MODEL_ROOT / "images/hippunfold/hippunfold-dev-v2.0.0.sif"
        if task == "hippunfold_adni_t1_subfields_mr":
            hippunfold_image = MODEL_ROOT / "images/hippunfold/hippunfold-v1.5.3.sif"
        hippunfold_command = [sys.executable, str(APP_ROOT / "run_hippunfold.py"),
             "--input", str(nifti), "--output", str(prediction),
             "--work-dir", str(work / "pipeline"),
             "--image", str(hippunfold_image),
             "--cache", str(MODEL_ROOT / "runtime-homes/hippunfold-cache"),
             "--model", hippunfold_tasks[task], "--app-root", str(APP_ROOT)]
        if task == "hippunfold_adni_t1_subfields_mr":
            hippunfold_command.append("--cpu")
        run(hippunfold_command)
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/hippunfold-bigbrain.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "siam_v3_head_mr_ct":
        work=job/"external"/task; input_dir,output_dir=work/"input",work/"output"
        input_dir.mkdir(parents=True,exist_ok=True); output_dir.mkdir(parents=True,exist_ok=True)
        nifti=input_dir/"case.nii.gz"
        run([sys.executable,str(APP_ROOT/"dicom_series_to_nifti.py"),str(job/"input"),str(nifti)])
        source=MODEL_ROOT/"sources/SIAM"; env=os.environ.copy()
        env.update(SIAM_MODEL_DIR=str(MODEL_ROOT/"weights/siam"),PYTHONPATH=str(source),
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
        run([str(RUNTIME_ROOT/"envs/segmentation-modern/bin/python"),"-m","SIAMpred.entry_point",
             "-i",str(input_dir),"-o",str(output_dir),"-m","0","-device","cuda","-nbthread","1"],env=env,cwd=source)
        prediction=output_dir/"siamV03_case.nii.gz"
        if not prediction.is_file(): raise SystemExit(f"SIAM did not create {prediction}")
        run([sys.executable,str(APP_ROOT/"convert_validate_rtstruct.py"),"--prediction",str(prediction),
             "--model-input",str(nifti),"--dicom",str(job/"input"),"--output",str(output),
             "--labels",str(APP_ROOT/"labels/siam-v3.json")])
        shutil.rmtree(work,ignore_errors=True)
    elif task in FASTSURFER_TASKS:
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        original, prediction = work / "input.nii.gz", work / "prediction.nii.gz"
        restored, labels = work / "restored.nii.gz", work / "labels.json"
        source = MODEL_ROOT / "sources/FastSurfer"
        runtime = RUNTIME_ROOT / "envs/segmentation-modern"
        overlay = MODEL_ROOT / "runtime-homes/fastsurfer-python"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(original)])
        run([str(runtime / "bin/python"), str(APP_ROOT / "run_fastsurfer.py"),
             "--input", str(original), "--output", str(prediction),
             "--source", str(source), "--runtime", str(runtime), "--overlay", str(overlay),
             "--work", str(work / "pipeline")])
        run([sys.executable, str(APP_ROOT / "resample_label_to_reference.py"),
             str(prediction), str(original), str(restored)])
        run([sys.executable, str(APP_ROOT / "labels_from_segmentation_lut.py"), str(restored),
             str(source / "FastSurferCNN/config/FreeSurferColorLUT.txt"), str(labels)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(restored), "--model-input", str(original),
             "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in GOUHFI_TASKS:
        spec=GOUHFI_TASKS[task]; work=job/"external"/task
        raw_dir, brain_dir, conformed_dir = work/"raw", work/"brain", work/"conformed"
        for directory in (raw_dir,brain_dir,conformed_dir): directory.mkdir(parents=True,exist_ok=True)
        original=raw_dir/"case_0000.nii.gz"; brain=brain_dir/"case_0000.nii.gz"
        prediction=work/"prediction.nii.gz"; restored=work/"restored.nii.gz"; labels=work/"labels.json"
        source=MODEL_ROOT/"sources/GOUHFI"; runtime=RUNTIME_ROOT/"envs/segmentation-modern"
        run([sys.executable,str(APP_ROOT/"dicom_series_to_nifti.py"),str(job/"input"),str(original)])
        env=os.environ.copy(); env["HOME"]=str(MODEL_ROOT/"runtime-homes/hd-bet")
        run([str(runtime/"bin/hd-bet"),"-i",str(original),"-o",str(brain),"--save_bet_mask"],env=env)
        run([str(runtime/"bin/python"),str(source/"data_utils/conform_images.py"),
             "-i",str(brain_dir),"-o",str(conformed_dir)],cwd=source)
        run([str(runtime/"bin/python"),str(APP_ROOT/"run_gouhfi.py"),
             "--input",str(conformed_dir/"case_0000.nii.gz"),"--output",str(prediction),
             "--source",str(source),"--runtime",str(runtime),"--mode",spec["mode"],"--work",str(work/"pipeline")])
        run([sys.executable,str(APP_ROOT/"resample_label_to_reference.py"),str(prediction),str(original),str(restored)])
        run([sys.executable,str(APP_ROOT/"lut_to_labels.py"),str(source/spec["lut"]),str(labels)])
        run([sys.executable,str(APP_ROOT/"convert_validate_rtstruct.py"),"--prediction",str(restored),
             "--model-input",str(original),"--dicom",str(job/"input"),"--output",str(output),"--labels",str(labels)])
        shutil.rmtree(work,ignore_errors=True)
    elif task in SCT_TASKS:
        spec = SCT_TASKS[task]
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction, labels = work / "input.nii.gz", work / "combined.nii.gz", work / "labels.json"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        sct_root = Path(os.environ.get("RTSEG_SCT_ROOT", MODEL_ROOT / "runtime-homes/sct/sct_7.3"))
        # SCT's relocatable install leaves console-script shebangs pointing at
        # the staging prefix. Invoke the bundled interpreter explicitly and
        # provide its relocated package/binary roots.
        sct_python = sct_root / "python/envs/venv_sct/bin/python3.10"
        command = [str(sct_python), str(sct_root / "bin/sct_deepseg"),
                   spec["sct_task"], "-i", str(nifti),
                   "-o", str(work / "prediction.nii.gz")]
        if spec.get("label_vert"): command += ["-label-vert", "1"]
        env = os.environ.copy()
        env.update(SCT_USE_GPU="1", SCT_DIR=str(sct_root),
                   PYTHONPATH=str(sct_root),
                   PATH=f"{sct_root / 'bin'}:{sct_root / 'python/envs/venv_sct/bin'}:{env.get('PATH', '')}")
        run(command, env=env)
        if spec.get("preserve_overlaps"):
            masks = {}
            for native in sorted(work.glob(spec.get("output_glob", "prediction*.nii.gz"))):
                key = re.sub(r"\.nii(?:\.gz)?$", "", native.name)
                key = re.sub(r"^prediction_?", "", key).strip("_")
                name = next((value for token, value in spec.get("output_names", {}).items()
                             if token.lower() in key.lower()), None)
                if name and native.is_file():
                    masks[name] = str(native)
            if not masks:
                raise SystemExit(f"{task} produced no named native masks")
            masks_json = work / "masks.json"
            masks_json.write_text(json.dumps(masks, indent=2, sort_keys=True) + "\n")
            run([sys.executable, str(APP_ROOT / "convert_masks_validate_rtstruct.py"),
                 "--masks-json", str(masks_json), "--model-input", str(nifti),
                 "--dicom", str(job / "input"), "--output", str(output),
                 "--threshold", str(spec.get("mask_threshold", 0.0))])
        else:
            multiclass_labels = spec.get("multiclass_labels", {})
            if spec.get("labels_file"):
                multiclass_labels = json.loads((APP_ROOT / spec["labels_file"]).read_text())
            native_glob = spec.get("output_glob", "prediction*.nii.gz")
            run([sys.executable, str(APP_ROOT / "combine_sct_outputs.py"),
                 "--glob", str(work / native_glob), "--output", str(prediction),
                 "--labels", str(labels), "--output-names", json.dumps(spec.get("output_names", {})),
                 "--multiclass-labels", json.dumps(multiclass_labels)])
            run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
                 "--prediction", str(prediction), "--model-input", str(nifti),
                 "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in ANTSPYNET_TASKS:
        spec = ANTSPYNET_TASKS[task]
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        model_input = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        labels = work / "labels.json"
        label_map = (json.loads((APP_ROOT / spec["labels_file"]).read_text())
                     if "labels_file" in spec else spec["labels"])
        labels.write_text(json.dumps(label_map, indent=2, sort_keys=True) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(model_input)])
        run([str(RUNTIME_ROOT / "envs/antspynet/bin/python"), str(APP_ROOT / "run_antspynet.py"),
             "--task", spec["runner_task"], "--input", str(model_input), "--output", str(prediction),
             "--cache", str(MODEL_ROOT / "weights/antspynet/cache")])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(model_input),
             "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "lungtumormask_ct":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        lungs = work / "lungs.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        env["TORCH_HOME"] = str(MODEL_ROOT / "weights/torch")
        run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/lungmask"),
             str(nifti), str(lungs), "--modelname", "R231", "--batchsize", "8",
             "--noprogress"], env=env)
        run([str(RUNTIME_ROOT / "envs/lnq/bin/python"),
             str(APP_ROOT / "run_lungtumormask.py"), "--input", str(nifti),
             "--lung-mask", str(lungs), "--checkpoint",
             str(MODEL_ROOT / "weights/lungtumormask/dc_student.pth"),
             "--output", str(prediction)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/lung-tumor.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "synthseg_v1_brain":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        model_input = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        restored = work / "restored.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(model_input)])
        source = MODEL_ROOT / "sources/SynthSeg"
        run([str(RUNTIME_ROOT / "envs/synthseg/bin/python"),
             str(source / "scripts/commands/SynthSeg_predict.py"),
             "--i", str(model_input), "--o", str(prediction), "--v1", "--cpu", "--threads", "8"],
            cwd=source)
        run([sys.executable, str(APP_ROOT / "resample_label_to_reference.py"),
             str(prediction), str(model_input), str(restored)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(restored), "--model-input", str(model_input),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/synthseg-v1.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in PYCAD_TASKS:
        spec = PYCAD_TASKS[task]
        work = job / "external" / task
        input_dir, prediction_dir = work / "input", work / "prediction"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        model_input = input_dir / "case_0000.nii.gz"
        labels = work / "labels.json"
        labels.write_text(json.dumps(spec["labels"], indent=2, sort_keys=True) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(model_input)])
        run([str(RUNTIME_ROOT / "envs/cads/bin/nnUNetv2_predict_from_modelfolder"),
             "-i", str(input_dir), "-o", str(prediction_dir),
             "-m", str(MODEL_ROOT / spec["model_folder"]), "-f", "0",
             "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1"])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction_dir / "case.nii.gz"), "--model-input", str(model_input),
             "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in MRANNOTATOR_TASKS:
        spec = MRANNOTATOR_TASKS[task]
        work = job / "external" / task
        input_dir, prediction_dir = work / "input", work / "prediction"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        original = work / "original.nii.gz"
        model_input = input_dir / "case_0000.nii.gz"
        labels = work / "labels.json"
        restored = work / "restored.nii.gz"
        labels.write_text(json.dumps(spec["labels"], indent=2, sort_keys=True) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(original)])
        run([sys.executable, str(APP_ROOT / "reorient_nifti.py"), str(original), str(model_input), "LAS"])
        run([str(RUNTIME_ROOT / "envs/cads/bin/nnUNetv2_predict_from_modelfolder"),
             "-i", str(input_dir), "-o", str(prediction_dir),
             "-m", str(MODEL_ROOT / spec["model_folder"]), "-f", "0",
             "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1"])
        prediction = prediction_dir / "case.nii.gz"
        run([sys.executable, str(APP_ROOT / "resample_label_to_reference.py"),
             str(prediction), str(original), str(restored)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(restored), "--model-input", str(original),
             "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in UNIVERSAL_MODEL_TASKS:
        spec = UNIVERSAL_MODEL_TASKS[task]
        work = job / "external" / task
        input_dir, mask_dir = work / "input", work / "masks"
        input_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        source = MODEL_ROOT / "sources/UniversalModel"
        run([str(RUNTIME_ROOT / "envs/universal-model/bin/python"),
             str(source / "pred_pseudo.py"), "--data_root_path", str(input_dir),
             "--result_save_path", str(mask_dir), "--resume", str(MODEL_ROOT / spec["checkpoint"]),
             "--backbone", spec["backbone"], "--num_workers", "1", "--log_name", str(work / "log")],
            cwd=source)
        run([sys.executable, str(APP_ROOT / "combine_universal_model.py"),
             str(mask_dir / "case"), str(nifti), str(prediction)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/universal-model.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in CADS_OPEN_TASKS:
        spec = CADS_OPEN_TASKS[task]
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        segmentation_dir = work / "segmentation"
        labels = work / "labels.json"
        labels.write_text(json.dumps(spec["labels"], indent=2, sort_keys=True) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        # A retry reuses this job directory; clear this job's previous mirror (dirs + file links only) first.
        shutil.rmtree(work / "cads-source", ignore_errors=True)
        source = writable_mirror(MODEL_ROOT / "sources/CADS", work / "cads-source")
        env["PYTHONPATH"] = os.pathsep.join([str(source), env.get("PYTHONPATH", "")])
        env["CADS_WEIGHTS_PATH"] = str(MODEL_ROOT / "weights/cads")
        run([str(RUNTIME_ROOT / "envs/cads/bin/python"), "-m", "cads.scripts.predict_images",
             "-in", str(nifti), "-out", str(segmentation_dir),
             "-task", str(spec["task_id"]), "-license", "open", "-np", "2", "-ns", "2"], env=env, cwd=source)
        prediction = segmentation_dir / "input" / f"input_part_{spec['task_id']}.nii.gz"
        if not prediction.exists():
            raise SystemExit("CADS did not create its expected task segmentation")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in lungmask_tasks:
        model, labels = lungmask_tasks[task]
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction = work / "input.nii.gz", work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        env["TORCH_HOME"] = str(MODEL_ROOT / "weights/torch")
        run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/lungmask"),
             str(nifti), str(prediction), "--modelname", model,
             "--batchsize", "8", "--noprogress"], env=env)
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in NNUNET_V1_TASKS:
        spec = NNUNET_V1_TASKS[task]
        work = job / "external" / task
        input_dir, prediction_dir = work / "input", work / "prediction"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case_0000.nii.gz"
        labels = work / "labels.json"
        labels.write_text(json.dumps(spec["labels"], indent=2, sort_keys=True) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        results = work / "nnunet-results"
        configuration = Path("nnUNet") / spec["configuration"]
        source = MODEL_ROOT / "weights/nnunet-v1/model-zoo" / configuration
        task_name = str(spec["task"])
        if task_name.isdigit():
            candidates = [p.name for p in source.iterdir()
                          if p.is_dir() and p.name.startswith(f"Task{int(task_name):03d}")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one nnU-Net task for {task_name}: {candidates}")
            task_name = candidates[0]
        shutil.rmtree(results, ignore_errors=True)
        writable_mirror(source / task_name, results / configuration / task_name)
        env["RESULTS_FOLDER"] = str(results)
        command = [str(RUNTIME_ROOT / "envs/nnunet-v1/bin/nnUNet_predict"),
                   "-i", str(input_dir), "-o", str(prediction_dir), "-t", spec["task"],
                   "-m", spec["configuration"], "-tr", spec["trainer"], "-p", spec["plans"],
                   "-chk", "model_final_checkpoint", "--num_threads_preprocessing", "1",
                   "--num_threads_nifti_save", "1"]
        if spec.get("disable_tta"):
            command.append("--disable_tta")
        run(command, env=env)
        prediction = prediction_dir / "case.nii.gz"
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in MONAI_BUNDLE_TASKS:
        spec = MONAI_BUNDLE_TASKS[task]
        work = job / "external" / task
        inference_output = work / "inference"
        work.mkdir(parents=True, exist_ok=True)
        inference_output.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        labels = work / "labels.json"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        bundle = MODEL_ROOT / "weights/monai-bundles" / spec["bundle"]
        if spec.get("labels_from_metadata"):
            metadata = json.loads((bundle / "configs/metadata.json").read_text())
            channel_def = metadata["network_data_format"]["outputs"]["pred"]["channel_def"]
            label_map = {str(k): v for k, v in channel_def.items() if str(k) != "0"}
        else:
            label_map = spec["labels"]
        labels.write_text(json.dumps(label_map, indent=2, sort_keys=True) + "\n")
        bundle_config = bundle / spec["config"]
        if spec.get("remove_network_img_size"):
            # MONAI <=1.4 SwinUNETR accepted img_size; MONAI 1.6 infers it
            # dynamically. Keep the official bundle immutable and make a
            # job-local compatibility config with only that obsolete key gone.
            compatible = json.loads(bundle_config.read_text())
            compatible["network_def"].pop("img_size", None)
            bundle_config = work / "inference-compatible.json"
            bundle_config.write_text(json.dumps(compatible, indent=2) + "\n")
        datalist = [{"image": str(nifti)}] if spec.get("datalist_records") else [str(nifti)]
        bundle_command = [str(RUNTIME_ROOT / "envs/lnq/bin/python"), "-m", "monai.bundle", "run",
                          "--config_file", str(bundle_config),
                          "--bundle_root", str(bundle), "--datalist", repr(datalist),
                          "--output_dir", str(inference_output), "--dataloader#num_workers", "0"]
        if spec.get("trusted_architecture_pickle"):
            # MONAI's official DiNTS search-code file contains NumPy objects
            # and predates PyTorch 2.6's weights_only=True default. Restrict
            # unsafe deserialization to the checksum-pinned bundle artifact.
            bundle_command += ["--arch_ckpt", "$torch.load(@arch_ckpt_path, map_location=torch.device('cuda'), weights_only=False)"]
        if spec.get("run_id"):
            bundle_command += ["--run_id", spec["run_id"]]
        for key, value in spec.get("overrides", {}).items():
            encoded = str(value) if isinstance(value, (bool, int, float)) else json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            bundle_command += [f"--{key}", encoded]
        run(bundle_command)
        prediction = inference_output / spec.get("output_relative", "input/input_trans.nii.gz")
        if not prediction.exists():
            raise SystemExit(f"MONAI bundle did not create {prediction}")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "hd_bet_mr":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "brain_bet.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        env["HOME"] = str(MODEL_ROOT / "runtime-homes/hd-bet")
        run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/hd-bet"),
             "-i", str(nifti), "-o", str(work / "brain.nii.gz"),
             "--save_bet_mask", "--no_bet_image"], env=env)
        if not prediction.exists():
            raise SystemExit(f"HD-BET did not create {prediction}")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/hd-bet.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "hd_ctbet_ct":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "brain_mask.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        env["HOME"] = str(MODEL_ROOT / "runtime-homes/hd-ctbet")
        run([str(RUNTIME_ROOT / "envs/nnunet-v1/bin/hd-ctbet"),
             "-i", str(nifti), "-o", str(work / "brain.nii.gz"),
             "-device", "0", "-mode", "fast", "-tta", "0"], env=env)
        if not prediction.exists():
            raise SystemExit(f"HD-CTBET did not create {prediction}")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/hd-bet.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "rano2_assist_t1c_mr":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        labels = work / "labels.json"
        labels.write_text(json.dumps({"1": "Nonenhancing_Tumor_Core", "2": "Peritumoral_Edema",
                                      "3": "Enhancing_Tumor_Core", "4": "Resection_Cavity"}, indent=2) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        source = MODEL_ROOT / "sources/rano2.0-assist/dynunet_pipeline"
        task_dir = MODEL_ROOT / "weights/rano2-assist/rano2.0-assist/dynunet_pipeline/data/tasks/task4000_brats24"
        run([str(RUNTIME_ROOT / "envs/rano2/bin/python"), str(APP_ROOT / "run_rano2.py"),
             "--input", str(nifti), "--work", str(work / "pipeline"), "--output", str(prediction),
             "--source", str(source), "--task-dir", str(task_dir),
             "--hd-bet", str(RUNTIME_ROOT / "envs/segmentation-modern/bin/hd-bet")])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "dotatate_pet_lesions":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        labels = work / "labels.json"
        labels.write_text(json.dumps({"1": "DOTATATE_Avid_Lesion"}, indent=2) + "\n")
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/dotatate/bin/python"), str(APP_ROOT / "run_dotatate_onnx.py"),
             "--input", str(nifti), "--output", str(prediction),
             "--model", str(MODEL_ROOT / "weights/dotatate-onnx/onnx_pretrained_models/Task521_PET_5foldEnsemble_nnunet.onnx"),
             "--source", str(MODEL_ROOT / "sources/dotatate-nnunet/infer_on_onnx_models")])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output), "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in MULTITALENT_V1_TASKS:
        spec = MULTITALENT_V1_TASKS[task]
        work = job / "external" / task
        input_dir, prediction_dir = work / "input", work / "prediction"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case_0000.nii.gz"
        masks_json = work / "masks.json"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([sys.executable, str(APP_ROOT / "run_multitalent_v1.py"),
             "--python", str(RUNTIME_ROOT / "envs/nnunet-v1/bin/python"),
             "--source", str(MODEL_ROOT / "sources/MultiTalentV1"),
             "--model-folder", str(MODEL_ROOT / spec["model_folder"]),
             "--input-dir", str(input_dir), "--output-dir", str(prediction_dir),
             "--labels", str(APP_ROOT / "labels/multitalent-v1-masks.json"),
             "--masks-json", str(masks_json), "--fold", "all"])
        run([sys.executable, str(APP_ROOT / "convert_masks_validate_rtstruct.py"),
             "--masks-json", str(masks_json), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output)])
        shutil.rmtree(work, ignore_errors=True)
    elif task in NNUNET_V2_TASKS:
        spec = NNUNET_V2_TASKS[task]
        if spec.get("channels") != 1:
            raise SystemExit(f"{task} requires an unsupported multi-series input contract")
        work = job / "external" / task
        input_dir, prediction_dir = work / "input", work / "prediction"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case_0000.nii.gz"
        original_nifti = work / "original.nii.gz"
        labels = work / "labels.json"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(original_nifti if spec.get("preprocessing") else nifti)])
        if spec.get("preprocessing") == "hd_bet_percentile":
            bet = work / "brain.nii.gz"
            env = os.environ.copy()
            env["HOME"] = str(MODEL_ROOT / "runtime-homes/hd-bet")
            run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/hd-bet"),
                 "-i", str(original_nifti), "-o", str(work / "brain.nii.gz"),
                 "--save_bet_mask"], env=env)
            if not bet.exists():
                raise SystemExit("HD-BET did not create MATTO preprocessing image")
            run([sys.executable, str(APP_ROOT / "normalize_masked_nifti.py"),
                 str(bet), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/python"),
             str(APP_ROOT / "run_nnunet_v2.py"), "--input-dir", str(input_dir),
             "--output-dir", str(prediction_dir), "--labels", str(labels),
             "--model-folder", str(MODEL_ROOT / spec["model_folder"]),
             "--checkpoint", spec.get("checkpoint", "checkpoint_final.pth"),
             *(["--keep-label", str(spec["keep_label"]), "--keep-name", spec["keep_name"]]
               if "keep_label" in spec else []),
             *(["--lowres-model-folder", str(MODEL_ROOT / spec["lowres_model_folder"])]
               if "lowres_model_folder" in spec else [])])
        prediction = prediction_dir / "case.nii.gz"
        if not prediction.is_file():
            raise SystemExit(f"{task} did not create case.nii.gz")
        reference_nifti = original_nifti if spec.get("preprocessing") else nifti
        if spec.get("preprocessing"):
            restored = work / "restored.nii.gz"
            run([sys.executable, str(APP_ROOT / "resample_label_to_reference.py"),
                 str(prediction), str(reference_nifti), str(restored)])
            prediction = restored
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(reference_nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task.startswith("moose_") or task == "dentalsegmentator":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        labels = work / "labels.json"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        moose_model = "clin_ct_dental" if task == "dentalsegmentator" else task.removeprefix("moose_")
        run([str(RUNTIME_ROOT / "envs/moose/bin/python"), str(APP_ROOT / "run_moose.py"),
             "--input", str(nifti), "--output", str(prediction),
             "--labels", str(labels), "--model", moose_model,
             "--model-root", str(MODEL_ROOT / "weights/moose")])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "totalspineseg":
        work = job / "external" / task
        prediction_dir = work / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/totalspineseg/bin/totalspineseg"),
             str(nifti), str(prediction_dir),
             "--data-dir", str(MODEL_ROOT / "weights/totalspineseg"),
             "--keep-only", "step2_output", "--max-workers", "2",
             "--max-workers-nnunet", "2", "--device", "cuda", "--quiet"])
        predictions = list((prediction_dir / "step2_output").glob("*.nii.gz"))
        if len(predictions) != 1:
            raise SystemExit(f"TotalSpineSeg created {len(predictions)} final prediction files; expected one")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(predictions[0]), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/totalspineseg.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "vibesegmentator":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction = work / "input.nii.gz", work / "prediction.nii.gz"
        labels = work / "labels.json"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/python"),
             str(APP_ROOT / "run_vibesegmentator.py"), "--input", str(nifti),
             "--output", str(prediction), "--labels", str(labels),
             "--model-root", str(MODEL_ROOT / "weights/vibesegmentator")])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "mrisegmentator":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction = work / "input.nii.gz", work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        env["MRISEGMENTATOR_DIR"] = str(MODEL_ROOT / "weights/mrisegmentator")
        run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/MRISegmentator"),
             "-i", str(nifti), "-o", str(prediction), "-d", "gpu"], env=env)
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/mrisegmentator.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in {"mrsegmentator", "mrsegmentator_body_comp"}:
        work = job / "external" / task
        prediction_dir = work / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        env["MRSEG_WEIGHTS_PATH"] = str(MODEL_ROOT / "weights/mrsegmentator")
        command = [str(RUNTIME_ROOT / "envs/mrsegmentator/bin/mrsegmentator"),
                   "--input", str(nifti), "--outdir", str(prediction_dir),
                   "--fast", "--batchsize", "1", "--nproc", "1", "--nproc_export", "1"]
        if task == "mrsegmentator_body_comp":
            command.append("--body_comp")
        run(command, env=env)
        predictions = list(prediction_dir.glob("*.nii")) + list(prediction_dir.glob("*.nii.gz"))
        if len(predictions) != 1:
            raise SystemExit(f"MRSegmentator created {len(predictions)} prediction files; expected one")
        labels = APP_ROOT / ("labels/mrsegmentator-body-comp.json" if task.endswith("body_comp")
                             else "labels/mrsegmentator.json")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(predictions[0]), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "medlsam_structseg_han_oars":
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction = work / "input.nii.gz", work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/pam/bin/python"), str(APP_ROOT / "run_medlsam.py"),
             "--input", str(nifti), "--output", str(prediction),
             "--source", str(MODEL_ROOT / "sources/MedLSAM"),
             "--medlam", str(MODEL_ROOT / "weights/medlsam/medlam.pth"),
             "--medsam", str(MODEL_ROOT / "weights/medlsam/medsam_vit_b.pth"),
             "--support-root", str(MODEL_ROOT / "testdata/medlsam-structseg-han/HaN_OAR"),
             "--work", str(work / "pipeline")])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/medlsam-structseg-han.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in {"pam_prompted", "sam_med2d_prompted", "sam_med3d_turbo_prompted", "medsam_vit_b_box_2d", "medical_sam3_3d_prompted", "medsam2_ct_lesion_prompted", "medical_sam2_prompted_ct", "sat3d_prompted", "nninteractive_v1_prompted", "vista3d_prompted_ct", "voxtell_v1_1_text_prompted", "segvol_prompted_ct"}:
        root = MODEL_ROOT
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        if task == "medical_sam3_3d_prompted":
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = ""
            run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/python"),
                 str(APP_ROOT / "run_medical_sam3.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--source", str(MODEL_ROOT / "sources/Medical-SAM3"),
                 "--checkpoint", str(MODEL_ROOT / "weights/medical-sam3/checkpoint_3D.pt"),
                 "--site-packages", str(MODEL_ROOT / "runtime-homes/medical-sam3-site")], env=env)
        elif task == "medsam_vit_b_box_2d":
            run([str(RUNTIME_ROOT / "envs/pam/bin/python"),
                 str(APP_ROOT / "run_medsam.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--source", str(MODEL_ROOT / "sources/MedSAM"),
                 "--checkpoint", str(MODEL_ROOT / "weights/medlsam/medsam_vit_b.pth")])
        elif task == "sam_med3d_turbo_prompted":
            run([str(RUNTIME_ROOT / "envs/pam/bin/python"),
                 str(APP_ROOT / "run_sam_med3d.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--source", str(MODEL_ROOT / "sources/SAM-Med3D"),
                 "--checkpoint", str(MODEL_ROOT / "weights/sam-med3d/sam_med3d_turbo.pth")])
        elif task in {"medsam2_ct_lesion_prompted", "medical_sam2_prompted_ct"}:
            imprint = task == "medical_sam2_prompted_ct"
            run([str(RUNTIME_ROOT / "envs/pam/bin/python"),
                 str(APP_ROOT / "run_medsam2.py"),
                 "--implementation", "imprintlab" if imprint else "wanglab",
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--source", str(MODEL_ROOT / ("sources/Medical-SAM2" if imprint else "sources/MedSAM2")),
                 "--checkpoint", str(MODEL_ROOT / ("weights/medical-sam2/MedSAM2_pretrain.pth" if imprint else "weights/medsam2/medsam2_FLARE25_RECIST_baseline.pt")),
                 "--site-packages", str(MODEL_ROOT / "runtime-homes/medsam2-site")])
        elif task == "segvol_prompted_ct":
            run([str(RUNTIME_ROOT / "envs/lnq/bin/python"),
                 str(APP_ROOT / "run_segvol.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--model-dir", str(MODEL_ROOT / "weights/segvol"),
                 "--monai-site", str(MODEL_ROOT / "runtime-homes/segvol-site"),
                 "--transformers-site", str(MODEL_ROOT / "runtime-homes/voxtell-env/lib/python3.11/site-packages")])
        elif task == "voxtell_v1_1_text_prompted":
            run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/python"),
                 str(APP_ROOT / "run_voxtell.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--model-dir", str(MODEL_ROOT / "weights/voxtell/voxtell_v1.1"),
                 "--embeddings", str(MODEL_ROOT / "weights/voxtell/embeddings/voxtell_v1.1/text_embeddings.npz"),
                 "--text-model", str(MODEL_ROOT / "weights/voxtell/qwen3-embedding-4b"),
                 "--site-packages", str(MODEL_ROOT / "runtime-homes/voxtell-env/lib/python3.11/site-packages"),
                 "--source", str(MODEL_ROOT / "sources/VoxTell"),
                 "--work", str(work / "prediction")])
        elif task == "vista3d_prompted_ct":
            run([str(RUNTIME_ROOT / "envs/lnq/bin/python"), str(APP_ROOT / "run_vista3d.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--source", str(MODEL_ROOT / "sources/NV-Segment-CTMR/NV-Segment-CT"),
                 "--checkpoint", str(MODEL_ROOT / "weights/vista3d/NV-Segment-CT/model.pt"),
                 "--runtime-python", str(RUNTIME_ROOT / "envs/lnq/bin/python"),
                 "--work", str(work / "pipeline")])
        elif task == "nninteractive_v1_prompted":
            run([str(RUNTIME_ROOT / "envs/segmentation-modern/bin/python"),
                 str(APP_ROOT / "run_nninteractive.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"),
                 "--model-dir", str(MODEL_ROOT / "weights/nninteractive/nnInteractive_v1.0"),
                 "--source", str(MODEL_ROOT / "sources/nnInteractive"),
                 "--site-packages", str(MODEL_ROOT / "runtime-homes/nninteractive-site")])
        elif task == "pam_prompted":
            modality = json.loads((job / "prompt.json").read_text()).get("modality")
            if modality not in {"CT", "MR", "PT"}:
                # Read modality from the selected DICOM rather than trusting browser input.
                import pydicom
                modality = str(pydicom.dcmread(next((job / "input").iterdir()), stop_before_pixels=True).Modality)
            run([str(RUNTIME_ROOT / "envs/pam/bin/python"), str(APP_ROOT / "run_pam.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"), "--modality", modality])
        elif task == "sam_med2d_prompted":
            run([str(RUNTIME_ROOT / "envs/pam/bin/python"), str(APP_ROOT / "run_sam_med2d.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json")])
        else:
            run([str(RUNTIME_ROOT / "envs/pam/bin/python"), str(APP_ROOT / "run_sat3d.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json")])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "prompted-target-label.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "hnlnl_2d3d":
        root = MODEL_ROOT
        work = job / "external" / task
        input_dir, output_2d, output_3d, ensemble = (
            work / "input", work / "2d", work / "3d", work / "ensemble")
        for directory in (input_dir, output_2d, output_3d, ensemble):
            directory.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case_0000.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        results = work / "nnunet-results"
        shutil.rmtree(results, ignore_errors=True)
        for configuration in ("2d", "3d_fullres"):
            relative = Path("nnUNet") / configuration / "Task110_HNLNFixed_MirrorBest"
            writable_mirror(root / "weights/hnlnl/model" / relative, results / relative)
        env["RESULTS_FOLDER"] = str(results)
        common = ["-i", str(input_dir), "-t", "Task110_HNLNFixed_MirrorBest",
                  "-tr", "nnUNetTrainerV2", "-p", "nnUNetPlansv2.1",
                  "-chk", "model_final_checkpoint", "-z"]
        predict = str(RUNTIME_ROOT / "envs/nnunet-v1/bin/nnUNet_predict")
        run([predict, *common, "-o", str(output_2d), "-m", "2d"], env=env)
        run([predict, *common, "-o", str(output_3d), "-m", "3d_fullres"], env=env)
        postprocessing = root / "weights/hnlnl/model/ensembles/Task110_HNLNFixed_MirrorBest/ensemble_3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1--2d__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json"
        run([str(RUNTIME_ROOT / "envs/nnunet-v1/bin/nnUNet_ensemble"),
             "-f", str(output_3d), str(output_2d), "-o", str(ensemble),
             "-pp", str(postprocessing)], env=env)
        prediction = ensemble / "case.nii.gz"
        if not prediction.exists():
            raise SystemExit("HNLNL ensemble did not create its expected mask")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "hnlnl-labels.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "dbdmp_lnq":
        root = MODEL_ROOT
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/lnq/bin/python"), str(APP_ROOT / "run_dbdmp.py"),
             "--source", str(root / "sources/LNQ2023_training_code/nnUNet"),
             "--model", str(root / "results/dbdmp/model/Dataset081_LNQ2023/nnUNetPreTrainerVNetv2__nnUNetPlans__3d_fullres"),
             "--input", str(nifti), "--output", str(prediction)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "mediastinal-node-label.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "compai_lnq":
        root = MODEL_ROOT
        source = root / "sources/CompAI_MediastinalLymphNodeSegmentation/lnq_segmentation/nnunet"
        model = source / "nnUNet_results/Dataset050_LNQ_Bouget_NSCLC/nnUNetTrainer__nnUNetPlans__3d_fullres"
        work = job / "external" / task
        input_dir, prediction_dir, lung_dir = work / "input", work / "prediction", work / "lungs"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        lung_dir.mkdir(parents=True, exist_ok=True)
        full_nifti = work / "full.nii.gz"
        nifti = input_dir / "case_0000.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(full_nifti)])
        run([TOTAL_SEGMENTATOR,
             "-i", str(full_nifti), "-o", str(lung_dir), "--roi_subset",
             "lung_lower_lobe_left", "lung_upper_lobe_left", "lung_lower_lobe_right",
             "lung_upper_lobe_right", "lung_middle_lobe_right"])
        lung_mask = work / "lung.nii.gz"
        run([str(Path(sys.executable).parent / "totalseg_combine_masks"),
             "-i", str(lung_dir), "-o", str(lung_mask), "-m", "lung"])
        run([sys.executable, str(APP_ROOT / "crop_to_mask.py"),
             str(full_nifti), str(lung_mask), str(nifti), "0", "0", "0"])
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join([str(source / "nnunet_code_changes"), env.get("PYTHONPATH", "")])
        run([str(RUNTIME_ROOT / "envs/lnq/bin/python"), str(APP_ROOT / "run_compai_predict.py"),
             "-i", str(input_dir), "-o", str(prediction_dir), "-m", str(model),
             "-f", "all", "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1"], env=env)
        cropped_prediction = prediction_dir / "case.nii.gz"
        if not cropped_prediction.exists():
            raise SystemExit("CompAI Model 7 did not create its expected mask")
        prediction = work / "prediction-full.nii.gz"
        run([sys.executable, str(APP_ROOT / "resample_label_to_reference.py"),
             str(cropped_prediction), str(full_nifti), str(prediction)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(full_nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "mediastinal-node-label.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in lnq_tasks:
        root = MODEL_ROOT
        spec, region, folds = lnq_tasks[task]
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction_nrrd = work / "prediction.nrrd"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        command = [str(RUNTIME_ROOT / "envs/lnq/bin/lnq-segmenter"), "predict", spec,
                   "--input", str(nifti), "--output", str(prediction_nrrd),
                   "--device", "cuda", "--yes"]
        if folds: command += ["--folds", *folds]
        run(command)
        run([sys.executable, str(APP_ROOT / "nrrd_to_nifti.py"),
             str(prediction_nrrd), str(prediction)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / f"lnq-{region}-label.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in pet_tasks:
        root = MODEL_ROOT
        fusion, weight_family, variant = pet_tasks[task]
        work = job / "external" / task
        data_dir = work / "data"
        bundle = work / "bundle"
        checkpoint_dir = bundle / "ckpt_f1"
        for directory in (data_dir, checkpoint_dir):
            directory.mkdir(parents=True, exist_ok=True)
        ct = data_dir / "planning-ct.nii.gz"
        pet_bqml = data_dir / "pet1-bqml.nii.gz"
        pet_suv = data_dir / "pet1-suv1000.nii.gz"
        pet_on_ct = data_dir / "pet1-suv1000-on-ct.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input_ct"), str(ct)])
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(pet_bqml)])
        run([sys.executable, str(APP_ROOT / "pet_bqml_to_suv1000.py"),
             str(pet_bqml), str(job / "input"), str(pet_suv)])
        run([sys.executable, str(APP_ROOT / "resample_image_to_reference.py"),
             str(pet_suv), str(ct), str(pet_on_ct)])
        datalist = work / "datalist.json"
        datalist.write_text(json.dumps({
            "labels": {"0": "background", "1": "CTV"},
            "modality": {"image": "PET", "image1": "CT", "image0": "CT"},
            "testing": [{"image": pet_on_ct.name, "image1": ct.name, "image0": ct.name}],
            "training": [],
        }))
        source = root / "sources/ISRT-CTV-AutoSeg" / fusion / "SwinUNETR"
        config = work / "config.yaml"
        run([sys.executable, str(APP_ROOT / "prepare_isrt_pet_config.py"),
             str(source / "configs/hyper_parameters.yaml"), str(data_dir),
             str(datalist), str(bundle), str(config)])
        probabilities = []
        checkpoint_link = checkpoint_dir / "model.pt"
        for fold in range(1, 4):
            checkpoint_link.unlink(missing_ok=True)
            checkpoint_link.symlink_to(
                root / "weights/isrt" / weight_family / fusion / f"model_{fold}.pt")
            shutil.rmtree(bundle / "prediction_f1", ignore_errors=True)
            run([str(RUNTIME_ROOT / "envs/lnq/bin/python"), "run.py",
                 f"--config_file={config}"], cwd=str(source))
            generated = bundle / "prediction_f1" / pet_on_ct.name
            if not generated.exists():
                raise SystemExit(f"ISRT PET1 {variant} fold {fold} did not create its probability map")
            saved = work / f"fold-{fold}-prob.nii.gz"
            shutil.copy2(generated, saved)
            probabilities.append(saved)
        probability = work / "ensemble-prob.nii.gz"
        run([sys.executable, str(APP_ROOT / "average_probabilities.py"), str(probability),
             *map(str, probabilities)])
        prediction = work / "ensemble-label.nii.gz"
        run([sys.executable, str(APP_ROOT / "threshold_probability.py"),
             str(probability), str(prediction), "0.5"])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(ct),
             "--dicom", str(job / "input_ct"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/isrt-ctv.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "uwlair_hntsmrg_pre_t2":
        root = MODEL_ROOT
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        run([str(RUNTIME_ROOT / "envs/lnq/bin/python"),
             str(APP_ROOT / "run_uwlair_pre.py"), str(nifti), str(prediction)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(APP_ROOT / "labels/hntsmrg-gtv.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in lnsegfm_tasks:
        root = MODEL_ROOT
        work = job / "external" / task
        input_dir = work / "input"
        prediction_dir = work / "prediction"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case_0000.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        model = root / "weights/ln-seg-fm" / lnsegfm_tasks[task]
        env = os.environ.copy()
        if task == "lnsegfm_swinunetr":
            env["PYTHONPATH"] = ":".join([
                str(root / "results/ln-seg-fm/monai130"),
                str(root / "results/ln-seg-fm/nnunet251"),
                env.get("PYTHONPATH", ""),
            ])
        run([str(RUNTIME_ROOT / "envs/lnq/bin/nnUNetv2_predict_from_modelfolder"),
             "-i", str(input_dir), "-o", str(prediction_dir), "-m", str(model),
             "-f", "0", "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1"], env=env)
        prediction = prediction_dir / "case.nii.gz"
        if not prediction.exists():
            raise SystemExit("LN-Seg-FM did not create its expected mask")
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(root / "results/ln-seg-fm/labels.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in {"raidionics_mediastinal_lymphnodes", "pediatric_thoracic_lymphoma", *isrt_tasks}:
        root = MODEL_ROOT
        work = job / "external" / task
        nifti_dir = work / "input"
        lungs_dir = work / "lungs"
        nodes_dir = work / "nodes"
        for directory in (nifti_dir, lungs_dir, nodes_dir):
            directory.mkdir(parents=True, exist_ok=True)
        nifti = nifti_dir / "input0.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        common = """[System]
gpu_id=-1
acceleration=cpu
inputs_folder={inputs}
output_folder={outputs}
model_folder={model}

[Runtime]
batch_size=1
folds_ensembling=false
reconstruction_method=thresholding
reconstruction_order=resample_first
test_time_augmentation_iteration=0
"""
        lungs_config = work / "lungs.ini"
        lungs_config.write_text(common.format(
            inputs=nifti_dir, outputs=lungs_dir,
            model=root / "weights/raidionics/lymphnodes/CT_Lungs/hr"))
        raidionics = str(RUNTIME_ROOT / "envs/raidionics/bin/raidionicsseg")
        run([raidionics, str(lungs_config)])
        lungs_mask = lungs_dir / "labels_Lungs.nii.gz"
        if not lungs_mask.exists():
            raise SystemExit("Raidionics lung prerequisite did not create its expected mask")
        if task == "raidionics_mediastinal_lymphnodes":
            nodes_config = work / "nodes.ini"
            nodes_config.write_text(common.format(
                inputs=nifti_dir, outputs=nodes_dir,
                model=root / "weights/raidionics/lymphnodes/CT_LymphNodes/hr")
                + f"\n[Mediastinum]\nlungs_segmentation_filename={lungs_dir / 'labels_Lungs.nii.gz'}\n")
            run([raidionics, str(nodes_config)])
            prediction = nodes_dir / "labels_LymphNodes.nii.gz"
            if not prediction.exists():
                raise SystemExit("Raidionics lymph-node inference did not create its expected mask")
            labels = APP_ROOT / "labels/raidionics-mediastinal-lymphnodes.json"
        elif task == "pediatric_thoracic_lymphoma":
            cropped_input = work / "lymphoma-input" / "case_0000.nii.gz"
            cropped_output = work / "lymphoma-output"
            cropped_output.mkdir(parents=True, exist_ok=True)
            run([sys.executable, str(APP_ROOT / "crop_to_mask.py"), str(nifti),
                 str(lungs_dir / "labels_Lungs.nii.gz"), str(cropped_input), "32", "32", "16"])
            nn_env = os.environ.copy()
            results = work / "nnunet-results"
            shutil.rmtree(results, ignore_errors=True)
            relative = Path("nnUNet/3d_fullres/Task066_Lymphoma")
            writable_mirror(root / "weights/lymphoma/results" / relative, results / relative)
            nn_env["RESULTS_FOLDER"] = str(results)
            run([str(RUNTIME_ROOT / "envs/nnunet-v1/bin/nnUNet_predict"),
                 "-i", str(cropped_input.parent), "-o", str(cropped_output),
                 "-t", "Task066_Lymphoma", "-m", "3d_fullres",
                 "-chk", "model_final_checkpoint"], env=nn_env)
            if not (cropped_output / "case.nii.gz").exists():
                raise SystemExit("Lymphoma inference did not create its expected mask")
            prediction = work / "lymphoma-full.nii.gz"
            run([sys.executable, str(APP_ROOT / "resample_label_to_reference.py"),
                 str(cropped_output / "case.nii.gz"), str(nifti), str(prediction)])
            labels = APP_ROOT / "labels/pediatric-thoracic-lymphoma.json"
        else:
            architecture, source_name = isrt_tasks[task]
            cropped_input = work / "isrt-data" / "case.nii.gz"
            run([sys.executable, str(APP_ROOT / "crop_to_mask.py"), str(nifti),
                 str(lungs_mask), str(cropped_input), "32", "32", "16"])
            datalist = work / "isrt-data" / "datalist.json"
            datalist.write_text(json.dumps({
                "labels": {"0": "background", "1": "CTV"},
                "modality": {"image": "CT"},
                "testing": [{"image": "case.nii.gz"}], "training": []}))
            bundle = work / "bundle"
            checkpoint_dir = bundle / "ckpt_f1"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            template = (APP_ROOT / f"isrt_ct_{architecture}.yaml").read_text()
            template = template.replace("__DATA_ROOT__", str(cropped_input.parent))
            template = template.replace("__BUNDLE_ROOT__", str(bundle))
            config = work / "config.yaml"
            config.write_text(template)
            source = root / "sources/ISRT-CTV-AutoSeg/CT_only" / source_name
            weight_dir_name = "SwinUNETR" if architecture == "swinunetr" else source_name
            probabilities = []
            checkpoint_link = checkpoint_dir / "model.pt"
            for fold in range(1, 4):
                checkpoint_link.unlink(missing_ok=True)
                checkpoint_link.symlink_to(root / "weights/isrt/CT_only" / weight_dir_name / f"model_{fold}.pt")
                run([str(RUNTIME_ROOT / "envs/lnq/bin/python"), "run.py",
                     f"--config_file={config}"], cwd=str(source))
                generated = bundle / "prediction_f1" / "case.nii.gz"
                if not generated.exists():
                    raise SystemExit(f"ISRT {architecture} fold {fold} did not create its probability map")
                saved = work / f"fold-{fold}-prob.nii.gz"
                shutil.copy2(generated, saved)
                probabilities.append(saved)
            probability = work / "ensemble-prob.nii.gz"
            run([sys.executable, str(APP_ROOT / "average_probabilities.py"), str(probability),
                 *map(str, probabilities)])
            cropped_label = work / "ensemble-label.nii.gz"
            run([sys.executable, str(APP_ROOT / "threshold_probability.py"),
                 str(probability), str(cropped_label), "0.5"])
            prediction = work / "ensemble-full.nii.gz"
            run([sys.executable, str(APP_ROOT / "resample_label_to_reference.py"),
                 str(cropped_label), str(nifti), str(prediction)])
            labels = APP_ROOT / "labels/isrt-ctv.json"
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    else:
        # TotalSegmentator's native RTSTRUCT helper reverses the DICOM row axis.
        # Keep inference in a source-grid multilabel NIfTI and use the portal's
        # geometry-verified direct [column,row,slice] -> [row,column,slice]
        # converter shared by every other model.
        from totalsegmentator.map_to_binary import class_map
        if task not in class_map:
            raise SystemExit(f"unknown TotalSegmentator task label map: {task}")
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti, prediction = work / "input.nii.gz", work / "prediction.nii.gz"
        labels = work / "labels.json"
        labels.write_text(json.dumps({str(k): v for k, v in class_map[task].items()}, indent=2))
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([TOTAL_SEGMENTATOR,
             "-i", str(nifti), "-o", str(prediction), "--task", task,
             "--ml", "--device", "gpu", "--nr_thr_saving", "1"])
        # TotalSegmentator strips auxiliary training labels only for tasks with a "<task>_auxiliary" class map;
        # appendicular_bones_mr has none, so keep exactly the labels this task's class map defines.
        run([sys.executable, str(APP_ROOT / "keep_labels.py"), str(prediction), str(labels)])
        run([sys.executable, str(APP_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    if task != "nv_reason_ct_report":
        if not output.exists():
            raise SystemExit(f"{task} did not create its expected RTSTRUCT")
        identify_rtstruct(output, task)
finally:
    if child is not None and child.poll() is None:
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    cleanup_scratch()
