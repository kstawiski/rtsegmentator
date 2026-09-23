import re
import os
import json
import signal
import shutil
import subprocess
import sys
from pathlib import Path

import pydicom

job=Path(sys.argv[1]); task=sys.argv[2]
if not re.fullmatch(r"[A-Za-z0-9_]+",task): raise SystemExit("Invalid task")
output=job/"output"/f"{task}_RTSTRUCT.dcm"
child = None
APP_ROOT = Path(os.environ.get("RTSEG_APP_ROOT", Path(__file__).resolve().parent))
MODEL_ROOT = Path(os.environ.get("RTSEG_MODEL_ROOT", "/models"))
RUNTIME_ROOT = Path(os.environ.get("RTSEG_RUNTIME_ROOT", MODEL_ROOT))
NNUNET_V2_TASKS = json.loads((APP_ROOT / "catalog/nnunet-v2.json").read_text())
NNUNET_V1_TASKS = json.loads((APP_ROOT / "catalog/nnunet-v1.json").read_text())
MONAI_BUNDLE_TASKS = json.loads((APP_ROOT / "catalog/monai-bundles.json").read_text())
CADS_OPEN_TASKS = json.loads((APP_ROOT / "catalog/cads-open.json").read_text())
UNIVERSAL_MODEL_TASKS = json.loads((APP_ROOT / "catalog/universal-model.json").read_text())
MRANNOTATOR_TASKS = json.loads((APP_ROOT / "catalog/mrannotator.json").read_text())
PYCAD_TASKS = json.loads((APP_ROOT / "catalog/pycad-model-zoo.json").read_text())
ANTSPYNET_TASKS = json.loads((APP_ROOT / "catalog/antspynet.json").read_text())
SCT_TASKS = json.loads((APP_ROOT / "catalog/sct.json").read_text())
GOUHFI_TASKS = json.loads((APP_ROOT / "catalog/gouhfi.json").read_text())

def run(cmd, env=None, cwd=None):
    global child
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
    if task == "siam_v3_head_mr_ct":
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
        command = [str(sct_root / "bin/sct_deepseg"), spec["sct_task"], "-i", str(nifti),
                   "-o", str(work / "prediction.nii.gz")]
        if spec.get("label_vert"): command += ["-label-vert", "1"]
        env = os.environ.copy(); env["SCT_USE_GPU"] = "1"
        run(command, env=env)
        multiclass_labels = spec.get("multiclass_labels", {})
        if spec.get("labels_file"):
            multiclass_labels = json.loads((APP_ROOT / spec["labels_file"]).read_text())
        run([sys.executable, str(APP_ROOT / "combine_sct_outputs.py"),
             "--glob", str(work / "prediction*.nii.gz"), "--output", str(prediction),
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
        env["CADS_WEIGHTS_PATH"] = str(MODEL_ROOT / "weights/cads")
        run([str(RUNTIME_ROOT / "envs/cads/bin/python"), "-m", "cads.scripts.predict_images",
             "-in", str(nifti), "-out", str(segmentation_dir),
             "-task", str(spec["task_id"]), "-license", "open", "-np", "2", "-ns", "2"], env=env)
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
        env["RESULTS_FOLDER"] = str(MODEL_ROOT / "weights/nnunet-v1/model-zoo")
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
    elif task in {"pam_prompted", "sam_med2d_prompted", "sat3d_prompted"}:
        root = MODEL_ROOT
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(APP_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        if task == "pam_prompted":
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
        env["RESULTS_FOLDER"] = str(root / "weights/hnlnl/model")
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
        run([str(Path(sys.executable).parent / "TotalSegmentator"),
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
            nn_env["RESULTS_FOLDER"] = str(root / "weights/lymphoma/results")
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
        run([str(Path(sys.executable).parent/"TotalSegmentator"),"-i",str(job/"input"),"-o",str(output),"--task",task,"--output_type","dicom_rtstruct","--device","gpu","--nr_thr_saving","1"])
    if not output.exists():
        raise SystemExit(f"{task} did not create its expected RTSTRUCT")
    identify_rtstruct(output, task)
finally:
    if child is not None and child.poll() is None:
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
