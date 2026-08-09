import re
import os
import json
import signal
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(os.environ.get("RTSEG_MODEL_ROOT", "/home/konrad/lymph-models"))
RUNTIME_ROOT = Path(os.environ.get("RTSEG_RUNTIME_ROOT", "/home/konrad/dicom-rt-seg"))

job=Path(sys.argv[1]); task=sys.argv[2]
if not re.fullmatch(r"[A-Za-z0-9_]+",task): raise SystemExit("Invalid task")
output=job/"output"/f"{task}_RTSTRUCT.dcm"
child = None

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
    if task in {"pam_prompted", "sam_med2d_prompted", "sat3d_prompted"}:
        root = MODEL_ROOT
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        if task == "pam_prompted":
            modality = json.loads((job / "prompt.json").read_text()).get("modality")
            if modality not in {"CT", "MR", "PT"}:
                # Read modality from the selected DICOM rather than trusting browser input.
                import pydicom
                modality = str(pydicom.dcmread(next((job / "input").iterdir()), stop_before_pixels=True).Modality)
            run([str(root / "envs/pam/bin/python"), str(RUNTIME_ROOT / "run_pam.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json"), "--modality", modality])
        elif task == "sam_med2d_prompted":
            run([str(root / "envs/pam/bin/python"), str(RUNTIME_ROOT / "run_sam_med2d.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json")])
        else:
            run([str(root / "envs/pam/bin/python"), str(RUNTIME_ROOT / "run_sat3d.py"),
                 "--input", str(nifti), "--output", str(prediction),
                 "--prompt", str(job / "prompt.json")])
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(RUNTIME_ROOT / "prompted-target-label.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "hnlnl_2d3d":
        root = MODEL_ROOT
        work = job / "external" / task
        input_dir, output_2d, output_3d, ensemble = (
            work / "input", work / "2d", work / "3d", work / "ensemble")
        for directory in (input_dir, output_2d, output_3d, ensemble):
            directory.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case_0000.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        env = os.environ.copy()
        env["RESULTS_FOLDER"] = str(root / "weights/hnlnl/model")
        common = ["-i", str(input_dir), "-t", "Task110_HNLNFixed_MirrorBest",
                  "-tr", "nnUNetTrainerV2", "-p", "nnUNetPlansv2.1",
                  "-chk", "model_final_checkpoint", "-z"]
        predict = str(root / "envs/nnunet-v1/bin/nnUNet_predict")
        run([predict, *common, "-o", str(output_2d), "-m", "2d"], env=env)
        run([predict, *common, "-o", str(output_3d), "-m", "3d_fullres"], env=env)
        postprocessing = root / "weights/hnlnl/model/ensembles/Task110_HNLNFixed_MirrorBest/ensemble_3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1--2d__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json"
        run([str(root / "envs/nnunet-v1/bin/nnUNet_ensemble"),
             "-f", str(output_3d), str(output_2d), "-o", str(ensemble),
             "-pp", str(postprocessing)], env=env)
        prediction = ensemble / "case.nii.gz"
        if not prediction.exists():
            raise SystemExit("HNLNL ensemble did not create its expected mask")
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(RUNTIME_ROOT / "hnlnl-labels.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "dbdmp_lnq":
        root = MODEL_ROOT
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(nifti)])
        run([str(root / "envs/lnq/bin/python"), str(RUNTIME_ROOT / "run_dbdmp.py"),
             "--source", str(root / "sources/LNQ2023_training_code/nnUNet"),
             "--model", str(root / "results/dbdmp/model/Dataset081_LNQ2023/nnUNetPreTrainerVNetv2__nnUNetPlans__3d_fullres"),
             "--input", str(nifti), "--output", str(prediction)])
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(RUNTIME_ROOT / "mediastinal-node-label.json")])
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
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"),
             str(job / "input"), str(full_nifti)])
        run([str(RUNTIME_ROOT / ".venv/bin/TotalSegmentator"),
             "-i", str(full_nifti), "-o", str(lung_dir), "--roi_subset",
             "lung_lower_lobe_left", "lung_upper_lobe_left", "lung_lower_lobe_right",
             "lung_upper_lobe_right", "lung_middle_lobe_right"])
        lung_mask = work / "lung.nii.gz"
        run([str(RUNTIME_ROOT / ".venv/bin/totalseg_combine_masks"),
             "-i", str(lung_dir), "-o", str(lung_mask), "-m", "lung"])
        run([sys.executable, str(RUNTIME_ROOT / "crop_to_mask.py"),
             str(full_nifti), str(lung_mask), str(nifti), "0", "0", "0"])
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join([str(source / "nnunet_code_changes"), env.get("PYTHONPATH", "")])
        run([str(root / "envs/lnq/bin/python"), str(RUNTIME_ROOT / "run_compai_predict.py"),
             "-i", str(input_dir), "-o", str(prediction_dir), "-m", str(model),
             "-f", "all", "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1"], env=env)
        cropped_prediction = prediction_dir / "case.nii.gz"
        if not cropped_prediction.exists():
            raise SystemExit("CompAI Model 7 did not create its expected mask")
        prediction = work / "prediction-full.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "resample_label_to_reference.py"),
             str(cropped_prediction), str(full_nifti), str(prediction)])
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(full_nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(RUNTIME_ROOT / "mediastinal-node-label.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in lnq_tasks:
        root = MODEL_ROOT
        spec, region, folds = lnq_tasks[task]
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nrrd"
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        command = [str(root / "envs/lnq/bin/lnq-segmenter"), "predict", spec,
                   "--input", str(nifti), "--output", str(prediction),
                   "--device", "cuda", "--yes"]
        if folds: command += ["--folds", *folds]
        run(command)
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(RUNTIME_ROOT / f"lnq-{region}-label.json")])
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
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"), str(job / "input_ct"), str(ct)])
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(pet_bqml)])
        run([sys.executable, str(RUNTIME_ROOT / "pet_bqml_to_suv1000.py"),
             str(pet_bqml), str(job / "input"), str(pet_suv)])
        run([sys.executable, str(RUNTIME_ROOT / "resample_image_to_reference.py"),
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
        run([sys.executable, str(RUNTIME_ROOT / "prepare_isrt_pet_config.py"),
             str(source / "configs/hyper_parameters.yaml"), str(data_dir),
             str(datalist), str(bundle), str(config)])
        probabilities = []
        checkpoint_link = checkpoint_dir / "model.pt"
        for fold in range(1, 4):
            checkpoint_link.unlink(missing_ok=True)
            checkpoint_link.symlink_to(
                root / "weights/isrt" / weight_family / fusion / f"model_{fold}.pt")
            shutil.rmtree(bundle / "prediction_f1", ignore_errors=True)
            run([str(root / "envs/lnq/bin/python"), "run.py",
                 f"--config_file={config}"], cwd=str(source))
            generated = bundle / "prediction_f1" / pet_on_ct.name
            if not generated.exists():
                raise SystemExit(f"ISRT PET1 {variant} fold {fold} did not create its probability map")
            saved = work / f"fold-{fold}-prob.nii.gz"
            shutil.copy2(generated, saved)
            probabilities.append(saved)
        probability = work / "ensemble-prob.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "average_probabilities.py"), str(probability),
             *map(str, probabilities)])
        prediction = work / "ensemble-label.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "threshold_probability.py"),
             str(probability), str(prediction), "0.5"])
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(ct),
             "--dicom", str(job / "input_ct"), "--output", str(output),
             "--labels", str(RUNTIME_ROOT / "labels/isrt-ctv.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task == "uwlair_hntsmrg_pre_t2":
        root = MODEL_ROOT
        work = job / "external" / task
        work.mkdir(parents=True, exist_ok=True)
        nifti = work / "input.nii.gz"
        prediction = work / "prediction.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        run([str(root / "envs/lnq/bin/python"),
             str(RUNTIME_ROOT / "run_uwlair_pre.py"), str(nifti), str(prediction)])
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(RUNTIME_ROOT / "labels/hntsmrg-gtv.json")])
        shutil.rmtree(work, ignore_errors=True)
    elif task in lnsegfm_tasks:
        root = MODEL_ROOT
        work = job / "external" / task
        input_dir = work / "input"
        prediction_dir = work / "prediction"
        input_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nifti = input_dir / "case_0000.nii.gz"
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
        model = root / "weights/ln-seg-fm" / lnsegfm_tasks[task]
        env = os.environ.copy()
        if task == "lnsegfm_swinunetr":
            env["PYTHONPATH"] = ":".join([
                str(root / "results/ln-seg-fm/monai130"),
                str(root / "results/ln-seg-fm/nnunet251"),
                env.get("PYTHONPATH", ""),
            ])
        run([str(root / "envs/lnq/bin/nnUNetv2_predict_from_modelfolder"),
             "-i", str(input_dir), "-o", str(prediction_dir), "-m", str(model),
             "-f", "0", "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1"], env=env)
        prediction = prediction_dir / "case.nii.gz"
        if not prediction.exists():
            raise SystemExit("LN-Seg-FM did not create its expected mask")
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
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
        run([sys.executable, str(RUNTIME_ROOT / "dicom_series_to_nifti.py"), str(job / "input"), str(nifti)])
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
        raidionics = str(root / "envs/raidionics/bin/raidionicsseg")
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
            labels = RUNTIME_ROOT / "labels/raidionics-mediastinal-lymphnodes.json"
        elif task == "pediatric_thoracic_lymphoma":
            cropped_input = work / "lymphoma-input" / "case_0000.nii.gz"
            cropped_output = work / "lymphoma-output"
            cropped_output.mkdir(parents=True, exist_ok=True)
            run([sys.executable, str(RUNTIME_ROOT / "crop_to_mask.py"), str(nifti),
                 str(lungs_dir / "labels_Lungs.nii.gz"), str(cropped_input), "32", "32", "16"])
            nn_env = os.environ.copy()
            nn_env["RESULTS_FOLDER"] = str(root / "weights/lymphoma/results")
            run([str(root / "envs/nnunet-v1/bin/nnUNet_predict"),
                 "-i", str(cropped_input.parent), "-o", str(cropped_output),
                 "-t", "Task066_Lymphoma", "-m", "3d_fullres",
                 "-chk", "model_final_checkpoint"], env=nn_env)
            if not (cropped_output / "case.nii.gz").exists():
                raise SystemExit("Lymphoma inference did not create its expected mask")
            prediction = work / "lymphoma-full.nii.gz"
            run([sys.executable, str(RUNTIME_ROOT / "resample_label_to_reference.py"),
                 str(cropped_output / "case.nii.gz"), str(nifti), str(prediction)])
            labels = RUNTIME_ROOT / "labels/pediatric-thoracic-lymphoma.json"
        else:
            architecture, source_name = isrt_tasks[task]
            cropped_input = work / "isrt-data" / "case.nii.gz"
            run([sys.executable, str(RUNTIME_ROOT / "crop_to_mask.py"), str(nifti),
                 str(lungs_mask), str(cropped_input), "32", "32", "16"])
            datalist = work / "isrt-data" / "datalist.json"
            datalist.write_text(json.dumps({
                "labels": {"0": "background", "1": "CTV"},
                "modality": {"image": "CT"},
                "testing": [{"image": "case.nii.gz"}], "training": []}))
            bundle = work / "bundle"
            checkpoint_dir = bundle / "ckpt_f1"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            template = (RUNTIME_ROOT / f"isrt_ct_{architecture}.yaml").read_text()
            template = template.replace("__DATAROOT__", str(cropped_input.parent))
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
                run([str(root / "envs/lnq/bin/python"), "run.py",
                     f"--config_file={config}"], cwd=str(source))
                generated = bundle / "prediction_f1" / "case.nii.gz"
                if not generated.exists():
                    raise SystemExit(f"ISRT {architecture} fold {fold} did not create its probability map")
                saved = work / f"fold-{fold}-prob.nii.gz"
                shutil.copy2(generated, saved)
                probabilities.append(saved)
            probability = work / "ensemble-prob.nii.gz"
            run([sys.executable, str(RUNTIME_ROOT / "average_probabilities.py"), str(probability),
                 *map(str, probabilities)])
            cropped_label = work / "ensemble-label.nii.gz"
            run([sys.executable, str(RUNTIME_ROOT / "threshold_probability.py"),
                 str(probability), str(cropped_label), "0.5"])
            prediction = work / "ensemble-full.nii.gz"
            run([sys.executable, str(RUNTIME_ROOT / "resample_label_to_reference.py"),
                 str(cropped_label), str(nifti), str(prediction)])
            labels = RUNTIME_ROOT / "labels/isrt-ctv.json"
        run([sys.executable, str(RUNTIME_ROOT / "convert_validate_rtstruct.py"),
             "--prediction", str(prediction), "--model-input", str(nifti),
             "--dicom", str(job / "input"), "--output", str(output),
             "--labels", str(labels)])
        shutil.rmtree(work, ignore_errors=True)
    else:
        run([str(Path(sys.executable).parent/"TotalSegmentator"),"-i",str(job/"input"),"-o",str(output),"--task",task,"--output_type","dicom_rtstruct","--device","gpu","--nr_thr_saving","1"])
finally:
    if child is not None and child.poll() is None:
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
