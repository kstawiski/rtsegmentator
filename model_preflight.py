"""Report whether each external model has its runtime, source, and weights installed."""

from __future__ import annotations

import argparse
import json
import shutil
import os
import sys
from pathlib import Path

from nv_segment_logic import MODES as NV_SEGMENT_MODES, validate_acceptance

MODEL_ROOT = Path(os.environ.get("RTSEG_MODEL_ROOT", "/models"))
RUNTIME_ROOT = Path(os.environ.get("RTSEG_RUNTIME_ROOT", MODEL_ROOT))
APP_ROOT = Path(os.environ.get("RTSEG_APP_ROOT", Path(__file__).resolve().parent))

NV_REASON_REVISION = "386b93e034983f6c1fc841a43833a1b6a0cd9c13"


def nv_reason_python():
    override = os.environ.get("RTSEG_NV_REASON_CT_PYTHON")
    if override: return Path(override)
    candidates = [RUNTIME_ROOT / "envs/nv-reason-ct/bin/python",
                  MODEL_ROOT / "runtime-envs/nv-reason-ct/bin/python"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def report_acceptance_ready():
    try:
        marker = json.loads((MODEL_ROOT / "validation/nv_reason_ct_report.json").read_text())
        return (marker.get("status") == "passed" and marker.get("revision") == NV_REASON_REVISION
                and {"chest", "abdomen", "pelvis", "cbct"} <= set(marker.get("checks", []))
                and marker.get("coverage_mode", "full") == os.environ.get("RTSEG_NV_REASON_CT_COVERAGE_MODE", "fast"))
    except (OSError, ValueError, TypeError): return False


COMMON = ["dicom_series_to_nifti.py", "convert_validate_rtstruct.py"]
REQUIREMENTS = {
    "medical_sam3_3d_prompted": [
        "@run_medical_sam3.py", "@prompted-target-label.json",
        "envs/segmentation-modern/bin/python",
        "runtime-homes/medical-sam3-site/ftfy/__init__.py",
        "runtime-homes/medical-sam3-site/pycocotools/_mask.cpython-311-x86_64-linux-gnu.so",
        "sources/Medical-SAM3/sam3/model_builder.py",
        "sources/Medical-SAM3/assets/bpe_simple_vocab_16e6.txt.gz",
        "weights/medical-sam3/checkpoint_3D.pt",
        "validation/medical_sam3_3d_prompted.json",
    ],
    "medsam_vit_b_box_2d": [
        "@run_medsam.py", "@prompted-target-label.json",
        "envs/pam/bin/python",
        "sources/MedSAM/segment_anything/build_sam.py",
        "weights/medlsam/medsam_vit_b.pth",
        "validation/medsam_vit_b_box_2d.json",
    ],
    "sam_med3d_turbo_prompted": [
        "@run_sam_med3d.py", "@prompted-target-label.json",
        "envs/pam/bin/python",
        "sources/SAM-Med3D/segment_anything/build_sam3D.py",
        "sources/SAM-Med3D/segment_anything/modeling/prompt_encoder3D.py",
        "weights/sam-med3d/sam_med3d_turbo.pth",
        "validation/sam_med3d_turbo_prompted.json",
    ],
    "medical_sam2_prompted_ct": [
        "@run_medsam2.py", "@prompted-target-label.json",
        "envs/pam/bin/python", "runtime-homes/medsam2-site/hydra/__init__.py",
        "sources/Medical-SAM2/sam2_train/build_sam.py",
        "sources/Medical-SAM2/sam2_train/sam2_hiera_t.yaml",
        "weights/medical-sam2/MedSAM2_pretrain.pth",
        "validation/medical_sam2_prompted_ct.json",
    ],
    "medsam2_ct_lesion_prompted": [
        "@run_medsam2.py", "@prompted-target-label.json",
        "envs/pam/bin/python",
        "runtime-homes/medsam2-site/hydra/__init__.py",
        "runtime-homes/medsam2-site/iopath/__init__.py",
        "sources/MedSAM2/sam2/build_sam.py",
        "sources/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml",
        "weights/medsam2/medsam2_FLARE25_RECIST_baseline.pt",
        "validation/medsam2_ct_lesion_prompted.json",
    ],
    "medlsam_structseg_han_oars": [
        "@run_medlsam.py", "@labels/medlsam-structseg-han.json",
        "envs/pam/bin/python",
        "sources/MedLSAM/MedLAM/Anatomy_detection.py",
        "sources/MedLSAM/MedSAM/segment_anything/build_sam.py",
        "weights/medlsam/medlam.pth",
        "weights/medlsam/medsam_vit_b.pth",
        "testdata/medlsam-structseg-han/HaN_OAR/1/data.nii.gz",
        "testdata/medlsam-structseg-han/HaN_OAR/1/label.nii.gz",
        "testdata/medlsam-structseg-han/HaN_OAR/2/data.nii.gz",
        "testdata/medlsam-structseg-han/HaN_OAR/2/label.nii.gz",
        "testdata/medlsam-structseg-han/HaN_OAR/3/data.nii.gz",
        "testdata/medlsam-structseg-han/HaN_OAR/3/label.nii.gz",
        "validation/medlsam_structseg_han_oars.json",
    ],
    "universeg_few_shot_2d": [
        "envs/segmentation-modern/bin/python",
        "sources/UniverSeg/universeg/model.py",
        "sources/UniverSeg/LICENSE-model",
        "weights/universeg/universeg_v1_nf64_ss64_STA.pt",
        "validation/universeg_few_shot_2d.json",
    ],
    "segvol_prompted_ct": [
        "@run_segvol.py", "@prompted-target-label.json",
        "envs/lnq/bin/python",
        "runtime-homes/segvol-site/monai/__init__.py",
        "runtime-homes/segvol-site/monai-0.9.0.dist-info/METADATA",
        "runtime-homes/voxtell-env/lib/python3.11/site-packages/transformers/__init__.py",
        "sources/SegVol/README.md",
        "weights/segvol/config.json",
        "weights/segvol/model_segvol_single.py",
        "weights/segvol/pytorch_model.bin",
        "validation/segvol_prompted_ct.json",
    ],
    "voxtell_v1_1_text_prompted": [
        "@run_voxtell.py", "@prompted-target-label.json",
        "envs/segmentation-modern/bin/python",
        "runtime-homes/voxtell-env/lib/python3.11/site-packages/transformers/__init__.py",
        "runtime-homes/voxtell-env/lib/python3.11/site-packages/positional_encodings/__init__.py",
        "sources/VoxTell/voxtell/inference/predictor.py",
        "weights/voxtell/voxtell_v1.1/plans.json",
        "weights/voxtell/voxtell_v1.1/fold_0/checkpoint_final.pth",
        "weights/voxtell/embeddings/voxtell_v1.1/text_embeddings.npz",
        "weights/voxtell/qwen3-embedding-4b/config.json",
        "validation/voxtell_v1_1_text_prompted.json",
    ],
    "vista3d_prompted_ct": [
        "@run_vista3d.py", "@prompted-target-label.json",
        "envs/lnq/bin/python",
        "sources/NV-Segment-CTMR/NV-Segment-CT/configs/inference.json",
        "sources/NV-Segment-CTMR/NV-Segment-CT/scripts/evaluator.py",
        "weights/vista3d/NV-Segment-CT/model.pt",
        "validation/vista3d_prompted_ct.json",
    ],
    "nninteractive_v1_prompted": [
        "@run_nninteractive.py", "@prompted-target-label.json",
        "envs/segmentation-modern/bin/python",
        "sources/nnInteractive/nnInteractive/inference/inference_session.py",
        "runtime-homes/nninteractive-site/nninteractive-2.5.1.dist-info/METADATA",
        "weights/nninteractive/nnInteractive_v1.0/fold_0/checkpoint_final.pth",
        "weights/nninteractive/nnInteractive_v1.0/plans.json",
        "weights/nninteractive/nnInteractive_v1.0/dataset.json",
        "validation/nninteractive_v1_prompted.json",
    ],
    "hippunfold_adni_t1_subfields_mr": [
        "@run_hippunfold.py", "@combine_hippunfold_subfields.py",
        "@resample_label_to_reference.py", "@labels/hippunfold-bigbrain.json",
        "sources/HippUnfold/hippunfold/workflow/Snakefile",
        "images/hippunfold/hippunfold-v1.5.3.sif",
        "weights/hippunfold/trained_model.3d_fullres.Task301_ADNI_T1w_successful.nnUNetTrainer.tar",
        "weights/hippunfold/t1-v1/nnUNet/3d_fullres/Task301_ADNI_T1w_successful/Task301_ADNI_T1w_successful__nnUNetPlansv2.1/fold_0/nnUNetTrainer.model",
        "validation/hippunfold_adni_t1_subfields_mr.json",
    ],
    "hippunfold_maguire_t2_subfields_mr": [
        "@run_hippunfold.py", "@combine_hippunfold_subfields.py",
        "@resample_label_to_reference.py", "@labels/hippunfold-bigbrain.json",
        "sources/HippUnfold/hippunfold/workflow/Snakefile",
        "images/hippunfold/hippunfold-dev-v2.0.0.sif",
        "runtime-homes/hippunfold-v2-overlay-dir",
        "weights/hippunfold/Dataset001_HippUnfoldMag.nnUNetTrainer__nnUNetPlans__3d_fullres.tar.gz",
        "weights/hippunfold/t2-v2/nnunet_v2/Dataset001_HippUnfoldMag/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
        "validation/hippunfold_maguire_t2_subfields_mr.json",
    ],
    "brats_toolkit_legacy_ensemble_mr": [
        "sources/BraTS-Toolkit/brats_toolkit/segmentor.py",
        "images/brats-toolkit/mic-dkfz.sif",
        "images/brats-toolkit/scan.sif",
        "images/brats-toolkit/xfeng.sif",
        "images/brats-toolkit/zyx-2019.sif",
        "images/brats-toolkit/scan-2019.sif",
        "images/brats-toolkit/econib.sif",
        "images/brats-toolkit/lfb-rwth.sif",
        "images/brats-toolkit/gbmnet.sif",
    ],
    "siam_v3_head_mr_ct": ["@labels/siam-v3.json", "envs/segmentation-modern/bin/python",
        "sources/SIAM/SIAMpred/entry_point.py", "sources/SIAM/SIAMpred/nn_prediction.py",
        "weights/siam/v0.3/pred_DS108_LcsfP_Ano/fold_0/checkpoint_final.pth",
        "weights/siam/v0.3/pred_DS108_LcsfP_Ano/fold_4/checkpoint_final.pth",
        "validation/siam_v3_head_mr_ct.json"],
    "lungtumormask_ct": ["@run_lungtumormask.py", "@labels/lung-tumor.json",
                           "envs/lnq/bin/python", "envs/segmentation-modern/bin/lungmask",
                           "sources/LungTumorMask/lungtumormask/network.py",
                           "weights/lungtumormask/dc_student.pth",
                           "validation/lungtumormask_ct.json"],
    "dotatate_pet_lesions": ["@run_dotatate_onnx.py", "envs/dotatate/bin/python", "sources/dotatate-nnunet/infer_on_onnx_models/main.py", "sources/dotatate-nnunet/infer_on_onnx_models/segment_utils.py", "weights/dotatate-onnx/onnx_pretrained_models/Task521_PET_5foldEnsemble_nnunet.onnx", "validation/dotatate_pet_lesions.json"],
    "rano2_assist_t1c_mr": ["@run_rano2.py", "envs/rano2/bin/python", "envs/segmentation-modern/bin/hd-bet", "sources/rano2.0-assist/dynunet_pipeline/src/inference.py", "weights/rano2-assist/rano2.0-assist/dynunet_pipeline/data/tasks/task4000_brats24/models/unmerged_model/fold0/checkpoint_epoch=1000.pt", "validation/rano2_assist_t1c_mr.json"],
    "synthseg_v1_brain": ["@labels/synthseg-v1.json", "envs/synthseg/bin/python",
                           "sources/SynthSeg/scripts/commands/SynthSeg_predict.py",
                           "sources/SynthSeg/models/synthseg_1.0.h5",
                           "validation/synthseg_v1_brain.json"],
    "mrsegmentator": ["@labels/mrsegmentator.json", "envs/mrsegmentator/bin/mrsegmentator", "weights/mrsegmentator/base/plans.json", "validation/mrsegmentator.json"],
    "mrsegmentator_body_comp": ["@labels/mrsegmentator-body-comp.json", "envs/mrsegmentator/bin/mrsegmentator", "weights/mrsegmentator/body_comp/plans.json", "validation/mrsegmentator_body_comp.json"],
    "mrisegmentator": ["@labels/mrisegmentator.json", "envs/segmentation-modern/bin/MRISegmentator", "weights/mrisegmentator/nnUNetTrainer__nnUNetPlans__3d_fullres_New/plans.json", "validation/mrisegmentator.json"],
    "vibesegmentator": ["@run_vibesegmentator.py", "envs/segmentation-modern/bin/python", "sources/VIBESegmentator/run_VIBESegmentator.py", "weights/vibesegmentator/nnUNet_results/Dataset100", "validation/vibesegmentator.json"],
    "dentalsegmentator": ["@run_moose.py", "envs/moose/bin/python", "weights/moose/Dataset112_DentalSegmentator_v100", "validation/dentalsegmentator.json"],
    "totalspineseg": ["@labels/totalspineseg.json", "envs/totalspineseg/bin/totalspineseg", "weights/totalspineseg/nnUNet/results/r20260730/Dataset101_TotalSpineSeg_step1", "weights/totalspineseg/nnUNet/results/r20260730/Dataset102_TotalSpineSeg_step2", "validation/totalspineseg.json"],
    "pam_prompted": ["@run_pam.py", "envs/pam/bin/python", "sources/PAM/tutorials/3d-propagation.ipynb", "weights/pam/propmask.pth"],
    "sam_med2d_prompted": ["@run_sam_med2d.py", "envs/pam/bin/python", "sources/SAM-Med2D/segment_anything/build_sam.py", "weights/sam-med2d/sam-med2d_b.pth"],
    "sat3d_prompted": ["@run_sat3d.py", "envs/pam/bin/python", "sources/SAT3D/networks/__init__.py", "weights/sat3d/sam_model_dice_best.pth", "weights/sat3d/critic_dice_best.pth"],
    "hnlnl_2d3d": ["envs/nnunet-v1/bin/nnUNet_predict", "envs/nnunet-v1/bin/nnUNet_ensemble", "weights/hnlnl/model", "validation/hnlnl_2d3d.json"],
    "dbdmp_lnq": ["@run_dbdmp.py", "envs/lnq/bin/python", "sources/LNQ2023_training_code/nnUNet", "results/dbdmp/model"],
    "compai_lnq": ["@run_compai_predict.py", "envs/lnq/bin/python", "sources/CompAI_MediastinalLymphNodeSegmentation/lnq_segmentation/nnunet", "weights/compai-model7"],
    # lnq-segmenter owns its versioned cache and downloads missing releases itself.
    "lnq_inguinal_v1": ["envs/lnq/bin/lnq-segmenter"],
    "lnq_abdominopelvic_v1": ["envs/lnq/bin/lnq-segmenter"],
    "lnq_axillary_v1": ["envs/lnq/bin/lnq-segmenter"],
    "lnq_mediastinal_v1": ["envs/lnq/bin/lnq-segmenter"],
    "raidionics_mediastinal_lymphnodes": ["envs/raidionics/bin/raidionicsseg", "weights/raidionics/lymphnodes/CT_Lungs/hr", "weights/raidionics/lymphnodes/CT_LymphNodes/hr"],
    "pediatric_thoracic_lymphoma": ["envs/raidionics/bin/raidionicsseg", "envs/nnunet-v1/bin/nnUNet_predict", "weights/raidionics/lymphnodes/CT_Lungs/hr", "weights/lymphoma/results"],
    "uwlair_hntsmrg_pre_t2": ["@run_uwlair_pre.py", "envs/lnq/bin/python", "sources/HNTS-MRG24-UWLAIR/inference/Task1_preRT/segmenter.py", "weights/hntsmrg-uwlair/extracted/preRT_models"],
}

for task in ("lnsegfm_nnunet", "lnsegfm_resenc_m", "lnsegfm_resenc_l", "lnsegfm_swinunetr"):
    REQUIREMENTS[task] = ["envs/lnq/bin/nnUNetv2_predict_from_modelfolder", "weights/ln-seg-fm"]
for task in ("isrt_ct_resunet", "isrt_ct_segresnet", "isrt_ct_swinunetr"):
    REQUIREMENTS[task] = ["envs/lnq/bin/python", "sources/ISRT-CTV-AutoSeg/CT_only", "weights/isrt/CT_only"]
for task in ("isrt_pet1_early_deform", "isrt_pet1_early_rigid", "isrt_pet1_late_deform", "isrt_pet1_late_rigid"):
    REQUIREMENTS[task] = ["envs/lnq/bin/python", "sources/ISRT-CTV-AutoSeg", "weights/isrt/PET_1_deform", "weights/isrt/PET_1_rigid"]

MOOSE_DATASETS = {
    "moose_clin_ct_body": "Dataset001_body",
    "moose_clin_ct_cardiac": "Dataset888_Cardiac",
    "moose_clin_ct_digestive": "Dataset999_Digestive",
    "moose_clin_ct_lungs": "Dataset333_HMS3dlungs",
    "moose_clin_ct_muscles": "Dataset555_Muscles",
    "moose_clin_ct_organs": "Dataset123_Organs",
    "moose_clin_ct_peripheral_bones": "Dataset666_Peripheral-Bones",
    "moose_clin_ct_ribs": "Dataset444_Ribs",
    "moose_clin_ct_vertebrae": "Dataset111_Vertebrae",
    "moose_clin_ct_body_composition": "Dataset778_Body_composition",
}
for task, dataset in MOOSE_DATASETS.items():
    REQUIREMENTS[task] = ["@run_moose.py", "envs/moose/bin/python", f"weights/moose/{dataset}",
                          f"validation/{task}.json"]
REQUIREMENTS["moose_clin_ct_body_composition"].append("weights/moose/Dataset112_FastVertebrae")

NNUNET_V2_TASKS = json.loads((APP_ROOT / "catalog/nnunet-v2.json").read_text())
for task, details in NNUNET_V2_TASKS.items():
    REQUIREMENTS[task] = ["@run_nnunet_v2.py", "envs/segmentation-modern/bin/nnUNetv2_predict_from_modelfolder",
                          details["model_folder"], f"validation/{task}.json"]
    if details.get("preprocessing") == "hd_bet_percentile":
        REQUIREMENTS[task] += ["@normalize_masked_nifti.py", "envs/segmentation-modern/bin/hd-bet",
                               "weights/hd-bet/release_2.0.0/fold_all/checkpoint_final.pth"]

NNUNET_V1_TASKS = json.loads((APP_ROOT / "catalog/nnunet-v1.json").read_text())
for task, details in NNUNET_V1_TASKS.items():
    model = f"weights/nnunet-v1/model-zoo/{details['configuration']}/{details['task']}/{details['trainer']}__{details['plans']}"
    REQUIREMENTS[task] = ["envs/nnunet-v1/bin/nnUNet_predict", model, f"validation/{task}.json"]

MULTITALENT_V1_TASKS = json.loads((APP_ROOT / "catalog/multitalent-v1.json").read_text())
for task, details in MULTITALENT_V1_TASKS.items():
    REQUIREMENTS[task] = ["@run_multitalent_v1.py", "@convert_masks_validate_rtstruct.py",
                          "envs/nnunet-v1/bin/python", "sources/MultiTalentV1",
                          details["model_folder"], "@labels/multitalent-v1-masks.json",
                          f"validation/{task}.json"]

MONAI_BUNDLE_TASKS = json.loads((APP_ROOT / "catalog/monai-bundles.json").read_text())
for task, details in MONAI_BUNDLE_TASKS.items():
    REQUIREMENTS[task] = ["envs/lnq/bin/python",
                          f"weights/monai-bundles/{details['bundle']}/{details['config']}",
                          f"weights/monai-bundles/{details['bundle']}/{details.get('checkpoint', 'models/model.pt')}",
                          f"validation/{task}.json"]

CADS_OPEN_TASKS = json.loads((APP_ROOT / "catalog/cads-open.json").read_text())
CADS_DATASETS = {
    551: "Dataset551_Totalseg251", 552: "Dataset552_Totalseg252",
    553: "Dataset553_Totalseg253", 554: "Dataset554_Totalseg254",
    555: "Dataset555_Totalseg255", 556: "Dataset556_GC256",
    557: "Dataset557_Brain257", 558: "Dataset558_OAR258",
    559: "Dataset559_Saros259",
}
for task, details in CADS_OPEN_TASKS.items():
    task_id = details["task_id"]
    requirements = ["envs/cads/bin/python", "sources/CADS/cads/scripts/predict_images.py",
                    f"weights/cads/open/{CADS_DATASETS[task_id]}", f"validation/{task}.json"]
    if task_id in (557, 558):
        requirements.append("weights/cads/open/Dataset553_Totalseg253")
    if task_id == 558:
        requirements.append("weights/cads/open/Dataset552_Totalseg252")
    REQUIREMENTS[task] = requirements

UNIVERSAL_MODEL_TASKS = json.loads((APP_ROOT / "catalog/universal-model.json").read_text())
for task, details in UNIVERSAL_MODEL_TASKS.items():
    REQUIREMENTS[task] = ["@combine_universal_model.py", "@labels/universal-model.json",
                          "envs/universal-model/bin/python", "sources/UniversalModel/pred_pseudo.py",
                          details["checkpoint"], f"validation/{task}.json"]

MRANNOTATOR_TASKS = json.loads((APP_ROOT / "catalog/mrannotator.json").read_text())
for task, details in MRANNOTATOR_TASKS.items():
    REQUIREMENTS[task] = ["@reorient_nifti.py", "envs/cads/bin/nnUNetv2_predict_from_modelfolder",
                          details["model_folder"], f"validation/{task}.json"]

PYCAD_TASKS = json.loads((APP_ROOT / "catalog/pycad-model-zoo.json").read_text())
for task, details in PYCAD_TASKS.items():
    REQUIREMENTS[task] = ["envs/cads/bin/nnUNetv2_predict_from_modelfolder",
                          details["model_folder"], f"validation/{task}.json"]

ANTSPYNET_TASKS = json.loads((APP_ROOT / "catalog/antspynet.json").read_text())
for task, details in ANTSPYNET_TASKS.items():
    REQUIREMENTS[task] = ["@run_antspynet.py", "envs/antspynet/bin/python",
                          "weights/antspynet/cache", f"validation/{task}.json"]
    if "labels_file" in details:
        REQUIREMENTS[task].append("@" + details["labels_file"])

SCT_TASKS = json.loads((APP_ROOT / "catalog/sct.json").read_text())
for task, details in SCT_TASKS.items():
    REQUIREMENTS[task] = ["@combine_sct_outputs.py", "@convert_masks_validate_rtstruct.py",
                          "runtime-homes/sct/sct_7.3/bin/sct_deepseg",
                          "runtime-homes/sct/sct_7.3/python/envs/venv_sct/bin/python3.10",
                          "runtime-homes/sct/sct_7.3/data/deepseg_models", f"validation/{task}.json"]
    if "labels_file" in details:
        REQUIREMENTS[task].append("@" + details["labels_file"])

GOUHFI_TASKS = json.loads((APP_ROOT / "catalog/gouhfi.json").read_text())
for task, details in GOUHFI_TASKS.items():
    requirements=["@run_gouhfi.py","@lut_to_labels.py","@resample_label_to_reference.py",
                  "envs/segmentation-modern/bin/python","envs/segmentation-modern/bin/hd-bet",
                  "sources/GOUHFI/run_inference/gouhfi_inference_postpro_reo.py",
                  f"sources/GOUHFI/{details['lut']}","sources/GOUHFI/trained_model/Dataset020_gouhfi_2p0n2",
                  f"validation/{task}.json"]
    if details["mode"] == "parcellation": requirements.append("sources/GOUHFI/trained_model/Dataset024_gouhfi_parc")
    REQUIREMENTS[task]=requirements

FASTSURFER_TASKS = json.loads((APP_ROOT / "catalog/fastsurfer.json").read_text())
for task in FASTSURFER_TASKS:
    REQUIREMENTS[task] = ["@run_fastsurfer.py", "@labels_from_segmentation_lut.py",
        "@resample_label_to_reference.py", "envs/segmentation-modern/bin/python",
        "runtime-homes/fastsurfer-python/h5py/__init__.py",
        "sources/FastSurfer/FastSurferCNN/run_prediction.py",
        "sources/FastSurfer/FastSurferCNN/config/FreeSurferColorLUT.txt",
        "sources/FastSurfer/checkpoints/aparc_vinn_axial_v2.0.0.pkl",
        "sources/FastSurfer/checkpoints/aparc_vinn_coronal_v2.0.0.pkl",
        "sources/FastSurfer/checkpoints/aparc_vinn_sagittal_v2.0.0.pkl",
        f"validation/{task}.json"]

CERR_TASKS = json.loads((APP_ROOT / "catalog/cerr.json").read_text())
for task, details in CERR_TASKS.items():
    if details.get("runner") == "smit":
        entrypoint = ("run_segmentation_rectum.py" if details["upstream_model"] == "MR_Rectum_GTV_SMIT"
                      else "run_segmentation.py")
        REQUIREMENTS[task] = [
            "@run_cerr_smit.py", "envs/segmentation-modern/bin/python",
            "runtime-homes/cerr/monai08-overlay/monai/__init__.py",
            f"sources/cerr/{details['upstream_model']}/{entrypoint}",
            details["checkpoint"], f"validation/{task}.json",
        ]
    elif details.get("runner") == "legacy_mrrn":
        REQUIREMENTS[task] = [
            "@run_cerr_legacy.py", "envs/segmentation-modern/bin/python",
            f"sources/cerr/{details['upstream_model']}/run_inference_lung_oars.py",
            details["checkpoint"], f"validation/{task}.json",
        ]
    elif details.get("runner") == "legacy_prostate":
        REQUIREMENTS[task] = [
            "@run_cerr_legacy.py", "runtime-homes/cerr/env6-prostate/bin/python",
            "runtime-homes/cerr/env6-prostate/lib/python3.5/site-packages/tensorflow/__init__.py",
            f"sources/cerr/{details['upstream_model']}/model_wrapper/run_inference_nii.py",
            details["checkpoint"], f"validation/{task}.json",
        ]

REQUIREMENTS["lungmask_r231"] = ["@labels/lungmask-lungs.json", "envs/segmentation-modern/bin/lungmask",
                                  "weights/torch/hub/checkpoints/unet_r231-d5d2fc3d.pth",
                                  "validation/lungmask_r231.json"]
REQUIREMENTS["lungmask_ltrc_lobes"] = ["@labels/lungmask-lobes.json", "envs/segmentation-modern/bin/lungmask",
                                       "weights/torch/hub/checkpoints/unet_ltrclobes-3a07043d.pth",
                                       "validation/lungmask_ltrc_lobes.json"]
REQUIREMENTS["lungmask_ltrc_lobes_r231"] = ["@labels/lungmask-lobes.json",
                                             "envs/segmentation-modern/bin/lungmask",
                                             "weights/torch/hub/checkpoints/unet_r231-d5d2fc3d.pth",
                                             "weights/torch/hub/checkpoints/unet_ltrclobes-3a07043d.pth",
                                             "validation/lungmask_ltrc_lobes_r231.json"]
REQUIREMENTS["hd_bet_mr"] = ["@labels/hd-bet.json",
                               "envs/segmentation-modern/bin/hd-bet",
                               "weights/hd-bet/release_2.0.0/fold_all/checkpoint_final.pth",
                               "validation/hd_bet_mr.json"]
REQUIREMENTS["hd_ctbet_ct"] = ["@labels/hd-bet.json",
                                "@models/patches/hd-ctbet-export-shape.patch",
                                "envs/nnunet-v1/bin/hd-ctbet",
                                "sources/HD-CTBET/HD_CTBET/data_loading.py",
                                "runtime-homes/hd-ctbet/hd-bet_params/CT_0.model",
                                "runtime-homes/hd-ctbet/hd-bet_params/CT_0.model.pkl",
                                "validation/hd_ctbet_ct.json"]
REQUIREMENTS["hecktor2020_petct_gtvt"] = ["@validate_hecktor_checkpoints.py",
    "sources/hecktor/src/models.py",
    "weights/hecktor-2020/s0_best_model_weights.pt",
    "weights/hecktor-2020/s1_best_model_weights.pt",
    "weights/hecktor-2020/s2_best_model_weights.pt",
    "weights/hecktor-2020/s3_best_model_weights.pt",
    "weights/hecktor-2020/ns0_best_model_weights.pt",
    "weights/hecktor-2020/ns1_best_model_weights.pt",
    "weights/hecktor-2020/ns2_best_model_weights.pt",
    "weights/hecktor-2020/ns3_best_model_weights.pt"]
REQUIREMENTS["totalsegmentator2d_projection_ct"] = [
    "sources/TotalSegmentator2D/ts2d/tool.py",
    "runtime-homes/ts2d-venv/bin/ts2d",
    "runtime-homes/ts2d-home/.ts2d/models/ts2d-v2-ep4000b2_cardiac/r001",
    "runtime-homes/ts2d-home/.ts2d/models/ts2d-v2-ep4000b2_muscles/r001",
    "runtime-homes/ts2d-home/.ts2d/models/ts2d-v2-ep4000b2_organs/r001",
    "runtime-homes/ts2d-home/.ts2d/models/ts2d-v2-ep4000b2_ribs/r001",
    "runtime-homes/ts2d-home/.ts2d/models/ts2d-v2-ep4000b2_vertebrae/r001",
    "validation/totalsegmentator2d_projection_ct.json"]


REQUIREMENTS["nv_reason_ct_report"] = ["@run_nv_reason_ct.py", "@nv_reason_ct_report.py",
    "@labels/nv-reason-ct-organs.json", "envs/nv-reason-ct/bin/python",
    *[f"weights/nv-reason-ct/{NV_REASON_REVISION}/{name}" for name in (
        "model.safetensors", "model.py", "processor.py", "config.json", "tokenizer.json",
        "tokenizer_config.json", "processor_config.json", "preprocessor_config.json",
        "video_preprocessor_config.json", "chat_template.jinja", "generation_config.json",
        "image_processor_3d/preprocessor_config.json")],
    "validation/nv_reason_ct_report.json"]


NV_SEGMENT_MANIFEST = json.loads((APP_ROOT / "nv_segment_manifest.json").read_text())["modes"]
for _task, _spec in NV_SEGMENT_MODES.items():
    REQUIREMENTS[_task] = [
        "@run_nv_segment.py", "@nv_segment_bundle.py", "@nv_segment_logic.py",
        "@nv_segment_manifest.json", f"@labels/{_task}.json", "envs/lnq/bin/python",
        *[f"{_spec['source']}/{name}" for name in NV_SEGMENT_MANIFEST[_task]["source_files"]],
        f"weights/vista3d/{_spec['bundle']}/model.pt", f"validation/{_task}.json",
    ]
    if _spec['upstream_mode'] == "MRI_BRAIN":
        REQUIREMENTS[_task] += [
            "runtime-envs/nv-segment/bin/python",
            *[f"runtime-envs/nv-segment/lib/python3.11/site-packages/{name}" for name in (
                "ants/__init__.py", "surfa/__init__.py", "pymedio/image.py",
                "pydicom/__init__.py", "zz_nv_segment_lnq.pth")],
            "runtime-envs/nv-segment/requirements-nv-segment.txt",
            "runtime-homes/nv-segment/synthstrip-1.8/manifest.json",
            "runtime-homes/nv-segment/synthstrip-1.8/freesurfer/mri_synthstrip",
            "runtime-homes/nv-segment/synthstrip-1.8/freesurfer/models/synthstrip.1.pt",
        ]


def audit(tasks=None) -> dict[str, dict]:
    report = {}
    for task, requirements in REQUIREMENTS.items():
        if tasks is not None and task not in tasks: continue
        missing = []
        for requirement in [*("@" + item for item in COMMON), *requirements]:
            if task == "nv_reason_ct_report" and requirement == "envs/nv-reason-ct/bin/python":
                path = nv_reason_python()
            elif requirement.startswith("@"):
                path = APP_ROOT / requirement[1:]
            elif requirement.startswith("envs/"):
                path = RUNTIME_ROOT / requirement
            else:
                path = MODEL_ROOT / requirement
            if not path.exists():
                missing.append(requirement.lstrip("@"))
        acceptance_ready = False
        if task in NV_SEGMENT_MODES:
            marker = MODEL_ROOT / "validation" / f"{task}.json"
            try:
                validate_acceptance(json.loads(marker.read_text()), task)
                acceptance_ready = True
            except (OSError, ValueError, TypeError, AttributeError):
                if f"validation/{task}.json" not in missing:
                    missing.append(f"validation/{task}.json (inconsistent acceptance contents)")
        orientation = MODEL_ROOT / "orientation-validation" / f"{task}.json"
        report[task] = {"ready": not missing, "missing": missing,
                        "orientation_ready": orientation.is_file()}
        if task == "nv_reason_ct_report":
            report[task]["acceptance_ready"] = report_acceptance_ready()
        elif task in NV_SEGMENT_MODES:
            report[task]["acceptance_ready"] = acceptance_ready
    # Native tasks have the same installed-runtime contract as the production catalog.
    for task in set(tasks or []) - REQUIREMENTS.keys():
        missing = []
        try:
            from totalsegmentator.map_to_binary import class_map
            if task not in class_map: missing.append("unknown TotalSegmentator task")
        except ImportError:
            missing.append("TotalSegmentator runtime")
        if not (Path(sys.executable).parent / "TotalSegmentator").is_file() and not shutil.which("TotalSegmentator"):
            missing.append("TotalSegmentator executable")
        missing.extend(name for name in COMMON if not (APP_ROOT / name).is_file())
        report[task] = {"ready": not missing, "missing": missing, "orientation_ready": True, "native": True}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", help="Audit only these external or native tasks")
    args = parser.parse_args()
    json.dump(audit(args.tasks), sys.stdout, sort_keys=True)
    print()
