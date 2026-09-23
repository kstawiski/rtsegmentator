"""Report whether each external model has its runtime, source, and weights installed."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODEL_ROOT = Path(os.environ.get("RTSEG_MODEL_ROOT", "/models"))
RUNTIME_ROOT = Path(os.environ.get("RTSEG_RUNTIME_ROOT", MODEL_ROOT))
APP_ROOT = Path(os.environ.get("RTSEG_APP_ROOT", Path(__file__).resolve().parent))

COMMON = ["dicom_series_to_nifti.py", "convert_validate_rtstruct.py"]
REQUIREMENTS = {
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
    "hnlnl_2d3d": ["envs/nnunet-v1/bin/nnUNet_predict", "envs/nnunet-v1/bin/nnUNet_ensemble", "weights/hnlnl/model"],
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
    REQUIREMENTS[task] = ["@combine_sct_outputs.py", "runtime-homes/sct/sct_7.3/bin/sct_deepseg",
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


def audit() -> dict[str, dict]:
    report = {}
    for task, requirements in REQUIREMENTS.items():
        missing = []
        for requirement in [*("@" + item for item in COMMON), *requirements]:
            if requirement.startswith("@"):
                path = APP_ROOT / requirement[1:]
            elif requirement.startswith("envs/"):
                path = RUNTIME_ROOT / requirement
            else:
                path = MODEL_ROOT / requirement
            if not path.exists():
                missing.append(requirement.lstrip("@"))
        orientation = MODEL_ROOT / "orientation-validation" / f"{task}.json"
        report[task] = {"ready": not missing, "missing": missing,
                        "orientation_ready": orientation.is_file()}
    return report


if __name__ == "__main__":
    json.dump(audit(), sys.stdout, sort_keys=True)
    print()
