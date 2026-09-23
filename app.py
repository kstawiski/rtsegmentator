from __future__ import annotations

import json
import io
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pydicom
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

BASE = Path(os.environ.get("PORTAL_DATA", Path(__file__).parent / "data"))
WORKER = os.environ.get("SEGMENTATION_WORKER", "worker@rtsegmentator-worker")
WORKER_MODE = os.environ.get("WORKER_MODE", "ssh").lower()
if WORKER_MODE not in {"local", "ssh"}:
    raise RuntimeError("WORKER_MODE must be 'local' or 'ssh'")
REMOTE_BASE = os.environ.get("REMOTE_JOB_ROOT", "/home/worker/jobs")
REMOTE_PYTHON = os.environ.get("REMOTE_PYTHON", "/opt/rtsegmentator/venv/bin/python")
REMOTE_RUNNER = os.environ.get("REMOTE_RUNNER", "/app/run_task.py")
REMOTE_APP_ROOT = os.environ.get("REMOTE_APP_ROOT", str(Path(REMOTE_RUNNER).parent))
REMOTE_MODEL_ROOT = os.environ.get("REMOTE_MODEL_ROOT", "/models")
REMOTE_RUNTIME_ROOT = os.environ.get("REMOTE_RUNTIME_ROOT", REMOTE_MODEL_ROOT)
LOCAL_PYTHON = os.environ.get("LOCAL_PYTHON", sys.executable)
LOCAL_RUNNER = os.environ.get("LOCAL_RUNNER", str(Path(__file__).parent / "run_task.py"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024**3)))
RETENTION_HOURS = int(os.environ.get("RETENTION_HOURS", "24"))
ENABLED_EXTERNAL_MODELS = {name.strip() for name in os.environ.get("ENABLED_EXTERNAL_MODELS", "").split(",") if name.strip()}
BASE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DICOM RT Segmentation Portal", version="1.0.0",
              docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")
queue_lock = threading.Lock()

FALLBACK_TASKS = ["total", "total_mr", "total_v3", "body", "body_mr", "vertebrae_mr", "lung_vessels", "lung_vessels_LEGACY", "cerebral_bleed", "hip_implant", "coronary_arteries", "coronary_arteries_LEGACY", "pleural_pericard_effusion", "head_glands_cavities", "head_muscles", "headneck_bones_vessels", "headneck_muscles", "liver_vessels", "liver_segments", "liver_segments_mr", "liver_lesions", "liver_lesions_mr", "oculomotor_muscles", "lung_nodules", "kidney_cysts", "breasts", "ventricle_parts", "tissue_types", "tissue_types_mr", "tissue_4_types", "brain_structures", "vertebrae_body", "face", "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr", "appendicular_bones", "appendicular_bones_mr", "aortic_sinuses", "heartchambers_highres", "trunk_cavities", "brain_aneurysm"]
EXTERNAL_TASKS = {
    "dotatate_pet_lesions": {
        "modality": "PT",
        "description": "Publisher-supplied PET-only five-fold ONNX ensemble for somatostatin-receptor PET",
        "structures": "DOTATATE-avid neuroendocrine-tumor lesions",
        "reference": "https://zenodo.org/records/7843469",
    },
    "rano2_assist_t1c_mr": {
        "modality": "MR",
        "description": "RANO2.0-assist single-sequence DynUNet for post-contrast T1 brain MRI, with automatic HD-BET preprocessing",
        "structures": "Non-enhancing tumor core, peritumoral edema, enhancing tumor core and resection cavity",
        "reference": "https://zenodo.org/records/15411078",
    },
    "lungtumormask_ct": {
        "modality": "CT",
        "description": "LungTumorMask v1.3.1 public CT checkpoint with lungmask localization",
        "structures": "Primary lung tumor candidate",
        "reference": "https://github.com/VemundFredriksen/LungTumorMask",
    },
    "synthseg_v1_brain": {
        "modality": "MR", "modalities": ["MR", "CT"],
        "description": "SynthSeg 1.0 contrast- and resolution-robust brain segmentation (CPU-isolated legacy runtime)",
        "structures": "31 bilateral cortical, white-matter, deep-gray, ventricular, cerebellar, hippocampal, amygdala and brainstem labels",
        "reference": "https://github.com/BBillot/SynthSeg",
    },
    "mrsegmentator": {
        "modality": "MR", "modalities": ["MR", "CT"],
        "description": "MRSegmentator v2 broad thoracic, abdominal and pelvic anatomy model",
        "structures": "40 organs, vessels, bones and muscle groups; validated by upstream for heterogeneous MRI and CT",
        "reference": "https://github.com/hhaentze/MRSegmentator",
    },
    "mrsegmentator_body_comp": {
        "modality": "MR",
        "description": "MRSegmentator v2 MRI body-composition model",
        "structures": "Subcutaneous and visceral fat, paired abdominal muscles, abdominal subcutaneous fat and gluteofemoral fat",
        "reference": "https://github.com/hhaentze/MRSegmentator",
    },
    "totalspineseg": {
        "modality": "MR", "modalities": ["MR", "CT"],
        "description": "Two-stage TotalSpineSeg instance segmentation and anatomical level labeling",
        "structures": "Spinal cord, spinal canal, C1-L7 vertebrae, sacrum and level-specific intervertebral discs",
        "reference": "https://github.com/neuropoly/totalspineseg",
    },
    "raidionics_mediastinal_lymphnodes": {
        "modality": "CT",
        "description": "Mediastinal visible lymph-node candidates (Bouget/Raidionics)",
    },
    "pediatric_thoracic_lymphoma": {
        "modality": "CT",
        "description": "Thoracic lymphoma candidates; pediatric contrast-CT model (protocol warning)",
    },
    "isrt_ct_resunet": {
        "modality": "CT",
        "description": "Pediatric Hodgkin ISRT CTV, CT-only ResUNet 3-model ensemble",
    },
    "isrt_ct_segresnet": {
        "modality": "CT",
        "description": "Pediatric Hodgkin ISRT CTV, CT-only SegResNet 3-model ensemble",
    },
    "isrt_ct_swinunetr": {
        "modality": "CT",
        "description": "Pediatric Hodgkin ISRT CTV, CT-only SwinUNETR 3-model ensemble",
    },
    "lnsegfm_nnunet": {
        "modality": "CT",
        "description": "Visible head-and-neck lymph-node candidates (LN-Seg-FM nnU-Net)",
    },
    "lnsegfm_resenc_m": {
        "modality": "CT",
        "description": "Visible head-and-neck lymph-node candidates (LN-Seg-FM ResEnc-M)",
    },
    "lnsegfm_resenc_l": {
        "modality": "CT",
        "description": "Visible head-and-neck lymph-node candidates (LN-Seg-FM ResEnc-L)",
    },
    "lnsegfm_swinunetr": {
        "modality": "CT",
        "description": "Visible head-and-neck lymph-node candidates (LN-Seg-FM SwinUNETR)",
    },
    "uwlair_hntsmrg_pre_t2": {
        "modality": "MR",
        "description": "Pre-treatment T2 head-and-neck GTVp/GTVn (UWLAIR 10-model ensemble)",
    },
    "isrt_pet1_early_deform": {
        "modality": "PT", "requires_ct": True,
        "description": "PET1+CT ISRT CTV, early-fusion SwinUNETR; deformable-registration weights",
    },
    "isrt_pet1_early_rigid": {
        "modality": "PT", "requires_ct": True,
        "description": "PET1+CT ISRT CTV, early-fusion SwinUNETR; rigid-registration weights",
    },
    "isrt_pet1_late_deform": {
        "modality": "PT", "requires_ct": True,
        "description": "PET1+CT ISRT CTV, late-fusion SwinUNETR; deformable-registration weights",
    },
    "isrt_pet1_late_rigid": {
        "modality": "PT", "requires_ct": True,
        "description": "PET1+CT ISRT CTV, late-fusion SwinUNETR; rigid-registration weights",
    },
}

MOOSE_TASKS = {
    "moose_clin_ct_body": "Body regions: legs, body, head and arms",
    "moose_clin_ct_cardiac": "Myocardium, cardiac chambers and major thoracoabdominal vessels",
    "moose_clin_ct_digestive": "Colon, duodenum, esophagus and small bowel",
    "moose_clin_ct_lungs": "Five pulmonary lobes",
    "moose_clin_ct_muscles": "Paired autochthonous, gluteal and iliopsoas muscles",
    "moose_clin_ct_organs": "19 major thoracic, abdominal and pelvic organs",
    "moose_clin_ct_peripheral_bones": "31 named peripheral bones and bone groups",
    "moose_clin_ct_ribs": "Left and right ribs plus sternum",
    "moose_clin_ct_vertebrae": "C1-L6 vertebrae, sacrum and hips",
    "moose_clin_ct_body_composition": "Skeletal muscle, subcutaneous fat and visceral fat",
}
for _name, _structures in MOOSE_TASKS.items():
    EXTERNAL_TASKS[_name] = {
        "modality": "CT", "description": "MOOSE 3.2 clinical CT segmentation model",
        "structures": _structures, "reference": "https://pypi.org/project/moosez/",
    }

NNUNET_V2_TASKS = json.loads((Path(__file__).parent / "catalog/nnunet-v2.json").read_text())
for _name, _details in NNUNET_V2_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"model_folder", "checkpoint", "channels"}}

NNUNET_V1_TASKS = json.loads((Path(__file__).parent / "catalog/nnunet-v1.json").read_text())
for _name, _details in NNUNET_V1_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"task", "configuration", "trainer", "plans", "labels", "disable_tta"}}

MONAI_BUNDLE_TASKS = json.loads((Path(__file__).parent / "catalog/monai-bundles.json").read_text())
for _name, _details in MONAI_BUNDLE_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"bundle", "config", "checkpoint", "labels", "labels_from_metadata", "trusted_architecture_pickle", "datalist_records", "output_relative", "run_id", "remove_network_img_size", "overrides"}}

CADS_OPEN_TASKS = json.loads((Path(__file__).parent / "catalog/cads-open.json").read_text())
for _name, _details in CADS_OPEN_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"task_id", "labels"}}

UNIVERSAL_MODEL_TASKS = json.loads((Path(__file__).parent / "catalog/universal-model.json").read_text())
for _name, _details in UNIVERSAL_MODEL_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"backbone", "checkpoint"}}

MRANNOTATOR_TASKS = json.loads((Path(__file__).parent / "catalog/mrannotator.json").read_text())
for _name, _details in MRANNOTATOR_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"model_folder", "labels"}}

PYCAD_TASKS = json.loads((Path(__file__).parent / "catalog/pycad-model-zoo.json").read_text())
for _name, _details in PYCAD_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"model_folder", "labels"}}

ANTSPYNET_TASKS = json.loads((Path(__file__).parent / "catalog/antspynet.json").read_text())
for _name, _details in ANTSPYNET_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"runner_task", "labels", "labels_file"}}

SCT_TASKS = json.loads((Path(__file__).parent / "catalog/sct.json").read_text())
for _name, _details in SCT_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"sct_task", "output_names", "multiclass_labels", "labels_file", "label_vert"}}

GOUHFI_TASKS = json.loads((Path(__file__).parent / "catalog/gouhfi.json").read_text())
for _name, _details in GOUHFI_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items() if k not in {"mode", "lut"}}

LUNGMASK_TASKS = {
    "lungmask_r231": ("R231", "Right and left lungs, robust to dense pathology"),
    "lungmask_ltrc_lobes": ("LTRCLobes", "Five pulmonary lobes"),
    "lungmask_ltrc_lobes_r231": ("LTRCLobes_R231", "Five pulmonary lobes fused with robust whole-lung R231 predictions"),
}
for _name, (_model, _structures) in LUNGMASK_TASKS.items():
    EXTERNAL_TASKS[_name] = {
        "modality": "CT", "description": f"lungmask {_model} chest CT segmentation",
        "structures": _structures, "reference": "https://github.com/JoHof/lungmask",
    }

EXTERNAL_TASKS["mrisegmentator"] = {
    "modality": "MR",
    "description": "MRISegmentator abdominal and thoracic T1-weighted MRI model",
    "structures": "62 organs, vessels, muscles, bones and vertebral/rib levels",
    "reference": "https://github.com/rsummers11/MRISegmenter",
}
EXTERNAL_TASKS["vibesegmentator"] = {
    "modality": "MR",
    "description": "VIBESegmentator model 100 for broad torso MRI/CT anatomy",
    "structures": "72 organs, vessels, bones, muscles, body-composition and spine classes",
    "reference": "https://github.com/robert-graf/VIBESegmentator",
}
EXTERNAL_TASKS["dentalsegmentator"] = {
    "modality": "CT",
    "description": "DentalSegmentator CT/CBCT dental and maxillofacial anatomy model",
    "structures": "Upper skull, mandible, upper teeth, lower teeth and mandibular canal",
    "reference": "https://zenodo.org/records/10829675",
}
EXTERNAL_TASKS["hd_bet_mr"] = {
    "modality": "MR",
    "description": "HD-BET v2 robust MRI brain-extraction model",
    "structures": "Brain mask",
    "reference": "https://github.com/MIC-DKFZ/HD-BET",
}
EXTERNAL_TASKS["hd_ctbet_ct"] = {
    "modality": "CT",
    "description": "HD-CTBET head-CT brain-extraction model",
    "structures": "Brain mask",
    "reference": "https://github.com/CAAI/HD-CTBET",
}
EXTERNAL_TASKS["siam_v3_head_mr_ct"] = {
    "modality": "MR", "modalities": ["MR", "CT"],
    "description": "SIAM v3 contrast- and resolution-robust whole-head tissue segmentation with anomaly candidate class",
    "structures": "Gray and white matter, CSF, deep nuclei, hippocampus, dura, vessels, skull, head and anomaly candidates",
    "reference": "https://github.com/romainVala/SIAM",
}

# Audited releases that must remain visible even when they cannot safely be
# submitted yet.  This prevents "not in the picker" from being mistaken for
# "not assessed" and gives users the exact upstream model and output meaning.
INFORMATIONAL_MODELS = {
    "sct_tumor_edema_cavity_t1_t2": {"modality":"MR", "enabled":False, "status":"requires_registered_t1_t2_spine_mr",
        "structures":"Spinal cord tumor, edema and cavity",
        "description":"The official SCT 7.3 source and checkpoint are installed. This task requires separate co-registered T1- and T2-weighted spine-MR volumes in a fixed two-channel contract; the current portal selects one DICOM series.",
        "reference":"https://spinalcordtoolbox.com/stable/user_section/command-line/sct_deepseg.html"},
    "matto_gbm_remaining_models": {"modality":"MR", "modalities":["MR","PT"], "enabled":False, "status":"installed_input_contracts_pending",
        "structures":"Edema, MR/PET PTV and rPTV, PET uptake, brain, brainstem, eyes, hippocampi, lenses and retinas",
        "description":"All twelve checksum-verified MATTO archives and inference source are installed on NFS. The enhancing-T1wCE checkpoint is enabled separately. Remaining models require FLAIR, SUV FET-PET, multiple derived segmentations, atlas/PSR orientation, or eyes-masked preprocessing contracts that are not yet represented by the portal.",
        "reference":"https://zenodo.org/records/21453119"},
    "trackrad_cine_mr_tracking": {"modality":"MR", "enabled":False, "status":"requires_cine_series_and_first_frame_mask",
        "structures":"Temporally propagated tumor mask for every sagittal cine-MRI frame",
        "description":"The checksum-verified 468 MB TrackRAD model and inference source are installed. The model requires a cine-MRI time series plus an explicit first-frame target mask and emits a temporal MHA sequence; the current portal accepts a static DICOM series and produces RTSTRUCT.",
        "reference":"https://zenodo.org/records/20664496"},
    "hd_glio_auto_four_sequence_mr": {"modality":"MR", "enabled":False, "status":"requires_four_mr_series_and_registration_pipeline",
        "structures":"Glioma tumor segmentation",
        "description":"The upstream inference source and checksum-verified v2 checkpoint are installed. Its published container contract requires native T1, post-contrast T1, T2 and FLAIR series, followed by brain extraction and rigid registration. A single selected series cannot satisfy this contract.",
        "reference":"https://github.com/NeuroAI-HD/HD-GLIO-AUTO"},
    "bamf_fdg_petct_lesions": {"modality":"PT", "enabled":False, "status":"requires_registered_pet_ct",
        "structures":"FDG-avid lesions plus released auxiliary organ classes",
        "description":"The checksum-verified five-fold Task762 checkpoint is installed. Its plans require two channels in fixed order—CT then PET—so it cannot run from the current single-series job contract.",
        "reference":"https://zenodo.org/records/8290055"},
    "myeloma_spine_lesions_vmi40": {"modality":"CT", "enabled":False, "status":"requires_registered_vmi40_and_spine_mask",
        "structures":"Osteolytic spinal multiple-myeloma lesions",
        "description":"All four released lesion checkpoints are installed. Each requires a VMI 40-keV volume plus a derived binary spine-mask channel and the publisher's spine-based crop/reconstruction pipeline; the current job contract accepts one selected DICOM series.",
        "reference":"https://zenodo.org/records/18598645"},
    "hd_bm_four_sequence_mr": {"modality":"MR", "enabled":False, "status":"requires_four_registered_mr_series",
        "structures":"Brain metastases",
        "description":"The complete five-fold HD-BM checkpoint and source are installed, but this model requires registered, brain-extracted T1, contrast-enhanced T1, FLAIR and T1-subtraction volumes.",
        "reference":"https://zenodo.org/records/7024601"},
    "hd_bm_slim_two_sequence_mr": {"modality":"MR", "enabled":False, "status":"requires_registered_t1ce_flair",
        "structures":"Brain metastases",
        "description":"The complete five-fold HD-BM Slim checkpoint and source are installed, but this model requires registered, brain-extracted contrast-enhanced T1 and FLAIR volumes. The current job contract selects one series.",
        "reference":"https://zenodo.org/records/7024601"},
    "dotatate_pet_acct_lesions": {"modality":"PT", "enabled":False, "status":"requires_registered_pet_ct",
        "structures":"DOTATATE-avid neuroendocrine-tumor lesions",
        "description":"The publisher's paired PET/attenuation-correction CT ONNX ensemble is installed, but requires two registered input series. The PET-only ensemble is enabled separately.",
        "reference":"https://zenodo.org/records/7843469"},
    "rano2_assist_four_sequence_mr": {"modality":"MR", "enabled":False, "status":"requires_four_registered_mr_series",
        "structures":"Non-enhancing tumor core, edema, enhancing tumor core and resection cavity",
        "description":"The four-modality RANO2.0-assist checkpoint is installed but requires registered T1c, T1, FLAIR and T2 volumes. The separately released T1c-only checkpoint is enabled.",
        "reference":"https://zenodo.org/records/15411078"},
    "accurate_petct_desktop": {"modality":"PT", "enabled":False, "status":"windows_idl_application_no_reusable_checkpoint",
        "structures":"Oncology PET/CT metabolic tumor volumes",
        "description":"The checksum-verified release is a compiled Windows IDL desktop application and runtime, not a portable model checkpoint or Linux inference package suitable for this service.",
        "reference":"https://zenodo.org/records/3908203"},
    "flare2023_coarse_to_fine_ct": {"modality":"CT", "enabled":False, "status":"checkpoint_only_missing_architecture_and_preprocessing",
        "structures":"FLARE abdominal organs and tumors",
        "description":"All four PyTorch state dictionaries are installed and checksum-verified, but the release provides no inference source, plans, dataset metadata, architecture constructor or label map needed to reproduce inference safely.",
        "reference":"https://zenodo.org/records/8372563"},
    "kidney_dixon_four_channel_mr": {"modality":"MR", "enabled":False, "status":"requires_four_registered_dixon_series",
        "structures":"Right and left kidneys",
        "description":"The nnU-Net v2 checkpoint is installed, but requires four registered post-contrast Dixon volumes in the exact order out-of-phase, in-phase, water and fat. The current job contract selects one series.",
        "reference":"https://zenodo.org/records/15328218"},
    "urinary_oars_t2_mr": {"modality":"MR", "enabled":False, "status":"weights_not_in_release",
        "structures":"Bladder, prostate, intraprostatic urethra, rectum, ureters, bladder neck, bladder trigone, bulbous urethra and membranous urethra",
        "description":"The public repository and Zenodo v1.0.0 release contain source, plans and the label map, but the sole 152 kB archive contains no trained checkpoint despite the release documentation calling it a weights download.",
        "reference":"https://zenodo.org/records/17749267"},
    "pelvic_13_oar_unet": {"modality":"CT", "enabled":False, "status":"no_checkpoint_thesis_only",
        "structures":"Thirteen pelvic organs at risk described by the thesis",
        "description":"Zenodo record 18875608 contains only a 4.7 MB thesis PDF and no source package or model checkpoint, so this catalog lead is not downloadable as a runnable model.",
        "reference":"https://zenodo.org/records/18875608"},
    "bamf_brats19_four_sequence_mr": {"modality":"MR", "enabled":False, "status":"requires_four_registered_mr_series",
        "structures":"Edema, enhancing tumor and non-enhancing tumor",
        "description":"BAMF nnU-Net v2 BraTS19 five-fold checkpoint is installed, but requires registered T1, post-contrast T1, T2 and FLAIR volumes in a fixed four-channel order.",
        "reference":"https://zenodo.org/records/11582627"},
    "antspynet_wmh_flair_t1": {"modality":"MR", "enabled":False, "status":"requires_registered_flair_t1",
        "structures":"White-matter hyperintensities",
        "description":"ANTsPyNet WMH segmentation requires a FLAIR series plus a co-registered T1 series. The single-series portal contract cannot safely supply both inputs.",
        "reference":"https://github.com/ANTsX/ANTsPyNet"},
    "mars_wmh_flair_t1_mr": {"modality":"MR", "enabled":False, "status":"requires_registered_flair_t1",
        "structures":"White-matter hyperintensities",
        "description":"MARS-WMH nnU-Net; official checkpoint and source are installed, but inference requires a selected FLAIR plus a T1-weighted series and registration of T1 into FLAIR space.",
        "reference":"https://github.com/miac-research/MARS-WMH"},
    "monai_brats_mri": {"modality":"MR", "enabled":False, "status":"requires_four_registered_mr_series",
        "structures":"Tumor core, whole tumor and enhancing tumor",
        "description":"MONAI BraTS SegResNet requires four co-registered MRI volumes in checkpoint order: T1c, T1, T2 and FLAIR. The complete bundle is installed, but a single selected DICOM series cannot satisfy its input contract.",
        "reference":"https://huggingface.co/MONAI/brats_mri_segmentation"},
    "monai_renal_cect_triphasic": {"modality":"CT", "enabled":False, "status":"requires_triphasic_ct",
        "structures":"Renal artery, vein, ureter, neoplasm and parenchyma",
        "description":"MONAI renalStructures_CECT SegResNet requires co-registered arterial, venous and excretory CT phases. The complete bundle is installed, but a single selected DICOM series cannot satisfy its three-channel contract.",
        "reference":"https://huggingface.co/MONAI/renalStructures_CECT_segmentation"},
    "hnlnl_2d3d": {"modality":"CT", "enabled":True, "status":"validated",
        "structures":"Ia, bilateral Ib, II, III, IVa, IVb, V, VIIb and VIII; plus VIa, VIb and VIIa",
        "description":"HNLNL five-fold 2D+3D ensemble for 20 elective head-and-neck nodal levels. The exact released label order was recovered from the paper's blinded-review supplement.",
        "reference":"https://github.com/putzfn/HNLNL_autosegmentation_trained_models"},
    "choi_tmli": {"modality":"CT", "enabled":False, "status":"validated_ontology_pending",
        "structures":"Cervical, axillary, mediastinal, para-aortic, iliac, obturator, presacral and inguinal lymphatic regions",
        "description":"Whole-body lymphatic-region model for total marrow and lymphoid irradiation. Inference was repaired by restoring the release's stored-pixel +1024 intensity convention; exact numeric-to-region/laterality mapping is still absent upstream.",
        "reference":"https://zenodo.org/records/7839889"},
    "clnet_235": {"modality":"CT", "enabled":False, "status":"packaging_blocked",
        "structures":"235 classes, reported to include 33 lymph-node stations; public station index map is incomplete",
        "description":"CL-Net whole-body continual segmentation checkpoint.",
        "reference":"https://github.com/alibaba-damo-academy/clNet"},
    "segrap2023_gtv": {"modality":"CT", "enabled":False, "status":"requires_paired_ct",
        "structures":"Primary nasopharyngeal GTVp and nodal GTVnd",
        "description":"SegRap 2023 Task 2; requires paired non-contrast and contrast-enhanced planning CT.",
        "reference":"https://github.com/Astarakee/segrap2023"},
    "lnq_inguinal_v1": {"modality":"CT", "enabled":True, "status":"validated_fold0",
        "structures":"Individual visible inguinal lymph-node candidates",
        "description":"LNQ candidate detector, validated with public fold 0; not an elective inguinal CTV model.",
        "reference":"https://github.com/pieper/lnq-segmenter"},
    "lnq_abdominopelvic_v1": {"modality":"CT", "enabled":True, "status":"validated_fold0",
        "structures":"Individual visible abdominal and pelvic lymph-node candidates",
        "description":"LNQ candidate detector, validated with public fold 0; not an elective pelvic CTV model.",
        "reference":"https://github.com/pieper/lnq-segmenter"},
    "lnq_axillary_v1": {"modality":"CT", "enabled":True, "status":"validated_fold0",
        "structures":"Individual visible axillary lymph-node candidates",
        "description":"LNQ candidate detector, validated with public fold 0; not an axillary-level CTV model.",
        "reference":"https://github.com/pieper/lnq-segmenter"},
    "lnq_mediastinal_v1": {"modality":"CT", "enabled":True, "status":"validated",
        "structures":"Individual visible mediastinal lymph-node candidates",
        "description":"LNQ five-fold mediastinal candidate detector.",
        "reference":"https://github.com/pieper/lnq-segmenter"},
    "compai_lnq": {"modality":"CT", "enabled":True, "status":"validated",
        "structures":"Visible mediastinal lymph-node candidates",
        "description":"CompAI technical-report Model 7, a one-channel CT nnU-Net for visible mediastinal lymph-node candidates.",
        "reference":"https://gitlab.lrz.de/compai/MediastinalLymphNodeSegmentation"},
    "dbdmp_lnq": {"modality":"CT", "enabled":True, "status":"validated",
        "structures":"Visible mediastinal lymph-node candidates",
        "description":"DBDMP VNetv2 for visible mediastinal lymph-node candidates; the portal repairs the release's erroneous crop geometry before inference and applies its official refinement.",
        "reference":"https://github.com/WltyBY/LNQ2023_training_code"},
    "lnq_atlas": {"modality":"CT", "enabled":False, "status":"weights_unavailable",
        "structures":"Visible mediastinal lymph-node candidates with probabilistic-atlas context",
        "description":"LNQ2023 probabilistic-atlas model; published weight share currently requires credentials.",
        "reference":"https://github.com/MICAI-IMI-UzL/LNQ2023"},
    "wcode_pia": {"modality":"CT", "enabled":False, "status":"weights_unavailable",
        "structures":"Visible mediastinal lymph-node candidates",
        "description":"Partial-label-aware LNQ model; checkpoint distribution is inaccessible from the worker.",
        "reference":"https://github.com/HiLab-git/WCODE-PIA"},
    "stunet_hntsmrg_pre": {"modality":"MR", "enabled":False, "status":"packaging_blocked",
        "structures":"Pre-treatment head-and-neck GTVp and GTVn",
        "description":"STU-Net HNTS-MRG checkpoint; released preprocessing plans and export metadata are missing.",
        "reference":"https://github.com/Duskwang/Weight"},
    "pam_prompted": {"modality":"CT", "modalities":["CT","MR","PT"], "enabled":True, "prompt":"mask_or_slice",
        "status":"validated", "structures":"One prompt-selected 3D tumour, lymph node, organ or other connected target volume",
        "description":"PAM propagates a positive point or 2D box seed from one axial slice through a CT, MR or PET volume. Tested end-to-end on MR with a strictly validated RTSTRUCT.",
        "reference":"https://github.com/czifan/PAM"},
    "sat3d_prompted": {"modality":"CT", "modalities":["CT","MR","PT"], "enabled":True, "prompt":"points",
        "status":"validated", "structures":"One prompt-selected 3D tumour volume, including primary tumour or GTVn where applicable",
        "description":"SAT3D interactive 3D tumour segmentation in a prompt-centred 128-voxel ROI. Requires at least one positive point and accepts additional negative points. Tested end-to-end on MR with a strictly validated RTSTRUCT.",
        "reference":"https://github.com/himashi92/SAT3D"},
    "sam_med2d_prompted": {"modality":"CT", "modalities":["CT","MR","PT"], "enabled":True, "prompt":"points_or_box",
        "status":"validated_slice_wise", "structures":"One prompt-selected 2D object contour on each explicitly prompted slice",
        "description":"SAM-Med2D slice-wise segmentation from positive/negative points or a box. Add prompts on every slice that should be contoured; it does not infer unprompted slices. Tested end-to-end on MR with a strictly validated RTSTRUCT.",
        "reference":"https://github.com/OpenGVLab/SAM-Med2D"},
    "biomedparse_prompted": {"modality":"CT", "modalities":["CT","MR","PT"], "enabled":False, "prompt":"text",
        "status":"weights_unavailable", "structures":"Text-selected biomedical object or tumour volume",
        "description":"BiomedParse text-prompted segmentation; checkpoint repository is approval-gated.",
        "reference":"https://github.com/microsoft/BiomedParse"},
}

def task_is_mr(name: str) -> bool:
    return name.endswith("_mr") or "_mr_" in name or name == "brain_aneurysm"

IMAGE_MODALITIES = {"CT", "MR", "PT"}
CLINICAL_TAGS = json.loads((Path(__file__).parent / "catalog/clinical-tags.json").read_text())
ORIENTATION_VALIDATION = Path(os.environ.get(
    "RTSEG_ORIENTATION_VALIDATION",
    "/models/orientation-validation"))

def clinical_tags(item: dict) -> list[dict[str, str]]:
    """Return curated disease-site discovery tags; tags describe usefulness, not model indication."""
    searchable = " ".join(str(item.get(key, "")) for key in ("name", "description", "structures")).lower()
    return [{"id": tag_id, "label": rule["label"]}
            for tag_id, rule in CLINICAL_TAGS.items()
            if any(term.lower() in searchable for term in rule["terms"])]

def task_modality(name: str) -> str:
    # TotalSegmentator currently has CT and MR task families. PET-capable
    # external models are added separately after their RTSTRUCT acceptance test.
    details = EXTERNAL_TASKS.get(name) or INFORMATIONAL_MODELS.get(name, {})
    return details.get("modality", "MR" if task_is_mr(name) else "CT")

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DICOM RT Segmentation</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#172033;background:#f4f7fb}body{max-width:1180px;margin:35px auto;padding:0 20px}h1{margin-bottom:5px}.sub{color:#617087;margin-top:0}.card{background:white;border:1px solid #dce4ee;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 5px 20px #2030500d}.drop{border:2px dashed #93a8c2;border-radius:10px;padding:24px;text-align:center}.tasks{max-height:520px;overflow:auto;border:1px solid #dce4ee;border-radius:10px}.task-group h3{position:sticky;top:0;background:#edf4fc;margin:0;padding:10px 12px;border-top:1px solid #d5e0ed;z-index:1}.model-table{width:100%;border-collapse:collapse;table-layout:fixed}.model-table th,.model-table td{padding:9px 10px;border-top:1px solid #e6ebf2;text-align:left;vertical-align:top}.model-table th{font-size:.8rem;color:#617087;background:#f8fafc}.model-table tr:hover{background:#f6f9fd}.model-table .pick{width:42px;text-align:center}.model-table .name{width:25%;overflow-wrap:anywhere}.model-table .status-col{width:14%}.unavailable{opacity:.62}.badge{display:inline-block;border-radius:999px;padding:2px 7px;background:#e8eef7;font-size:.75rem}.badge.ready{background:#def5e8;color:#17623a}.prompt-help{background:#f5f8fc;border:1px solid #d9e3ef;border-radius:9px;padding:12px;margin:12px 0}.prompt-help pre{white-space:pre-wrap;background:#172033;color:#eef5ff;padding:9px;border-radius:6px;font-size:.78rem}.progress-track{height:10px;background:#e2e8f0;border-radius:99px;overflow:hidden;margin:8px 0;max-width:620px}.progress-bar{height:100%;background:#1769d2;transition:width .4s}.progress-bar.active{width:35%;animation:activity 1.5s ease-in-out infinite}@keyframes activity{0%{transform:translateX(-110%)}100%{transform:translateX(300%)}}button{background:#1769d2;color:white;border:0;border-radius:8px;padding:11px 18px;font-weight:650;cursor:pointer}button:disabled{opacity:.5}input[type=file]{margin:12px}.job{border-top:1px solid #e2e8f0;padding:12px 0}.status{font-weight:650;text-transform:capitalize}.error{color:#a21b1b;white-space:pre-wrap}.warn{background:#fff7db;border-left:4px solid #e3a008;padding:10px}small{color:#68768a}@media(max-width:700px){.model-table .output{display:none}.model-table .name{width:auto}.model-table .status-col{width:24%}}</style></head><body>
<style>.site-tag{display:inline-block;border-radius:999px;padding:2px 7px;margin:4px 3px 0 0;background:#eef2ff;color:#384aa3;font-size:.7rem}.model-filter{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}.model-filter select,.model-filter input{padding:9px;border:1px solid #b9c6d6;border-radius:7px;background:white;min-width:230px}</style>
<h1>DICOM RT Segmentation</h1><p class="sub">Local or remote GPU inference · validated RTSTRUCT output</p>
<div class="warn"><b>Research use only.</b> TotalSegmentator is not a medical device. Review every contour before clinical use. Uploaded studies may contain PHI and are deleted automatically after 24 hours.</div>
<div class="card"><h2>1. Upload and inspect locally</h2><form id="uploadForm"><div class="drop">Select DICOM files or ZIP archives. Nothing is sent to the GPU worker during inspection.<br><input id="files" name="files" type="file" multiple accept=".dcm,.zip,application/dicom,application/zip" required></div><p><button id="inspect">Inspect upload</button></p></form><div id="assessment"></div></div>
<div id="clinicalTools" class="model-filter" hidden><label>Useful for tumor site <select id="clinicalFilter"><option value="">All tumor sites</option></select></label><label>Find model <input id="modelSearch" type="search" placeholder="Name, output, or tag"></label><small>Tags indicate potential planning usefulness, not that a model segments the tumor.</small></div>
<form id="modelForm" class="card" hidden><h2>2. Select one detected series</h2><div id="series"></div><h2>3. Select compatible models</h2><p><button type="button" id="all">Select all available for modality</button> <button type="button" id="none">Clear</button></p><div id="tasks" class="tasks">Loading model catalog…</div><div id="promptPanel" class="card" hidden><h3>Interactive prompt</h3><div id="promptGuidance" class="prompt-help"></div><p>Choose a slice and prompt type, then click the image. The portal builds <code>[column, row, slice]</code> coordinates automatically; you normally do not need to edit the JSON. A box is made with two clicks on the same slice.</p><div><input id="promptSlice" type="range" min="0" max="0" value="0" style="width:75%"> <span id="promptSliceLabel">slice 0</span></div><p><select id="promptMode"><option value="positive">Positive point (inside target)</option><option value="negative">Negative point (exclude area)</option><option value="box">Box corners</option></select> <button type="button" id="clearPrompt">Clear prompt</button></p><img id="promptImage" alt="Selected DICOM slice" style="max-width:512px;width:100%;cursor:crosshair;border:1px solid #8392a8"><details><summary>Advanced: inspect or edit prompt JSON</summary><label>Prompt JSON<br><textarea name="prompt_json" id="promptJson" rows="6" style="width:100%"></textarea></label><pre>{"positive_points":[[256,240,84]],"negative_points":[[310,240,84]],"box":null,"text":"lymph node"}</pre></details></div><p id="modelNote"><small><code>total</code> is selected for CT and <code>total_mr</code> for MR. PET1 ISRT tasks require a CT series in the same Study and Frame of Reference; it is paired automatically and unmatched PET is rejected before transfer. Disabled entries remain visible with their exact validation or upstream-blocker status.</small></p><button id="submit">Queue selected series</button></form><div class="card"><h2>Jobs</h2><div id="jobs">No jobs submitted in this browser.</div></div>
<script>
const taskbox=document.querySelector('#tasks'), jobsEl=document.querySelector('#jobs'); let ids=JSON.parse(localStorage.getItem('dicomJobs')||'[]'), uploadId=null,prompt={positive_points:[],negative_points:[],box:null,text:'lymph node'},boxCorner=null;
function writePrompt(){document.querySelector('#promptJson').value=JSON.stringify(prompt)}
function loadPromptSlice(){let s=document.querySelector('[name=series_key]:checked'),i=+document.querySelector('#promptSlice').value;if(!s||!uploadId)return;document.querySelector('#promptSliceLabel').textContent=`slice ${i+1}/${s.dataset.slices}`;document.querySelector('#promptImage').src=`/api/uploads/${uploadId}/preview?series_key=${encodeURIComponent(s.value)}&index=${i}&_=${Date.now()}`}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function refreshPrompt(){let selected=[...document.querySelectorAll('[name=models]:checked')],prompted=selected.filter(x=>x.dataset.prompt),show=prompted.length>0;document.querySelector('#promptPanel').hidden=!show;if(show){let names=prompted.map(x=>x.value),notes=[];if(names.includes('pam_prompted'))notes.push('<b>PAM:</b> place one positive point inside the target, or draw a tight two-corner box. It propagates that seed through the 3D volume.');if(names.includes('sat3d_prompted'))notes.push('<b>SAT3D:</b> requires at least one positive point inside the tumour. Add negative points in nearby structures that must be excluded.');if(names.includes('sam_med2d_prompted'))notes.push('<b>SAM-Med2D:</b> use points or a box on every slice you want contoured; it does not propagate to unprompted slices.');document.querySelector('#promptGuidance').innerHTML=notes.join('<br><br>')+'<br><br><small><b>Examples:</b> positive point <code>[256,240,84]</code>; positive plus exclusion point <code>[310,240,84]</code>; box corners <code>[[210,190,84],[330,300,84]]</code>.</small>';loadPromptSlice()}}
function isMrTask(n){return n.endsWith('_mr')||n.includes('_mr_')||n==='brain_aneurysm'} function applyModality(reset=true){let chosen=document.querySelector('[name=series_key]:checked');if(!chosen)return;let modality=chosen.dataset.modality;document.querySelectorAll('[name=models]').forEach(x=>{let compatible=x.dataset.modalities.split(',').includes(modality);x.disabled=!compatible||x.dataset.enabled!=='true';if(reset||x.disabled)x.checked=false});let d=document.querySelector(`[name=models][value="${modality==='MR'?'total_mr':modality==='CT'?'total':'__none__'}"]`);if(reset&&d&&!d.disabled)d.checked=true;document.querySelector('#submit').disabled=!document.querySelector('[name=models]:not(:disabled)');refreshPrompt()}
function category(t){if(t.prompt)return 'Interactive prompted models';if(t.enabled===false)return 'Unavailable / pending models';if(t.name.startsWith('gouhfi_'))return 'GOUHFI — brain MRI';if(t.name.startsWith('sct_'))return 'Spinal Cord Toolbox';if(t.name.startsWith('cads_open_'))return 'CADS — open whole-body CT';if(t.name.startsWith('universal_model_'))return 'UniversalModel — organs and tumours';if(t.name.startsWith('pycad_'))return 'PYCAD Model Zoo';if(t.name.startsWith('isrt_'))return 'Involved-site radiotherapy (CTV)';if(t.name.startsWith('moose_'))return 'MOOSE — clinical CT';if(t.name.startsWith('mrannotator_')||['mrsegmentator','mrsegmentator_body_comp','mrisegmentator','vibesegmentator'].includes(t.name))return 'Packaged broad MRI models';if(t.name==='totalspineseg')return 'Spine models';if(t.name.startsWith('lungmask_'))return 'Lung and thoracic models';if(t.name.startsWith('nnunetv1_'))return 'Legacy nnU-Net v1 model zoo';if(t.name.startsWith('bamf_')||t.name==='cervix_cbct')return 'Packaged nnU-Net v2 models';if(/ln|lymph|gtv|tumou?r|hnlnl|raidionics|pediatric/.test(t.name+' '+(t.description||'')))return 'Lymph nodes and tumours';return `TotalSegmentator — ${t.modality}`}
let modelCatalog=[];
function renderCatalog(){let selected=new Set([...document.querySelectorAll('[name=models]:checked')].map(x=>x.value)),site=document.querySelector('#clinicalFilter').value,q=document.querySelector('#modelSearch').value.trim().toLowerCase(),visible=modelCatalog.filter(t=>(!site||(t.clinical_tags||[]).some(x=>x.id===site))&&(!q||[t.name,t.description,t.structures,...(t.clinical_tags||[]).map(x=>x.label)].join(' ').toLowerCase().includes(q))),groups={};visible.forEach(t=>(groups[category(t)]??=[]).push(t));taskbox.innerHTML=visible.length?Object.entries(groups).map(([group,tasks])=>`<section class="task-group"><h3>${esc(group)} <small>(${tasks.length})</small></h3><table class="model-table"><thead><tr><th class="pick">Use</th><th class="name">Model</th><th class="output">Purpose / output</th><th class="status-col">Status</th></tr></thead><tbody>${tasks.map(t=>`<tr class="${t.enabled===false?'unavailable':''}"><td class="pick"><input aria-label="Select ${esc(t.name)}" type="checkbox" name="models" value="${esc(t.name)}" data-modality="${esc(t.modality)}" data-modalities="${esc((t.modalities||[t.modality]).join(','))}" data-enabled="${t.enabled!==false}" data-prompt="${esc(t.prompt||'')}" ${selected.has(t.name)?'checked':''}></td><td class="name"><b>${esc(t.name)}</b>${t.licensed?' 🔑':''}${t.reference?`<br><small><a href="${esc(t.reference)}" target="_blank" rel="noopener">Reference ↗</a></small>`:''}<br>${(t.clinical_tags||[]).map(x=>`<span class="site-tag">${esc(x.label)}</span>`).join('')}</td><td class="output"><small>${esc(t.description||'')}${t.structures?`<br><b>Output:</b> ${esc(t.structures)}`:''}</small></td><td><span class="badge ${t.enabled!==false?'ready':''}">${esc(t.enabled===false?(t.status||'unavailable'):(t.status||'available'))}</span></td></tr>`).join('')}</tbody></table></section>`).join(''):'<p style="padding:12px">No models match this tumor site and search.</p>';document.querySelectorAll('[name=models]').forEach(x=>x.onchange=refreshPrompt);applyModality(false)}
fetch('/api/tasks').then(r=>r.json()).then(x=>{modelCatalog=x.tasks;let f=document.querySelector('#clinicalFilter'),tools=document.querySelector('#clinicalTools');f.innerHTML='<option value="">All tumor sites</option>'+x.clinical_tags.map(t=>`<option value="${esc(t.id)}">${esc(t.label)}</option>`).join('');document.querySelector('#modelForm h2:nth-of-type(2)').after(tools);tools.hidden=false;f.onchange=renderCatalog;document.querySelector('#modelSearch').oninput=renderCatalog;renderCatalog()});
document.querySelector('#all').onclick=()=>{document.querySelectorAll('[name=models]:not(:disabled)').forEach(x=>x.checked=true);refreshPrompt()}; document.querySelector('#none').onclick=()=>{document.querySelectorAll('[name=models]').forEach(x=>x.checked=false);refreshPrompt()};
document.querySelector('#promptSlice').oninput=loadPromptSlice;document.querySelector('#clearPrompt').onclick=()=>{prompt={positive_points:[],negative_points:[],box:null,text:'lymph node'};boxCorner=null;writePrompt()};document.querySelector('#promptImage').onclick=e=>{let s=document.querySelector('[name=series_key]:checked'),r=e.target.getBoundingClientRect(),p=[Math.max(0,Math.min(+s.dataset.cols-1,Math.round((e.clientX-r.left)/r.width*+s.dataset.cols))),Math.max(0,Math.min(+s.dataset.rows-1,Math.round((e.clientY-r.top)/r.height*+s.dataset.rows))),+document.querySelector('#promptSlice').value],m=document.querySelector('#promptMode').value;if(m==='box'){if(!boxCorner)boxCorner=p;else{if(boxCorner[2]!==p[2]){alert('Both box corners must be on the same slice');return}prompt.box=[boxCorner,p];boxCorner=null}}else prompt[m+'_points'].push(p);writePrompt()};writePrompt();
document.querySelector('#uploadForm').onsubmit=async e=>{e.preventDefault();let b=document.querySelector('#inspect');b.disabled=true;b.textContent='Uploading and inspecting…';try{let r=await fetch('/api/uploads',{method:'POST',body:new FormData(e.target)}),x=await r.json();if(!r.ok)throw Error(x.detail||'Inspection failed');uploadId=x.upload_id;document.querySelector('#assessment').innerHTML=`<b>${x.series.length} valid series detected.</b> ${x.ignored} invalid or non-image file(s) will be ignored.`;document.querySelector('#series').innerHTML=x.series.map((s,i)=>`<label class="task" style="display:block"><input type="radio" name="series_key" value="${s.key}" data-modality="${s.modality}" data-slices="${s.slices}" data-rows="${s.rows}" data-cols="${s.columns}" ${i===0?'checked':''}> <b>${s.modality}</b> · ${s.slices} slices · ${s.rows}×${s.columns} · ${s.description||'No series description'} <small>${s.series_uid}</small></label>`).join('');document.querySelectorAll('[name=series_key]').forEach(x=>x.onchange=()=>{applyModality();let s=document.querySelector('[name=series_key]:checked');document.querySelector('#promptSlice').max=+s.dataset.slices-1;document.querySelector('#promptSlice').value=Math.floor(+s.dataset.slices/2);loadPromptSlice()});document.querySelector('#modelForm').hidden=false;let s=document.querySelector('[name=series_key]:checked');document.querySelector('#promptSlice').max=+s.dataset.slices-1;document.querySelector('#promptSlice').value=Math.floor(+s.dataset.slices/2);applyModality()}catch(x){alert(x.message)}finally{b.disabled=false;b.textContent='Inspect upload'}};
document.querySelector('#modelForm').onsubmit=async e=>{e.preventDefault();let b=document.querySelector('#submit');b.disabled=true;b.textContent='Queueing…';try{let fd=new FormData(e.target);let r=await fetch(`/api/uploads/${uploadId}/start`,{method:'POST',body:fd}),x=await r.json();if(!r.ok)throw Error(x.detail||'Could not start job');ids.unshift(x.id);localStorage.setItem('dicomJobs',JSON.stringify(ids.slice(0,30)));document.querySelector('#modelForm').hidden=true;document.querySelector('#assessment').textContent='Selected series queued; unselected files were deleted.';poll()}catch(x){alert(x.message)}finally{b.disabled=false;b.textContent='Queue selected series'}};
function formatBytes(n){if(!Number.isFinite(n))return '';let u=['B','KiB','MiB','GiB'],i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return `${n.toFixed(i?1:0)} ${u[i]}`}
async function retryJob(id){let r=await fetch(`/api/jobs/${encodeURIComponent(id)}/retry`,{method:'POST'}),x=await r.json();if(!r.ok)alert(x.detail||'Retry failed');poll()}
async function poll(){try{let r=await fetch('/api/jobs'),x=await r.json();ids=x.jobs.map(j=>j.id)}catch(e){}let rows=[];for(const id of ids){try{let r=await fetch('/api/jobs/'+id),j=await r.json(),busy=['queued','transferring','processing','retrieving'].includes(j.status),pct=Number.isFinite(j.progress)?j.progress:null,elapsed=busy?Math.max(0,Math.round((Date.now()/1000-(j.updated||j.created||Date.now()/1000)))):0,bar=busy||pct!==null?`<div class="progress-track"><div class="progress-bar ${pct===null?'active':''}" style="${pct!==null?`width:${pct}%`:''}"></div></div>`:'',transfer=j.status==='transferring'&&j.total_bytes?`<small>${formatBytes(j.transferred_bytes||0)} / ${formatBytes(j.total_bytes)} · ${pct||0}%</small>`:busy?`<small>Active · status updated ${elapsed}s ago</small>`:'';rows.push(`<div class="job"><span class="status">${esc(j.status)}</span> · ${esc(id)}<br><small>${j.models.map(esc).join(', ')}</small>${j.detail?`<div>${esc(j.detail)}</div>`:''}${bar}${transfer}${j.error?`<div class="error">${esc(j.error)}</div>`:''}${j.status==='failed'?`<p><button type="button" onclick="retryJob('${esc(id)}')">Retry using retained input</button></p>`:''}${j.status==='complete'?`<p><a href="/api/jobs/${encodeURIComponent(id)}/download"><button>Download RTSTRUCT ZIP</button></a></p>`:''}</div>`)}catch(e){}}jobsEl.innerHTML=rows.join('')||'No server jobs yet.'}poll();setInterval(poll,4000);
</script></body></html>'''

def metadata(job: Path) -> dict:
    try: return json.loads((job / "job.json").read_text())
    except Exception: raise HTTPException(404, "Job not found")

def save_metadata(job: Path, **changes):
    data = metadata(job); data.update(changes); data["updated"] = time.time(); (job / "job.json").write_text(json.dumps(data, indent=2))

def transfer_to_worker(job: Path, source: Path, destination: str):
    """Copy one archive while publishing rsync's real byte progress."""
    total = source.stat().st_size
    save_metadata(job, status="transferring", detail="Sending DICOM series to GPU worker",
                  progress=0, transferred_bytes=0, total_bytes=total)
    command = ["rsync", "-a", "--info=progress2", "--no-inc-recursive",
               str(source), f"{WORKER}:{destination}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    last = -1
    assert process.stdout is not None
    for line in process.stdout:
        match = re.search(r"([0-9,]+)\s+(\d+)%", line)
        if not match: continue
        transferred = int(match.group(1).replace(",", ""))
        percent = int(match.group(2))
        if percent != last:
            save_metadata(job, progress=percent, transferred_bytes=min(transferred, total))
            last = percent
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    save_metadata(job, progress=100, transferred_bytes=total,
                  detail="Transfer complete; preparing worker input")

def run_worker_task(job: Path, remote: str, task: str, index: int, total: int):
    """Run inference while keeping the portal heartbeat visibly current."""
    detail = f"Running {task} ({index}/{total})"
    log = job / f"{task}.log"
    started = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(["ssh", WORKER, "env",
                                    f"RTSEG_APP_ROOT={REMOTE_APP_ROOT}",
                                    f"RTSEG_MODEL_ROOT={REMOTE_MODEL_ROOT}",
                                    f"RTSEG_RUNTIME_ROOT={REMOTE_RUNTIME_ROOT}",
                                    REMOTE_PYTHON, REMOTE_RUNNER, remote, task],
                                   stdout=output, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            if time.monotonic() - started > 12 * 3600:
                process.kill()
                raise subprocess.TimeoutExpired(process.args, 12 * 3600)
            save_metadata(job, status="processing", detail=detail, progress=None,
                          model_index=index, model_total=total)
            time.sleep(5)
    if process.returncode:
        error = log.read_text(errors="replace")[-6000:]
        raise RuntimeError(error or f"{task} exited with status {process.returncode}")

def run_local_task(job: Path, task: str, index: int, total: int):
    detail = f"Running {task} locally ({index}/{total})"
    log = job / f"{task}.log"
    started = time.monotonic()
    with log.open("w") as output:
        environment = os.environ.copy()
        environment.setdefault("RTSEG_APP_ROOT", str(Path(LOCAL_RUNNER).parent))
        environment.setdefault("RTSEG_MODEL_ROOT", "/models")
        environment.setdefault("RTSEG_RUNTIME_ROOT", environment["RTSEG_MODEL_ROOT"])
        process = subprocess.Popen([LOCAL_PYTHON, LOCAL_RUNNER, str(job), task], env=environment,
                                   stdout=output, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            if time.monotonic() - started > 12 * 3600:
                process.kill()
                raise subprocess.TimeoutExpired(process.args, 12 * 3600)
            save_metadata(job,status="processing",detail=detail,progress=None,
                          model_index=index,model_total=total)
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(log.read_text(errors="replace")[-6000:] or
                           f"{task} exited with status {process.returncode}")

def finalize_results(job: Path):
    results = sorted((job / "output").glob("*.dcm"))
    if not results:
        raise RuntimeError("Worker completed without producing an RTSTRUCT file")
    with zipfile.ZipFile(job/"rtstruct-results.zip","w",zipfile.ZIP_DEFLATED) as z:
        for result in results: z.write(result,result.name)
    shutil.rmtree(job/"input",ignore_errors=True)
    shutil.rmtree(job/"input_ct",ignore_errors=True)
    save_metadata(job,status="complete",detail="RTSTRUCT files ready for download",progress=100)

def safe_extract(archive: Path, destination: Path, budget: int = MAX_BYTES):
    with zipfile.ZipFile(archive) as z:
        expanded = sum(i.file_size for i in z.infolist() if not i.is_dir())
        if expanded > budget: raise ValueError("Expanded ZIP exceeds the configured 8 GiB limit")
        for info in z.infolist():
            if info.is_dir(): continue
            name = Path(info.filename).name
            if not name or name.startswith("."): continue
            target = destination / f"{secrets.token_hex(4)}_{name}"
            with z.open(info) as src, target.open("wb") as dst: shutil.copyfileobj(src, dst)

def inspect_upload(input_dir: Path) -> dict:
    groups = {}; ignored = 0
    for f in input_dir.iterdir():
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            modality = str(getattr(ds, "Modality", ""))
            uid = str(getattr(ds, "SeriesInstanceUID", ""))
            if modality not in IMAGE_MODALITIES or not uid or not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                ignored += 1; continue
            key = f"{uid}|{modality}"
            g = groups.setdefault(key, {"key": key, "series_uid": uid, "modality": modality,
                "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "frame_uid": str(getattr(ds, "FrameOfReferenceUID", "")),
                "description": str(getattr(ds, "SeriesDescription", "")), "study_date": str(getattr(ds, "StudyDate", "")),
                "files": [], "rows": int(ds.Rows), "columns": int(ds.Columns)})
            g["files"].append(f.name)
        except Exception: ignored += 1
    series=[]
    for g in groups.values():
        g["slices"]=len(g["files"]); series.append(g)
    series.sort(key=lambda x:(x["modality"],x["description"],x["series_uid"]))
    return {"series":series,"ignored":ignored}

def validate_series(input_dir: Path) -> tuple[int, str]:
    valid = 0; series = set(); modalities = set()
    for f in input_dir.iterdir():
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if getattr(ds, "Modality", "") in IMAGE_MODALITIES and hasattr(ds, "PixelData") is False:
                # PixelData is beyond stop_before_pixels; image tags below are sufficient.
                if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
                    valid += 1; series.add(str(getattr(ds, "SeriesInstanceUID", ""))); modalities.add(ds.Modality)
        except Exception: pass
    if valid == 0: raise ValueError("No CT/MR/PET DICOM image slices were found.")
    if len(series) != 1: raise ValueError(f"Upload exactly one image series; found {len(series)} SeriesInstanceUID values.")
    if len(modalities) != 1: raise ValueError("Upload exactly one modality (CT, MR, or PET).")
    return valid, next(iter(modalities))

def process(job: Path):
    with queue_lock:
        jid=job.name; remote=f"{REMOTE_BASE}/{jid}"
        try:
            if WORKER_MODE == "local":
                data=metadata(job)
                save_metadata(job,status="processing",detail="Starting local GPU segmentation",progress=None)
                for i,task in enumerate(data["models"],1):
                    run_local_task(job, task, i, len(data["models"]))
                finalize_results(job)
                return
            save_metadata(job,status="transferring",detail="Preparing selected DICOM series",progress=0)
            subprocess.run(["ssh",WORKER,"mkdir","-p",f"{remote}/input",f"{remote}/output"],check=True,timeout=30)
            archive=job/"selected-series.tar"
            archive_members = ["input"]
            if (job / "input_ct").is_dir(): archive_members.append("input_ct")
            if (job / "prompt.json").is_file(): archive_members.append("prompt.json")
            subprocess.run(["tar","-C",str(job),"-cf",str(archive),*archive_members],check=True,timeout=600)
            transfer_to_worker(job, archive, f"{remote}/selected-series.tar")
            archive.unlink(missing_ok=True)
            subprocess.run(["ssh",WORKER,f"tar -C {remote} -xf {remote}/selected-series.tar && rm -f {remote}/selected-series.tar"],check=True,timeout=600)
            data=metadata(job); save_metadata(job,status="processing",detail="GPU segmentation in progress",progress=None)
            for i,task in enumerate(data["models"],1):
                run_worker_task(job, remote, task, i, len(data["models"]))
            save_metadata(job,status="retrieving",detail="Retrieving RTSTRUCT results from GPU worker",progress=None)
            subprocess.run(["scp","-q",f"{WORKER}:{remote}/output/*.dcm",str(job/"output")+"/"],check=True,timeout=600)
            finalize_results(job)
            shutil.rmtree(job/"input",ignore_errors=True)
            subprocess.run(["ssh",WORKER,"rm","-rf",remote],timeout=30)
        except Exception as exc:
            if WORKER_MODE == "ssh":
                subprocess.run(["ssh",WORKER,"rm","-rf",remote],timeout=30)
            save_metadata(job,status="failed",detail="Processing failed",error=str(exc))

@app.get("/",response_class=HTMLResponse)
def home(): return HTML

@app.get("/api/tasks")
@app.get("/api/v1/models", tags=["automation"])
def tasks():
    # Catalog is sourced from the installed worker package when available.
    code="from totalsegmentator.map_to_binary import class_map,commercial_models; import json; print(json.dumps({'tasks':sorted(class_map),'licensed':list(commercial_models),'structures':{k:list(v.values()) if isinstance(v,dict) else list(v) for k,v in class_map.items()}}))"
    try:
        if WORKER_MODE == "local":
            cp=subprocess.run([LOCAL_PYTHON,"-c",code],capture_output=True,text=True,timeout=20,check=True)
        else:
            command=f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(code)}"
            cp=subprocess.run(["ssh",WORKER,command],capture_output=True,text=True,timeout=20,check=True)
        remote=json.loads(cp.stdout.strip().splitlines()[-1]); licensed=set(remote["licensed"])
        internal={"test","total_highres_test","total_v1","vertebrae_pp","vertebrae_pp_refined"}
        names=[n for n in remote["tasks"] if n not in internal and not n.endswith("_auxiliary")]
    except Exception: names=FALLBACK_TASKS; licensed=set()
    preflight = {}
    try:
        if WORKER_MODE == "local":
            environment = os.environ.copy()
            environment.setdefault("RTSEG_APP_ROOT", str(Path(LOCAL_RUNNER).parent))
            environment.setdefault("RTSEG_MODEL_ROOT", "/models")
            environment.setdefault("RTSEG_RUNTIME_ROOT", environment["RTSEG_MODEL_ROOT"])
            cp = subprocess.run([LOCAL_PYTHON, str(Path(LOCAL_RUNNER).parent / "model_preflight.py")],
                                env=environment, capture_output=True, text=True, timeout=20, check=True)
        else:
            command = ["ssh", WORKER, "env", f"RTSEG_APP_ROOT={REMOTE_APP_ROOT}",
                       f"RTSEG_RUNTIME_ROOT={REMOTE_RUNTIME_ROOT}",
                       f"RTSEG_MODEL_ROOT={REMOTE_MODEL_ROOT}",
                       REMOTE_PYTHON,
                       str(Path(REMOTE_APP_ROOT) / "model_preflight.py")]
            cp = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
        preflight = json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    structure_map = remote.get("structures",{}) if 'remote' in locals() else {}
    catalog=[{"name":n,"licensed":n in licensed,"modality":task_modality(n),"enabled":True,
              "description":f"TotalSegmentator {task_modality(n)} segmentation task producing the structures listed below.",
              "structures":", ".join(structure_map.get(n,[])) or "Task-specific TotalSegmentator structures",
              "reference":"https://github.com/wasserth/TotalSegmentator"} for n in names]
    for name, details in EXTERNAL_TASKS.items():
        item={"name":name,"licensed":False,"enabled":True,**details}
        if WORKER_MODE == "local" and name not in ENABLED_EXTERNAL_MODELS:
            item.update(enabled=False, status="model_bundle_required")
        check = preflight.get(name)
        if check and not check["ready"]:
            item.update(enabled=False, status="dependencies_missing", missing=check["missing"])
        orientation_ready = (check.get("orientation_ready", False) if check is not None
                             else (ORIENTATION_VALIDATION / f"{name}.json").is_file())
        if not orientation_ready:
            item.update(enabled=False, status="orientation_revalidation_pending")
        if name.startswith("lnsegfm_"):
            item.update(structures="Visible/pathologic head-and-neck lymph-node candidate mask (not elective nodal levels)", reference="https://github.com/HiLab-git/LN-Seg-FM")
        elif name.startswith("isrt_"):
            item.update(structures="Pediatric Hodgkin lymphoma involved-site radiotherapy clinical target volume (CTV)", reference="https://github.com/xtie97/ISRT-CTV-AutoSeg")
        elif name == "raidionics_mediastinal_lymphnodes":
            item.update(structures="Binary visible/enlarged mediastinal lymph-node candidate mask", reference="https://github.com/dbouget/ct_mediastinal_structures_segmentation")
        elif name == "pediatric_thoracic_lymphoma":
            item.update(structures="Binary pediatric thoracic lymphoma mass", reference="https://github.com/fast-radiology/lymphoma")
        elif name == "uwlair_hntsmrg_pre_t2":
            item.update(structures="Primary gross tumour volume (GTVp) and metastatic nodal gross tumour volume (GTVn)", reference="https://github.com/xtie97/HNTS-MRG24-UWLAIR")
        catalog.append(item)
    for name, details in INFORMATIONAL_MODELS.items():
        item={"name":name,"licensed":False,**details}
        if WORKER_MODE == "local" and name not in ENABLED_EXTERNAL_MODELS:
            item.update(enabled=False, status="model_bundle_required")
        check = preflight.get(name)
        if check and not check["ready"]:
            item.update(enabled=False, status="dependencies_missing", missing=check["missing"])
        catalog.append(item)
    for item in catalog:
        item["clinical_tags"] = clinical_tags(item)
    return {"tasks":catalog, "clinical_tags":[{"id": key, "label": value["label"]}
                                                for key, value in CLINICAL_TAGS.items()]}

@app.post("/api/uploads")
async def assess_upload(files: list[UploadFile]=File(...)):
    uid="upload-"+time.strftime("%Y%m%d-%H%M%S-")+secrets.token_hex(4); stage=BASE/uid; inp=stage/"input"; inp.mkdir(parents=True)
    total=0
    try:
        for upload in files:
            name=Path(upload.filename or "upload").name; target=stage/(secrets.token_hex(4)+"_"+name)
            with target.open("wb") as dst:
                while chunk:=await upload.read(4*1024**2):
                    total+=len(chunk)
                    if total>MAX_BYTES: raise ValueError("Upload exceeds configured 8 GiB limit")
                    dst.write(chunk)
            if zipfile.is_zipfile(target): safe_extract(target,inp,MAX_BYTES); target.unlink()
            else: shutil.move(target,inp/(secrets.token_hex(4)+"_"+name))
        report=inspect_upload(inp)
        if not report["series"]: raise ValueError("No valid CT/MR/PET DICOM image series were found")
        private={"id":uid,"created":time.time(),**report}; (stage/"upload.json").write_text(json.dumps(private,indent=2))
        public={k:v for k,v in report.items()}; public["series"]=[{k:v for k,v in g.items() if k!="files"} for g in report["series"]]
        return {"upload_id":uid,**public}
    except Exception as exc:
        shutil.rmtree(stage,ignore_errors=True); raise HTTPException(400,str(exc))

@app.get("/api/uploads/{uid}/preview")
def preview_slice(uid: str, series_key: str, index: int = 0):
    """Render one locally staged DICOM slice for interactive prompt placement."""
    if not re.fullmatch(r"upload-[A-Za-z0-9-]+", uid): raise HTTPException(404)
    stage = BASE / uid
    try: data = json.loads((stage / "upload.json").read_text())
    except Exception: raise HTTPException(404, "Upload not found or expired")
    selected = next((g for g in data["series"] if g["key"] == series_key), None)
    if not selected: raise HTTPException(404, "Series not found")
    ordered = []
    for name in selected["files"]:
        path = stage / "input" / name
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            position = tuple(float(v) for v in getattr(ds, "ImagePositionPatient", []))
            order = (position[-1] if position else float(getattr(ds, "InstanceNumber", 0)),
                     float(getattr(ds, "InstanceNumber", 0)), name)
            ordered.append((order, path))
        except Exception: pass
    ordered.sort(key=lambda item: item[0])
    if not ordered: raise HTTPException(422, "Series has no renderable slices")
    index = max(0, min(index, len(ordered)-1))
    ds = pydicom.dcmread(ordered[index][1])
    try: pixels = ds.pixel_array.astype(np.float32)
    except Exception as exc: raise HTTPException(422, f"DICOM pixel decoding failed: {exc}")
    pixels = pixels * float(getattr(ds,"RescaleSlope",1)) + float(getattr(ds,"RescaleIntercept",0))
    def scalar(value):
        try: return float(value[0])
        except Exception: return float(value)
    try:
        center, width = scalar(ds.WindowCenter), max(1.0, scalar(ds.WindowWidth))
        low, high = center-width/2, center+width/2
    except Exception:
        finite = pixels[np.isfinite(pixels)]
        low, high = np.percentile(finite, [1,99]) if finite.size else (0,1)
        if high <= low: high = low+1
    rendered = np.clip((pixels-low)/(high-low)*255,0,255).astype(np.uint8)
    if str(getattr(ds,"PhotometricInterpretation","")) == "MONOCHROME1": rendered = 255-rendered
    image = Image.fromarray(rendered)
    payload = io.BytesIO(); image.save(payload, format="PNG")
    return Response(payload.getvalue(), media_type="image/png", headers={
        "X-Slice-Index":str(index), "X-Slice-Count":str(len(ordered)),
        "X-Columns":str(int(ds.Columns)), "X-Rows":str(int(ds.Rows))})

@app.post("/api/uploads/{uid}/start")
async def start_assessed(uid: str, series_key: str=Form(...), models: list[str]=Form(...), prompt_json: str=Form("")):
    if not re.fullmatch(r"upload-[A-Za-z0-9-]+",uid): raise HTTPException(404)
    stage=BASE/uid
    try: data=json.loads((stage/"upload.json").read_text())
    except Exception: raise HTTPException(404,"Upload not found or expired")
    selected=next((g for g in data["series"] if g["key"]==series_key),None)
    if not selected: raise HTTPException(400,"Selected series was not found")
    available={x["name"] for x in tasks()["tasks"] if x.get("enabled",True)}
    if not models or any(m not in available for m in models): raise HTTPException(400,"Invalid model selection")
    modality=selected["modality"]
    wrong=[]
    for model in models:
        details = EXTERNAL_TASKS.get(model) or INFORMATIONAL_MODELS.get(model, {})
        supported = details.get("modalities")
        if supported:
            if modality not in supported: wrong.append(model)
        elif task_modality(model) != modality:
            wrong.append(model)
    if wrong: raise HTTPException(400,f"Incompatible {modality} tasks: {', '.join(wrong)}")
    needs_ct = any(EXTERNAL_TASKS.get(m, {}).get("requires_ct") for m in models)
    prompt_models = [m for m in models if (EXTERNAL_TASKS.get(m) or INFORMATIONAL_MODELS.get(m,{})).get("prompt")]
    prompt_data = None
    if prompt_models:
        try: prompt_data = json.loads(prompt_json)
        except Exception: raise HTTPException(400,"Prompt JSON is required and must be valid JSON")
        columns, rows, slices = selected["columns"], selected["rows"], selected["slices"]
        coordinates = list(prompt_data.get("positive_points",[])) + list(prompt_data.get("negative_points",[]))
        if prompt_data.get("box"): coordinates += list(prompt_data["box"])
        if not coordinates and not str(prompt_data.get("text","")).strip():
            raise HTTPException(400,"The selected prompted model requires a point, box, mask, or text prompt")
        if "pam_prompted" in prompt_models and not (prompt_data.get("positive_points") or prompt_data.get("box")):
            raise HTTPException(400,"PAM requires a positive point or a two-corner box on one axial slice")
        coordinate_prompt_models = {"sam_med2d_prompted", "sat3d_prompted"}
        if coordinate_prompt_models.intersection(prompt_models) and not coordinates:
            raise HTTPException(400,"The selected model requires at least one point or a two-corner box")
        if "sat3d_prompted" in prompt_models and not prompt_data.get("positive_points"):
            raise HTTPException(400,"SAT3D requires at least one positive point")
        for point in coordinates:
            if not isinstance(point,list) or len(point)!=3 or any(not isinstance(v,(int,float)) for v in point):
                raise HTTPException(400,"Every prompt coordinate must be [column,row,slice]")
            if not (0<=point[0]<columns and 0<=point[1]<rows and 0<=point[2]<slices):
                raise HTTPException(400,f"Prompt coordinate {point} is outside the selected series")
    companion = None
    if needs_ct:
        companion = next((g for g in data["series"]
            if g["modality"] == "CT"
            and g.get("study_uid") == selected.get("study_uid")
            and g.get("frame_uid") == selected.get("frame_uid")), None)
        if companion is None:
            raise HTTPException(400, "This PET model requires a CT series in the same Study and Frame of Reference")
    jid=time.strftime("%Y%m%d-%H%M%S-")+secrets.token_hex(4); job=BASE/jid; inp=job/"input"; out=job/"output"; inp.mkdir(parents=True); out.mkdir()
    for name in selected["files"]: shutil.move(stage/"input"/name,inp/name)
    detail = f"{selected['slices']} {modality} DICOM slices selected"
    if companion:
        ct_dir = job / "input_ct"; ct_dir.mkdir()
        for name in companion["files"]: shutil.move(stage/"input"/name, ct_dir/name)
        detail += f"; paired with {companion['slices']} co-referenced CT slices"
    if prompt_data: (job/"prompt.json").write_text(json.dumps(prompt_data,indent=2))
    (job/"job.json").write_text(json.dumps({"id":jid,"status":"queued","detail":detail,"models":models,"created":time.time(),"error":None,"prompted":bool(prompt_data)},indent=2))
    shutil.rmtree(stage,ignore_errors=True); threading.Thread(target=process,args=(job,),daemon=True).start(); return {"id":jid}

@app.post("/api/jobs")
@app.post("/api/v1/jobs", status_code=202, tags=["automation"])
async def create_job(files: list[UploadFile]=File(...), models: list[str]=Form(...)):
    if not models: raise HTTPException(400,"Select at least one model")
    available={x["name"] for x in tasks()["tasks"] if x.get("enabled",True)}
    if any(not re.fullmatch(r"[A-Za-z0-9_]+",m) or m not in available for m in models): raise HTTPException(400,"Invalid model selection")
    if "total" in models and "total_mr" in models: raise HTTPException(400,"Select either total (CT) or total_mr (MR), not both")
    jid=time.strftime("%Y%m%d-%H%M%S-")+secrets.token_hex(4); job=BASE/jid; inp=job/"input"; out=job/"output"; inp.mkdir(parents=True); out.mkdir()
    total=0
    try:
        for upload in files:
            name=Path(upload.filename or "upload").name; target=job/(secrets.token_hex(4)+"_"+name)
            with target.open("wb") as dst:
                while chunk:=await upload.read(4*1024**2):
                    total+=len(chunk)
                    if total>MAX_BYTES: raise ValueError("Upload exceeds configured 8 GiB limit")
                    dst.write(chunk)
            if zipfile.is_zipfile(target): safe_extract(target,inp); target.unlink()
            else: shutil.move(target,inp/(secrets.token_hex(4)+"_"+name))
        slices, modality=validate_series(inp)
        wrong=[m for m in models if task_modality(m) != modality]
        if wrong: raise ValueError(f"The uploaded series is {modality}, but incompatible tasks were selected: {', '.join(wrong)}")
        (job/"job.json").write_text(json.dumps({"id":jid,"status":"queued","detail":f"{slices} {modality} DICOM slices received","models":models,"created":time.time(),"error":None},indent=2))
    except Exception as exc:
        shutil.rmtree(job,ignore_errors=True); raise HTTPException(400,str(exc))
    threading.Thread(target=process,args=(job,),daemon=True).start(); return {"id":jid}

@app.get("/api/jobs/{jid}")
@app.get("/api/v1/jobs/{jid}", tags=["automation"])
def get_job(jid: str):
    if not re.fullmatch(r"[A-Za-z0-9-]+",jid): raise HTTPException(404)
    return metadata(BASE/jid)

@app.get("/api/jobs")
@app.get("/api/v1/jobs", tags=["automation"])
def list_jobs():
    found=[]
    for path in BASE.iterdir():
        if not path.is_dir() or path.name.startswith("upload-"): continue
        try:
            data=metadata(path)
            found.append({"id":data["id"],"status":data["status"],"created":data.get("created",0)})
        except Exception: pass
    found.sort(key=lambda x:x["created"],reverse=True)
    return {"jobs":found[:100]}

@app.post("/api/jobs/{jid}/retry", status_code=202)
@app.post("/api/v1/jobs/{jid}/retry", status_code=202, tags=["automation"])
def retry_job(jid: str):
    if not re.fullmatch(r"[A-Za-z0-9-]+", jid): raise HTTPException(404)
    job=BASE/jid; data=metadata(job)
    if data.get("status") != "failed": raise HTTPException(409,"Only failed jobs can be retried")
    if not (job/"input").is_dir() or not any((job/"input").iterdir()):
        raise HTTPException(409,"The retained input series is unavailable; upload it again")
    shutil.rmtree(job/"output",ignore_errors=True); (job/"output").mkdir()
    save_metadata(job,status="queued",detail="Retry queued",error=None,progress=0,model_index=0,
                  recovery_count=int(data.get("recovery_count",0)))
    threading.Thread(target=process,args=(job,),daemon=True).start()
    return {"id":jid,"status":"queued"}

@app.get("/api/jobs/{jid}/download")
@app.get("/api/v1/jobs/{jid}/result", tags=["automation"])
def download(jid: str):
    job=BASE/jid; data=metadata(job); result=job/"rtstruct-results.zip"
    if data["status"]!="complete" or not result.exists(): raise HTTPException(409,"Result is not ready")
    return FileResponse(result,filename=f"{jid}-RTSTRUCT.zip",media_type="application/zip")

@app.get("/api/v1/health", tags=["automation"])
def health():
    worker_ok = True
    detail = "local worker runtime"
    if WORKER_MODE == "ssh":
        check = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                                WORKER, "true"], capture_output=True, timeout=5)
        worker_ok = check.returncode == 0
        detail = WORKER
    return {"status":"ok" if worker_ok else "degraded", "worker_mode":WORKER_MODE,
            "worker":detail, "worker_reachable":worker_ok, "version":app.version}

def cleanup_loop():
    while True:
        cutoff=time.time()-RETENTION_HOURS*3600
        for job in BASE.iterdir():
            try:
                if not job.is_dir(): continue
                meta=job/("upload.json" if job.name.startswith("upload-") else "job.json")
                if json.loads(meta.read_text()).get("created",time.time())<cutoff: shutil.rmtree(job)
            except Exception: pass
        time.sleep(3600)

def recover_interrupted_jobs():
    """Requeue interrupted jobs when their retained input permits a safe rerun."""
    interrupted = []
    for job in BASE.iterdir():
        if not job.is_dir() or job.name.startswith("upload-"):
            continue
        try:
            data = metadata(job)
            if data.get("status") in {"queued", "transferring", "processing", "retrieving"}:
                interrupted.append((job, data.get("status", "unknown")))
        except Exception:
            continue
    retry = []
    for job, previous_status in interrupted:
        if WORKER_MODE == "ssh":
            remote = f"{REMOTE_BASE}/{job.name}"
            subprocess.run(["ssh", WORKER, "rm", "-rf", remote], timeout=30)
        data = metadata(job); recovery_count = int(data.get("recovery_count",0)) + 1
        has_input = (job/"input").is_dir() and any((job/"input").iterdir())
        if has_input and recovery_count <= 3:
            shutil.rmtree(job/"output",ignore_errors=True); (job/"output").mkdir()
            save_metadata(job,status="queued",detail=f"Restarting after service interruption ({recovery_count}/3)",
                          error=None,progress=0,model_index=0,recovery_count=recovery_count,
                          interrupted_status=previous_status)
            retry.append(job)
        else:
            reason = "retained input is unavailable" if not has_input else "automatic recovery limit was reached"
            save_metadata(job,status="failed",detail="Processing was interrupted by a portal service restart",
                          error=f"Could not restart automatically because {reason}.",
                          recovery_count=recovery_count,interrupted_status=previous_status)
    for job in retry:
        threading.Thread(target=process,args=(job,),daemon=True).start()

@app.on_event("startup")
def startup():
    recover_interrupted_jobs()
    threading.Thread(target=cleanup_loop,daemon=True).start()
