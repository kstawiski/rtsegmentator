from __future__ import annotations

import json
import logging
import io
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import tarfile
import zipfile
from pathlib import Path

import pydicom
from nv_reason_ct_report import TASK as REPORT_TASK, detect_cbct
import numpy as np
from PIL import Image
import asyncio
import fcntl
from contextlib import contextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, JSONResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from crop_prostate_sbrt import NoSelectedROIError, crop_ct_rtstruct, discover_dicom
from viewer import ViewerService

BASE = Path(os.environ.get("PORTAL_DATA", Path(__file__).parent / "data"))
WORKER = os.environ.get("SEGMENTATION_WORKER", "worker@rtsegmentator-worker")
WORKER_MODE = os.environ.get("WORKER_MODE", "local").lower()
if WORKER_MODE not in {"local", "ssh", "colab", "auto"}:
    raise RuntimeError("WORKER_MODE must be 'local', 'ssh', 'colab', or 'auto'")
REMOTE_GPU_TASKS = {name.strip() for name in os.environ.get("REMOTE_GPU_TASKS", "").split(",") if name.strip()}
REMOTE_NV_REASON_CT_COVERAGE_MODE = os.environ.get("RTSEG_NV_REASON_CT_COVERAGE_MODE", "fast")
REMOTE_BASE = os.environ.get("REMOTE_JOB_ROOT", "/home/worker/jobs")
REMOTE_PYTHON = os.environ.get("REMOTE_PYTHON", "/opt/rtsegmentator/venv/bin/python")
REMOTE_RUNNER = os.environ.get("REMOTE_RUNNER", "/app/run_task.py")
REMOTE_APP_ROOT = os.environ.get("REMOTE_APP_ROOT", str(Path(REMOTE_RUNNER).parent))
REMOTE_MODEL_ROOT = os.environ.get("REMOTE_MODEL_ROOT", "/models")
REMOTE_RUNTIME_ROOT = os.environ.get("REMOTE_RUNTIME_ROOT", REMOTE_MODEL_ROOT)
REMOTE_SCRATCH_ROOT = os.environ.get("REMOTE_SCRATCH_ROOT", "")
REMOTE_NV_REASON_CT_CACHE = os.environ.get("REMOTE_NV_REASON_CT_CACHE", "")
REMOTE_TOTALSEG_WEIGHTS_PATH = os.environ.get("REMOTE_TOTALSEG_WEIGHTS_PATH", "")
LOCAL_PYTHON = os.environ.get("LOCAL_PYTHON", sys.executable)
LOCAL_RUNNER = os.environ.get("LOCAL_RUNNER", str(Path(__file__).parent / "run_task.py"))
COLAB_CLI = os.environ.get("COLAB_CLI", "colab")
COLAB_GPU = os.environ.get("COLAB_GPU", "T4")
COLAB_AUTH = os.environ.get("COLAB_AUTH", "oauth2")
COLAB_ORCHESTRATOR = os.environ.get("COLAB_ORCHESTRATOR", "worker").lower()
if COLAB_ORCHESTRATOR not in {"direct", "worker", "nfs"}:
    raise RuntimeError("COLAB_ORCHESTRATOR must be 'direct', 'worker', or 'nfs'")
COLAB_EXTERNAL_TASKS = {name.strip() for name in os.environ.get(
    "COLAB_EXTERNAL_TASKS", "lungmask_r231").split(",") if name.strip()}
GPU_RELEASE_COMMAND = os.environ.get("GPU_RELEASE_COMMAND", "")
GPU_RELEASE_TIMEOUT = int(os.environ.get("GPU_RELEASE_TIMEOUT", "90"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024**3)))
RETENTION_HOURS = int(os.environ.get("RETENTION_HOURS", "24"))
ENABLED_EXTERNAL_MODELS = {name.strip() for name in os.environ.get("ENABLED_EXTERNAL_MODELS", "").split(",") if name.strip()}
ENABLED_MODELS_FILE = Path(os.environ.get("ENABLED_MODELS_FILE", str(BASE / "enabled-models.txt")))
CATALOG_TTL_SECONDS = int(os.environ.get("CATALOG_TTL_SECONDS", "600"))
PREFLIGHT_TIMEOUT_SECONDS = int(os.environ.get("PREFLIGHT_TIMEOUT_SECONDS", "900"))
TASK_TIME_LIMIT_SECONDS = int(os.environ.get("TASK_TIME_LIMIT_SECONDS", str(12 * 3600)))
TASK_STOP_GRACE_SECONDS = int(os.environ.get("TASK_STOP_GRACE_SECONDS", "45"))
BASE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DICOM RT Segmentation Portal", version="1.0.0",
              docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

queue_lock = threading.Lock()
metadata_lock = threading.Lock()
logger = logging.getLogger(__name__)
viewer = ViewerService(globals())
app.include_router(viewer.router)

FALLBACK_TASKS = ["total", "total_mr", "total_v3", "body", "body_mr", "vertebrae_mr", "lung_vessels", "lung_vessels_LEGACY", "cerebral_bleed", "hip_implant", "coronary_arteries", "coronary_arteries_LEGACY", "pleural_pericard_effusion", "head_glands_cavities", "head_muscles", "headneck_bones_vessels", "headneck_muscles", "liver_vessels", "liver_segments", "liver_segments_mr", "liver_lesions", "liver_lesions_mr", "oculomotor_muscles", "lung_nodules", "kidney_cysts", "breasts", "ventricle_parts", "tissue_types", "tissue_types_mr", "tissue_4_types", "brain_structures", "vertebrae_body", "face", "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr", "appendicular_bones", "appendicular_bones_mr", "aortic_sinuses", "heartchambers_highres", "trunk_cavities", "brain_aneurysm"]
EXTERNAL_TASKS = {
    "nv_reason_ct_report": {
        "modality": "CT", "modalities": ["CT"], "output": "report",
        "description": "NVIDIA NV-Reason-CT research-only AI draft report; requires review by a radiologist. "
                       "Segmentation checks anatomical coverage and guides the crop.",
        "structures": "Report (JSON, Markdown and HTML with crop thumbnails); no RTSTRUCT",
        "reference": "https://huggingface.co/nvidia/NV-Reason-CT", "license": "OpenMDW-1.1",
        "limitations": "Diagnostic chest/abdomen CT only; CBCT refused. May describe absent organs or invent "
                       "findings on unreadable images. Segmentation checks do not establish diagnostic accuracy.",
    },
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

NV_SEGMENT_TASKS = json.loads((Path(__file__).parent / "catalog/nv-segment.json").read_text())
EXTERNAL_TASKS.update(NV_SEGMENT_TASKS)
LNCTVSEG_TASKS = json.loads((Path(__file__).parent / "catalog/lnctvseg.json").read_text())
EXTERNAL_TASKS.update(LNCTVSEG_TASKS)

NNUNET_V2_TASKS = json.loads((Path(__file__).parent / "catalog/nnunet-v2.json").read_text())
for _name, _details in NNUNET_V2_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"model_folder", "checkpoint", "channels"}}

NNUNET_V1_TASKS = json.loads((Path(__file__).parent / "catalog/nnunet-v1.json").read_text())
for _name, _details in NNUNET_V1_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"task", "configuration", "trainer", "plans", "labels", "disable_tta"}}

MULTITALENT_V1_TASKS = json.loads((Path(__file__).parent / "catalog/multitalent-v1.json").read_text())
for _name, _details in MULTITALENT_V1_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items() if k != "model_folder"}

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
                             if k not in {"sct_task", "output_names", "multiclass_labels", "labels_file", "label_vert", "output_glob", "preserve_overlaps"}}

GOUHFI_TASKS = json.loads((Path(__file__).parent / "catalog/gouhfi.json").read_text())
for _name, _details in GOUHFI_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items() if k not in {"mode", "lut"}}

FASTSURFER_TASKS = json.loads((Path(__file__).parent / "catalog/fastsurfer.json").read_text())
for _name, _details in FASTSURFER_TASKS.items():
    EXTERNAL_TASKS[_name] = _details

CERR_TASKS = json.loads((Path(__file__).parent / "catalog/cerr.json").read_text())
for _name, _details in CERR_TASKS.items():
    EXTERNAL_TASKS[_name] = {
        key: value for key, value in _details.items()
        if key not in {"runner", "upstream_model", "checkpoint", "labels"}
    }
    if _details.get("blocked_status"):
        EXTERNAL_TASKS[_name]["enabled"] = False

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
EXTERNAL_TASKS["medlsam_structseg_han_oars"] = {
    "modality": "CT",
    "description": "MedLSAM fixed-support StructSeg head-and-neck OAR research pre-contouring; held-out accuracy varied substantially by structure and every contour requires review",
    "structures": "22 head-and-neck OARs including brainstem, eyes, lenses, optic apparatus, temporal lobes, pituitary, parotids, ears, TMJs, spinal cord and mandible",
    "reference": "https://github.com/openmedlab/MedLSAM",
    "status": "validated",
    "clinical_tag_ids": ["head_neck_cancer", "brain_metastases", "glioblastoma_glioma"],
}

# Audited releases that must remain visible even when they cannot safely be
# submitted yet.  This prevents "not in the picker" from being mistaken for
# "not assessed" and gives users the exact upstream model and output meaning.
INFORMATIONAL_MODELS = {
    "medical_sam3_3d_prompted": {"modality":"CT", "modalities":["CT","MR","PT"],
        "enabled":True, "licensed":False, "prompt":"text_and_box", "status":"validated_research",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","glioblastoma_glioma","brain_metastases","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","neuroendocrine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One medical text-concept and box-selected object over the released eight-slice 3-D window",
        "description":"Official Medical-SAM3 full 3-D checkpoint and bundled SAM3 implementation. It uses the publisher's eight-frame train/evaluation protocol centered on a boxed seed slice. Public MSD spleen acceptance achieved Dice 0.967 across all eight output slices.",
        "reference":"https://huggingface.co/ChongCong/Medical-SAM3"},
    "sam_med3d_original_unreleased": {"modality":"CT", "modalities":["CT","MR","PT"],
        "enabled":False, "status":"original_checkpoint_not_released",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","glioblastoma_glioma","brain_metastases","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","neuroendocrine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"Class-agnostic 3-D point-prompted object segmentation",
        "description":"The catalog's original SAM-Med3D entry has no separately published checkpoint in the official repository or Hugging Face model. Upstream now exposes only SAM-Med3D Turbo and its CVPR/bounding-box variants; Turbo is installed as a separate validated task.",
        "reference":"https://github.com/uni-medical/SAM-Med3D"},
    "medsam_vit_b_box_2d": {"modality":"CT", "modalities":["CT","MR","PT"],
        "enabled":True, "licensed":False, "prompt":"box", "status":"validated_slice_wise",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","glioblastoma_glioma","brain_metastases","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","neuroendocrine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One box-selected 2-D object contour on the explicitly prompted slice",
        "description":"Official MedSAM ViT-B box-prompted model. It runs only on the explicitly boxed slice and does not propagate through a volume. The Zenodo checkpoint passed strict loading and achieved Dice 0.954 on a held-out MSD spleen slice before DICOM RTSTRUCT geometry acceptance.",
        "reference":"https://zenodo.org/records/10689643"},
    "sam_med3d_turbo_prompted": {"modality":"CT", "modalities":["CT","MR","PT"],
        "enabled":True, "licensed":False, "prompt":"points_or_box", "status":"validated_research",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","glioblastoma_glioma","brain_metastases","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","neuroendocrine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One point-prompted 3-D organ, tumour, lesion or other object",
        "description":"Official 100.5M-parameter SAM-Med3D Turbo checkpoint. A positive 3-D point selects the object; a submitted box is converted to its 3-D centre point. The public AMOS fixture passed strict checkpoint loading and inference, followed by source-grid DICOM RTSTRUCT validation.",
        "reference":"https://huggingface.co/blueyo0/SAM-Med3D"},
    "universeg_few_shot_2d": {"modality":"CT", "modalities":["CT","MR","PT"],
        "enabled":False, "status":"requires_support_image_mask_uploads",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","glioblastoma_glioma","brain_metastases","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","neuroendocrine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"Class defined at inference by one or more paired 2-D support images and binary masks",
        "description":"The official 1.18M-parameter UniverSeg checkpoint is installed and passed strict state loading plus 64-shot OASIS inference (Dice 0.910). It remains disabled because every submitted task requires user-provided support image-mask examples; the portal currently accepts only a target DICOM series and spatial/text prompts, so silently substituting unrelated examples would change the requested class.",
        "reference":"https://github.com/JJGO/UniverSeg"},
    "segvol_prompted_ct": {"modality":"CT", "enabled":True, "licensed":False,
        "prompt":"text_points_or_box", "status":"validated",
        "prompt_contract":{"allow_box_with_points":False},
        "limitations":"Use points or a 3-D box, without combining them. Text may accompany either prompt type.",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One text-, point-, or 3-D-box-selected CT organ, tumour, lesion or other object",
        "description":"Official SegVol CT foundation model supporting more than 200 reported categories. The released checkpoint passed exact MONAI 0.9 state loading and H100 text-prompted inference on its public fixture (spleen Dice 0.980), followed by direct source-grid RTSTRUCT geometry acceptance.",
        "reference":"https://huggingface.co/BAAI/SegVol"},
    "medsam2_ct_lesion_prompted": {"noncommercial":True, "modality":"CT", "enabled":True, "licensed":False,
        "prompt":"points_or_box", "status":"validated_research",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One point- or box-prompted 3-D CT lesion",
        "description":"Official MedSAM2 CT lesion model with forward and reverse volumetric propagation. Research/education use only; prompts must be placed on one axial seed slice.",
        "reference":"https://huggingface.co/wanglab/MedSAM2"},
    "medical_sam2_prompted_ct": {"noncommercial":True, "modality":"CT", "enabled":True, "licensed":False,
        "prompt":"points_or_box", "status":"validated_research",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One point- or box-prompted 3-D CT object",
        "description":"ImprintLab Medical-SAM2 prompt-propagation checkpoint. Research use only; technical fixture accuracy was substantially below the WangLab CT-lesion checkpoint.",
        "reference":"https://github.com/ImprintLab/Medical-SAM2"},
    "voxtell_v1_1_text_prompted": {"modality":"CT", "modalities":["CT","MR","PT"],
        "enabled":True, "licensed":False, "prompt":"text", "status":"validated",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","glioblastoma_glioma","brain_metastases","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","neuroendocrine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One free-text-selected 3-D anatomical structure, tumour or lesion",
        "description":"Official VoxTell v1.1 free-text 3-D segmentation model for CT, MRI and PET. The prompt ‘spleen’ passed H100 inference, direct source-grid RTSTRUCT conversion and per-ROI physical-geometry validation on the public MSD fixture.",
        "reference":"https://huggingface.co/mrokuss/VoxTell"},
    "vista3d_prompted_ct": {"modality":"CT", "enabled":True, "licensed":False,
        "prompt":"points", "status":"validated",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One point-prompted 3-D organ, tumour, lesion or other object",
        "description":"Official NV-Segment-CT/VISTA3D point-prompted model. It passed H100 inference, source-grid RTSTRUCT conversion and strict orientation acceptance on the public MSD spleen CT fixture.",
        "reference":"https://huggingface.co/nvidia/NV-Segment-CT"},
    "nninteractive_v1_prompted": {"noncommercial":True, "modality":"CT", "modalities":["CT","MR","PT"], "licensed":False,
        "enabled":True, "prompt":"points_or_box", "status":"validated",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","glioblastoma_glioma","brain_metastases","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","neuroendocrine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"One prompt-selected 3-D organ, tumour, lymph node or other object",
        "description":"Official nnInteractive v1.0 class-agnostic model using positive/negative points or a 2-D box. The CC BY-NC-SA 4.0 checkpoint passed H100 inference, source-grid RTSTRUCT conversion and strict orientation acceptance.",
        "reference":"https://github.com/MIC-DKFZ/nnInteractive"},
    "hippunfold_adni_t1_subfields_mr": {"modality":"MR", "enabled":True,
        "status":"validated",
        "clinical_tag_ids":["glioblastoma_glioma","brain_metastases"],
        "structures":"Bilateral hippocampal subfields, dentate gyrus, stratum radiatum/lacunosum/moleculare and cyst labels",
        "description":"HippUnfold whole-brain T1 workflow using the released five-fold ADNI checkpoint. All five folds produced 16 bilateral subfield labels on the public ds002168 fixture; direct source-grid RTSTRUCT conversion matched every ROI physical bounding box within 0.001 mm.",
        "reference":"https://zenodo.org/records/15297857"},
    "hippunfold_maguire_t2_subfields_mr": {"modality":"MR", "enabled":True,
        "status":"validated",
        "clinical_tag_ids":["glioblastoma_glioma","brain_metastases"],
        "structures":"Bilateral hippocampal subfields, dentate gyrus, stratum radiatum/lacunosum/moleculare and cyst labels",
        "description":"HippUnfold T2 workflow using the released five-fold Maguire nnU-Net v2 checkpoint. All five folds completed on a public whole-field OpenNeuro T2 scan; 15 non-empty bilateral subfield labels were exported to RTSTRUCT with source-grid geometry matching within 0.0002 mm. Limited-field T2 scans may still require a registered T1 series.",
        "reference":"https://zenodo.org/records/18432751"},
    "cerebrum7t_t1_mr": {"modality":"MR", "enabled":False, "status":"missing_voxelwise_training_normalization",
        "structures":"Gray matter, basal ganglia, white matter, ventricles, cerebellum and brainstem",
        "description":"The official source and 14.9 MB trained HDF5 model are installed. Inference requires 256x352x224 7T T1 MRI plus voxelwise training-cohort mean and standard-deviation volumes named in the release metadata; those normalization volumes are not included in the repository or model release, so substituting scalar normalization would not reproduce the model.",
        "reference":"https://github.com/rockNroll87q/cerebrum7t"},
    "deepbrainseg_brats_four_sequence_mr": {"modality":"MR", "enabled":False, "status":"requires_four_registered_mr_series",
        "structures":"Whole tumor, tumor core and active/enhancing tumor constituents",
        "description":"The corrected official source repository, all four release checkpoints and BraTS fixtures are installed; all checkpoint state dictionaries load in the current CUDA runtime. The model requires co-registered T1, post-contrast T1, T2 and FLAIR volumes plus brain masking, while the portal currently selects one DICOM series.",
        "reference":"https://github.com/koriavinash1/DeepBrainSeg"},
    "hecktor2020_petct_gtvt": {"modality":"PT", "requires_ct":True, "enabled":False,
        "status":"requires_patient_specific_headneck_bbox",
        "clinical_tag_ids":["head_neck_cancer"],
        "structures":"Combined head-and-neck gross tumor volume (GTVt) foreground within the supplied 144 mm challenge bounding box",
        "description":"The official source and all eight PET/CT ensemble checkpoints are installed and load strictly. Published inference requires registered PET and CT plus a patient-specific 144x144x144 mm physical bounding box supplied separately by HECKTOR. The challenge dataset and bounding boxes are currently access-restricted, and inferring an arbitrary crop would violate the model contract.",
        "reference":"https://github.com/iantsen/hecktor"},
    "totalsegmentator2d_projection_ct": {"modality":"CT", "enabled":False,
        "status":"projection_only_no_volumetric_rtstruct",
        "clinical_tag_ids":["lung_cancer","prostate_cancer","bladder_cancer","head_neck_cancer","breast_cancer","liver_cancer","pancreatic_cancer","kidney_cancer","colorectal_cancer","cervical_cancer","lymphoma","spine_tumors","sarcoma_soft_tissue","whole_body_metastatic"],
        "structures":"117 anatomy classes on a coronal CT intensity projection",
        "description":"Official TotalSegmentator2D v2 source and all five checkpoint groups (cardiac, muscles, organs, ribs and vertebrae) are installed. Offline composite H100 inference passed on the publisher fixture and produced all 117 segment channels. The result has one voxel across the anterior-posterior axis and encodes projected occupancy, so it cannot be reconstructed into patient-specific 3-D contours or a geometrically valid RTSTRUCT.",
        "reference":"https://github.com/risc-mi/totalsegmentator2D"},
    "brats_toolkit_legacy_ensemble_mr": {"modality":"MR", "enabled":False, "status":"requires_four_registered_preprocessed_mr_series",
        "structures":"BraTS whole tumor, tumor core and enhancing tumor candidate segmentations from eight legacy algorithms",
        "description":"The official BraTS Toolkit source and its eight downloadable legacy algorithm images are preserved on NFS. Each algorithm expects skull-stripped, co-registered FLAIR, T1, post-contrast T1 and T2 NIfTI volumes in SRI-24 space; the portal currently selects one DICOM series and therefore cannot satisfy or safely reconstruct this input contract.",
        "reference":"https://github.com/BrainLesion/BraTS-Toolkit"},
    "tractseg_dwi_peaks": {"modality":"MR", "enabled":False, "status":"requires_diffusion_peaks_or_dwi_gradients",
        "structures":"72 white-matter tract bundle masks",
        "description":"All nine official TractSeg 2.10 pretrained weight sets and source are installed and the released 9-channel peaks fixture passed all-72-bundle inference. The portal currently selects one scalar 3D DICOM series and cannot safely supply a 4D CSD/bedpostX peaks volume or raw DWI with matched b-values/b-vectors.",
        "reference":"https://github.com/MIC-DKFZ/TractSeg"},
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
    "choi_tmli": {"modality":"CT", "enabled":False, "requires_preflight":True, "status":"ontology_review_required",
        "structures":"Cervical, axillary, mediastinal, para-aortic, iliac, obturator, presacral and inguinal lymphatic regions",
        "description":"Whole-body lymphatic-region model for total marrow and lymphoid irradiation. Numeric output can be reviewed with the anatomical landmark report and clinician review sheet. Named export requires a confirmed 18-region mapping; the task remains disabled pending review.",
        "reference":"https://zenodo.org/records/7839889"},
    "clnet_235": {"modality":"CT", "enabled":False, "status":"nodal_decoders_unavailable",
        "structures":"33 reported lymph-node stations; nodal checkpoints absent from the public archive",
        "description":"Checked 2026-10-15: Hugging Face revision a18539d2a8473c0eca1b4e6160523b1de12c65f3 still supplies the same 985573570-byte archive. Its seven decoder maps contain no lymph-node labels; Task1007 and Task1008 weights are absent.",
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
    "lnq_atlas": {"noncommercial":True, "modality":"CT", "enabled":False, "requires_preflight":True, "status":"public_weights_download_pending",
        "structures":"Visible mediastinal lymph-node candidates with probabilistic-atlas context",
        "license_note":"CC BY-NC-ND weights (model ledger).",
        "description":"Checked 2026-10-15: the public share returns HTTP 200 without credentials and lists five checkpoints plus the atlas. Download, pinned artifact verification, TotalSegmentator 1.5.7 runtime and GPU checks remain required; the adapter is staged and disabled.",
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
    str(Path(os.environ.get("RTSEG_MODEL_ROOT", "/models")) / "orientation-validation")))

def clinical_tags(item: dict) -> list[dict[str, str]]:
    """Return curated disease-site discovery tags; tags describe usefulness, not model indication."""
    searchable = " ".join(str(item.get(key, "")) for key in ("name", "description", "structures")).lower()
    explicit = set(item.get("clinical_tag_ids", []))
    return [{"id": tag_id, "label": rule["label"]}
            for tag_id, rule in CLINICAL_TAGS.items()
            if tag_id in explicit or any(term.lower() in searchable for term in rule["terms"])]

def task_modality(name: str) -> str:
    # TotalSegmentator currently has CT and MR task families. PET-capable
    # external models are added separately after their RTSTRUCT acceptance test.
    details = EXTERNAL_TASKS.get(name) or INFORMATIONAL_MODELS.get(name, {})
    return details.get("modality", "MR" if task_is_mr(name) else "CT")



def metadata(job: Path) -> dict:
    try: return json.loads((job / "job.json").read_text())
    except Exception: raise HTTPException(404, "Job not found")

def save_metadata(job: Path, **changes):
    with metadata_lock:
        target = job / "job.json"
        data = json.loads(target.read_text())
        data.update(changes)
        data["updated"] = time.time()
        incoming = job / "job.json.incoming"
        incoming.write_text(json.dumps(data, indent=2))
        incoming.replace(target)

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

def proc_stat(pid: int):
    """(state, parent pid, start time) of a process from /proc, or None once it is gone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[1]), fields[19]
    except (OSError, IndexError, ValueError):
        return None

def process_tree(root: int) -> dict[int, str]:
    """Live processes descended from root, root included, mapped to their start times."""
    children, starts = {}, {}
    for entry in os.listdir("/proc"):
        stat = proc_stat(int(entry)) if entry.isdigit() else None
        if stat and stat[0] != "Z":
            children.setdefault(stat[1], []).append(int(entry))
            starts[int(entry)] = stat[2]
    tree, frontier = {}, [root]
    while frontier:
        pid = frontier.pop()
        if pid in starts and pid not in tree:
            tree[pid] = starts[pid]
            frontier.extend(children.get(pid, []))
    return tree

def stop_process_tree(process: subprocess.Popen):
    """Stop a task runner and every process it started.

    run_task.py answers SIGTERM by stopping its model's process group. The model runs in its own session, so killing
    the runner alone would leave the model on the GPU. Whatever is still alive after the grace period is killed from
    a snapshot of the tree taken before the signal; start times guard against reused pids."""
    tree = process_tree(process.pid)
    process.terminate()
    try:
        process.wait(timeout=TASK_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    for pid, started in tree.items():
        stat = proc_stat(pid)
        if stat and stat[0] != "Z" and stat[2] == started:
            for survivor in process_tree(pid):
                try: os.kill(survivor, signal.SIGKILL)
                except ProcessLookupError: pass
    if process.poll() is None:
        process.kill()
    process.wait()

# Runs on the worker: SIGTERM to the run_task.py process whose arguments match this job.
REMOTE_STOP_SCRIPT = """
import os, signal, sys
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\\0")[:-1]
    except OSError:
        continue
    if [arg.decode(errors="replace") for arg in argv[1:]] == sys.argv[1:]:
        os.kill(int(pid), signal.SIGTERM)
"""

def stop_remote_task(remote: str, task: str):
    """Stop this job's run_task.py on the worker; killing the local ssh client would leave it running there."""
    command = shlex.join([REMOTE_PYTHON, "-c", REMOTE_STOP_SCRIPT, REMOTE_RUNNER, remote, task])
    try:
        subprocess.run(["ssh", WORKER, command], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass

def remote_command(*args):
    """One shell-quoted command: OpenSSH joins command arguments before remote execution."""
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", WORKER, shlex.join(args)]


def remote_environment():
    optional = {"RTSEG_SCRATCH_ROOT": REMOTE_SCRATCH_ROOT, "RTSEG_NV_REASON_CT_CACHE": REMOTE_NV_REASON_CT_CACHE,
                "TOTALSEG_WEIGHTS_PATH": REMOTE_TOTALSEG_WEIGHTS_PATH}
    return ["env", *[f"{name}={value}" for name, value in optional.items() if value], f"RTSEG_APP_ROOT={REMOTE_APP_ROOT}", f"RTSEG_MODEL_ROOT={REMOTE_MODEL_ROOT}",
            f"RTSEG_RUNTIME_ROOT={REMOTE_RUNTIME_ROOT}",
            f"RTSEG_NV_REASON_CT_COVERAGE_MODE={REMOTE_NV_REASON_CT_COVERAGE_MODE}"]


def task_uses_remote(task):
    return WORKER_MODE in {"local", "auto"} and task in REMOTE_GPU_TASKS


def ordered_tasks(models):
    return sorted(dict.fromkeys(models), key=lambda name: name == REPORT_TASK)


def reject_report_cbct(input_dir, models):
    if REPORT_TASK in models and any(g["cbct"] for g in inspect_upload(input_dir)["series"]):
        raise ValueError("CBCT is not supported for AI reports; diagnostic CT is required")


def reset_job_output(job, data):
    if REPORT_TASK in data.get("models", []):
        shutil.rmtree(job / "output/report", ignore_errors=True)
        save_metadata(job, report_status="pending", report_reason="")
    else:
        shutil.rmtree(job / "output", ignore_errors=True)
    (job / "output").mkdir(exist_ok=True)


def worker_archive_members(job, tasks):
    members = ["input"]
    if (job / "input_ct").is_dir(): members.append("input_ct")
    if (job / "prompt.json").is_file(): members.append("prompt.json")
    # The report needs finished contours, not bulky crop packages or previous drafts.
    if REPORT_TASK in tasks:
        members.extend(p.relative_to(job).as_posix() for p in sorted((job / "output").glob("*_RTSTRUCT.dcm")))
    return members


def prepare_worker_job(job, remote, tasks):
    save_metadata(job, status="transferring", detail="Preparing selected DICOM series", progress=0)
    subprocess.run(remote_command("mkdir", "-p", f"{remote}/input", f"{remote}/output"), check=True, timeout=30)
    archive = job / "selected-series.tar"
    try:
        subprocess.run(["tar", "-C", str(job), "-cf", str(archive), *worker_archive_members(job, tasks)],
                       check=True, timeout=600)
        transfer_to_worker(job, archive, f"{remote}/selected-series.tar")
        subprocess.run(remote_command("tar", "-C", remote, "-xf", f"{remote}/selected-series.tar"),
                       check=True, timeout=600)
        subprocess.run(remote_command("rm", "-f", f"{remote}/selected-series.tar"), check=True, timeout=30)
    finally:
        archive.unlink(missing_ok=True)


def retrieve_worker_results(job, remote, task=None):
    """Whole SSH jobs copy all output; routed tasks copy only the output they own."""
    (job / "output").mkdir(exist_ok=True)
    source = "output/." if task is None else ("output/report" if task == REPORT_TASK else f"output/{task}_RTSTRUCT.dcm*")
    # The task-owned prefix includes the orientation record and also supports
    # recovery from older workers that emitted only the DICOM file.
    sources = [f"{WORKER}:{remote}/{source}"]
    if task == "lnctvseg_nnunet_v1":
        sources.append(f"{WORKER}:{remote}/output/{task}")
    subprocess.run(["scp", "-q", "-r", *sources, str(job / "output") + "/"],
                   check=True, timeout=600)


def remove_worker_job(job, remote):
    subprocess.run(remote_command("rm", "-rf", "--", remote), check=True, timeout=30)
    save_metadata(job, remote_task_active=None, remote_results_retained=False)


def recover_remote_results(job, remote, task=None):
    try:
        retrieve_worker_results(job, remote, task)
        remove_worker_job(job, remote)
    except Exception:
        save_metadata(job, remote_results_retained=True)
        raise


def run_remote_gpu_task(job, task, index, total):
    remote = f"{REMOTE_BASE}/{job.name}"
    prior = metadata(job)
    if prior.get("remote_results_retained"):
        recover_remote_results(job, remote, prior.get("remote_task_active"))
    save_metadata(job, remote_task_active=task, execution_backend="ssh_task")
    try:
        prepare_worker_job(job, remote, [task])
        run_worker_task(job, remote, task, index, total)
        save_metadata(job, status="retrieving", detail=f"Retrieving {task} results", progress=None)
        recover_remote_results(job, remote, task)
    except Exception:
        # Preserve local contours. If retrieval cannot finish, retain the remote copy for recovery.
        try: recover_remote_results(job, remote, task)
        except Exception: pass
        raise


def execute_job_task(job, task, index, total, remote=None):
    data = metadata(job)
    report_job = REPORT_TASK in data["models"]
    if (report_job and task != REPORT_TASK and task in data.get("completed_models", [])
            and (job / "output" / f"{task}_RTSTRUCT.dcm").is_file()):
        return
    if task == REPORT_TASK: save_metadata(job, report_status="running", report_reason="")
    if remote is not None:
        run_worker_task(job, remote, task, index, total)
    elif task_uses_remote(task):
        run_remote_gpu_task(job, task, index, total)
    else:
        run_local_task(job, task, index, total)
    if report_job:
        done = metadata(job).get("completed_models", [])
        save_metadata(job, completed_models=list(dict.fromkeys([*done, task])))


def run_worker_task(job: Path, remote: str, task: str, index: int, total: int):
    """Run inference while keeping the portal heartbeat visibly current."""
    detail = f"Running {task} ({index}/{total})"
    log = job / f"{task}.log"
    started = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(remote_command(*remote_environment(), REMOTE_PYTHON, REMOTE_RUNNER, remote, task),
                                   stdout=output, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            if time.monotonic() - started > TASK_TIME_LIMIT_SECONDS:
                stop_remote_task(remote, task)
                stop_process_tree(process)
                raise subprocess.TimeoutExpired(process.args, TASK_TIME_LIMIT_SECONDS)
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
        worker_local = WORKER_MODE == "auto"
        runner = REMOTE_RUNNER if worker_local else LOCAL_RUNNER
        python = REMOTE_PYTHON if worker_local else LOCAL_PYTHON
        environment.setdefault("RTSEG_APP_ROOT", REMOTE_APP_ROOT if worker_local else str(Path(runner).parent))
        environment.setdefault("RTSEG_MODEL_ROOT", REMOTE_MODEL_ROOT if worker_local else "/models")
        environment.setdefault("RTSEG_RUNTIME_ROOT", REMOTE_RUNTIME_ROOT if worker_local else environment["RTSEG_MODEL_ROOT"])
        process = subprocess.Popen([python, runner, str(job), task], env=environment,
                                   stdout=output, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            if time.monotonic() - started > TASK_TIME_LIMIT_SECONDS:
                stop_process_tree(process)
                raise subprocess.TimeoutExpired(process.args, TASK_TIME_LIMIT_SECONDS)
            save_metadata(job,status="processing",detail=detail,progress=None,
                          model_index=index,model_total=total)
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(log.read_text(errors="replace")[-6000:] or
                           f"{task} exited with status {process.returncode}")

def is_gpu_oom(exc: Exception) -> bool:
    """Recognize GPU allocation failures without hiding unrelated model errors."""
    message = str(exc).lower()
    markers = ("cuda out of memory", "cuda error: out of memory", "cudnn_status_alloc_failed",
               "cublas_status_alloc_failed", "hip out of memory", "gpu out of memory",
               "failed to allocate cuda", "cuda_error_out_of_memory")
    return any(marker in message for marker in markers)

def gpu_memory_mb() -> tuple[int | None, int | None]:
    """Return aggregate (free, total) GPU memory without making health depend on it."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        values = [tuple(int(value.strip()) for value in line.split(","))
                  for line in result.stdout.splitlines() if line.strip()]
        return sum(value[0] for value in values), sum(value[1] for value in values)
    except Exception:
        return None, None

def release_worker_gpu(job: Path, task: str) -> None:
    """Run an explicitly configured GPU-release hook and wait for free VRAM."""
    if not GPU_RELEASE_COMMAND.strip():
        raise RuntimeError("GPU release command is not configured")
    save_metadata(job, status="processing",
                  detail=f"Releasing local LLM GPU memory before retrying {task}",
                  progress=None, gpu_release_attempted=True)
    result = subprocess.run(shlex.split(GPU_RELEASE_COMMAND), capture_output=True,
                            text=True, timeout=30)
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode and "not running" not in combined.lower():
        raise RuntimeError(f"Could not release worker GPU: {combined or result.returncode}")
    deadline = time.monotonic() + GPU_RELEASE_TIMEOUT
    last_free, total = gpu_memory_mb()
    while time.monotonic() < deadline:
        free, total = gpu_memory_mb()
        if free is not None:
            last_free = free
            if total and free >= min(30 * 1024, int(total * 0.70)):
                break
        time.sleep(2)
    save_metadata(job, detail=f"Retrying {task} on worker GPU after releasing local LLM",
                  gpu_free_mb=last_free, gpu_total_mb=total)

def colab_command(*args: str) -> list[str]:
    return [COLAB_CLI, f"--auth={COLAB_AUTH}", *args]

def run_colab_job(job: Path, tasks: list[str] | None = None, finalize: bool = True):
    """Run a complete job in an ephemeral Colab GPU session."""
    session = f"rtseg-{job.name[-12:]}"
    source = Path(__file__).resolve().parent
    app_archive = job / "colab-app.tar.gz"
    job_archive = job / "colab-job.tar.gz"
    result_archive = job / "colab-results.zip"
    status_file = job / "colab-status.json"
    save_metadata(job, status="transferring", detail=f"Provisioning Colab {COLAB_GPU} runtime", progress=0)
    try:
        subprocess.run(colab_command("new", "-s", session, "--gpu", COLAB_GPU), check=True, timeout=300)
        with tarfile.open(app_archive, "w:gz") as archive:
            for path in source.iterdir():
                if path.name in {".git", ".venv", "data", "models", "runtimes", "__pycache__"}:
                    continue
                if path.is_file() or path.name in {"catalog", "labels"}:
                    archive.add(path, arcname=f"app/{path.name}")
        with tarfile.open(job_archive, "w:gz") as archive:
            for name in ("input", "input_ct", "prompt.json"):
                path = job / name
                if path.exists(): archive.add(path, arcname=f"job/{name}")
            job_data = metadata(job)
            job_data["models"] = tasks or job_data["models"]
            payload = json.dumps(job_data, indent=2).encode()
            info = tarfile.TarInfo("job/job.json")
            info.size = len(payload)
            info.mtime = int(time.time())
            archive.addfile(info, io.BytesIO(payload))
        total = app_archive.stat().st_size + job_archive.stat().st_size
        save_metadata(job, detail="Uploading selected series and worker code to Colab",
                      total_bytes=total, transferred_bytes=0, progress=5)
        subprocess.run(colab_command("upload", "-s", session, str(app_archive),
                                     "/content/rtseg-app.tar.gz"), check=True, timeout=3600)
        save_metadata(job, transferred_bytes=app_archive.stat().st_size, progress=15)
        subprocess.run(colab_command("upload", "-s", session, str(job_archive),
                                     "/content/rtseg-job.tar.gz"), check=True, timeout=3600)
        save_metadata(job, status="processing", detail="Colab GPU segmentation in progress",
                      transferred_bytes=total, progress=None)
        log = job / "colab.log"
        with log.open("w") as output:
            subprocess.run(colab_command("exec", "-s", session, "-f",
                                         str(source / "colab_worker.py"), "--timeout", str(12 * 3600)),
                           check=True, timeout=12 * 3600 + 300, stdout=output,
                           stderr=subprocess.STDOUT, text=True)
        subprocess.run(colab_command("download", "-s", session,
                                     "/content/rtseg-status.json", str(status_file)),
                       check=True, timeout=300)
        remote_status = json.loads(status_file.read_text())
        if remote_status.get("status") != "complete":
            raise RuntimeError(remote_status.get("error") or "Colab worker failed without an error message")
        save_metadata(job, status="retrieving", detail="Retrieving RTSTRUCT results from Colab")
        subprocess.run(colab_command("download", "-s", session,
                                     "/content/rtseg-results.zip", str(result_archive)),
                       check=True, timeout=3600)
        with zipfile.ZipFile(result_archive) as archive:
            for info in archive.infolist():
                if info.is_dir() or Path(info.filename).name != info.filename or not info.filename.endswith((".dcm", ".dcm.orientation.json")):
                    raise RuntimeError("Colab returned an unsafe or unexpected result archive")
            archive.extractall(job / "output")
        if finalize:
            finalize_results(job)
    finally:
        app_archive.unlink(missing_ok=True)
        job_archive.unlink(missing_ok=True)
        result_archive.unlink(missing_ok=True)
        status_file.unlink(missing_ok=True)
        subprocess.run(colab_command("stop", "-s", session), timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_colab_job_via_worker(job: Path, tasks: list[str] | None = None, finalize: bool = True):
    """Stage assets from worker NFS so large models never touch portal storage."""
    session = f"rtseg-{job.name[-12:]}"
    remote = f"{REMOTE_BASE}/{job.name}"
    archive = job / "selected-series.tar"
    members = ["input", "job.json"]
    if (job / "input_ct").is_dir(): members.append("input_ct")
    if (job / "prompt.json").is_file(): members.append("prompt.json")
    if tasks:
        data = metadata(job); data["models"] = tasks
        task_metadata = job / "colab-job.json"
        task_metadata.write_text(json.dumps(data, indent=2))
        members.remove("job.json"); members.append("colab-job.json")
    try:
        subprocess.run(["tar", "-C", str(job), "-cf", str(archive), *members], check=True, timeout=600)
        subprocess.run(["ssh", WORKER, "mkdir", "-p", f"{remote}/input", f"{remote}/output"], check=True, timeout=30)
        transfer_to_worker(job, archive, f"{remote}/selected-series.tar")
        extract_command = f"tar -C {shlex.quote(remote)} -xf {shlex.quote(remote + '/selected-series.tar')}"
        if tasks:
            extract_command += f" && mv {shlex.quote(remote + '/colab-job.json')} {shlex.quote(remote + '/job.json')}"
        subprocess.run(["ssh", WORKER, extract_command], check=True, timeout=600)
        command = ["ssh", WORKER, REMOTE_PYTHON, str(Path(REMOTE_APP_ROOT) / "run_colab_remote.py"),
                   remote, "--session", session, "--gpu", COLAB_GPU,
                   "--app-root", REMOTE_APP_ROOT, "--model-root", REMOTE_MODEL_ROOT,
                   "--runtime-root", REMOTE_RUNTIME_ROOT]
        log = job / "colab.log"
        with log.open("w") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True)
            while process.poll() is None:
                save_metadata(job, status="processing", detail="Colab GPU segmentation orchestrated by model worker",
                              progress=None)
                time.sleep(10)
        if process.returncode:
            raise RuntimeError(log.read_text(errors="replace")[-12000:] or
                               f"Remote Colab orchestrator exited with status {process.returncode}")
        subprocess.run(["scp", "-q", f"{WORKER}:{remote}/output/*.dcm*", str(job / "output") + "/"],
                       check=True, timeout=3600)
        if finalize: finalize_results(job)
    finally:
        archive.unlink(missing_ok=True)
        (job / "colab-job.json").unlink(missing_ok=True)
        subprocess.run(["ssh", WORKER, "rm", "-rf", remote], timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_colab_job_on_nfs_worker(job: Path, tasks: list[str] | None = None, finalize: bool = True):
    """Run the NFS-aware Colab orchestrator directly on the model worker."""
    session = f"rtseg-{job.name[-12:]}"
    if tasks:
        data = metadata(job)
        data["models"] = tasks
        (job / "job.json").write_text(json.dumps(data, indent=2))
    command = [REMOTE_PYTHON, str(Path(REMOTE_APP_ROOT) / "run_colab_remote.py"),
               str(job), "--session", session, "--gpu", COLAB_GPU,
               "--app-root", REMOTE_APP_ROOT, "--model-root", REMOTE_MODEL_ROOT,
               "--runtime-root", REMOTE_RUNTIME_ROOT, "--colab", COLAB_CLI]
    log = job / "colab-orchestrator.log"
    progress_file = job / "colab-progress.json"
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            progress = None
            detail = "Colab GPU segmentation orchestrated on model worker"
            try:
                staged = json.loads(progress_file.read_text())
                progress = staged.get("progress")
                detail = staged.get("detail", detail)
            except Exception:
                pass
            save_metadata(job, status="processing", detail=detail, progress=progress,
                          execution_backend="colab")
            time.sleep(2)
    if process.returncode:
        raise RuntimeError(log.read_text(errors="replace")[-12000:] or
                           f"Colab orchestrator exited with status {process.returncode}")
    if finalize:
        finalize_results(job)
    progress_file.unlink(missing_ok=True)

def colab_capable(task: str) -> bool:
    """Colab has the portable TotalSegmentator runtime, not the NFS model bundle."""
    if task == REPORT_TASK: return False  # The portable Colab worker does not carry this runtime.
    return (task not in EXTERNAL_TASKS and task not in INFORMATIONAL_MODELS) or task in COLAB_EXTERNAL_TASKS

EMPTY_PREDICTION_MARKERS = ("empty prediction: no RTSTRUCT created",
                            "all supplied masks are empty: no RTSTRUCT created",
                            "no contourable structures: no RTSTRUCT created")

def is_empty_prediction(exc: Exception) -> bool:
    """Recognize completed empty predictions or explicitly reported un-contourable ROIs."""
    message = str(exc)
    return any(marker in message for marker in EMPTY_PREDICTION_MARKERS)

def conversion_notes(job: Path) -> list[str]:
    """Surface converter omissions from retained local/SSH/Colab worker logs."""
    notes = []
    for path in sorted(job.glob("*.log")):
        with path.open(errors="replace") as log:
            for line in log:
                if not line.startswith("RTSTRUCT_CONVERSION_NOTE "):
                    continue
                try:
                    record = json.loads(line.removeprefix("RTSTRUCT_CONVERSION_NOTE "))
                    model = record["output"].removesuffix("_RTSTRUCT.dcm")
                    for roi in record["omitted_rois"]:
                        note = (f"{model}: omitted ROI {roi['name']} ({int(roi['voxels'])} voxels; "
                                f"{roi['reason']}).")
                        if note not in notes:
                            notes.append(note)
                except (ValueError, KeyError, TypeError, AttributeError):
                    logger.warning("Could not parse an RTSTRUCT conversion note in %s", path.name)
    return notes

def finalize_results(job: Path, empty=()):
    results = sorted((job / "output").glob("*.dcm"))
    empty = list(empty)
    report_path = job / "output/report" / (REPORT_TASK + ".json")
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    if not results and not empty and report is None:
        raise RuntimeError("Worker completed without producing an RTSTRUCT file")
    omissions = conversion_notes(job)
    with zipfile.ZipFile(job/"rtstruct-results.zip","w",zipfile.ZIP_DEFLATED) as z:
        nested = sorted(path for path in (job / "output").rglob("*")
                        if path.is_file() and path.parent != job / "output")
        orientation_records = sorted((job / "output").glob("*.dcm.orientation.json"))
        for result in results + orientation_records + nested:
            z.write(result, result.relative_to(job / "output").as_posix())
        if empty:
            z.writestr("NO_STRUCTURES_FOUND.txt",
                       "These models ran to completion but found no structures in the selected series, "
                       "so no RTSTRUCT was created for them:\n" + "".join(f"- {name}\n" for name in empty))
        if omissions:
            z.writestr("CONVERSION_NOTES.txt", "\n".join(omissions) + "\n")
    # Retain the selected input for the viewer until job retention or explicit deletion.
    shutil.rmtree(job/"input_ct",ignore_errors=True)
    if not results:
        detail = f"No structures found by {', '.join(empty)}; no RTSTRUCT was created"
    elif empty:
        detail = f"RTSTRUCT files ready for download; no structures found by {', '.join(empty)}"
    else:
        detail = "RTSTRUCT files ready for download"
    if report is not None:
        detail = "Results ready; AI draft " + report["status"].replace("_", " ")
        save_metadata(job, report_status=report["status"], report_reason=report.get("reason", ""))
    conversion_note = job / "output/lnctvseg_nnunet_v1/OMITTED_LABELS.txt"
    if conversion_note.is_file():
        omissions = omissions + [conversion_note.read_text().strip()]
    notes = metadata(job).get("postprocess_notes", []) + omissions
    if notes: detail += "; " + "; ".join(notes)
    save_metadata(job,status="complete",detail=detail,progress=100,no_structures_found=empty,
                  conversion_notes=omissions)

def validate_postprocess(value: str, models: list[str]) -> list[str]:
    if not value: return []
    if value != "prostate_sbrt_crop": raise HTTPException(400, "Unknown postprocess value")
    if "total" not in models: raise HTTPException(400, "prostate_sbrt_crop postprocessing requires the total model")
    if WORKER_MODE != "local" or task_uses_remote("total"):
        raise HTTPException(400, "Prostate SBRT postprocessing is available only with local total inference")
    return [value]

def note_prostate_not_cropped(job: Path, reason: str):
    note = f"Prostate SBRT crop not performed: {reason}"
    output = job / "output" / "prostate_sbrt_crop"; output.mkdir(parents=True, exist_ok=True)
    (output / "NOT_CROPPED.txt").write_text(note + "\n")
    save_metadata(job, postprocess_notes=metadata(job).get("postprocess_notes", []) + [note])

def postprocess_prostate_crop(job: Path):
    save_metadata(job, detail="Cropping total CT + RTSTRUCT to prostate ±15 cm", progress=None)
    ct_files, _ = discover_dicom(job / "input")
    try:
        crop_ct_rtstruct(ct_files, [job / "output" / "total_RTSTRUCT.dcm"],
                         job / "output" / "prostate_sbrt_crop", roi_mode="prostate",
                         margin_mm=150, keep_uids=False)
    except NoSelectedROIError as exc:
        note_prostate_not_cropped(job, str(exc))

def process_prostate_sbrt_crop(job: Path):
    """CPU-only tool; deliberately independent of the segmentation queue lock."""
    incoming = job / "rtstruct-results.zip.incoming"
    try:
        data = metadata(job)
        if data.get("tool") != "prostate_sbrt_crop": raise ValueError("Unknown tool job")
        save_metadata(job, status="processing", detail="Cropping CT + RTSTRUCT on CPU", progress=None)
        ct_files, rt_files = discover_dicom(job / "input")
        report = crop_ct_rtstruct(ct_files, rt_files, job / "output",
                                  margin_mm=data.get("margin_mm", 150), keep_uids=data.get("keep_uids", False))
        with zipfile.ZipFile(incoming, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((job / "output").rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(job / "output").as_posix())
        incoming.replace(job / "rtstruct-results.zip")
        shutil.rmtree(job / "input", ignore_errors=True)
        low, high = report["kept_z_range"]
        detail = f"Prostate SBRT crop ready: {report['slices_kept']}/{report['slices_total']} slices, z {low:g} to {high:g} mm"
        save_metadata(job, status="complete", detail=detail, progress=100, error=None)
    except Exception as exc:
        incoming.unlink(missing_ok=True)
        save_metadata(job, status="failed", detail="Cropping failed", error=str(exc), progress=None)

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
            check = detect_cbct(ds)
            g.setdefault("cbct", False)
            g.setdefault("cbct_reason", check["reason"])
            if check["is_cbct"]: g.update(cbct=True, cbct_reason=check["reason"])
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
            tasks_to_run = ordered_tasks(metadata(job)["models"])
            if WORKER_MODE == "auto":
                prior = metadata(job)
                if prior.get("fallback_reason") == "gpu_oom":
                    unsupported = [name for name in tasks_to_run if not task_uses_remote(name) and not colab_capable(name)]
                    if unsupported:
                        raise RuntimeError("Worker GPU previously ran out of memory, but these models are not "
                                           f"validated for Colab: {', '.join(unsupported)}")
                    save_metadata(job, status="transferring",
                                  detail="Resuming Colab fallback after previously confirmed worker GPU OOM",
                                  progress=0, execution_backend="colab")
                    local_tasks = [name for name in tasks_to_run if not task_uses_remote(name)]
                    if local_tasks: run_colab_job_on_nfs_worker(job, tasks=local_tasks, finalize=False)
                    for index, task in enumerate(tasks_to_run, 1):
                        if task_uses_remote(task): execute_job_task(job, task, index, len(tasks_to_run))
                    finalize_results(job)
                    return
                save_metadata(job, status="processing", detail="Starting segmentation on worker GPU",
                              progress=None, execution_backend="local")
                for index, task in enumerate(tasks_to_run, 1):
                    if task_uses_remote(task):
                        execute_job_task(job, task, index, len(tasks_to_run))
                        continue
                    before = set((job / "output").iterdir())
                    try:
                        execute_job_task(job, task, index, len(tasks_to_run))
                    except Exception as exc:
                        for path in set((job / "output").iterdir()) - before:
                            if path.is_file(): path.unlink(missing_ok=True)
                            elif path.is_dir(): shutil.rmtree(path, ignore_errors=True)
                        if not is_gpu_oom(exc):
                            raise
                        if GPU_RELEASE_COMMAND.strip():
                            release_worker_gpu(job, task)
                            try:
                                execute_job_task(job, task, index, len(tasks_to_run))
                                continue
                            except Exception as retry_exc:
                                for path in set((job / "output").iterdir()) - before:
                                    if path.is_file(): path.unlink(missing_ok=True)
                                    elif path.is_dir(): shutil.rmtree(path, ignore_errors=True)
                                if not is_gpu_oom(retry_exc):
                                    raise
                                exc = retry_exc
                        remaining = [name for name in tasks_to_run[index - 1:] if not task_uses_remote(name)]
                        unsupported = [name for name in remaining if not colab_capable(name)]
                        if unsupported:
                            raise RuntimeError("Worker GPU ran out of memory, but these remaining models are not "
                                               f"validated for Colab: {', '.join(unsupported)}") from exc
                        save_metadata(job, status="transferring",
                                      detail=f"Worker GPU out of memory during {task}; escalating remaining models to Colab",
                                      progress=0, execution_backend="colab", fallback_reason="gpu_oom")
                        run_colab_job_on_nfs_worker(job, tasks=remaining, finalize=False)
                        for remote_index, remote_task in enumerate(tasks_to_run[index - 1:], index):
                            if task_uses_remote(remote_task):
                                execute_job_task(job, remote_task, remote_index, len(tasks_to_run))
                        break
                finalize_results(job)
                return
            if WORKER_MODE == "local":
                data=metadata(job)
                save_metadata(job,status="processing",detail="Starting local GPU segmentation",progress=None,postprocess_notes=[])
                empty=[]
                for i,task in enumerate(tasks_to_run,1):
                    try:
                        execute_job_task(job, task, i, len(tasks_to_run))
                    except Exception as exc:
                        if not is_empty_prediction(exc):
                            raise
                        empty.append(task)
                        if task == "total" and "prostate_sbrt_crop" in data.get("postprocess", []):
                            note_prostate_not_cropped(job, "total found no structures; no prostate ROI with contours")
                    else:
                        if task == "total" and "prostate_sbrt_crop" in data.get("postprocess", []):
                            postprocess_prostate_crop(job)
                finalize_results(job, empty)
                return
            if WORKER_MODE == "colab":
                unsupported = [task for task in tasks_to_run if not colab_capable(task)]
                if unsupported:
                    raise RuntimeError(f"Models require the managed worker and cannot run in Colab: {', '.join(unsupported)}")
                if COLAB_ORCHESTRATOR == "worker":
                    run_colab_job_via_worker(job)
                elif COLAB_ORCHESTRATOR == "nfs":
                    run_colab_job_on_nfs_worker(job)
                else:
                    run_colab_job(job)
                return
            prepare_worker_job(job, remote, tasks_to_run)
            save_metadata(job,status="processing",detail="GPU segmentation in progress",progress=None)
            for i,task in enumerate(tasks_to_run,1):
                execute_job_task(job, task, i, len(tasks_to_run), remote=remote)
            save_metadata(job,status="retrieving",detail="Retrieving RTSTRUCT results from GPU worker",progress=None)
            retrieve_worker_results(job, remote)
            remove_worker_job(job, remote)
            finalize_results(job)
        except Exception as exc:
            report_job = REPORT_TASK in metadata(job).get("models", [])
            if WORKER_MODE == "ssh" and report_job:
                try: recover_remote_results(job, remote)
                except Exception: pass
            elif WORKER_MODE == "ssh" or (WORKER_MODE == "colab" and COLAB_ORCHESTRATOR == "worker"):
                try: remove_worker_job(job, remote)
                except Exception: save_metadata(job, remote_results_retained=True)
            changes = {"report_status": "failed", "report_reason": "Task failed; see job error"} if report_job else {}
            save_metadata(job,status="failed",detail="Processing failed",error=str(exc), **changes)

@app.get("/",response_class=HTMLResponse)
def home(): return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

def enabled_external_models():
    """ENABLED_EXTERNAL_MODELS plus one model name per line in ENABLED_MODELS_FILE ('#' comments allowed)."""
    names = set(ENABLED_EXTERNAL_MODELS)
    try:
        names |= {line.strip() for line in ENABLED_MODELS_FILE.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")}
    except OSError:
        pass
    return names

def enabled_models_mtime():
    try: return ENABLED_MODELS_FILE.stat().st_mtime_ns
    except OSError: return None

catalog_build_lock = threading.Lock()
catalog_flag_lock = threading.Lock()
catalog_state = {"value": None, "built": 0.0, "enabled_mtime": None, "refreshing": False, "error": None}


def refresh_catalog(*, only_if_missing=False):
    with catalog_build_lock:
        if only_if_missing and catalog_state["value"] is not None:
            return catalog_state["value"]
        with catalog_flag_lock:
            catalog_state["refreshing"] = True
        try:
            mtime = enabled_models_mtime()
            value = build_catalog()
            with catalog_flag_lock:
                catalog_state.update(value=value, built=time.time(), enabled_mtime=mtime, error=None)
            return value
        except Exception as exc:
            # Exception messages may contain paths, command lines or credentials.
            with catalog_flag_lock:
                catalog_state["error"] = f"Catalog refresh failed ({type(exc).__name__})"
            logger.exception("Model catalog refresh failed; retaining the last good catalog")
            raise
        finally:
            with catalog_flag_lock:
                catalog_state["refreshing"] = False


def refresh_catalog_in_background():
    with catalog_flag_lock:
        if catalog_state["refreshing"]: return False
        catalog_state["refreshing"] = True
        cold = catalog_state["value"] is None
    def run():
        try: refresh_catalog(only_if_missing=cold)
        except Exception:
            pass  # refresh_catalog logs the traceback and records the public error.
    threading.Thread(target=run, daemon=True).start()
    return True


@app.get("/api/tasks")
@app.get("/api/v1/models", tags=["automation"])
def tasks():
    """Serve the last good catalog while a stale catalog refreshes in the background."""
    if catalog_state["value"] is None:
        try:
            refresh_catalog(only_if_missing=True)
        except Exception as exc:
            raise HTTPException(503, "Model catalog is unavailable; retry after the refresh succeeds") from exc
    elif (time.time() - catalog_state["built"] > CATALOG_TTL_SECONDS
          or enabled_models_mtime() != catalog_state["enabled_mtime"]):
        refresh_catalog_in_background()
    return catalog_state["value"]

def remote_preflight():
    selected = ["--tasks", *sorted(REMOTE_GPU_TASKS)] if WORKER_MODE in {"local", "auto"} and REMOTE_GPU_TASKS else []
    cp = subprocess.run(remote_command(*remote_environment(), REMOTE_PYTHON,
                        str(Path(REMOTE_APP_ROOT) / "model_preflight.py"), *selected),
                        capture_output=True, text=True, timeout=PREFLIGHT_TIMEOUT_SECONDS, check=True)
    result = json.loads(cp.stdout.strip().splitlines()[-1])
    if not isinstance(result, dict): raise ValueError("Remote preflight did not return an object")
    return result


def apply_model_gates(item, check, enabled):
    name = item["name"]
    blocked = item.pop("blocked_status", None)
    requires_preflight = item.pop("requires_preflight", False)
    if check and not check["ready"]:
        item["missing"] = check["missing"]
    if blocked:
        item.update(enabled=False, status=blocked)
    elif not item.get("enabled", False):
        pass  # Preserve the catalog's declared restriction.
    elif requires_preflight and not check:
        item.update(enabled=False, status="preflight_unavailable")
    elif check and not check["ready"]:
        item.update(enabled=False, status="dependencies_missing", missing=check["missing"])
    elif name == REPORT_TASK:
        if WORKER_MODE == "colab":
            item.update(enabled=False, status="report_requires_managed_gpu_worker")
        elif not check or not check.get("acceptance_ready"):
            item.update(enabled=False, status="report_acceptance_pending")
    else:
        orientation_ready = (check.get("orientation_ready", False) if check is not None
                             else (ORIENTATION_VALIDATION / f"{name}.json").is_file())
        if not orientation_ready:
            item.update(enabled=False, status="orientation_revalidation_pending")
    if item.get("enabled", False):
        if WORKER_MODE == "local" and name not in enabled:
            item.update(enabled=False, status="not_enabled_on_this_server")
        elif WORKER_MODE == "colab" and name not in COLAB_EXTERNAL_TASKS:
            item.update(enabled=False, status="model_bundle_required")


def build_catalog():
    # Catalog is sourced from the installed worker package when available.
    enabled = enabled_external_models()
    code="from totalsegmentator.map_to_binary import class_map,commercial_models; import json, importlib.util; C=__import__('totalsegmentator.map_tasks_config',fromlist=['TASK_CONFIGS']).TASK_CONFIGS if importlib.util.find_spec('totalsegmentator.map_tasks_config') else None; print(json.dumps({'tasks':sorted(t for t in class_map if C is None or t in C),'licensed':list(commercial_models),'structures':{k:list(v.values()) if isinstance(v,dict) else list(v) for k,v in class_map.items()}}))"
    try:
        if WORKER_MODE in {"local", "auto"} or (WORKER_MODE == "colab" and COLAB_ORCHESTRATOR == "nfs"):
            cp=subprocess.run([LOCAL_PYTHON,"-c",code],capture_output=True,text=True,timeout=20,check=True)
        elif WORKER_MODE in {"ssh", "colab"}:
            command=f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(code)}"
            cp=subprocess.run(["ssh",WORKER,command],capture_output=True,text=True,timeout=20,check=True)
        else:
            raise RuntimeError("Colab runtimes are provisioned only while processing jobs")
        remote=json.loads(cp.stdout.strip().splitlines()[-1]); licensed=set(remote["licensed"])
        internal={"test","total_highres_test","total_v1","vertebrae_pp","vertebrae_pp_refined"}
        names=[n for n in remote["tasks"] if n not in internal and not n.endswith("_auxiliary")]
    except Exception: names=FALLBACK_TASKS; licensed=set()
    preflight = {}
    try:
        if WORKER_MODE in {"local", "auto"} or (WORKER_MODE == "colab" and COLAB_ORCHESTRATOR == "nfs"):
            environment = os.environ.copy()
            app_root = REMOTE_APP_ROOT if COLAB_ORCHESTRATOR == "nfs" else str(Path(LOCAL_RUNNER).parent)
            python = REMOTE_PYTHON if COLAB_ORCHESTRATOR == "nfs" else LOCAL_PYTHON
            environment.setdefault("RTSEG_APP_ROOT", app_root)
            environment.setdefault("RTSEG_MODEL_ROOT", REMOTE_MODEL_ROOT if COLAB_ORCHESTRATOR == "nfs" else "/models")
            environment.setdefault("RTSEG_RUNTIME_ROOT", REMOTE_RUNTIME_ROOT if COLAB_ORCHESTRATOR == "nfs" else environment["RTSEG_MODEL_ROOT"])
            cp = subprocess.run([python, str(Path(app_root) / "model_preflight.py")],
                                env=environment, capture_output=True, text=True, timeout=PREFLIGHT_TIMEOUT_SECONDS, check=True)
        elif WORKER_MODE in {"ssh", "colab"}:
            preflight = remote_preflight()
            cp = None
        else:
            raise RuntimeError("No persistent Colab runtime for model preflight")
        if cp is not None: preflight = json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    routed_preflight = {}
    routed_error = None
    if WORKER_MODE in {"local", "auto"} and REMOTE_GPU_TASKS:
        try: routed_preflight = remote_preflight()
        except Exception as exc: routed_error = type(exc).__name__
        # A local marker must never stand in for evidence on the remote worker.
        for name in REMOTE_GPU_TASKS:
            preflight[name] = routed_preflight.get(name, {"ready": False, "missing": ["remote preflight result"],
                                                         "orientation_ready": False, "acceptance_ready": False})
    structure_map = remote.get("structures",{}) if 'remote' in locals() else {}
    catalog=[{"name":n,"licensed":n in licensed,"modality":task_modality(n),
              "enabled":not (WORKER_MODE == "colab" and n in licensed),
              "status":"license_not_configured_in_colab" if WORKER_MODE == "colab" and n in licensed else "available",
              "description":f"TotalSegmentator {task_modality(n)} segmentation task producing the structures listed below.",
              "structures":", ".join(structure_map.get(n,[])) or "Task-specific TotalSegmentator structures",
              "reference":"https://github.com/wasserth/TotalSegmentator"} for n in names]
    for name, details in EXTERNAL_TASKS.items():
        item={"name":name,"licensed":False,"enabled":True,**details}
        apply_model_gates(item, preflight.get(name), enabled)
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
        apply_model_gates(item, preflight.get(name), enabled)
        catalog.append(item)
    for item in catalog:
        if task_uses_remote(item["name"]):
            item["execution_backend"] = "ssh"
            check = routed_preflight.get(item["name"])
            if routed_error:
                item.update(enabled=False, status="remote_worker_unreachable_or_preflight_failed",
                            remote_preflight_error=routed_error)
            elif not check:
                item.update(enabled=False, status="remote_preflight_missing")
            elif not check.get("ready"):
                item.update(enabled=False, status="dependencies_missing", missing=check.get("missing", []))
        item["clinical_tags"] = clinical_tags(item)
    return {"tasks":catalog, "clinical_tags":[{"id": key, "label": value["label"]}
                                                for key, value in CLINICAL_TAGS.items()]}

@app.post("/api/tools/prostate-sbrt-crop", status_code=202)
@app.post("/api/v1/tools/prostate-sbrt-crop", status_code=202, tags=["automation"])
async def create_prostate_sbrt_crop(files: list[UploadFile]=File(...), margin_mm: float=Form(150.0), keep_uids: bool=Form(False)):
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise HTTPException(400, "margin_mm must be finite and non-negative")
    jid=time.strftime("%Y%m%d-%H%M%S-")+secrets.token_hex(4); job=BASE/jid; inp=job/"input"; inp.mkdir(parents=True)
    (job / "output").mkdir()
    total=0; expanded=0
    try:
        for upload in files:
            name=Path(upload.filename or "upload").name; target=job/(secrets.token_hex(4)+"_"+name)
            with target.open("wb") as dst:
                while chunk:=await upload.read(4*1024**2):
                    total+=len(chunk)
                    if total>MAX_BYTES: raise ValueError("Upload exceeds configured 8 GiB limit")
                    dst.write(chunk)
            if zipfile.is_zipfile(target):
                with zipfile.ZipFile(target) as archive:
                    size=sum(i.file_size for i in archive.infolist() if not i.is_dir())
                safe_extract(target, inp, MAX_BYTES-expanded); expanded+=size; target.unlink()
            else:
                expanded+=target.stat().st_size
                if expanded>MAX_BYTES: raise ValueError("Expanded upload exceeds configured 8 GiB limit")
                shutil.move(target, inp/(secrets.token_hex(4)+"_"+name))
        (job/"job.json").write_text(json.dumps({"id":jid,"status":"queued","detail":"Prostate SBRT crop queued on CPU",
            "models":["prostate_sbrt_crop"],"created":time.time(),"error":None,"tool":"prostate_sbrt_crop",
            "margin_mm":margin_mm,"keep_uids":keep_uids},indent=2))
    except Exception as exc:
        shutil.rmtree(job,ignore_errors=True); raise HTTPException(400,str(exc))
    threading.Thread(target=process_prostate_sbrt_crop,args=(job,),daemon=True).start()
    return {"id":jid}

MAX_REQUEST_BYTES = 32 * 1024**2
UPLOAD_LOCK_WAIT_SECONDS = 4.0


@contextmanager
def upload_session(session_id: str):
    # Directory flock coordinates threads AND cooperating uvicorn processes. The browser sends up to four
    # slices at once and each writer holds the lock only for one append, so wait briefly for the writer
    # ahead before answering 429. Blocking: call from worker threads, never on the event loop.
    if not re.fullmatch(r"session-\d{8}-\d{6}-[0-9a-f]+", session_id):
        raise HTTPException(404, "Upload session not found or expired")
    session = BASE / session_id
    try: fd = os.open(session, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError: raise HTTPException(404, "Upload session not found or expired")
    try:
        deadline = time.monotonic() + UPLOAD_LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() >= deadline: raise HTTPException(429, "Upload session busy; retry")
                time.sleep(0.01)
        try:
            data = json.loads((session / "session.json").read_text())
            if data["id"] != session_id or not isinstance(data["files"], dict): raise ValueError()
            if data["status"] not in {"receiving", "assessing", "ready", "failed", "used"}: raise ValueError()
            if not isinstance(data["created"], (int, float)) or not np.isfinite(data["created"]): raise ValueError()
            if not isinstance(data["received_bytes"], int) or data["received_bytes"] < 0: raise ValueError()
            if data["error"] is not None and not isinstance(data["error"], str): raise ValueError()
            if data["result"] is not None and not isinstance(data["result"], dict): raise ValueError()
            for key, item in data["files"].items():
                if not re.fullmatch(r"0|[1-9][0-9]{0,5}", key): raise ValueError()
                if not isinstance(item["name"], str) or item["name"] in {"", ".", ".."} or any(c in item["name"] for c in ("/", "\\", "\x00")): raise ValueError()
                if any(not isinstance(item[k], int) or item[k] < 0 for k in ("size", "received")): raise ValueError()
                if item["received"] > item["size"]: raise ValueError()
        except Exception: raise HTTPException(404, "Upload session not found or expired")
        yield session, data
    finally: os.close(fd)


def save_session(session: Path, data: dict):
    target = session / "session.json"
    incoming = session / "session.json.incoming"
    incoming.write_text(json.dumps(data))
    incoming.replace(target)


@app.post("/api/upload-sessions")
@app.post("/api/v1/upload-sessions", tags=["automation"])
def create_upload_session():
    sid = "session-" + time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(8)
    session = BASE / sid; session.mkdir(); (session / "files").mkdir()
    save_session(session, {"id":sid, "created":time.time(), "status":"receiving",
                           "files":{}, "received_bytes":0, "error":None, "result":None})
    return {"session_id":sid, "max_request_bytes":MAX_REQUEST_BYTES, "max_total_bytes":MAX_BYTES}


def commit_upload_chunk(session_id: str, file_index: int, name: str, offset: int, size: int, body: bytes):
    with upload_session(session_id) as (session, data):
        if data["status"] != "receiving": raise HTTPException(409, "Upload session is no longer receiving")
        if not 0 <= file_index <= 999999 or offset < 0 or size < 0:
            raise HTTPException(400, "Invalid file index, offset or size")
        # Only the numeric index is used on disk until staging. A supplied basename
        # cannot overwrite metadata or another file, including on Windows clients.
        name = Path(name.replace("\\", "/")).name
        if name in {"", ".", ".."} or "\x00" in name: name = "upload"
        key = str(file_index); record = data["files"].get(key)
        target = session / "files" / key
        received = target.stat().st_size if target.exists() else 0
        if record and (record["size"] != size or record["name"] != name):
            raise HTTPException(400, "File name and size must stay the same")
        if offset != received:
            return JSONResponse(status_code=409, content={"detail":"Offset does not match stored bytes", "expected_offset":received})
        if offset + len(body) > size: raise HTTPException(400, "Chunk exceeds declared file size")
        if len(body) > MAX_REQUEST_BYTES: raise HTTPException(413, "Request exceeds 32 MiB limit")
        # Reconcile a write whose metadata commit was interrupted before a retry.
        previous = record["received"] if record else 0
        total = data["received_bytes"] + received - previous + len(body)
        if total > MAX_BYTES or size > MAX_BYTES: raise HTTPException(413, "Upload exceeds configured 8 GiB limit")
        if not record:
            data["files"][key] = {"name":name, "size":size, "received":received}
            save_session(session, data)
        try:
            with target.open("ab") as dst: dst.write(body)
        except Exception:
            if target.exists():
                with target.open("r+b") as dst: dst.truncate(received)
            raise
        data["files"][key]["received"] = received + len(body)
        data["received_bytes"] = total
        save_session(session, data)
        return {"received":received + len(body), "complete":received + len(body) == size}


@app.put("/api/upload-sessions/{session_id}/files/{file_index}")
@app.put("/api/v1/upload-sessions/{session_id}/files/{file_index}", tags=["automation"])
async def put_upload_chunk(session_id: str, file_index: int, request: Request, name: str, offset: int, size: int):
    await run_in_threadpool(require_receiving_session, session_id)
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError: raise HTTPException(400, "Invalid Content-Length")
    if length > MAX_REQUEST_BYTES: raise HTTPException(413, "Request exceeds 32 MiB limit")
    # A 32 MiB slice over a slow link can take minutes (Tailscale users reach the portal directly), so only a
    # body that stops arriving is rejected.
    body, stream = bytearray(), request.stream().__aiter__()
    while True:
        try: chunk = await asyncio.wait_for(stream.__anext__(), timeout=UPLOAD_IDLE_TIMEOUT_SECONDS)
        except StopAsyncIteration: break
        except asyncio.TimeoutError: raise HTTPException(408, "Upload stalled; retry")
        if len(body) + len(chunk) > MAX_REQUEST_BYTES: raise HTTPException(413, "Request exceeds 32 MiB limit")
        body.extend(chunk)
    return await run_in_threadpool(commit_upload_chunk, session_id, file_index, name, offset, size, body)


def require_receiving_session(session_id: str):
    with upload_session(session_id) as (_, data):
        if data["status"] != "receiving": raise HTTPException(409, "Upload session is no longer receiving")


UPLOAD_IDLE_TIMEOUT_SECONDS = 60


@app.get("/api/upload-sessions/{session_id}")
@app.get("/api/v1/upload-sessions/{session_id}", tags=["automation"])
def get_upload_session(session_id: str):
    with upload_session(session_id) as (_, data):
        return {"session_id":session_id, "status":data["status"], "files":len(data["files"]),
                "received_bytes":data["received_bytes"], "error":data["error"], "result":data["result"]}


def complete_session_files(session: Path, data: dict):
    if data["status"] != "receiving": raise HTTPException(409, "Upload session is no longer receiving")
    if not data["files"] or any(not (session / "files" / key).exists() or
            (session / "files" / key).stat().st_size != item["size"] for key, item in data["files"].items()):
        raise HTTPException(409, "Upload session has no files or incomplete files")


def stage_session_files(session: Path, data: dict, stage: Path, crop: bool = False):
    inp = stage / "input"; expanded = 0
    for key, item in data["files"].items():
        target = session / "files" / key
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as archive:
                size = sum(i.file_size for i in archive.infolist() if not i.is_dir())
            safe_extract(target, inp, MAX_BYTES-expanded if crop else MAX_BYTES)
            expanded += size; target.unlink()
        else:
            expanded += target.stat().st_size
            if crop and expanded > MAX_BYTES: raise ValueError("Expanded upload exceeds configured 8 GiB limit")
            shutil.move(target, inp / (key + "_" + item["name"]))


def finish_session_assessment(session: Path, data: dict):
    uid = "upload-" + time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(8)
    stage = BASE / uid
    try:
        (stage / "input").mkdir(parents=True)
        stage_session_files(session, data, stage)
        report = inspect_upload(stage / "input")
        if not report["series"]: raise ValueError("No valid CT/MR/PET DICOM image series were found")
        (stage / "upload.json").write_text(json.dumps({"id":uid, "created":time.time(), **report}, indent=2))
        public = {**report, "series":[{k:v for k,v in g.items() if k != "files"} for g in report["series"]]}
        changes = {"status":"ready", "result":{"upload_id":uid, **public}, "error":None}
    except Exception as exc:
        shutil.rmtree(stage, ignore_errors=True)
        changes = {"status":"failed", "error":str(exc), "result":None}
    # Polling uses a nonblocking lock; a background writer may wait for that brief read.
    while session.exists():
        try:
            with upload_session(session.name) as (_, current):
                current.update(changes); save_session(session, current)
            return
        except HTTPException as exc:
            if exc.status_code != 429: return
            time.sleep(0.01)


@app.post("/api/upload-sessions/{session_id}/assess", status_code=202)
@app.post("/api/v1/upload-sessions/{session_id}/assess", status_code=202, tags=["automation"])
def assess_upload_session(session_id: str):
    with upload_session(session_id) as (session, data):
        # Idempotent finalization makes a lost POST response safe to retry.
        if data["status"] in {"assessing", "ready", "failed"} and data.get("operation") == "assess":
            return {"status":"assessing"}
        complete_session_files(session, data)
        data.update(status="assessing", operation="assess"); save_session(session, data)
        threading.Thread(target=finish_session_assessment, args=(session, data), daemon=True).start()
    return {"status":"assessing"}


def finish_session_job(session: Path, data: dict, job: Path, crop: bool):
    try:
        stage_session_files(session, data, job, crop)
        if crop: process_prostate_sbrt_crop(job)
        else:
            slices, modality = validate_series(job / "input")
            validate_job_series(metadata(job)["models"], inspect_upload(job / "input")["series"][0])
            save_metadata(job, detail=f"{slices} {modality} DICOM slices received")
            process(job)
    except Exception as exc:
        save_metadata(job, status="failed", detail="Cropping failed" if crop else "Upload staging failed", error=str(exc))


def queue_session_job(session_id: str, options: dict, crop: bool):
    with upload_session(session_id) as (session, data):
        operation = "crop" if crop else "jobs"
        if data["status"] == "used" and data.get("operation") == operation and data.get("options") == options:
            return {"id":data["job_id"]}
        complete_session_files(session, data)
        jid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(8); job = BASE / jid
        (job / "input").mkdir(parents=True); (job / "output").mkdir()
        (job / "job.json").write_text(json.dumps({"id":jid, "status":"queued", "created":time.time(),
            "detail":"Prostate SBRT crop queued on CPU" if crop else "Staging uploaded files", "error":None, **options}, indent=2))
        data.update(status="used", operation=operation, options=options, job_id=jid); save_session(session, data)
        threading.Thread(target=finish_session_job, args=(session, data, job, crop), daemon=True).start()
        return {"id":jid}


@app.post("/api/upload-sessions/{session_id}/prostate-sbrt-crop", status_code=202)
@app.post("/api/v1/upload-sessions/{session_id}/prostate-sbrt-crop", status_code=202, tags=["automation"])
def crop_upload_session(session_id: str, margin_mm: float=Form(150.0), keep_uids: bool=Form(False)):
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise HTTPException(400, "margin_mm must be finite and non-negative")
    return queue_session_job(session_id, {"models":["prostate_sbrt_crop"], "tool":"prostate_sbrt_crop",
                                         "margin_mm":margin_mm, "keep_uids":keep_uids}, True)


@app.post("/api/v1/upload-sessions/{session_id}/jobs", status_code=202, tags=["automation"])
async def create_session_job(session_id: str, models: list[str]=Form(...), postprocess: str=Form("")):
    postprocessing = validate_postprocess(postprocess, models)
    if not models: raise HTTPException(400, "Select at least one model")
    def check_session():
        with upload_session(session_id): pass
    await run_in_threadpool(check_session)
    # A cold catalog may run minutes of preflight. Keep legacy validation, but
    # bound the wait; its existing build lock prevents duplicate catalog builds.
    try: catalog = await asyncio.wait_for(run_in_threadpool(tasks), timeout=4)
    except asyncio.TimeoutError: raise HTTPException(503, "Model catalog is warming; retry")
    validate_job_selection(models, catalog)
    return queue_session_job(session_id, {"models":ordered_tasks(models),"report_status":"pending" if REPORT_TASK in models else None, "postprocess":postprocessing}, False)


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
    """Use the same normal-based order and pixel grid as the viewer and prompt runners."""
    return viewer.preview(uid, series_key, index)

def model_details(name: str) -> dict:
    return EXTERNAL_TASKS.get(name) or INFORMATIONAL_MODELS.get(name, {})


def validate_job_selection(models, catalog, *, assessed=False):
    """Validate every creation route against the same catalog and input contract."""
    items = {item["name"]: {**model_details(item["name"]), **item} for item in catalog["tasks"]}
    if not models:
        raise HTTPException(400, "Select at least one model")
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) or name not in items
           or not items[name].get("enabled", True) for name in models):
        raise HTTPException(400, "Invalid model selection")
    if "total" in models and "total_mr" in models:
        raise HTTPException(400, "Select either total (CT) or total_mr (MR), not both")
    if not assessed:
        if any(items[name].get("prompt") for name in models):
            raise HTTPException(400, "A valid prompt is required; use the assessed-upload start route with prompt_json")
        if any(items[name].get("requires_ct") for name in models):
            raise HTTPException(400, "A paired CT series is required; use the assessed-upload start route")
    return items


def validate_job_series(models, selected, items=None):
    items = items or {name: model_details(name) for name in models}
    if REPORT_TASK in models and selected.get("cbct"):
        raise ValueError("CBCT is not supported for AI reports; diagnostic CT is required")
    modality = selected["modality"]
    wrong = [name for name in models if modality not in
             (items[name].get("modalities") or [items[name].get("modality", task_modality(name))])]
    if wrong:
        raise ValueError(f"Incompatible {modality} tasks: {', '.join(wrong)}")


def validate_assessed_prompt(models, items, selected, prompt_json):
    prompt_models = [name for name in models if items[name].get("prompt")]
    prompt_data = None
    if prompt_models:
        try: prompt_data = json.loads(prompt_json)
        except Exception: raise HTTPException(400,"Prompt JSON is required and must be valid JSON")
        if not isinstance(prompt_data, dict):
            raise HTTPException(400,"Prompt JSON must be an object")
        columns, rows, slices = selected["columns"], selected["rows"], selected["slices"]
        positives, negatives, box = (prompt_data.get("positive_points", []),
                                     prompt_data.get("negative_points", []),
                                     prompt_data.get("box"))
        if not isinstance(positives, list) or not isinstance(negatives, list):
            raise HTTPException(400,"positive_points and negative_points must be lists")
        if box is not None and (not isinstance(box, list) or len(box) != 2):
            raise HTTPException(400,"A box prompt must contain exactly two [column,row,slice] corners")
        coordinates = positives + negatives + (box or [])
        text_prompt = prompt_data.get("text", "")
        if not isinstance(text_prompt, str):
            raise HTTPException(400, "Prompt text must be a string")
        if not coordinates and not text_prompt.strip():
            raise HTTPException(400,"The selected prompted model requires a point, box, or text prompt")
        for name in prompt_models:
            kind = items[name]["prompt"]
            contract = items[name].get("prompt_contract", {})
            if box and (positives or negatives) and contract.get("allow_box_with_points") is False:
                raise HTTPException(400, f"{name} cannot combine box and point prompts; use points or a box")
            if kind in {"points", "points_or_box"} and not (positives or (box and kind == "points_or_box")):
                raise HTTPException(400, f"{name} requires a positive point" +
                                    (" or a two-corner box prompt" if kind == "points_or_box" else " prompt"))
            if kind in {"box", "text_and_box"} and not box:
                raise HTTPException(400, f"{name} requires a two-corner box prompt")
            if kind in {"text", "text_and_box"} and not text_prompt.strip():
                raise HTTPException(400, f"{name} requires a non-empty text prompt")
        if "pam_prompted" in prompt_models and not (prompt_data.get("positive_points") or prompt_data.get("box")):
            raise HTTPException(400,"PAM requires a positive point or a two-corner box on one axial slice")
        coordinate_prompt_models = {"sam_med2d_prompted", "medsam_vit_b_box_2d", "medical_sam3_3d_prompted", "sat3d_prompted", "nninteractive_v1_prompted", "vista3d_prompted_ct"}
        if coordinate_prompt_models.intersection(prompt_models) and not coordinates:
            raise HTTPException(400,"The selected model requires at least one point or a two-corner box")
        if {"sat3d_prompted", "vista3d_prompted_ct"}.intersection(prompt_models) and not positives:
            raise HTTPException(400,"The selected 3-D prompted model requires at least one positive point")
        if "nninteractive_v1_prompted" in prompt_models and not (positives or box):
            raise HTTPException(400,"nnInteractive requires a positive point or a two-corner box on one axial slice")
        if "medsam_vit_b_box_2d" in prompt_models and not box:
            raise HTTPException(400,"MedSAM requires a two-corner box on one axial slice")
        if "medical_sam3_3d_prompted" in prompt_models:
            if not box:
                raise HTTPException(400,"Medical-SAM3 requires a two-corner box on one axial slice")
            text_prompt = prompt_data.get("text")
            if not isinstance(text_prompt, str) or not text_prompt.strip():
                raise HTTPException(400,"Medical-SAM3 requires a non-empty medical concept text prompt")
        if "voxtell_v1_1_text_prompted" in prompt_models:
            text_prompt = prompt_data.get("text")
            if not isinstance(text_prompt, str) or not text_prompt.strip():
                raise HTTPException(400,"VoxTell requires a non-empty text prompt")
            if len(text_prompt) > 512 or any(ord(char) < 32 for char in text_prompt):
                raise HTTPException(400,"VoxTell text must be at most 512 printable characters")
        for point in coordinates:
            if not isinstance(point,list) or len(point)!=3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in point):
                raise HTTPException(400,"Every prompt coordinate must be [column,row,slice]")
            if not (0<=point[0]<columns and 0<=point[1]<rows and 0<=point[2]<slices):
                raise HTTPException(400,f"Prompt coordinate {point} is outside the selected series")
        if box and box[0][2] != box[1][2] and any(m != "segvol_prompted_ct" for m in prompt_models):
            raise HTTPException(400,"Both box corners must be on the same slice")
    return prompt_data


@app.post("/api/uploads/{uid}/start")
async def start_assessed(uid: str, series_key: str=Form(...), models: list[str]=Form(...), prompt_json: str=Form(""), postprocess: str=Form("")):
    postprocessing = validate_postprocess(postprocess, models)
    if not re.fullmatch(r"upload-[A-Za-z0-9-]+",uid): raise HTTPException(404)
    stage=BASE/uid
    try: data=json.loads((stage/"upload.json").read_text())
    except Exception: raise HTTPException(404,"Upload not found or expired")
    selected=next((g for g in data["series"] if g["key"]==series_key),None)
    if not selected: raise HTTPException(400,"Selected series was not found")
    items = validate_job_selection(models, tasks(), assessed=True)
    try:
        validate_job_series(models, selected, items)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    modality = selected["modality"]
    needs_ct = any(items[name].get("requires_ct") for name in models)
    prompt_data = validate_assessed_prompt(models, items, selected, prompt_json)
    companion = None
    if needs_ct:
        companion = next((g for g in data["series"]
            if g["modality"] == "CT"
            and selected.get("study_uid") and selected.get("frame_uid")
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
    (job/"job.json").write_text(json.dumps({"id":jid,"status":"queued","detail":detail,"models":ordered_tasks(models),"report_status":"pending" if REPORT_TASK in models else None,"created":time.time(),"error":None,"prompted":bool(prompt_data),"postprocess":postprocessing},indent=2))
    shutil.rmtree(stage,ignore_errors=True); threading.Thread(target=process,args=(job,),daemon=True).start(); return {"id":jid}

@app.post("/api/jobs")
@app.post("/api/v1/jobs", status_code=202, tags=["automation"])
async def create_job(files: list[UploadFile]=File(...), models: list[str]=Form(...), postprocess: str=Form("")):
    postprocessing = validate_postprocess(postprocess, models)
    items = validate_job_selection(models, tasks())
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
        selected = inspect_upload(inp)["series"][0]
        validate_job_series(models, selected, items)
        (job/"job.json").write_text(json.dumps({"id":jid,"status":"queued","detail":f"{slices} {modality} DICOM slices received","models":ordered_tasks(models),"report_status":"pending" if REPORT_TASK in models else None,"created":time.time(),"error":None,"postprocess":postprocessing},indent=2))
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
            found.append({"id":data["id"],"status":data["status"],"created":data.get("created",0),
                          "report_status":data.get("report_status"), "report_reason":data.get("report_reason", "")})
        except Exception: pass
    found.sort(key=lambda x:x["created"],reverse=True)
    return {"jobs":found[:100]}

@app.post("/api/jobs/{jid}/retry", status_code=202)
@app.post("/api/v1/jobs/{jid}/retry", status_code=202, tags=["automation"])
@viewer.serialize_job_mutation
def retry_job(jid: str):
    if not re.fullmatch(r"[A-Za-z0-9-]+", jid): raise HTTPException(404)
    job=BASE/jid; data=metadata(job)
    if data.get("status") != "failed": raise HTTPException(409,"Only failed jobs can be retried")
    if not (job/"input").is_dir() or not any((job/"input").iterdir()):
        raise HTTPException(409,"The retained input series is unavailable; upload it again")
    reset_job_output(job, data)
    save_metadata(job,status="queued",detail="Retry queued",error=None,progress=0,model_index=0,
                  recovery_count=int(data.get("recovery_count",0)), fallback_reason=None,
                  execution_backend=None, gpu_release_attempted=False, postprocess_notes=[])
    target = process_prostate_sbrt_crop if data.get("tool") else process
    threading.Thread(target=target,args=(job,),daemon=True).start()
    return {"id":jid,"status":"queued"}

@app.get("/api/jobs/{jid}/download")
@app.get("/api/v1/jobs/{jid}/result", tags=["automation"])
def download(jid: str):
    job=BASE/jid; data=metadata(job); result=job/"rtstruct-results.zip"
    if data["status"]!="complete" or not result.exists(): raise HTTPException(409,"Result is not ready")
    filename = f"{jid}-prostate-SBRT-crop.zip" if data.get("tool") else f"{jid}-RTSTRUCT.zip"
    return FileResponse(result,filename=filename,media_type="application/zip")

def report_file(jid, extension):
    if not re.fullmatch(r"[A-Za-z0-9-]+", jid): raise HTTPException(404)
    job = BASE / jid
    data = metadata(job)
    path = job / "output/report" / (REPORT_TASK + extension)
    if data["status"] != "complete" or not path.is_file():
        raise HTTPException(409, "Report is not ready")
    return path


@app.get("/api/v1/jobs/{jid}/report", tags=["automation"])
def get_report(jid: str):
    return FileResponse(report_file(jid, ".json"), media_type="application/json")


@app.get("/api/v1/jobs/{jid}/report/html", response_class=HTMLResponse, tags=["automation"])
@app.get("/jobs/{jid}/report", response_class=HTMLResponse)
def view_report(jid: str):
    return FileResponse(report_file(jid, ".html"), media_type="text/html",
                        headers={"Content-Security-Policy": "default-src 'none'; img-src data:; style-src 'unsafe-inline'"})


def remote_worker_reachable():
    try:
        check = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                                WORKER, "true"], capture_output=True, timeout=5)
        return check.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@app.get("/api/config")
@app.get("/api/v1/config", tags=["automation"])
def config():
    return {"retention_hours": RETENTION_HOURS, "max_upload_bytes": MAX_BYTES,
            "worker_mode": WORKER_MODE, "remote_gpu_tasks": sorted(REMOTE_GPU_TASKS),
            "version": app.version, "report_task": REPORT_TASK}


@app.get("/api/v1/health", tags=["automation"])
def health():
    worker_ok = True
    detail = "local worker runtime"
    if WORKER_MODE == "ssh":
        worker_ok = remote_worker_reachable()
        detail = WORKER
    if WORKER_MODE == "colab":
        command = (["ssh", WORKER, str(Path.home() / ".local/bin/colab"), "--auth=oauth2", "sessions"]
                   if COLAB_ORCHESTRATOR == "worker" else colab_command("sessions"))
        check = subprocess.run(command, capture_output=True, text=True, timeout=20)
        colab_ok = check.returncode == 0
        worker_ok = colab_ok
        detail = f"Colab {COLAB_GPU} via {COLAB_AUTH} authentication"
    if WORKER_MODE == "auto":
        local = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        colab_check = subprocess.run(colab_command("sessions"), capture_output=True, text=True, timeout=20)
        worker_ok = local.returncode == 0 and colab_check.returncode == 0
        detail = f"worker GPU first; Colab {COLAB_GPU} fallback on GPU OOM"
    free_mb, total_mb = gpu_memory_mb() if WORKER_MODE in {"local", "auto"} else (None, None)
    remote_ok = remote_worker_reachable() if REMOTE_GPU_TASKS and WORKER_MODE != "ssh" else (worker_ok if WORKER_MODE == "ssh" else None)
    with catalog_flag_lock:
        catalog_health = {"catalog_built_at": catalog_state["built"] or None,
                          "catalog_refreshing": catalog_state["refreshing"],
                          "catalog_error": catalog_state["error"]}
    return {"status":"ok" if worker_ok and remote_ok is not False and not catalog_health["catalog_error"] else "degraded", "worker_mode":WORKER_MODE,
            "worker":detail, "worker_reachable":worker_ok, "gpu_free_mb":free_mb,
            "gpu_total_mb":total_mb, "gpu_release_command_configured":bool(GPU_RELEASE_COMMAND.strip()),
            "remote_gpu_tasks":sorted(REMOTE_GPU_TASKS), "remote_worker_reachable":remote_ok,
            "remote_worker":WORKER if REMOTE_GPU_TASKS else None, "version":app.version, **catalog_health}

def cleanup_once():
    cutoff=time.time()-RETENTION_HOURS*3600
    for job in BASE.iterdir():
        fd = None
        try:
            if not job.is_dir() or job.is_symlink(): continue
            is_session = job.name.startswith("session-")
            if is_session:
                fd = os.open(job, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta=job/("session.json" if is_session else "upload.json" if job.name.startswith("upload-") else "job.json")
            try: created = float(json.loads(meta.read_text())["created"])
            except Exception: created = job.stat().st_mtime
            if created < cutoff: shutil.rmtree(job)
        except Exception: pass
        finally:
            if fd is not None: os.close(fd)


def cleanup_loop():
    while True:
        cleanup_once()
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
        data = metadata(job)
        is_tool = bool(data.get("tool"))
        active_remote = data.get("remote_task_active")
        if not is_tool and active_remote:
            remote = f"{REMOTE_BASE}/{job.name}"
            stop_remote_task(remote, active_remote)
            try: recover_remote_results(job, remote, active_remote)
            except Exception:
                save_metadata(job, status="failed", detail="Remote recovery needs inspection",
                              error="Could not recover remote results; retained on worker")
                continue
        if not is_tool and (WORKER_MODE == "ssh" or (WORKER_MODE == "colab" and COLAB_ORCHESTRATOR == "worker")):
            remote = f"{REMOTE_BASE}/{job.name}"
            if WORKER_MODE == "colab":
                session = f"rtseg-{job.name[-12:]}"
                subprocess.run(["ssh", WORKER, str(Path.home() / ".local/bin/colab"),
                                "--auth=oauth2", "stop", "-s", session], timeout=60)
            if WORKER_MODE == "ssh" and REPORT_TASK in data.get("models", []):
                for task in data["models"]: stop_remote_task(remote, task)
                try: recover_remote_results(job, remote)
                except Exception:
                    save_metadata(job, status="failed", detail="Remote recovery needs inspection",
                                  error="Could not recover remote results; retained on worker")
                    continue
            else:
                subprocess.run(["ssh", WORKER, "rm", "-rf", remote], timeout=30)
        elif not active_remote and not is_tool and WORKER_MODE in {"auto", "colab"} and COLAB_ORCHESTRATOR == "nfs":
            session = f"rtseg-{job.name[-12:]}"
            subprocess.run(colab_command("stop", "-s", session), timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(job / "colab-stage", ignore_errors=True)
        data = metadata(job); recovery_count = int(data.get("recovery_count",0)) + 1
        has_input = (job/"input").is_dir() and any((job/"input").iterdir())
        if has_input and recovery_count <= 3:
            reset_job_output(job, data)
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
        target = process_prostate_sbrt_crop if metadata(job).get("tool") else process
        threading.Thread(target=target,args=(job,),daemon=True).start()

@app.on_event("startup")
def startup():
    refresh_catalog_in_background()  # Warm the catalog; log and expose any initial failure.
    recover_interrupted_jobs()
    threading.Thread(target=cleanup_loop,daemon=True).start()
