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
from collections import deque
from pathlib import Path
from threading import Event, current_thread

import pydicom
from nv_reason_ct_report import TASK as REPORT_TASK, detect_cbct
import numpy as np
from PIL import Image
import asyncio
import fcntl
from contextlib import contextmanager
from functools import wraps
from job_progress import TaskActivity, tail_log, stage_phrase, task_detail
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, JSONResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from crop_prostate_sbrt import NoSelectedROIError, crop_ct_rtstruct, discover_dicom
import result_manifest
from rtseg_version import __version__
from viewer import ViewerService
from mri_inputs import (mr_instance, image_types, finish_mr_series, public_mr_fields, role_contract,
                        resolve_roles, input_directories, input_identity)
from prostate_models import artifact_names as prostate_artifact_names
from gtrc_models import (
    artifact_names as gtrc_artifact_names,
    is_gtrc_task,
    resolve_job_suv_threshold,
)

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
REMOTE_CLEANUP_TIMEOUT_SECONDS = int(os.environ.get("REMOTE_CLEANUP_TIMEOUT_SECONDS", "300"))
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
MAX_RESULT_BYTES = int(os.environ.get("MAX_RESULT_BYTES", str(8 * 1024**3)))
RETENTION_HOURS = int(os.environ.get("RETENTION_HOURS", "24"))
ENABLED_EXTERNAL_MODELS = {name.strip() for name in os.environ.get("ENABLED_EXTERNAL_MODELS", "").split(",") if name.strip()}
ENABLED_MODELS_FILE = Path(os.environ.get("ENABLED_MODELS_FILE", str(BASE / "enabled-models.txt")))
CATALOG_TTL_SECONDS = int(os.environ.get("CATALOG_TTL_SECONDS", "600"))
PREFLIGHT_TIMEOUT_SECONDS = int(os.environ.get("PREFLIGHT_TIMEOUT_SECONDS", "900"))
TASK_TIME_LIMIT_SECONDS = int(os.environ.get("TASK_TIME_LIMIT_SECONDS", str(12 * 3600)))
RTSEG_STALL_SECONDS = max(1, int(os.environ.get("RTSEG_STALL_SECONDS", "1200")))
RTSEG_STALL_RETRIES = max(0, int(os.environ.get("RTSEG_STALL_RETRIES", "1")))
TASK_MONITOR_SECONDS = 5
TASK_STOP_GRACE_SECONDS = int(os.environ.get("TASK_STOP_GRACE_SECONDS", "45"))
BASE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RTsegmentator DICOM RT Segmentation Portal", version=__version__,
              docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")
STRUCTURE_CATALOG = json.loads((Path(__file__).resolve().parent / "catalog/structures.json").read_text())
NODAL_DESCRIPTIVE_FIELDS = frozenset({
    "code_license", "weights_license", "weights_license_unresolved", "description",
    "limitations", "evidence_note", "evidence_sources", "noncommercial", "code_noncommercial",
})


def load_nodal_model_details(path):
    details = json.loads(path.read_text())
    if not isinstance(details, dict):
        raise ValueError("nodal-models.json must contain an object keyed by model name")
    for name, fields in details.items():
        if not isinstance(fields, dict):
            raise ValueError(f"nodal-models.json: {name} must contain a descriptive object")
        unknown = fields.keys() - NODAL_DESCRIPTIVE_FIELDS
        if unknown:
            raise ValueError(f"nodal-models.json: {name} has unsupported fields: {', '.join(sorted(unknown))}")
    return details


NODAL_MODEL_DETAILS = load_nodal_model_details(Path(__file__).resolve().parent / "catalog/nodal-models.json")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

queue_guard = threading.RLock()
submission_lock = threading.RLock()
pending_jobs = deque()
queue_running = False
queue_active = False
queue_current = None
queue_thread = None
queue_sequence = 0
recovery_barriers = set()
metadata_lock = threading.Lock()
job_workers = {}
job_workers_lock = threading.Lock()
watchdog_stop = threading.Event()


def tracked_job_worker(function):
    """Register a worker while it executes a job, including task stall retries."""
    @wraps(function)
    def tracked(job, *args, **kwargs):
        with job_workers_lock:
            job_workers[job.name] = current_thread()
        try:
            return function(job, *args, **kwargs)
        finally:
            with job_workers_lock:
                if job_workers.get(job.name) is current_thread():
                    job_workers.pop(job.name, None)
    return tracked

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
        "structures": "Bilateral cortical, white-matter, deep-gray, ventricular, cerebellar, hippocampal, amygdala and brainstem labels",
        "reference": "https://github.com/BBillot/SynthSeg",
    },
    "mrsegmentator": {
        "modality": "MR", "modalities": ["MR", "CT"],
        "description": "MRSegmentator v2 broad thoracic, abdominal and pelvic anatomy model",
        "structures": "Organs, vessels, bones and muscle groups; validated by upstream for heterogeneous MRI and CT",
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
PROSTATE_TASKS = json.loads((Path(__file__).parent / "catalog/prostate.json").read_text())
EXTERNAL_TASKS.update(PROSTATE_TASKS)
GTRC_TASKS = json.loads((Path(__file__).parent / "catalog/gtrc.json").read_text())
EXTERNAL_TASKS.update(GTRC_TASKS)

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
                             if k not in {"bundle", "config", "checkpoint", "labels", "labels_from_metadata", "trusted_architecture_pickle", "datalist_records", "output_relative", "run_id", "remove_network_img_size", "overrides", "postprocess_on_cpu", "input_axcodes"}}

CADS_OPEN_TASKS = json.loads((Path(__file__).parent / "catalog/cads-open.json").read_text())
for _name, _details in CADS_OPEN_TASKS.items():
    EXTERNAL_TASKS[_name] = {k: v for k, v in _details.items()
                             if k not in {"task_id", "labels"}}

CADS_RESEARCH_TASKS = json.loads((Path(__file__).parent / "catalog/cads-research.json").read_text())
for _name, _details in CADS_RESEARCH_TASKS.items():
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
    "structures": "Organs, vessels, muscles, bones and vertebral/rib levels",
    "reference": "https://github.com/rsummers11/MRISegmenter",
}
EXTERNAL_TASKS["vibesegmentator"] = {
    "modality": "MR",
    "description": "VIBESegmentator model 100 for broad torso MRI/CT anatomy",
    "structures": "Organs, vessels, bones, muscles, body-composition and spine classes",
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
    "structures": "Head-and-neck OARs including brainstem, eyes, lenses, optic apparatus, temporal lobes, pituitary, parotids, ears, TMJs, spinal cord and mandible",
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
        "description":"Official 100.5M-parameter SAM-Med3D Turbo checkpoint. A positive 3-D point selects the object. Points take precedence over a box; the box centre is used only when no positive or negative points are given. A box is not a spatial boundary. The public AMOS fixture passed strict checkpoint loading and inference, followed by source-grid DICOM RTSTRUCT validation.",
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
        "structures":"Anatomy classes on a coronal CT intensity projection",
        "description":"Official TotalSegmentator2D v2 source and all five checkpoint groups (cardiac, muscles, organs, ribs and vertebrae) are installed. Offline composite H100 inference passed on the publisher fixture and produced all 117 segment channels. The result has one voxel across the anterior-posterior axis and encodes projected occupancy, so it cannot be reconstructed into patient-specific 3-D contours or a geometrically valid RTSTRUCT.",
        "reference":"https://github.com/risc-mi/totalsegmentator2D"},
    "brats_toolkit_legacy_ensemble_mr": {"modality":"MR", "enabled":False, "status":"requires_four_registered_preprocessed_mr_series",
        "structures":"BraTS whole tumor, tumor core and enhancing tumor candidate segmentations from eight legacy algorithms",
        "description":"The official BraTS Toolkit source and its eight downloadable legacy algorithm images are preserved on NFS. Each algorithm expects skull-stripped, co-registered FLAIR, T1, post-contrast T1 and T2 NIfTI volumes in SRI-24 space; the portal currently selects one DICOM series and therefore cannot satisfy or safely reconstruct this input contract.",
        "reference":"https://github.com/BrainLesion/BraTS-Toolkit"},
    "tractseg_dwi_peaks": {"modality":"MR", "enabled":False, "status":"requires_diffusion_peaks_or_dwi_gradients",
        "structures":"White-matter tract bundle masks",
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
        "structures":"Reported lymph-node stations; nodal checkpoints absent from the public archive",
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
    "lnq_atlas": {"noncommercial":True, "modality":"CT", "enabled":True, "requires_preflight":True, "status":"technical_checks_recorded",
        "structures":"Visible mediastinal lymph-node candidates with probabilistic-atlas context",
        "license_note":"Code CC BY-NC-ND 4.0. Archive weight terms are not stated. Owner accepted these terms on 2026-10-22.",
        "description":"Segments visible mediastinal lymph-node candidates on one CT series. The output is one aggregate ROI, Mediastinal_Lymph_Node_Candidate, not an elective nodal region or a station map. No remaining candidate completes with no RTSTRUCT. Code licence CC BY-NC-ND 4.0. Archive weight terms are not stated. Owner accepted these terms on 2026-10-22. This task is routed to the H100 worker. On the P6000 portal (compute capability 6.1), keep lnq_atlas in REMOTE_GPU_TASKS. An acceptance marker recorded on the H100 does not enable the task on a P6000. Engelson et al., MELBA 2024, sections 4.1-4.3 and Tables 2 and 4: on the LNQ test set of 100 patients and 861 reference nodes, the ablation that uses all modifications, a binarization threshold of 0.2 and no minimum-diameter filter had Dice 0.6033 and found 57% of reference nodes (LN-found 0.5654), compared with 27% (LN-found 0.2749) when training on CT only. The challenge submission, a five-model ensemble with threshold 0.3 and a 5 mm minimum diameter, had Dice 0.5690 and ASSD 6.8976 mm on that same test set. The authors state that post-processing fit on the small validation set did not carry over to the test set. The installed adapter uses the submission settings (threshold 0.3, 5 mm), not the 0.2 ablation. Selectable only after the GPU acceptance marker and the orientation marker pass.",
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
    fields = [str(item.get(key, "")) for key in ("name", "description", "structures")]
    if item.get("reference") == "https://github.com/wasserth/TotalSegmentator":
        # TotalSegmentator formerly supplied the full class list in its prose summary.
        fields.extend(item.get("structure_names", []))
    searchable = " ".join(fields).lower()
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

_rsync_zstd_supported: dict[str, bool] = {}


def worker_rsync_command(source, destination, *, compressed=True):
    command = ["rsync", "-a", "--info=progress2", "--no-inc-recursive"]
    if compressed:
        command.extend(["--compress", "--compress-choice=zstd", "--compress-level=3"])
    return [*command, str(source), f"{WORKER}:{destination}"]


def rsync_compression_rejected(output):
    text = output.lower()
    return (any(word in text for word in ("compress", "zstd")) and
            any(word in text for word in ("unknown option", "unrecognized option", "unknown compress",
                                         "not support", "unsupported", "failed to negotiate",
                                         "rejected", "not compiled with", "invalid compress")))


def transfer_to_worker(job: Path, source: Path, destination: str):
    """Copy one archive while publishing rsync's real byte progress."""
    total = source.stat().st_size
    save_metadata(job, status="transferring", detail="Sending DICOM series to GPU worker",
                  progress=0, transferred_bytes=0, total_bytes=total)
    compressed = _rsync_zstd_supported.get(WORKER, True)
    while True:
        command = worker_rsync_command(source, destination, compressed=compressed)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        last, diagnostics = -1, ""
        assert process.stdout is not None
        for line in process.stdout:
            diagnostics = (diagnostics + line)[-8192:]
            for match in re.finditer(r"([0-9,]+)\s+(\d+)%", line):
                # progress2 counts original file bytes, even with compression.
                transferred = int(match.group(1).replace(",", ""))
                percent = int(match.group(2))
                if percent != last:
                    save_metadata(job, progress=percent, transferred_bytes=min(transferred, total))
                    last = percent
        returncode = process.wait()
        if not returncode:
            _rsync_zstd_supported[WORKER] = compressed
            break
        if compressed and rsync_compression_rejected(diagnostics):
            _rsync_zstd_supported[WORKER] = False
            compressed = False
            save_metadata(job, progress=0, transferred_bytes=0)
            continue
        raise subprocess.CalledProcessError(returncode, command, output=diagnostics)
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


def stop_recorded_task_tree(data):
    """Stop an unmonitored runner only while its recorded process identity matches."""
    pid, started = data.get('task_pid'), data.get('task_pid_started')
    def alive(pid, started):
        stat = proc_stat(pid)
        return bool(stat and stat[0] != 'Z' and stat[2] == started)
    if not pid or not started or not alive(pid, started):
        return
    tree = process_tree(pid)
    if tree.get(pid) != started or not alive(pid, started):
        return
    def signal_verified(pid, started, sig):
        if alive(pid, started):
            try: os.kill(pid, sig)
            except ProcessLookupError: pass
    signal_verified(pid, started, signal.SIGTERM)
    deadline = time.monotonic() + TASK_STOP_GRACE_SECONDS
    while any(alive(p, s) for p, s in tree.items()) and time.monotonic() < deadline:
        time.sleep(.05)
    # Retain detached children from the original snapshot, and include children
    # started during shutdown before killing any remaining verified process.
    for p, s in list(tree.items()):
        if alive(p, s):
            descendants = process_tree(p)
            # A PID can be reused during the /proc scan. Do not adopt the new
            # process or its children into the tree we are authorized to stop.
            if descendants.get(p) == s and alive(p, s):
                tree.update(descendants)
    for p, s in tree.items():
        signal_verified(p, s, signal.SIGKILL)
    deadline = time.monotonic() + max(1, min(5, TASK_STOP_GRACE_SECONDS))
    while any(alive(p, s) for p, s in tree.items()) and time.monotonic() < deadline:
        time.sleep(.05)
    if any(alive(p, s) for p, s in tree.items()):
        raise RuntimeError('Could not confirm that the interrupted task tree stopped')

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
    data = metadata(job)
    members = [name for name in input_directories(data) if (job / name).is_dir()]
    members.append('job.json')
    if (job / "prompt.json").is_file(): members.append("prompt.json")
    # The report needs finished contours, not bulky crop packages or previous drafts.
    if REPORT_TASK in tasks:
        members.extend(p.relative_to(job).as_posix() for p in sorted((job / "output").glob("*_RTSTRUCT.dcm")))
    return members


def prepare_worker_job(job, remote, tasks):
    data = metadata(job)
    if data.get('input_roles') and not input_manifest_complete(job, data):
        raise ValueError(INPUT_PREPARATION_ERROR)
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


def remove_worker_job(job, remote, timeout=30):
    subprocess.run(remote_command("rm", "-rf", "--", remote), check=True, timeout=timeout)
    save_metadata(job, remote_task_active=None, remote_results_retained=False, remote_cleanup_pending=False)


def cleanup_worker_job(job, remote):
    """Remove the worker copy after results are safely local; a slow or failed removal must not fail the job."""
    try:
        remove_worker_job(job, remote, timeout=REMOTE_CLEANUP_TIMEOUT_SECONDS)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Worker cleanup for %s did not finish (%s); it is retried when the job is deleted",
                       job.name, exc)
        save_metadata(job, remote_task_active=None, remote_results_retained=False, remote_cleanup_pending=True)


def recover_remote_results(job, remote, task=None):
    try:
        retrieve_worker_results(job, remote, task)
    except Exception:
        save_metadata(job, remote_results_retained=True)
        raise
    cleanup_worker_job(job, remote)


def run_remote_gpu_task(job, task, index, total):
    remote = f"{REMOTE_BASE}/{job.name}"
    prior = metadata(job)
    if prior.get("remote_results_retained"):
        recover_remote_results(job, remote, prior.get("remote_task_active"))
    save_metadata(job, remote_task_active=task, execution_backend="ssh_task")
    empty = None
    try:
        prepare_worker_job(job, remote, [task])
        try:
            run_worker_task(job, remote, task, index, total)
        except Exception as exc:
            if not is_empty_prediction(exc):
                raise
            empty = exc
        save_metadata(job, status="retrieving", detail=f"Retrieving {task} results", progress=None)
        recover_remote_results(job, remote, task)
    except Exception:
        # Preserve local contours. If retrieval cannot finish, retain the remote copy for recovery.
        try: recover_remote_results(job, remote, task)
        except Exception: pass
        raise
    if empty is not None:
        raise empty


def execute_job_task(job, task, index, total, remote=None):
    data = metadata(job)
    result = job / "output" / ("report/nv_reason_ct_report.json" if task == REPORT_TASK else f"{task}_RTSTRUCT.dcm")
    if task in data.get("completed_models", []) and result.is_file():
        return
    if task == REPORT_TASK: save_metadata(job, report_status="running", report_reason="")
    if remote is not None:
        run_worker_task(job, remote, task, index, total)
    elif task_uses_remote(task):
        run_remote_gpu_task(job, task, index, total)
    else:
        run_local_task(job, task, index, total)
    done = metadata(job).get("completed_models", [])
    save_metadata(job, completed_models=list(dict.fromkeys([*done, task])))


def process_cpu_times(root):
    """CPU ticks by (PID, start-time); child exit and PID reuse cannot hide activity."""
    samples = {}
    for pid, started in process_tree(root).items():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[19] == started:
                # Include waited-for children, which may start and exit between polls.
                samples[(pid, started)] = sum(int(fields[i]) for i in (11, 12, 13, 14))
        except (OSError, ValueError, IndexError):
            pass
    return samples


def reset_task_output(job, task, remote=None):
    """Discard only this task's partial results and work; keep completed models."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", task): raise ValueError("Invalid task")
    relative = [f"external/{task}", "output/report" if task == REPORT_TASK else f"output/{task}"]
    relative.extend(p.relative_to(job).as_posix() for p in (job / "output").glob(f"{task}_RTSTRUCT.dcm*"))
    for name in relative:
        path = job / name
        if path.is_dir(): shutil.rmtree(path)
        else: path.unlink(missing_ok=True)
    if remote is not None:
        # Shell expansion is restricted to a validated task identifier.
        prefix = shlex.quote(f"{remote}/output/{task}_RTSTRUCT.dcm") + "*"
        paths = [f"{remote}/external/{task}", f"{remote}/output/{'report' if task == REPORT_TASK else task}"]
        command = "rm -rf -- " + " ".join(shlex.quote(path) for path in paths) + " " + prefix
        subprocess.run(["ssh", WORKER, command], check=True, timeout=30)


def run_monitored_task(job, task, index, total, command, *, environment=None, remote=None):
    log = job / f"{task}.log"
    data = metadata(job)
    retries = sum(event.get('task') == task and event.get('action') == 'retry'
                  for event in data.get('stall_events', []))
    deadline = time.monotonic() + TASK_TIME_LIMIT_SECONDS
    while True:
        if time.monotonic() >= deadline:
            finish_task_timing(job, task, 'failed')
            raise subprocess.TimeoutExpired(command, TASK_TIME_LIMIT_SECONDS)
        now = time.time()
        activity = TaskActivity(now)
        timings = metadata(job).get('task_timings', {})
        prior = timings.get(task, {})
        timings[task] = {**prior, 'started': prior.get('started', now), 'attempt_started': now,
                         'attempts': prior.get('attempts', 0) + 1, 'status': 'running'}
        save_metadata(job, status='processing', current_task=task, task_started=now, task_remote=remote is not None,
                      model_index=index, model_total=total, progress=None,
                      task_timings=timings, stage=activity.stage['stage'], stage_key='staging',
                      stage_started=now, stage_done=None, stage_total=None,
                      last_activity=now, last_cpu_activity=now, stall_warning=False,
                      stall_threshold=RTSEG_STALL_SECONDS,
                      detail=f"{task} ({index}/{total}): preparing model input")
        warned = False
        stalled = False
        # Append on a stall retry so the failed attempt remains inspectable.
        with log.open('a' if retries else 'w') as output:
            if retries:
                output.write(f"\nAutomatic stall retry {retries} for {task}\n")
                output.flush()
            activity.position = log.stat().st_size
            attempt_log_start = activity.position
            process = subprocess.Popen(command, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, text=True)
            pid = getattr(process, 'pid', None)
            stat = proc_stat(pid) if pid else None
            save_metadata(job, task_pid=pid, task_pid_started=stat[2] if stat else None)
            stopped = False
            try:
                while process.poll() is None:
                    now = time.time()
                    fields = activity.sample(log, now, None if remote else process_cpu_times(process.pid))
                    idle = activity.stalled(now, RTSEG_STALL_SECONDS, remote=remote is not None)
                    save_metadata(job, **fields, stall_warning=idle,
                                  detail=task_detail(task, index, total, fields))
                    if time.monotonic() >= deadline:
                        if remote: stop_remote_task(remote, task)
                        stop_process_tree(process)
                        stopped = True
                        finish_task_timing(job, task, 'failed')
                        raise subprocess.TimeoutExpired(command, TASK_TIME_LIMIT_SECONDS)
                    if idle and warned:
                        stalled = True
                        if remote: stop_remote_task(remote, task)
                        stop_process_tree(process)
                        stopped = True
                        break
                    warned = idle  # Publish a warning for one monitor interval before stopping.
                    time.sleep(TASK_MONITOR_SECONDS)
                fields = activity.sample(log, time.time(), None if remote or not pid else process_cpu_times(pid))
                save_metadata(job, **fields, detail=task_detail(task, index, total, fields))
            finally:
                # Never leave a running inference child if monitoring itself fails.
                if not stopped and process.poll() is None:
                    if remote: stop_remote_task(remote, task)
                    stop_process_tree(process)
                save_metadata(job, task_pid=None, task_pid_started=None)
        if time.monotonic() >= deadline:
            finish_task_timing(job, task, 'failed')
            raise subprocess.TimeoutExpired(command, TASK_TIME_LIMIT_SECONDS)
        if stalled:
            data = metadata(job)
            retry = retries < RTSEG_STALL_RETRIES
            event = {'task': task, 'stage': activity.stage['stage'], 'time': time.time(),
                     'action': 'retry' if retry else 'failed'}
            save_metadata(job, stall_events=[*data.get('stall_events', []), event])
            if retry:
                reset_task_output(job, task, remote)
                retries += 1
                continue
            duration = f"{RTSEG_STALL_SECONDS / 60:g} min"
            suffix = '; retried once' if retries == 1 else (f'; retried {retries} times' if retries else '')
            message = f"{task} stalled during {stage_phrase(activity.stage['stage'])} for {duration}{suffix}"
            finish_task_timing(job, task, 'failed')
            raise RuntimeError(message)
        if process.returncode:
            finish_task_timing(job, task, 'failed')
            with log.open('rb') as source:
                source.seek(max(attempt_log_start, source.seek(0, 2) - 24000))
                error = source.read(24000).decode('utf-8', errors='replace')[-6000:]
            raise RuntimeError(error or f"{task} exited with status {process.returncode}")
        finish_task_timing(job, task, 'done')
        save_metadata(job, stall_warning=False)
        return


def finish_task_timing(job, task, status):
    timings = metadata(job).get('task_timings', {})
    timing = timings.get(task, {})
    now = time.time()
    timings[task] = {**timing, 'finished': now, 'duration': max(0, now - timing.get('started', now)),
                     'status': status}
    save_metadata(job, task_timings=timings)


def run_worker_task(job: Path, remote: str, task: str, index: int, total: int):
    command = remote_command(*remote_environment(), REMOTE_PYTHON, REMOTE_RUNNER, remote, task)
    run_monitored_task(job, task, index, total, command, remote=remote)


def run_local_task(job: Path, task: str, index: int, total: int):
    environment = os.environ.copy()
    worker_local = WORKER_MODE == "auto"
    runner = REMOTE_RUNNER if worker_local else LOCAL_RUNNER
    python = REMOTE_PYTHON if worker_local else LOCAL_PYTHON
    environment.setdefault("RTSEG_APP_ROOT", REMOTE_APP_ROOT if worker_local else str(Path(runner).parent))
    environment.setdefault("RTSEG_MODEL_ROOT", REMOTE_MODEL_ROOT if worker_local else "/models")
    environment.setdefault("RTSEG_RUNTIME_ROOT", REMOTE_RUNTIME_ROOT if worker_local else environment["RTSEG_MODEL_ROOT"])
    run_monitored_task(job, task, index, total, [python, runner, str(job), task], environment=environment)

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

def reject_multirole_colab(job):
    # The separate Colab runners still package only legacy inputs/results.
    if metadata(job).get('input_roles'):
        raise RuntimeError("Multi-sequence MRI jobs require local or SSH execution; Colab role transport is not supported")

def run_colab_job(job: Path, tasks: list[str] | None = None, finalize: bool = True):
    """Run a complete job in an ephemeral Colab GPU session."""
    reject_multirole_colab(job)
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
            for name in (*input_directories(metadata(job)), "prompt.json"):
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
                if info.is_dir() or Path(info.filename).name != info.filename or not info.filename.endswith((".dcm", ".dcm.orientation.json", ".dcm.lesions.json")):
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
    reject_multirole_colab(job)
    session = f"rtseg-{job.name[-12:]}"
    remote = f"{REMOTE_BASE}/{job.name}"
    archive = job / "selected-series.tar"
    members = [name for name in input_directories(metadata(job)) if (job / name).is_dir()] + ["job.json"]
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
    reject_multirole_colab(job)
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
    save_metadata(job, status="retrieving", detail="Preparing results for download", progress=None,
                  stage="Exporting results", stage_key="export", stage_started=time.time(),
                  stage_done=None, stage_total=None, stall_warning=False)
    output = job / "output"
    nested = sorted(path for path in output.rglob("*") if path.is_file() and path.parent != output)
    sidecars = sorted(output.glob("*.dcm.orientation.json")) + sorted(output.glob("*.dcm.lesions.json"))
    declared = [output / name for task in metadata(job).get("models", [])
                for name in (*prostate_artifact_names(task), *gtrc_artifact_names(task))
                if (output / name).is_file()]
    artifacts = sorted(set(results + sidecars + nested + declared))
    if any(path.is_symlink() or output.resolve() not in path.resolve().parents for path in artifacts):
        raise ValueError("Result artifacts must be files inside the output directory")
    if sum(path.stat().st_size for path in artifacts) > MAX_RESULT_BYTES:
        raise ValueError("Result artifacts exceed the configured download size limit")
    members = [(result.relative_to(output).as_posix(), result) for result in artifacts]
    if empty:
        members.append(("NO_STRUCTURES_FOUND.txt",
                        ("These models ran to completion but found no structures in the selected series, "
                         "so no RTSTRUCT was created for them:\n" + "".join(f"- {name}\n" for name in empty)).encode()))
    if omissions:
        members.append(("CONVERSION_NOTES.txt", ("\n".join(omissions) + "\n").encode()))
    record = metadata(job)
    manifest = result_manifest.build_manifest(
        record, members, version=__version__, catalog=manifest_catalog(record.get("models", [])),
        structure_catalog=STRUCTURE_CATALOG, empty_models=empty,
        notes=list(record.get("postprocess_notes", [])) + omissions, created=time.time())
    with zipfile.ZipFile(job/"rtstruct-results.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name, source in members:
            if isinstance(source, Path):
                z.write(source, name)
            else:
                z.writestr(name, source)
        z.writestr(result_manifest.MANIFEST_NAME, result_manifest.dumps(manifest))
    # Retain the selected input for the viewer until job retention or explicit deletion.
    for name in input_directories(metadata(job)):
        if name != 'input':
            shutil.rmtree(job / name, ignore_errors=True)
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

def manifest_catalog(models) -> dict:
    """Descriptive model fields for the result manifest, from static metadata and the cached catalog.

    Finalization never rebuilds the catalog; a model absent from the cache keeps its static fields."""
    cached = {item.get("name"): item for item in (catalog_state.get("value") or {}).get("tasks", [])}
    return {name: {**model_details(name), **NODAL_MODEL_DETAILS.get(name, {}), **cached.get(name, {})}
            for name in models}

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

@tracked_job_worker
def process_prostate_sbrt_crop(job: Path):
    """CPU-only tool; deliberately independent of the segmentation FIFO dispatcher."""
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
        names = report["selected_roi_names"]
        around = ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
        detail = (f"Prostate SBRT crop ready: {report['slices_kept']}/{report['slices_total']} slices, "
                  f"z {low:g} to {high:g} mm, around {around}")
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
    groups = {}; ignored = 0; unattributed = []; rejected_by_uid = {}

    def image_evidence(ds):
        if any(tag in ds for tag in ("PixelData", "FloatPixelData", "DoubleFloatPixelData")):
            return True
        sop = str(getattr(ds, "SOPClassUID", "") or
                  getattr(getattr(ds, "file_meta", None), "MediaStorageSOPClassUID", ""))
        if sop:
            name = pydicom.uid.UID(sop).name
            if "Image Storage" in name:
                return True
            if name != sop:
                # A known non-image object (e.g. a DICOMDIR media index) is not a lost image.
                return False
        # Unknown or absent SOP class: a DICOM preamble is the remaining image evidence.
        return getattr(ds, "preamble", None) is not None

    def recover_identity(path):
        # Deferred pixels are used only as tag evidence, never decoded or loaded.
        return pydicom.dcmread(path, force=True, defer_size=1, specific_tags=[
            "Modality", "SeriesInstanceUID", "StudyInstanceUID", "FrameOfReferenceUID",
            "SeriesDescription", "StudyDate", "SOPClassUID", "PixelData",
            "FloatPixelData", "DoubleFloatPixelData"])

    def identify(ds):
        modality = str(getattr(ds, "Modality", ""))
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if modality not in IMAGE_MODALITIES or not uid:
            return None
        key = f"{uid}|{modality}"
        return groups.setdefault(key, {"key": key, "series_uid": uid, "modality": modality,
            "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
            "frame_uid": str(getattr(ds, "FrameOfReferenceUID", "")),
            "description": str(getattr(ds, "SeriesDescription", "")),
            "study_date": str(getattr(ds, "StudyDate", "")), "image_type": [],
            "files": [], "rows": 0, "columns": 0, **({"instances": []} if modality == "MR" else {})})

    for f in sorted(input_dir.iterdir()):
        g = None; identified_image = False
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            # Keep the series identity before detailed metadata can reject an instance.
            g = identify(ds)
            if g is None:
                uid = str(getattr(ds, "SeriesInstanceUID", ""))
                if uid:
                    rejected_by_uid.setdefault(uid, []).append({
                        "file": f.name, "reason": "Instance has missing or unsupported Modality"})
                elif str(getattr(ds, "Modality", "")) == "MR":
                    raise ValueError("MR instance has no SeriesInstanceUID")
                elif image_evidence(ds) or image_evidence(recover_identity(f)):
                    identified_image = True
                    raise ValueError("Image instance has no recoverable series identity")
                ignored += 1; continue
            instance = mr_instance(ds, f.name) if g["modality"] == "MR" else None
            rows, columns = int(ds.Rows), int(ds.Columns)
            if min(rows, columns) < 1:
                raise ValueError("Invalid image dimensions")
            g.update(rows=rows, columns=columns)
            g["image_type"] = list(dict.fromkeys(g["image_type"] + image_types(ds)))
            check = detect_cbct(ds)
            g.setdefault("cbct", False)
            g.setdefault("cbct_reason", check["reason"])
            if check["is_cbct"]: g.update(cbct=True, cbct_reason=check["reason"])
            g["files"].append(f.name)
            if instance is not None:
                g["instances"].append(instance)
        except Exception as exc:
            ignored += 1
            if g is None:
                try:
                    identity = recover_identity(f)
                    identified_image = identified_image or image_evidence(identity)
                    g = identify(identity)
                except Exception:
                    pass
            known_reasons = {"Invalid NumberOfFrames", "Invalid image dimensions",
                             "MR instance has no SeriesInstanceUID",
                             "Image instance has no recoverable series identity"}
            reason = str(exc) if str(exc) in known_reasons else f"Unreadable or rejected image metadata ({type(exc).__name__})"
            error = {"file": f.name, "reason": reason}
            if g is not None:
                g.setdefault("assessment_errors", []).append(error)
            else:
                # A damaged DICOM with no recoverable identity cannot safely be
                # assigned to a surviving MR series. Unrelated non-DICOM files remain ignored.
                try:
                    with f.open("rb") as stream:
                        is_dicom = stream.read(132)[128:] == b"DICM"
                    if identified_image or is_dicom or f.suffix.lower() == ".dcm":
                        unattributed.append(error)
                except OSError:
                    pass
    series=[]
    for g in groups.values():
        if g["series_uid"] in rejected_by_uid:
            g.setdefault("assessment_errors", []).extend(rejected_by_uid[g["series_uid"]])
        if g["modality"] == "MR" and unattributed:
            g.setdefault("assessment_errors", []).extend(unattributed)
        if g.get("assessment_errors"):
            reasons = "; ".join(dict.fromkeys(e["reason"] for e in g["assessment_errors"]))
            g["unavailable_reason"] = (f"Series assessment rejected {len(g['assessment_errors'])} image instance(s): "
                                       f"{reasons}. Upload a complete readable series")
        if g["modality"] == "MR":
            finish_mr_series(g)
        g["slices"]=len(g["files"]); series.append(g)
    series.sort(key=lambda x:(x["modality"],x["description"],x["series_uid"]))
    return {"series":series,"ignored":ignored}


def require_series_assessment(series):
    if series.get("unavailable_reason") or series.get("assessment_errors"):
        raise ValueError(series.get("unavailable_reason") or "Series assessment rejected image instances")


def validate_staged_mri(job):
    """Reassess copied/moved MRI before publishing the job as runnable."""
    data = metadata(job)
    if not data.get("input_roles") and (data.get("series") or {}).get("modality") != "MR":
        return
    for directory in {"input"} | {r["dir"] for r in data.get("input_roles", {}).values()}:
        report = inspect_upload(job / directory)
        if report["ignored"] or len(report["series"]) != 1:
            raise ValueError(f"{directory}: staged series assessment rejected image instances")
        require_series_assessment(report["series"][0])

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

def series_metadata(series):
    if series is None:
        return None
    result = {key: series.get(key) for key in ("key", "series_uid", "modality", "description",
        "slices", "rows", "columns", "study_date", "cbct")}
    if series.get('modality') == 'MR':
        result.update(public_mr_fields(series))
    return result


def public_assessment(report):
    return {**report, 'series': [{**{k: v for k, v in s.items() if k not in {'files', 'instances', 'hbv_options'}},
        **(public_mr_fields(s) if s['modality'] == 'MR' else {})} for s in report['series']]}


def new_batch_id():
    return "batch-" + time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(8)


def single_series_batch(series):
    return {"batch_id": new_batch_id(), "batch_index": 1, "batch_size": 1, "series": series_metadata(series)}


INPUT_PREPARATION_ERROR = "Interrupted while preparing input; start the job again"


def directory_manifest(directory, names=None):
    names = sorted(names if names is not None else (p.name for p in directory.iterdir() if p.is_file()))
    return [{'name': name, 'size': (directory / name).stat().st_size} for name in names]


def job_input_manifest(job):
    data = metadata(job) if (job / 'job.json').is_file() else {}
    return {name: directory_manifest(job / name) for name in input_directories(data) if (job / name).is_dir()}


def legacy_input_complete(job, data):
    """Jobs created before input manifests: their metadata was written only after the series was in place, so a
    nonempty input holding one image series is the original input. Jobs from this version always carry a manifest
    (or a staging marker), so the fallback never applies to them."""
    if 'input_manifest' in data or 'input_roles' in data or 'staging_upload' in data or data.get('input_preparation_interrupted'):
        return False
    try:
        slices, _ = validate_series(job / 'input')
        expected = (data.get('series') or {}).get('slices')
        return expected is None or slices == expected
    except Exception:
        return False


def input_manifest_complete(job, data):
    """Exact names and sizes prevent dispatch after an interrupted move or copy."""
    try:
        if data.get('input_preparation_interrupted'):
            return False
        manifest = data.get('input_manifest')
        # Never create a manifest from a possibly interrupted input directory.
        if not isinstance(manifest, dict):
            return legacy_input_complete(job, data)
        allowed = set(input_directories(data))
        required = {'input'} | {r['dir'] for r in data.get('input_roles', {}).values()}
        return (required <= set(manifest) <= allowed and all(manifest[name] for name in required)
                and (not data.get('input_roles') or data.get('input_identity') == input_identity(data))
                and all(directory_manifest(job / name) == expected for name, expected in manifest.items()))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def fail_input_preparation(job):
    save_metadata(job, status='failed', detail=INPUT_PREPARATION_ERROR,
                  error=INPUT_PREPARATION_ERROR, input_preparation_interrupted=True)


def rollback_staged_batch(jobs):
    """Restore primary inputs only if the complete assessed upload can be rebuilt."""
    try:
        records = [(job, metadata(job)) for job in jobs]
        uid = records[0][1].get('staging_upload', '')
        if not re.fullmatch(r'upload-[A-Za-z0-9-]+', uid):
            return False
        if any(data.get('staging_upload') != uid for _, data in records):
            return False
        stage = BASE / uid
        upload = json.loads((stage / 'upload.json').read_text())
        available = {p.name: p for p in (stage / 'input').iterdir() if p.is_file()}
        moves = []
        for job, data in records:
            for item in data['input_manifest']['input']:
                name = item['name']
                source = job / 'input' / name
                if source.exists():
                    if name in available or not source.is_file() or source.stat().st_size != item['size']:
                        return False
                    available[name] = source
                    moves.append((source, stage / 'input' / name))
                elif name not in available or available[name].stat().st_size != item['size']:
                    return False
        if any(name not in available for series in upload['series'] for name in series['files']):
            return False
        for source, destination in moves:
            shutil.move(source, destination)
        for job, data in records:
            for name in input_directories(data):
                if name != 'input':
                    shutil.rmtree(job / name, ignore_errors=True)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning('Could not roll back interrupted input preparation', exc_info=True)
        return False


def new_job_id():
    # Nanoseconds preserve creation order within one timestamp second on recovery.
    return time.strftime("%Y%m%d-%H%M%S-") + f"{time.time_ns():020d}-" + secrets.token_hex(4)


def queue_detail(ahead):
    if ahead == 0:
        return "Next in queue"
    return f"Queued behind {ahead} earlier {'job' if ahead == 1 else 'jobs'}"


def update_queue_details():
    """Called under queue_guard; the active job also counts as an earlier job."""
    ahead = int(queue_active)
    for job, _, _ in pending_jobs:
        try:
            if metadata(job).get("status") not in {"queued", "staging"}:
                continue
            save_metadata(job, detail=queue_detail(ahead))
            ahead += 1
        except (OSError, ValueError, HTTPException):
            # Retention may remove an old queued job; it must not stop the dispatcher.
            logger.warning("Could not update queued job %s", job.name, exc_info=True)


def drain_jobs():
    global queue_running, queue_active, queue_current
    while True:
        with queue_guard:
            if recovery_barriers:
                queue_running = False
                return
            if not pending_jobs:
                queue_running = False
                return
            job, target, args = pending_jobs.popleft()
            queue_active = True
            queue_current = (job, current_thread())
            update_queue_details()
        try:
            status = metadata(job).get("status")
            if status != "queued" and not (status == "staging" and target == process_prepared_session):
                continue
            if status == "queued" and not input_manifest_complete(job, metadata(job)):
                fail_input_preparation(job)
                continue
            save_metadata(job, detail="Preparing queued job")
            target(*args)
        except Exception as exc:
            logger.exception("Queued job failed: %s", job.name)
            try:
                save_metadata(job, status="failed", detail="Processing failed", error=str(exc))
            except (OSError, ValueError, HTTPException):
                logger.warning("Could not save failed job %s", job.name, exc_info=True)
        finally:
            with queue_guard:
                # Keep an interrupted worker's reservation until the watchdog
                # recovers or fails it. A surviving task must not be passed.
                if sys.exc_info()[0] is None:
                    queue_active = False
                    queue_current = None
                update_queue_details()


def dispatcher_can_restart():
    """Do not pass an active job whose dispatcher died, even before it is stale."""
    if recovery_barriers:
        return False
    if queue_current is None:
        return True
    job, _ = queue_current
    try:
        return (metadata(job).get("status") not in {"staging", "queued", "transferring", "processing", "retrieving"}
                or any(queued == job for queued, _, _ in pending_jobs))
    except HTTPException:
        return True


def enqueue_jobs(jobs, target=None, args=None, *, recovery=False):
    """Reserve the entire batch before starting one FIFO dispatcher."""
    global queue_running, queue_thread, queue_current, queue_active, queue_sequence
    with submission_lock, queue_guard:
        def creation_order(entry):
            try:
                return metadata(entry[0]).get("created", 0), entry[0].name
            except HTTPException:
                return 0, entry[0].name
        entries = list(pending_jobs)
        additions = [(job, target or process, args if args is not None else (job,)) for job in jobs]
        for entry in sorted(additions, key=creation_order) if recovery else additions:
            job = entry[0]
            try:
                prior_sequence = metadata(job).get('queue_sequence', 0)
                queue_sequence = max(queue_sequence, prior_sequence)
                if not recovery or not prior_sequence:
                    queue_sequence += 1
                    save_metadata(job, queue_sequence=queue_sequence)
            except HTTPException:
                pass  # A removed reservation is skipped by the dispatcher.
            if recovery:
                # Insert recovered jobs by original creation time without
                # changing the acceptance order of jobs already waiting.
                position = next((i for i, waiting in enumerate(entries)
                                 if creation_order(waiting) > creation_order(entry)), len(entries))
                entries.insert(position, entry)
            else:
                entries.append(entry)
        pending_jobs.clear()
        pending_jobs.extend(entries)
        start = ((not queue_running or queue_thread is None or not queue_thread.is_alive())
                 and dispatcher_can_restart())
        queue_running = True
        if start:
            # A dead dispatcher must not strand reservations or a recovered job.
            queue_current = None
            queue_active = False
            queue_thread = threading.Thread(target=drain_jobs, args=(), daemon=True)
        update_queue_details()
        if start:
            queue_thread.start()


@tracked_job_worker
def process(job: Path):
    # Production callers execute through enqueue_jobs; task execution is unchanged.
    jid=job.name; remote=f"{REMOTE_BASE}/{jid}"
    try:
        data = metadata(job)
        if data.get('input_roles') and not input_manifest_complete(job, data):
            fail_input_preparation(job)
            return
        tasks_to_run = ordered_tasks(data["models"])
        empty = []

        def run_completed_task(task, index, remote=None):
            """A declared empty prediction completes the task in every backend."""
            try:
                if remote is None:
                    execute_job_task(job, task, index, len(tasks_to_run))
                else:
                    execute_job_task(job, task, index, len(tasks_to_run), remote=remote)
            except Exception as exc:
                if not is_empty_prediction(exc):
                    raise
                empty.append(task)
                done = metadata(job).get("completed_models", [])
                save_metadata(job, completed_models=list(dict.fromkeys([*done, task])))
                return False
            return True

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
                    if task_uses_remote(task): run_completed_task(task, index)
                finalize_results(job, empty)
                return
            save_metadata(job, status="processing", detail="Starting segmentation on worker GPU",
                          progress=None, execution_backend="local")
            for index, task in enumerate(tasks_to_run, 1):
                if task_uses_remote(task):
                    run_completed_task(task, index)
                    continue
                before = set((job / "output").iterdir())
                try:
                    run_completed_task(task, index)
                except Exception as exc:
                    for path in set((job / "output").iterdir()) - before:
                        if path.is_file(): path.unlink(missing_ok=True)
                        elif path.is_dir(): shutil.rmtree(path, ignore_errors=True)
                    if not is_gpu_oom(exc):
                        raise
                    if GPU_RELEASE_COMMAND.strip():
                        release_worker_gpu(job, task)
                        try:
                            run_completed_task(task, index)
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
                            run_completed_task(remote_task, remote_index)
                    break
            finalize_results(job, empty)
            return
        if WORKER_MODE == "local":
            data=metadata(job)
            save_metadata(job,status="processing",detail="Starting local GPU segmentation",progress=None,postprocess_notes=[])
            for i,task in enumerate(tasks_to_run,1):
                if not run_completed_task(task, i):
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
            run_completed_task(task, i, remote=remote)
        save_metadata(job,status="retrieving",detail="Retrieving RTSTRUCT results from GPU worker",progress=None)
        recover_remote_results(job, remote)
        finalize_results(job, empty)
    except Exception as exc:
        report_job = REPORT_TASK in metadata(job).get("models", [])
        if WORKER_MODE == "ssh" and report_job:
            try: recover_remote_results(job, remote)
            except Exception: pass
        elif ((WORKER_MODE == "ssh" or (WORKER_MODE == "colab" and COLAB_ORCHESTRATOR == "worker"))
              and not metadata(job).get("remote_results_retained")):
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
    elif is_gtrc_task(name) and check is not None and check.get("acceptance_ready") is False:
        item.update(enabled=False, status="acceptance_validation_pending")
    elif is_gtrc_task(name) and check is not None and check.get("cuda_ready") is False:
        item.update(enabled=False, status="gpu_runtime_unavailable")
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
    code="from totalsegmentator.map_to_binary import class_map,commercial_models; import json, importlib.util; C=__import__('totalsegmentator.map_tasks_config',fromlist=['TASK_CONFIGS']).TASK_CONFIGS if importlib.util.find_spec('totalsegmentator.map_tasks_config') else None; print(json.dumps({'tasks':sorted(t for t in class_map if C is None or t in C),'licensed':list(commercial_models)}))"
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
    catalog=[{"name":n,"licensed":n in licensed,"modality":task_modality(n),
              "enabled":not (WORKER_MODE == "colab" and n in licensed),
              "status":"license_not_configured_in_colab" if WORKER_MODE == "colab" and n in licensed else "available",
              "description":f"TotalSegmentator {task_modality(n)} segmentation task producing the structures listed below.",
              "structures":"Task-specific TotalSegmentator structures",
              "reference":"https://github.com/wasserth/TotalSegmentator"} for n in names]
    for name, details in EXTERNAL_TASKS.items():
        item={"name":name,"licensed":False,"enabled":True,**details}
        apply_model_gates(item, preflight.get(name), enabled)
        if name.startswith("lnsegfm_"):
            item.update(structures="Visible head-and-neck lymph-node candidate mask; no malignancy classification or elective levels", reference="https://github.com/HiLab-git/LN-Seg-FM")
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
        nodal = NODAL_MODEL_DETAILS.get(item["name"])
        if nodal:
            # Descriptions never grant availability or replace a gate's blocker.
            item.update({key: value for key, value in nodal.items() if key in NODAL_DESCRIPTIVE_FIELDS})
            item["license"] = f"Code: {nodal['code_license']}; checkpoints: {nodal['weights_license']}"
            item.pop("license_note", None)
            if item.get("enabled") and item.get("status", "").startswith("validated"):
                item["status"] = ("technical_review_required" if item["name"] == "hnlnl_2d3d"
                                  else "technical_checks_recorded")
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
        contract = STRUCTURE_CATALOG.get(item["name"], {
            "kind": "unresolved", "names": [], "count": None,
            "source": "Worker catalog", "note": "This worker task has no generated exporter label contract; rebuild the structure catalog."
        })
        item.update(structure_kind=contract["kind"], structure_names=contract["names"],
                    structure_count=contract["count"], structure_note=contract["note"],
                    structure_source=contract["source"])
        if contract["kind"] == "unresolved" and item.get("enabled"):
            item.update(enabled=False, status="structure_metadata_unavailable")
        item["clinical_tags"] = clinical_tags(item)
    return {"tasks":catalog, "clinical_tags":[{"id": key, "label": value["label"]}
                                                for key, value in CLINICAL_TAGS.items()]}

@app.post("/api/tools/prostate-sbrt-crop", status_code=202)
@app.post("/api/v1/tools/prostate-sbrt-crop", status_code=202, tags=["automation"])
async def create_prostate_sbrt_crop(files: list[UploadFile]=File(...), margin_mm: float=Form(150.0), keep_uids: bool=Form(False)):
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise HTTPException(400, "margin_mm must be finite and non-negative")
    jid=new_job_id(); job=BASE/jid; inp=job/"input"; inp.mkdir(parents=True)
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
            "input_manifest": job_input_manifest(job),
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
        public = public_assessment(report)
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


def process_prepared_session(job: Path, ready: Event):
    ready.wait()
    data = metadata(job)
    if data.get("status") == "queued":
        if input_manifest_complete(job, data):
            process(job)
        else:
            fail_input_preparation(job)


def finish_session_job(session: Path, data: dict, job: Path, crop: bool, ready: Event | None = None):
    try:
        stage_session_files(session, data, job, crop)
        if crop:
            save_metadata(job, status='queued', input_manifest=job_input_manifest(job))
            process_prostate_sbrt_crop(job)
        else:
            slices, modality = validate_series(job / "input")
            selected = inspect_upload(job / "input")["series"][0]
            validate_job_series(metadata(job)["models"], selected)
            with queue_guard:
                save_metadata(job, status='queued', series=series_metadata(selected),
                              input_manifest=job_input_manifest(job))
            if ready is None:
                save_metadata(job, detail=f"{slices} {modality} DICOM slices received")
                process(job)
    except Exception as exc:
        with queue_guard:
            save_metadata(job, status="failed", detail="Cropping failed" if crop else "Upload staging failed", error=str(exc))
    finally:
        if ready is not None:
            with queue_guard:
                ready.set()
                update_queue_details()


def queue_session_job(session_id: str, options: dict, crop: bool):
    with submission_lock, upload_session(session_id) as (session, data):
        operation = "crop" if crop else "jobs"
        if data["status"] == "used" and data.get("operation") == operation and data.get("options") == options:
            return {"id":data["job_id"]}
        complete_session_files(session, data)
        jid = new_job_id(); job = BASE / jid
        (job / "input").mkdir(parents=True); (job / "output").mkdir()
        # input_manifest stays null until staging completes, which marks this as a manifest-era job:
        # a failed or interrupted staging can never pass the pre-manifest fallback.
        (job / "job.json").write_text(json.dumps({"id":jid, "status":"staging", "created":time.time(),
            "input_manifest": None,
            "detail":"Prostate SBRT crop queued on CPU" if crop else "Staging uploaded files", "error":None,
            **({} if crop else single_series_batch(None)), **options}, indent=2))
        data.update(status="used", operation=operation, options=options, job_id=jid); save_session(session, data)
        if crop:
            threading.Thread(target=finish_session_job, args=(session, data, job, crop), daemon=True).start()
        else:
            # Stage immediately, retaining inputs for restart recovery even while GPU jobs wait.
            # Hold the reservation lock across staging-thread startup and FIFO registration.
            with queue_guard:
                ready = Event()
                threading.Thread(target=finish_session_job, args=(session, data, job, crop, ready), daemon=True).start()
                enqueue_jobs([job], target=process_prepared_session, args=(job, ready))
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
    return await run_in_threadpool(queue_session_job, session_id,
        {"models":ordered_tasks(models),"report_status":"pending" if REPORT_TASK in models else None, "postprocess":postprocessing}, False)


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
        public = public_assessment(report)
        return {"upload_id":uid,**public}
    except Exception as exc:
        shutil.rmtree(stage,ignore_errors=True); raise HTTPException(400,str(exc))

@app.get("/api/uploads/{uid}/preview")
def preview_slice(uid: str, series_key: str, index: int = 0):
    """Use the same normal-based order and pixel grid as the viewer and prompt runners."""
    return viewer.preview(uid, series_key, index)

def model_details(name: str) -> dict:
    return EXTERNAL_TASKS.get(name) or INFORMATIONAL_MODELS.get(name, {})


def validate_job_selection(models, catalog, *, assessed=False, batch=False):
    """Validate every creation route against the same catalog and input contract."""
    items = {item["name"]: {**model_details(item["name"]), **item} for item in catalog["tasks"]}
    if not models:
        raise HTTPException(400, "Select at least one model")
    invalid = [name for name in models if not re.fullmatch(r"[A-Za-z0-9_]+", name)
               or name not in items or not items[name].get("enabled", True)]
    if invalid:
        raise HTTPException(400, "Unavailable or invalid models: " + ", ".join(invalid))
    if not batch and "total" in models and "total_mr" in models:
        raise HTTPException(400, "Select either total (CT) or total_mr (MR), not both")
    if not assessed:
        if any(items[name].get("prompt") for name in models):
            raise HTTPException(400, "A valid prompt is required; use the assessed-upload start route with prompt_json")
        if any(items[name].get("requires_ct") for name in models):
            raise HTTPException(400, "A paired CT series is required; use the assessed-upload start route")
        if any(items[name].get("input_roles") for name in models):
            raise HTTPException(400, "Sequence roles are required; use the assessed-upload batch route")
    return items


def validate_job_series(models, selected, items=None):
    require_series_assessment(selected)
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
            if name == "sam_med3d_turbo_prompted" and negatives and box and not positives:
                raise HTTPException(400, "SAM-Med3D ignores the box when points are present. Add a positive point, "
                                    "or remove all points to use only the box centre; negative points alone are insufficient.")
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


def resolve_gtrc_job_options(
    models: list[str],
    task_options: str | dict | None = "",
    *,
    suv_threshold: str = "",
    fdg_suv_threshold: str = "",
    psma_suv_threshold: str = "",
    ct_series_key: str = "",
) -> tuple[dict[str, dict[str, object]], str | None, dict[str, str]]:
    """Parse and validate per-job GTRC SUVbw threshold options and optional companion CT choices."""
    raw_opts: dict[str, object] = {}
    if isinstance(task_options, str) and task_options.strip():
        try:
            parsed = json.loads(task_options)
        except ValueError as exc:
            raise HTTPException(400, "task_options must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(400, "task_options must be a JSON object")
        raw_opts = dict(parsed)
    elif isinstance(task_options, dict):
        raw_opts = dict(task_options)
    elif task_options not in ("", None):
        raise HTTPException(400, "task_options must be a JSON object")

    global_ct_choice: str | None = str(ct_series_key).strip() if ct_series_key and str(ct_series_key).strip() else None
    per_pet_ct_choice: dict[str, str] = {}
    if "ct_series_key" in raw_opts and raw_opts["ct_series_key"] is not None:
        if not isinstance(raw_opts["ct_series_key"], str):
            raise HTTPException(400, "ct_series_key in task_options must be a string")
        if raw_opts["ct_series_key"].strip():
            global_ct_choice = raw_opts["ct_series_key"].strip()
    if "ct_series_by_pet" in raw_opts and raw_opts["ct_series_by_pet"] is not None:
        if not isinstance(raw_opts["ct_series_by_pet"], dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in raw_opts["ct_series_by_pet"].items()
        ):
            raise HTTPException(400, "ct_series_by_pet in task_options must map PET series keys to CT series keys")
        per_pet_ct_choice.update({k: v.strip() for k, v in raw_opts["ct_series_by_pet"].items() if v.strip()})

    raw_thresholds: dict[str, object] = {}
    raw_sources: dict[str, str | None] = {}
    for key, val in raw_opts.items():
        if key in {"ct_series_key", "ct_series_by_pet", "suv_threshold"}:
            continue
        if not is_gtrc_task(key):
            raise HTTPException(400, f"Unsupported task_options key: {key}")
        if isinstance(val, dict):
            if "suv_threshold" in val and val["suv_threshold"] is not None and val["suv_threshold"] != "":
                raw_thresholds[key] = val["suv_threshold"]
            if "source" in val and val["source"] is not None:
                if not isinstance(val["source"], str) or not val["source"].strip():
                    raise HTTPException(400, f"source for {key} must be a non-empty string")
                raw_sources[key] = val["source"].strip()
            if "ct_series_key" in val and val["ct_series_key"] is not None:
                if not isinstance(val["ct_series_key"], str):
                    raise HTTPException(400, f"ct_series_key for {key} must be a string")
                if val["ct_series_key"].strip() and not global_ct_choice:
                    global_ct_choice = val["ct_series_key"].strip()
        elif val is not None and val != "":
            raw_thresholds[key] = val
            raw_sources[key] = "user-entered"

    top_suv = raw_opts.get("suv_threshold")
    if suv_threshold and str(suv_threshold).strip():
        top_suv = str(suv_threshold).strip()
    if fdg_suv_threshold and str(fdg_suv_threshold).strip():
        raw_thresholds["gtrc_fdg_petct_ttb"] = str(fdg_suv_threshold).strip()
        raw_sources["gtrc_fdg_petct_ttb"] = "user-entered"
    if psma_suv_threshold and str(psma_suv_threshold).strip():
        raw_thresholds["gtrc_psma_petct_ttb"] = str(psma_suv_threshold).strip()
        raw_sources["gtrc_psma_petct_ttb"] = "user-entered"

    resolved: dict[str, dict[str, object]] = {}
    for name in models:
        if not is_gtrc_task(name):
            continue
        explicit_val = raw_thresholds.get(name, top_suv if top_suv not in (None, "") else None)
        explicit_src = raw_sources.get(name)
        if explicit_src is None and explicit_val is not None:
            explicit_src = "user-entered"
        try:
            val, src = resolve_job_suv_threshold(
                name,
                explicit_threshold=explicit_val,
                explicit_source=explicit_src,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        resolved[name] = {"suv_threshold": val, "source": src}

    return resolved, global_ct_choice, per_pet_ct_choice


def create_assessed_jobs(uid, series_keys, models, prompt_json="", postprocess="", *, legacy=False, roles=None,
                         task_options="", suv_threshold="", fdg_suv_threshold="", psma_suv_threshold="",
                         ct_series_key=""):
    # Serialize consumption and submission so requests cannot consume the same upload twice.
    with submission_lock:
        if not re.fullmatch(r"upload-[A-Za-z0-9-]+", uid):
            raise HTTPException(404, "Upload not found or expired")
        stage = BASE / uid
        try:
            data = json.loads((stage / "upload.json").read_text())
        except (OSError, ValueError):
            raise HTTPException(404, "Upload not found or expired")
        if not series_keys:
            raise HTTPException(400, "Select at least one series")
        if len(set(series_keys)) != len(series_keys):
            duplicates = sorted({key for key in series_keys if series_keys.count(key) > 1})
            raise HTTPException(400, "Duplicate series keys: " + ", ".join(duplicates))
        by_key = {series["key"]: series for series in data["series"]}
        unknown = [key for key in series_keys if key not in by_key]
        if unknown:
            raise HTTPException(400, "Unknown series keys: " + ", ".join(unknown))
        selected = [by_key[key] for key in series_keys]
        for series in selected:
            try:
                require_series_assessment(series)
            except ValueError as exc:
                raise HTTPException(400, f"Series {series['key']}: {exc}") from exc
        items = validate_job_selection(models, tasks(), assessed=True, batch=not legacy)
        models = list(dict.fromkeys(models))
        gtrc_options, global_ct_choice, per_pet_ct_choice = resolve_gtrc_job_options(
            models,
            task_options,
            suv_threshold=suv_threshold,
            fdg_suv_threshold=fdg_suv_threshold,
            psma_suv_threshold=psma_suv_threshold,
            ct_series_key=ct_series_key,
        )
        prompted = [name for name in models if items[name].get("prompt")]
        if prompted and len(selected) != 1:
            raise HTTPException(400, "Select a single series to use interactive models: " + ", ".join(prompted))
        if isinstance(roles, str):
            try:
                roles = json.loads(roles) if roles else {}
            except ValueError:
                raise HTTPException(400, "roles must be a JSON object")
        if roles is None:
            roles = {}
        if not isinstance(roles, dict) or set(roles) - set(series_keys):
            raise HTTPException(400, "roles must map selected primary series keys to role selections")
        roles = {k: dict(v) if isinstance(v, dict) else v for k, v in roles.items()}
        for series in selected:
            choices = roles.get(series['key'], {})
            if not isinstance(choices, dict):
                raise HTTPException(400, "Each primary series role selection must be an object")
            if "ct" in choices and any(items[name].get("requires_ct") for name in models):
                ct_val = choices.pop("ct")
                if not isinstance(ct_val, str) or not ct_val.strip():
                    raise HTTPException(400, f"Series {series['key']}: companion CT role must be a non-empty series key")
                per_pet_ct_choice[series['key']] = ct_val.strip()
            allowed = {}
            try:
                for name in models:
                    contract, reference = role_contract(items[name])
                    for role in contract:
                        if role['role'] != reference:
                            allowed[role['role']] = (role, contract, reference)
                for role_name, choice in choices.items():
                    if role_name not in allowed:
                        raise ValueError(f"Role {role_name} is not a companion role of a selected model")
                    role, contract, reference = allowed[role_name]
                    # Validate every explicit choice even if another required role is absent.
                    ref = next(r for r in contract if r['role'] == reference)
                    resolve_roles(series, data['series'], {'input_roles': [ref, role], 'reference_role': reference},
                                  {role_name: choice})
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, f"Series {series['key']}: {exc}") from exc
        plans, applied, role_blocked = [], set(), set()
        for series in selected:
            label = f"{series.get('description') or series['key']} ({series['modality']}; {series['key']})"
            subset = models if legacy else [name for name in models if series["modality"] in
                (items[name].get("modalities") or [items[name].get("modality", task_modality(name))])
                and not (name == REPORT_TASK and (series["modality"] != "CT" or series.get("cbct")))]
            role_records, role_sources, skipped = {}, {}, {}
            for name in list(subset):
                try:
                    records, sources, error = resolve_roles(series, data['series'], items[name], roles.get(series['key'], {}))
                    if error:
                        skipped[name] = error
                        subset = [model for model in subset if model != name]
                        role_blocked.add(name)
                        continue
                    for role, record in records.items():
                        if role in role_records and role_records[role] != record:
                            raise ValueError(f"Selected models have conflicting input role {role}")
                        if record['dir'] == 'input' and any(r['dir'] == 'input' and key != role for key, r in role_records.items()):
                            raise ValueError("Selected models have different reference roles")
                        role_records[role] = record
                    role_sources.update(sources)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(400, f"Series {label}: {exc}") from exc
            if not subset and skipped:
                raise HTTPException(400, f"Series {label}: " + '; '.join(f"{name}: {error}" for name, error in skipped.items()))
            if not subset:
                raise HTTPException(400, f"Series {label} has no compatible models to run from {', '.join(models)} (AI reports exclude CBCT)")
            try:
                validate_job_series(subset, series, items)
            except ValueError as exc:
                raise HTTPException(400, f"Series {label}: {exc}") from exc
            try:
                prompt_data = validate_assessed_prompt(subset, items, series, prompt_json)
            except HTTPException as exc:
                raise HTTPException(400, f"Series {label}: {exc.detail}") from exc
            companion = None
            if any(items[name].get("requires_ct") for name in subset):
                candidates = [
                    candidate for candidate in data["series"]
                    if candidate["modality"] == "CT"
                    and series.get("study_uid") and series.get("frame_uid")
                    and candidate.get("study_uid") == series["study_uid"]
                    and candidate.get("frame_uid") == series["frame_uid"]
                ]
                explicit_ct = per_pet_ct_choice.get(series["key"]) or global_ct_choice
                if explicit_ct:
                    companion = next(
                        (c for c in candidates if c["key"] == explicit_ct or c.get("series_uid") == explicit_ct),
                        None,
                    )
                    if companion is None:
                        raise HTTPException(
                            400,
                            f"Series {label}: selected companion CT {explicit_ct!r} is not a CT series in the same Study and Frame of Reference",
                        )
                elif len(candidates) > 1 and any(is_gtrc_task(name) for name in subset):
                    gtrc_names = ", ".join(name for name in subset if is_gtrc_task(name))
                    raise HTTPException(
                        400,
                        f"Series {label}: {len(candidates)} CT series match StudyInstanceUID and FrameOfReferenceUID for {gtrc_names}; "
                        "select the matching CT series or upload only the matching CT series",
                    )
                else:
                    companion = candidates[0] if candidates else None
                if companion is None:
                    paired = ", ".join(name for name in subset if items[name].get("requires_ct"))
                    raise HTTPException(400, f"Series {label}: {paired} requires a CT series in the same Study and Frame of Reference")
            try:
                postprocessing = validate_postprocess(postprocess, subset)
            except HTTPException:
                postprocessing = []
            if companion and 'input_ct' in role_sources:
                raise HTTPException(400, "A CT companion conflicts with a declared input role")
            plans.append((series, ordered_tasks(subset), companion, prompt_data, postprocessing,
                          role_records, role_sources, skipped))
            applied.update(subset)
        unused = [name for name in models if name not in applied and name not in role_blocked]
        if unused:
            raise HTTPException(400, "Models apply to no selected series: " + ", ".join(unused))
        if postprocess and not any(plan[4] for plan in plans):
            # Preserve the precise postprocess contract error.
            validate_postprocess(postprocess, models)
        # Check every source before allocating any job directories.
        fresh = {s["key"]: s for s in inspect_upload(stage / "input")["series"]}
        for series, _, companion, _, _, records, role_sources, _ in plans:
            selected_sources = {s["key"]: s for s in [series] + ([companion] if companion else [])}
            selected_sources.update({r["series_key"]: by_key[r["series_key"]] for r in records.values()})
            for source in selected_sources.values():
                try:
                    require_series_assessment(source)
                    current = fresh.get(source["key"])
                    if current is None:
                        raise ValueError("Series assessment failed; selected images are no longer readable")
                    require_series_assessment(current)
                    if (set(current["files"]) != set(source["files"]) or
                            current.get("instances") != source.get("instances")):
                        raise ValueError("Series assessment changed; upload and review the complete series again")
                except ValueError as exc:
                    raise HTTPException(400, f"Series {source['key']}: {exc}") from exc
            for source in [series] + ([companion] if companion else []):
                if any(not (stage / "input" / name).is_file() for name in source["files"]):
                    raise HTTPException(400, f"Series {source['key']}: staged files are missing; upload again")
            for names in role_sources.values():
                if any(not (stage / 'input' / name).is_file() for name in names):
                    raise HTTPException(400, "Companion files are missing; upload again")
        batch_id = new_batch_id()
        created, moved, results = [], [], []
        try:
            # Persist non-runnable metadata before copying or moving any input.
            for index, (series, subset, companion, prompt_data, postprocessing, role_records, role_sources, skipped) in enumerate(plans, 1):
                jid = new_job_id()
                job = BASE / jid
                created.append(job)
                (job / "input").mkdir(parents=True)
                (job / "output").mkdir()
                if prompt_data:
                    (job / "prompt.json").write_text(json.dumps(prompt_data, indent=2))
                public_series = series_metadata(series)
                manifest = {'input': directory_manifest(stage / 'input', series['files'])}
                if companion:
                    manifest['input_ct'] = directory_manifest(stage / 'input', companion['files'])
                for directory, names in role_sources.items():
                    manifest[directory] = directory_manifest(stage / 'input', names)
                role_data = {'input_roles': role_records} if role_records else {}
                if role_records:
                    role_data['input_identity'] = input_identity({**role_data, 'input_manifest': manifest})
                subset_task_options = {name: gtrc_options[name] for name in subset if name in gtrc_options}
                (job / "job.json").write_text(json.dumps({"id": jid, "status": "staging", "detail": "Preparing input",
                    **role_data, **({'skipped_models': skipped} if skipped else {}),
                    **({'task_options': subset_task_options} if subset_task_options else {}),
                    **({'paired_series_key': companion['key']} if companion else {}),
                    "input_manifest": manifest, "staging_upload": uid,
                    "models": subset, "report_status": "pending" if REPORT_TASK in subset else None,
                    "created": time.time(), "error": None, "prompted": bool(prompt_data), "postprocess": postprocessing,
                    "batch_id": batch_id, "batch_index": index, "batch_size": len(plans), "series": public_series}, indent=2))
                results.append({"id": jid, "series_key": series["key"], "models": subset, **({"skipped_models": skipped} if skipped else {})})
            # Copies precede every move: selected CT can also accompany several PET jobs.
            for job, (_, _, companion, _, _, _, role_sources, _) in zip(created, plans):
                if companion:
                    (job / "input_ct").mkdir()
                    for name in companion['files']:
                        shutil.copy2(stage / 'input' / name, job / 'input_ct' / name)
                for directory, names in role_sources.items():
                    if directory == 'input':
                        continue
                    (job / directory).mkdir()
                    for name in names:
                        shutil.copy2(stage / 'input' / name, job / directory / name)
            for job, (series, _, _, _, _, _, _, _) in zip(created, plans):
                for name in series["files"]:
                    source, destination = stage / "input" / name, job / "input" / name
                    shutil.move(source, destination)
                    moved.append((source, destination))
            if not all(input_manifest_complete(job, metadata(job)) for job in created):
                raise ValueError('Prepared input does not match its manifest')
            for job in created:
                validate_staged_mri(job)
            for job in created:
                save_metadata(job, status='queued', detail='Next in queue')
        except Exception as exc:
            for source, destination in reversed(moved):
                shutil.move(destination, source)
            for job in created:
                shutil.rmtree(job, ignore_errors=True)
            if isinstance(exc, ValueError):
                raise HTTPException(400, str(exc)) from exc
            raise
        shutil.rmtree(stage, ignore_errors=True)
        enqueue_jobs(created)
        return {"batch_id": batch_id, "ids": [result["id"] for result in results], "jobs": results}


@app.post("/api/v1/uploads/{uid}/jobs", status_code=202, tags=["automation"])
def start_assessed_batch(uid: str, series_keys: list[str]=Form(...), models: list[str]=Form(...),
                         prompt_json: str=Form(""), postprocess: str=Form(""), roles: str=Form(""),
                         task_options: str=Form(""), suv_threshold: str=Form(""),
                         fdg_suv_threshold: str=Form(""), psma_suv_threshold: str=Form(""),
                         ct_series_key: str=Form("")):
    return create_assessed_jobs(
        uid, series_keys, models, prompt_json, postprocess, roles=roles,
        task_options=task_options, suv_threshold=suv_threshold,
        fdg_suv_threshold=fdg_suv_threshold, psma_suv_threshold=psma_suv_threshold,
        ct_series_key=ct_series_key,
    )


@app.post("/api/uploads/{uid}/start")
async def start_assessed(uid: str, series_key: str=Form(...), models: list[str]=Form(...),
                         prompt_json: str=Form(""), postprocess: str=Form(""),
                         task_options: str=Form(""), suv_threshold: str=Form(""),
                         fdg_suv_threshold: str=Form(""), psma_suv_threshold: str=Form(""),
                         ct_series_key: str=Form("")):
    result = await run_in_threadpool(
        create_assessed_jobs, uid, [series_key], models, prompt_json, postprocess,
        legacy=True, task_options=task_options, suv_threshold=suv_threshold,
        fdg_suv_threshold=fdg_suv_threshold, psma_suv_threshold=psma_suv_threshold,
        ct_series_key=ct_series_key,
    )
    return {"id": result["ids"][0]}

def queue_ready_job(job: Path, data: dict):
    """Publish a fully staged direct job and reserve its position with batch creation."""
    with submission_lock:
        (job / "job.json").write_text(json.dumps({**data, "created": time.time(),
                                                "input_manifest": job_input_manifest(job)}, indent=2))
        enqueue_jobs([job])


@app.post("/api/jobs")
@app.post("/api/v1/jobs", status_code=202, tags=["automation"])
async def create_job(files: list[UploadFile]=File(...), models: list[str]=Form(...), postprocess: str=Form("")):
    postprocessing = validate_postprocess(postprocess, models)
    items = validate_job_selection(models, tasks())
    jid=new_job_id(); job=BASE/jid; inp=job/"input"; out=job/"output"; inp.mkdir(parents=True); out.mkdir()
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
        await run_in_threadpool(queue_ready_job, job,
            {"id":jid,"status":"queued","detail":f"{slices} {modality} DICOM slices received",
             "models":ordered_tasks(models),"report_status":"pending" if REPORT_TASK in models else None,
             "error":None,"postprocess":postprocessing, **single_series_batch(selected)})
    except Exception as exc:
        shutil.rmtree(job,ignore_errors=True); raise HTTPException(400,str(exc))
    return {"id":jid}

@app.get("/api/jobs/{jid}")
@app.get("/api/v1/jobs/{jid}", tags=["automation"])
def get_job(jid: str):
    if not re.fullmatch(r"[A-Za-z0-9-]+",jid): raise HTTPException(404)
    return metadata(BASE/jid)

@app.get("/api/v1/jobs/{jid}/log", tags=["automation"])
def get_job_log(jid: str, task: str | None = None, lines: int = 40):
    if not re.fullmatch(r"[A-Za-z0-9-]+", jid): raise HTTPException(404)
    job = BASE / jid
    data = metadata(job)
    task = task or data.get('current_task')
    if (not isinstance(task, str) or not re.fullmatch(r"[A-Za-z0-9_]+", task)
            or task not in data.get('models', [])):
        raise HTTPException(404, "Unknown job task")
    path = job / f"{task}.log"
    if path.is_symlink(): raise HTTPException(404, "Task log unavailable")
    try:
        content = tail_log(path, lines)
    except FileNotFoundError:
        content = []
    return JSONResponse({'task': task, 'lines': content}, headers={'Cache-Control': 'no-store'})


@app.get("/api/jobs")
@app.get("/api/v1/jobs", tags=["automation"])
def list_jobs(batch_id: str | None = None):
    found=[]
    for path in BASE.iterdir():
        if not path.is_dir() or path.name.startswith(("upload-", ".")): continue
        try:
            data=metadata(path)
            if batch_id is not None and data.get("batch_id") != batch_id:
                continue
            found.append({"id":data["id"],"status":data["status"],"created":data.get("created",0),
                          "report_status":data.get("report_status"), "report_reason":data.get("report_reason", ""),
                          **{key: data.get(key) for key in ("batch_id", "batch_index", "batch_size", "series")}})
        except Exception: pass
    groups = {}
    for data in found:
        groups.setdefault(data.get("batch_id") or data["id"], []).append(data)
    ordered = sorted(groups.values(), key=lambda group: max(data["created"] for data in group), reverse=True)
    recent = []
    for group in ordered:
        if len(recent) >= 100:
            break
        recent.extend(sorted(group, key=lambda data: data.get("batch_index") or 0))
    return {"jobs": recent}

@app.post("/api/jobs/{jid}/retry", status_code=202)
@app.post("/api/v1/jobs/{jid}/retry", status_code=202, tags=["automation"])
@viewer.serialize_job_mutation
def retry_job(jid: str):
    with submission_lock:
        if not re.fullmatch(r"[A-Za-z0-9-]+", jid): raise HTTPException(404)
        job=BASE/jid; data=metadata(job)
        if data.get("status") != "failed": raise HTTPException(409,"Only failed jobs can be retried")
        if not input_manifest_complete(job, data):
            raise HTTPException(409,"The retained input series is unavailable or incomplete; upload it again")
        reset_job_output(job, data)
        save_metadata(job,status="queued",detail="Retry queued",error=None,progress=0,model_index=0,
                      recovery_count=int(data.get("recovery_count",0)), fallback_reason=None,
                      execution_backend=None, gpu_release_attempted=False, postprocess_notes=[])
        target = process_prostate_sbrt_crop if data.get("tool") else process
        if data.get("tool"):
            threading.Thread(target=target,args=(job,),daemon=True).start()
        else:
            enqueue_jobs([job])
        return {"id":jid,"status":"queued"}


@app.get("/api/jobs/{jid}/download")
@app.get("/api/v1/jobs/{jid}/result", tags=["automation"])
def download(jid: str):
    if not re.fullmatch(r"[A-Za-z0-9-]+", jid): raise HTTPException(404)
    job=BASE/jid; data=metadata(job); result=job/"rtstruct-results.zip"
    if data["status"]!="complete" or not result.exists(): raise HTTPException(409,"Result is not ready")
    filename = f"{jid}-prostate-SBRT-crop.zip" if data.get("tool") else f"{jid}-RTSTRUCT.zip"
    return FileResponse(result,filename=filename,media_type="application/zip")

@app.get("/api/v1/jobs/{jid}/manifest", tags=["automation"])
def get_result_manifest(jid: str):
    """The result manifest packaged in the job's ZIP (segmentation jobs only)."""
    if not re.fullmatch(r"[A-Za-z0-9-]+", jid): raise HTTPException(404)
    job = BASE / jid; data = metadata(job); result = job / "rtstruct-results.zip"
    if data["status"] != "complete" or not result.exists(): raise HTTPException(409, "Result is not ready")
    try:
        with zipfile.ZipFile(result) as archive:
            content = archive.read(result_manifest.MANIFEST_NAME)
    except KeyError:
        raise HTTPException(404, "This result package has no manifest")
    return Response(content, media_type="application/json")


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

REMOTE_CLEANUP_QUEUE = ".remote-cleanup-pending.json"


def retry_remote_cleanup(additions=()):
    """Remove worker copies of expired jobs; keep failures queued for the next hourly pass."""
    queue = BASE / REMOTE_CLEANUP_QUEUE
    try: pending = json.loads(queue.read_text())
    except (OSError, ValueError): pending = []
    pending = list(dict.fromkeys([p for p in pending if isinstance(p, str)] + list(additions)))
    remaining = []
    for remote in pending:
        try: subprocess.run(remote_command("rm", "-rf", "--", remote), check=True, timeout=REMOTE_CLEANUP_TIMEOUT_SECONDS)
        except (subprocess.SubprocessError, OSError):
            logger.warning("Worker copy %s of an expired job is still present; retrying next hour", remote)
            remaining.append(remote)
    if remaining: queue.write_text(json.dumps(remaining))
    else: queue.unlink(missing_ok=True)


# Statuses that dispatch, recovery and the watchdog act on. Retention removes only other (finished) jobs, so it
# never races those paths; a stuck active job is first recovered or failed by the watchdog.
ACTIVE_JOB_STATUSES = {'staging', 'queued', 'transferring', 'processing', 'retrieving'}
EXPIRED_DIR = '.expired'


def cleanup_once():
    cutoff=time.time()-RETENTION_HOURS*3600
    expired_remote = []
    expired = BASE / EXPIRED_DIR
    if expired.is_dir():  # leftovers from an interrupted earlier pass
        for leftover in expired.iterdir(): shutil.rmtree(leftover, ignore_errors=True)
    for job in BASE.iterdir():
        fd = None
        try:
            if not job.is_dir() or job.is_symlink() or job.name.startswith("."): continue
            is_session = job.name.startswith("session-")
            if is_session:
                fd = os.open(job, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta=job/("session.json" if is_session else "upload.json" if job.name.startswith("upload-") else "job.json")
            try: data = json.loads(meta.read_text())
            except Exception: data = {}
            try: created = float(data["created"])
            except Exception: created = job.stat().st_mtime
            if created < cutoff:
                if not is_session and not job.name.startswith("upload-"):
                    # Decide and detach under the locks used by recovery, dispatch and retry; delete outside them.
                    with submission_lock, queue_guard:
                        if job in recovery_barriers: continue
                        try: data = json.loads(meta.read_text())
                        except Exception: data = {}
                        if data.get("status") in ACTIVE_JOB_STATUSES: continue
                        expired.mkdir(exist_ok=True)
                        target = expired / job.name
                        job.rename(target)
                    job = target
                if data.get("remote_cleanup_pending") or data.get("remote_results_retained"):
                    expired_remote.append(f"{REMOTE_BASE}/{job.name}")
                shutil.rmtree(job)
        except Exception: pass
        finally:
            if fd is not None: os.close(fd)
    if expired_remote or (BASE / REMOTE_CLEANUP_QUEUE).exists():
        retry_remote_cleanup(expired_remote)


def cleanup_loop():
    while True:
        cleanup_once()
        time.sleep(3600)

def recover_interrupted_jobs(candidates=None, *, orphaned=False):
    """Requeue interrupted jobs when their retained input permits a safe rerun."""
    global queue_sequence
    # Keep later reservations from dispatching while older jobs are reset.
    with submission_lock, queue_guard:
        interrupted = []
        for job in (BASE.iterdir() if candidates is None else candidates):
            if not job.is_dir() or job.name.startswith(("upload-", ".")):
                continue
            try:
                data = metadata(job)
                queue_sequence = max(queue_sequence, data.get('queue_sequence', 0))
                if not job_is_live(job, data) and data.get("status") in {"staging", "queued", "transferring", "processing", "retrieving"}:
                    interrupted.append((job, data.get("status", "unknown")))
            except Exception:
                continue
        interrupted.sort(key=lambda entry: (metadata(entry[0]).get("created", 0), entry[0].name))
        # A surviving PID is not a monitor. Reconcile every orphan before any
        # recovered job can start, including jobs whose recovery limit is spent.
        for job, _ in interrupted:
            recovery_barriers.add(job)
            stop_recorded_task_tree(metadata(job))
            save_metadata(job, task_pid=None, task_pid_started=None)
            recovery_barriers.discard(job)
        handled = set()
        for job, previous_status in interrupted:
            if previous_status != 'staging' or job in handled:
                continue
            data = metadata(job)
            group = [job]
            if data.get('batch_id') and data.get('staging_upload'):
                group = []
                for sibling in BASE.iterdir():
                    try:
                        record = metadata(sibling)
                        if (record.get('batch_id') == data['batch_id'] and
                                record.get('status') in {'staging', 'queued'} and not job_is_live(sibling, record)):
                            group.append(sibling)
                    except HTTPException:
                        continue
            restored = rollback_staged_batch(group)
            for sibling in group:
                fail_input_preparation(sibling)
                save_metadata(sibling, input_rolled_back=restored)
            handled.update(group)
        retry = []
        for job, previous_status in interrupted:
            if job in handled:
                continue
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
            has_input = input_manifest_complete(job, data)
            if not has_input:
                fail_input_preparation(job)
                save_metadata(job, recovery_count=recovery_count, interrupted_status=previous_status)
                continue
            if has_input and recovery_count <= 3:
                completed = data.get("completed_models", [])
                if not completed or (REPORT_TASK in data.get("models", []) and REPORT_TASK not in completed):
                    reset_job_output(job, data)
                elif data.get("current_task") in data.get("models", []) and data["current_task"] not in completed:
                    reset_task_output(job, data["current_task"])
                reason = "worker stopped unexpectedly" if orphaned else "service interruption"
                save_metadata(job,status="queued",detail=f"Restarting after {reason} ({recovery_count}/3)",
                              error=None,progress=0,model_index=0,recovery_count=recovery_count,
                              interrupted_status=previous_status)
                retry.append(job)
            else:
                detail = "Job worker stopped unexpectedly" if orphaned else "Processing was interrupted by a portal service restart"
                save_metadata(job,status="failed",detail=detail,
                              error="Could not restart automatically because automatic recovery limit was reached.",
                              recovery_count=recovery_count,interrupted_status=previous_status)
        segmentation_jobs = [job for job in retry if not metadata(job).get("tool")]
        if segmentation_jobs:
            enqueue_jobs(segmentation_jobs, recovery=True)
        for job in retry:
            if metadata(job).get("tool"):
                threading.Thread(target=process_prostate_sbrt_crop, args=(job,), daemon=True).start()

def job_is_live(job, data):
    """Only a process-local reservation or live monitor owns a running job."""
    with queue_guard:
        if any(queued == job for queued, _, _ in pending_jobs):
            return True
        if queue_current and queue_current[0] == job and queue_current[1].is_alive():
            return True
        with job_workers_lock:
            worker = job_workers.get(job.name)
            if worker and worker.is_alive():
                return True
        return False


def watchdog_once(now=None):
    """Recover all stale orphans together, preserving creation order in the FIFO."""
    now = time.time() if now is None else now
    # Creation, staging reservation and recovery share this lock. A job cannot
    # become visible to this scan before its FIFO reservation is published.
    with submission_lock, queue_guard:
        orphaned = []
        for job in BASE.iterdir():
            if not job.is_dir() or job.name.startswith(('upload-', '.')): continue
            try:
                data = metadata(job)
                if data.get('status') not in {'staging', 'queued', 'transferring', 'processing', 'retrieving'}: continue
                if (job not in recovery_barriers and
                        now - data.get('updated', data.get('created', now)) < RTSEG_STALL_SECONDS): continue
                if job_is_live(job, data): continue
                events = data.get('stall_events', [])
                save_metadata(job, stall_events=[*events, {'task': data.get('current_task'),
                              'stage': data.get('stage'), 'time': now, 'action': 'orphan_recovery'}])
                orphaned.append(job)
            except Exception:
                logger.exception('Orphaned-job watchdog could not inspect %s', job.name)
        if orphaned:
            try:
                recover_interrupted_jobs(orphaned, orphaned=True)
            except Exception:
                logger.exception('Orphaned-job watchdog could not recover interrupted jobs')
        if pending_jobs and queue_thread is not None and not queue_thread.is_alive():
            enqueue_jobs([])


def watchdog_loop():
    while not watchdog_stop.wait(min(30, RTSEG_STALL_SECONDS)):
        try:
            watchdog_once()
        except OSError:
            logger.exception("Orphaned-job watchdog could not scan jobs")


@app.on_event("startup")
def startup():
    refresh_catalog_in_background()  # Warm the catalog; log and expose any initial failure.
    recover_interrupted_jobs()
    threading.Thread(target=cleanup_loop,daemon=True).start()
    watchdog_stop.clear()
    threading.Thread(target=watchdog_loop,daemon=True).start()


@app.on_event("shutdown")
def stop_watchdog():
    watchdog_stop.set()
