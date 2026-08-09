from __future__ import annotations

import json
import io
import os
import re
import secrets
import shlex
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import pydicom
import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

BASE = Path(os.environ.get("PORTAL_DATA", Path(__file__).parent / "data"))
WORKER = os.environ.get("SEGMENTATION_WORKER", "konrad@cpd-konrad-worker")
WORKER_MODE = os.environ.get("WORKER_MODE", "ssh").lower()
WORKER_SSH_PORT = os.environ.get("WORKER_SSH_PORT", "")
REMOTE_BASE = os.environ.get("REMOTE_JOB_ROOT", "/home/konrad/dicom-rt-jobs")
REMOTE_PYTHON = os.environ.get("REMOTE_PYTHON", "/home/konrad/dicom-rt-seg/.venv/bin/python")
REMOTE_RUNNER = os.environ.get("REMOTE_RUNNER", "/home/konrad/dicom-rt-seg/run_task.py")
LOCAL_PYTHON = os.environ.get("LOCAL_PYTHON", REMOTE_PYTHON)
LOCAL_RUNNER = os.environ.get("LOCAL_RUNNER", REMOTE_RUNNER)
API_KEY = os.environ.get("RTSEG_API_KEY", "")
SSH = ["ssh", *(["-p", WORKER_SSH_PORT] if WORKER_SSH_PORT else []), WORKER]
SCP = ["scp", "-q", *(["-P", WORKER_SSH_PORT] if WORKER_SSH_PORT else [])]
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024**3)))
RETENTION_HOURS = int(os.environ.get("RETENTION_HOURS", "24"))
BASE.mkdir(parents=True, exist_ok=True)

if WORKER_MODE not in {"local", "ssh"}:
    raise RuntimeError("WORKER_MODE must be 'local' or 'ssh'")

app = FastAPI(
    title="RTsegmentator API",
    version="1.0.0",
    description="DICOM series assessment, GPU segmentation and validated RTSTRUCT export.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
queue_lock = threading.Lock()


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Protect automation endpoints when RTSEG_API_KEY is configured."""
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(401, "Missing or invalid X-API-Key header")

FALLBACK_TASKS = ["total", "total_mr", "total_v3", "body", "body_mr", "vertebrae_mr", "lung_vessels", "lung_vessels_LEGACY", "cerebral_bleed", "hip_implant", "coronary_arteries", "coronary_arteries_LEGACY", "pleural_pericard_effusion", "head_glands_cavities", "head_muscles", "headneck_bones_vessels", "headneck_muscles", "liver_vessels", "liver_segments", "liver_segments_mr", "liver_lesions", "liver_lesions_mr", "oculomotor_muscles", "lung_nodules", "kidney_cysts", "breasts", "ventricle_parts", "tissue_types", "tissue_types_mr", "tissue_4_types", "brain_structures", "vertebrae_body", "face", "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr", "appendicular_bones", "appendicular_bones_mr", "aortic_sinuses", "heartchambers_highres", "trunk_cavities", "brain_aneurysm"]
EXTERNAL_TASKS = {
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

# Audited releases that must remain visible even when they cannot safely be
# submitted yet.  This prevents "not in the picker" from being mistaken for
# "not assessed" and gives users the exact upstream model and output meaning.
INFORMATIONAL_MODELS = {
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

def task_modality(name: str) -> str:
    # TotalSegmentator currently has CT and MR task families. PET-capable
    # external models are added separately after their RTSTRUCT acceptance test.
    details = EXTERNAL_TASKS.get(name) or INFORMATIONAL_MODELS.get(name, {})
    return details.get("modality", "MR" if task_is_mr(name) else "CT")

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DICOM RT Segmentation</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#172033;background:#f4f7fb}body{max-width:1050px;margin:35px auto;padding:0 20px}h1{margin-bottom:5px}.sub{color:#617087;margin-top:0}.card{background:white;border:1px solid #dce4ee;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 5px 20px #2030500d}.drop{border:2px dashed #93a8c2;border-radius:10px;padding:24px;text-align:center}.tasks{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:7px;max-height:330px;overflow:auto;padding:8px}.task{padding:7px;border-radius:6px}.task:hover{background:#eef4fb}button{background:#1769d2;color:white;border:0;border-radius:8px;padding:11px 18px;font-weight:650;cursor:pointer}button:disabled{opacity:.5}input[type=file]{margin:12px}.job{border-top:1px solid #e2e8f0;padding:12px 0}.status{font-weight:650}.error{color:#a21b1b;white-space:pre-wrap}.warn{background:#fff7db;border-left:4px solid #e3a008;padding:10px}small{color:#68768a}</style></head><body>
<h1>DICOM RT Segmentation</h1><p class="sub">GPU inference on cpd-konrad-worker · RTSTRUCT output for manual ARIA import</p>
<div class="warn"><b>Research use only.</b> TotalSegmentator is not a medical device. Review every contour before clinical use. Uploaded studies may contain PHI and are deleted automatically after 24 hours.</div>
<div class="card"><h2>1. Upload and inspect locally</h2><form id="uploadForm"><div class="drop">Select DICOM files or ZIP archives. Nothing is sent to the GPU worker during inspection.<br><input id="files" name="files" type="file" multiple accept=".dcm,.zip,application/dicom,application/zip" required></div><p><button id="inspect">Inspect upload</button></p></form><div id="assessment"></div></div>
<form id="modelForm" class="card" hidden><h2>2. Select one detected series</h2><div id="series"></div><h2>3. Select compatible models</h2><p><button type="button" id="all">Select all available for modality</button> <button type="button" id="none">Clear</button></p><div id="tasks" class="tasks">Loading model catalog…</div><div id="promptPanel" class="card" hidden><h3>Prompt</h3><p><small>Move to the target slice, choose a prompt type, then click the image. Coordinates use <code>[column,row,slice]</code>. A box requires two clicks on one slice.</small></p><div><input id="promptSlice" type="range" min="0" max="0" value="0" style="width:75%"> <span id="promptSliceLabel">slice 0</span></div><p><select id="promptMode"><option value="positive">Positive point</option><option value="negative">Negative point</option><option value="box">Box corners</option></select> <button type="button" id="clearPrompt">Clear prompt</button></p><img id="promptImage" alt="Selected DICOM slice" style="max-width:512px;width:100%;cursor:crosshair;border:1px solid #8392a8"><label>Prompt JSON<br><textarea name="prompt_json" id="promptJson" rows="6" style="width:100%"></textarea></label></div><p id="modelNote"><small><code>total</code> is selected for CT and <code>total_mr</code> for MR. PET1 ISRT tasks require a CT series in the same Study and Frame of Reference; it is paired automatically and unmatched PET is rejected before transfer. Disabled entries remain visible with their exact validation or upstream-blocker status.</small></p><button id="submit">Queue selected series</button></form><div class="card"><h2>Jobs</h2><div id="jobs">No jobs submitted in this browser.</div></div>
<script>
const taskbox=document.querySelector('#tasks'), jobsEl=document.querySelector('#jobs'); let ids=JSON.parse(localStorage.getItem('dicomJobs')||'[]'), uploadId=null,prompt={positive_points:[],negative_points:[],box:null,text:'lymph node'},boxCorner=null;
function writePrompt(){document.querySelector('#promptJson').value=JSON.stringify(prompt)}
function loadPromptSlice(){let s=document.querySelector('[name=series_key]:checked'),i=+document.querySelector('#promptSlice').value;if(!s||!uploadId)return;document.querySelector('#promptSliceLabel').textContent=`slice ${i+1}/${s.dataset.slices}`;document.querySelector('#promptImage').src=`/api/uploads/${uploadId}/preview?series_key=${encodeURIComponent(s.value)}&index=${i}&_=${Date.now()}`}
function refreshPrompt(){let selected=[...document.querySelectorAll('[name=models]:checked')],show=selected.some(x=>x.dataset.prompt);document.querySelector('#promptPanel').hidden=!show;if(show)loadPromptSlice()}
function isMrTask(n){return n.endsWith('_mr')||n.includes('_mr_')||n==='brain_aneurysm'} function applyModality(){let chosen=document.querySelector('[name=series_key]:checked');if(!chosen)return;let modality=chosen.dataset.modality;document.querySelectorAll('[name=models]').forEach(x=>{let compatible=x.dataset.modalities.split(',').includes(modality);x.disabled=!compatible||x.dataset.enabled!=='true';x.checked=false});let d=document.querySelector(`[name=models][value="${modality==='MR'?'total_mr':modality==='CT'?'total':'__none__'}"]`);if(d&&!d.disabled)d.checked=true;document.querySelector('#submit').disabled=!document.querySelector('[name=models]:not(:disabled)');refreshPrompt()}
fetch('/api/tasks').then(r=>r.json()).then(x=>{taskbox.innerHTML=x.tasks.map(t=>`<label class="task" title="${t.description||''}"><input type="checkbox" name="models" value="${t.name}" data-modality="${t.modality}" data-modalities="${(t.modalities||[t.modality]).join(',')}" data-enabled="${t.enabled!==false}" data-prompt="${t.prompt||''}"> <b>${t.name}</b>${t.licensed?' 🔑':''}${t.enabled===false?` <small>[${t.status||'unavailable'}]</small>`:''}${t.description?`<br><small>${t.description}</small>`:''}${t.structures?`<br><small><b>Output:</b> ${t.structures}</small>`:''}${t.reference?`<br><small><a href="${t.reference}" target="_blank" rel="noopener">Model reference ↗</a></small>`:''}</label>`).join('');document.querySelectorAll('[name=models]').forEach(x=>x.onchange=refreshPrompt)});
document.querySelector('#all').onclick=()=>{document.querySelectorAll('[name=models]:not(:disabled)').forEach(x=>x.checked=true);refreshPrompt()}; document.querySelector('#none').onclick=()=>{document.querySelectorAll('[name=models]').forEach(x=>x.checked=false);refreshPrompt()};
document.querySelector('#promptSlice').oninput=loadPromptSlice;document.querySelector('#clearPrompt').onclick=()=>{prompt={positive_points:[],negative_points:[],box:null,text:'lymph node'};boxCorner=null;writePrompt()};document.querySelector('#promptImage').onclick=e=>{let s=document.querySelector('[name=series_key]:checked'),r=e.target.getBoundingClientRect(),p=[Math.max(0,Math.min(+s.dataset.cols-1,Math.round((e.clientX-r.left)/r.width*+s.dataset.cols))),Math.max(0,Math.min(+s.dataset.rows-1,Math.round((e.clientY-r.top)/r.height*+s.dataset.rows))),+document.querySelector('#promptSlice').value],m=document.querySelector('#promptMode').value;if(m==='box'){if(!boxCorner)boxCorner=p;else{if(boxCorner[2]!==p[2]){alert('Both box corners must be on the same slice');return}prompt.box=[boxCorner,p];boxCorner=null}}else prompt[m+'_points'].push(p);writePrompt()};writePrompt();
document.querySelector('#uploadForm').onsubmit=async e=>{e.preventDefault();let b=document.querySelector('#inspect');b.disabled=true;b.textContent='Uploading and inspecting…';try{let r=await fetch('/api/uploads',{method:'POST',body:new FormData(e.target)}),x=await r.json();if(!r.ok)throw Error(x.detail||'Inspection failed');uploadId=x.upload_id;document.querySelector('#assessment').innerHTML=`<b>${x.series.length} valid series detected.</b> ${x.ignored} invalid or non-image file(s) will be ignored.`;document.querySelector('#series').innerHTML=x.series.map((s,i)=>`<label class="task" style="display:block"><input type="radio" name="series_key" value="${s.key}" data-modality="${s.modality}" data-slices="${s.slices}" data-rows="${s.rows}" data-cols="${s.columns}" ${i===0?'checked':''}> <b>${s.modality}</b> · ${s.slices} slices · ${s.rows}×${s.columns} · ${s.description||'No series description'} <small>${s.series_uid}</small></label>`).join('');document.querySelectorAll('[name=series_key]').forEach(x=>x.onchange=()=>{applyModality();let s=document.querySelector('[name=series_key]:checked');document.querySelector('#promptSlice').max=+s.dataset.slices-1;document.querySelector('#promptSlice').value=Math.floor(+s.dataset.slices/2);loadPromptSlice()});document.querySelector('#modelForm').hidden=false;let s=document.querySelector('[name=series_key]:checked');document.querySelector('#promptSlice').max=+s.dataset.slices-1;document.querySelector('#promptSlice').value=Math.floor(+s.dataset.slices/2);applyModality()}catch(x){alert(x.message)}finally{b.disabled=false;b.textContent='Inspect upload'}};
document.querySelector('#modelForm').onsubmit=async e=>{e.preventDefault();let b=document.querySelector('#submit');b.disabled=true;b.textContent='Queueing…';try{let fd=new FormData(e.target);let r=await fetch(`/api/uploads/${uploadId}/start`,{method:'POST',body:fd}),x=await r.json();if(!r.ok)throw Error(x.detail||'Could not start job');ids.unshift(x.id);localStorage.setItem('dicomJobs',JSON.stringify(ids.slice(0,30)));document.querySelector('#modelForm').hidden=true;document.querySelector('#assessment').textContent='Selected series queued; unselected files were deleted.';poll()}catch(x){alert(x.message)}finally{b.disabled=false;b.textContent='Queue selected series'}};
async function poll(){try{let r=await fetch('/api/jobs'),x=await r.json();ids=x.jobs.map(j=>j.id)}catch(e){}let rows=[];for(const id of ids){try{let r=await fetch('/api/jobs/'+id),j=await r.json();rows.push(`<div class="job"><span class="status">${j.status}</span> · ${id}<br><small>${j.models.join(', ')}</small>${j.detail?`<div>${j.detail}</div>`:''}${j.error?`<div class="error">${j.error}</div>`:''}${j.status==='complete'?`<p><a href="/api/jobs/${id}/download"><button>Download RTSTRUCT ZIP</button></a></p>`:''}</div>`)}catch(e){}}jobsEl.innerHTML=rows.join('')||'No server jobs yet.'}poll();setInterval(poll,4000);
</script></body></html>'''

def metadata(job: Path) -> dict:
    try: return json.loads((job / "job.json").read_text())
    except Exception: raise HTTPException(404, "Job not found")

def save_metadata(job: Path, **changes):
    data = metadata(job); data.update(changes); (job / "job.json").write_text(json.dumps(data, indent=2))

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
            if WORKER_MODE == "ssh":
                save_metadata(job,status="transferring",detail="Sending selected DICOM series to GPU worker")
                subprocess.run([*SSH,"mkdir","-p",f"{remote}/input",f"{remote}/output"],check=True,timeout=30)
                archive=job/"selected-series.tar"
                archive_members = ["input"]
                if (job / "input_ct").is_dir(): archive_members.append("input_ct")
                if (job / "prompt.json").is_file(): archive_members.append("prompt.json")
                subprocess.run(["tar","-C",str(job),"-cf",str(archive),*archive_members],check=True,timeout=600)
                subprocess.run([*SCP,str(archive),f"{WORKER}:{remote}/selected-series.tar"],check=True,timeout=1800)
                archive.unlink(missing_ok=True)
                subprocess.run([*SSH,f"tar -C {remote} -xf {remote}/selected-series.tar && rm -f {remote}/selected-series.tar"],check=True,timeout=600)
                runner_job = remote
            else:
                save_metadata(job,status="processing",detail="Using local GPU worker")
                runner_job = str(job)
            data=metadata(job); save_metadata(job,status="processing",detail="GPU segmentation in progress")
            for i,task in enumerate(data["models"],1):
                save_metadata(job,status="processing",detail=f"Running {task} ({i}/{len(data['models'])})")
                command = ([*SSH,REMOTE_PYTHON,REMOTE_RUNNER,runner_job,task]
                           if WORKER_MODE == "ssh"
                           else [LOCAL_PYTHON,LOCAL_RUNNER,runner_job,task])
                cp=subprocess.run(command,text=True,capture_output=True,timeout=12*3600)
                if cp.returncode: raise RuntimeError((cp.stdout+"\n"+cp.stderr)[-6000:])
            if WORKER_MODE == "ssh":
                subprocess.run([*SCP,f"{WORKER}:{remote}/output/*.dcm",str(job/"output")+"/"],check=True,timeout=600)
            with zipfile.ZipFile(job/"rtstruct-results.zip","w",zipfile.ZIP_DEFLATED) as z:
                for f in (job/"output").glob("*.dcm"): z.write(f,f.name)
            if not any((job/"output").glob("*.dcm")):
                raise RuntimeError("Worker completed without producing any RTSTRUCT files")
            shutil.rmtree(job/"input",ignore_errors=True)
            if WORKER_MODE == "ssh": subprocess.run([*SSH,"rm","-rf",remote],timeout=30)
            save_metadata(job,status="complete",detail="RTSTRUCT files ready for download")
        except Exception as exc:
            if WORKER_MODE == "ssh": subprocess.run([*SSH,"rm","-rf",remote],timeout=30)
            save_metadata(job,status="failed",detail="Processing failed",error=str(exc))

@app.get("/",response_class=HTMLResponse)
def home(): return HTML

@app.get("/api/tasks")
def tasks():
    # Catalog is sourced from the installed worker package when available.
    code="from totalsegmentator.map_to_binary import class_map,commercial_models; import json; print(json.dumps({'tasks':sorted(class_map),'licensed':list(commercial_models),'structures':{k:list(v.values()) if isinstance(v,dict) else list(v) for k,v in class_map.items()}}))"
    try:
        if WORKER_MODE == "ssh":
            command=f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(code)}"
            cp=subprocess.run([*SSH,command],capture_output=True,text=True,timeout=20,check=True)
        else:
            cp=subprocess.run([LOCAL_PYTHON,"-c",code],capture_output=True,text=True,timeout=20,check=True)
        remote=json.loads(cp.stdout.strip().splitlines()[-1]); licensed=set(remote["licensed"])
        internal={"test","total_highres_test","total_v1","vertebrae_pp","vertebrae_pp_refined"}
        names=[n for n in remote["tasks"] if n not in internal and not n.endswith("_auxiliary")]
    except Exception: names=FALLBACK_TASKS; licensed=set()
    structure_map = remote.get("structures",{}) if 'remote' in locals() else {}
    catalog=[{"name":n,"licensed":n in licensed,"modality":task_modality(n),"enabled":True,
              "description":f"TotalSegmentator {task_modality(n)} segmentation task producing the structures listed below.",
              "structures":", ".join(structure_map.get(n,[])) or "Task-specific TotalSegmentator structures",
              "reference":"https://github.com/wasserth/TotalSegmentator"} for n in names]
    for name, details in EXTERNAL_TASKS.items():
        item={"name":name,"licensed":False,"enabled":True,**details}
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
    catalog.extend({"name":name,"licensed":False,**details} for name,details in INFORMATIONAL_MODELS.items())
    return {"tasks":catalog}

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
async def create_job(files: list[UploadFile]=File(...), models: list[str]=Form(...)):
    if not models: raise HTTPException(400,"Select at least one model")
    available={x["name"] for x in tasks()["tasks"] if x.get("enabled",True)}
    if any(not re.fullmatch(r"[A-Za-z0-9_]+",m) or m not in available for m in models): raise HTTPException(400,"Invalid model selection")
    if "total" in models and "total_mr" in models: raise HTTPException(400,"Select either total (CT) or total_mr (MR), not both")
    if any((EXTERNAL_TASKS.get(m) or INFORMATIONAL_MODELS.get(m, {})).get("prompt") for m in models):
        raise HTTPException(400, "Prompted models require the two-stage /api/v1/uploads workflow")
    if any(EXTERNAL_TASKS.get(m, {}).get("requires_ct") for m in models):
        raise HTTPException(400, "Paired PET/CT models require the two-stage /api/v1/uploads workflow")
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
def get_job(jid: str):
    if not re.fullmatch(r"[A-Za-z0-9-]+",jid): raise HTTPException(404)
    return metadata(BASE/jid)

@app.get("/api/jobs")
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

@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    job=BASE/jid; data=metadata(job); result=job/"rtstruct-results.zip"
    if data["status"]!="complete" or not result.exists(): raise HTTPException(409,"Result is not ready")
    return FileResponse(result,filename=f"{jid}-RTSTRUCT.zip",media_type="application/zip")


# Stable automation API. The browser endpoints above remain backward-compatible;
# these versioned routes are documented in OpenAPI and can be protected with an
# X-API-Key header by setting RTSEG_API_KEY.
@app.get("/api/v1/models", dependencies=[Depends(require_api_key)], tags=["automation"])
def api_models():
    return tasks()


@app.post("/api/v1/uploads", dependencies=[Depends(require_api_key)], tags=["automation"])
async def api_assess_upload(files: list[UploadFile] = File(...)):
    return await assess_upload(files)


@app.post("/api/v1/uploads/{uid}/jobs", dependencies=[Depends(require_api_key)], tags=["automation"])
async def api_start_assessed(
    uid: str,
    series_key: str = Form(...),
    models: list[str] = Form(...),
    prompt_json: str = Form(""),
):
    return await start_assessed(uid, series_key, models, prompt_json)


@app.post("/api/v1/jobs", dependencies=[Depends(require_api_key)], tags=["automation"])
async def api_create_job(
    files: list[UploadFile] = File(...),
    models: list[str] = Form(...),
):
    """Submit one already-isolated CT, MR or PET DICOM series."""
    return await create_job(files, models)


@app.get("/api/v1/jobs", dependencies=[Depends(require_api_key)], tags=["automation"])
def api_list_jobs():
    return list_jobs()


@app.get("/api/v1/jobs/{jid}", dependencies=[Depends(require_api_key)], tags=["automation"])
def api_get_job(jid: str):
    return get_job(jid)


@app.get("/api/v1/jobs/{jid}/result", dependencies=[Depends(require_api_key)], tags=["automation"])
def api_download(jid: str):
    return download(jid)


@app.get("/api/v1/health", dependencies=[Depends(require_api_key)], tags=["automation"])
def api_health():
    return {"status": "ok", "worker_mode": WORKER_MODE, "worker": WORKER if WORKER_MODE == "ssh" else "local"}

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
    """Fail jobs whose in-process worker vanished with a portal restart.

    Processing is intentionally owned by this service process.  A stale status
    must never imply that inference is still running, and the remote job tree
    may contain PHI, so clean it up during recovery as well.
    """
    interrupted = []
    for job in BASE.iterdir():
        if not job.is_dir() or job.name.startswith("upload-"):
            continue
        try:
            data = metadata(job)
            if data.get("status") in {"queued", "transferring", "processing"}:
                interrupted.append((job, data.get("status", "unknown")))
        except Exception:
            continue
    for job, previous_status in interrupted:
        remote = f"{REMOTE_BASE}/{job.name}"
        if WORKER_MODE == "ssh":
            subprocess.run([*SSH, "rm", "-rf", remote], timeout=30)
        save_metadata(
            job,
            status="failed",
            detail="Processing was interrupted by a portal service restart",
            error=(f"Recovered stale {previous_status} job at service startup. "
                   "Submit it again to rerun safely."),
        )

@app.on_event("startup")
def startup():
    recover_interrupted_jobs()
    threading.Thread(target=cleanup_loop,daemon=True).start()
