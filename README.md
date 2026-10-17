# DICOM RT Segmentation Portal

Upload DICOM CT, MR or PET series, select compatible segmentation models, and download RTSTRUCT contours checked against the source images. Run the portal with a local GPU or an SSH GPU worker.

> **Research use only.** Review every contour before clinical use. Technical execution and geometry checks do not establish clinical accuracy or manuscript readiness.

## Features

- Separate uploaded studies into image series before selecting models. PET1 ISRT tasks require a CT in the same Study and Frame of Reference.
- Upload folders, extensionless DICOM files and ZIP archives. Resumable upload sessions accept chunks up to 32 MiB and report upload progress.
- Discover TotalSegmentator tasks from the installed runtime and external tasks from committed catalogs. Catalog results are cached; stale results remain available during a background refresh.
- View uploaded and completed-job series with patient orientation markers (R/L, A/P, H/F). Standard orientation is on by default: axial A/R, coronal H/R and sagittal H/A at top/left. The toggle restores stored order; oblique series stay as stored with direction markers. Contours follow the image display transform.
- Select prompted models through the slice viewer or the assessed-upload API. Each model retains its input contract and limitation text.
- Run serialized GPU jobs with progress, retries using retained input, and per-task time limits. Timeout handling stops the runner and its process tree.
- Use read-only model bundles. CADS and selected nnU-Net v1 pipelines create writable directory mirrors in each job while linking immutable model files.
- Filter TotalSegmentator auxiliary labels against the task label map before RTSTRUCT export.
- Convert masks with direct source-grid row/column mapping. Verify source references, physical bounds and a contour-to-mask round trip; gate external tasks on orientation evidence.
- Run automatic NV-Segment-CT and NV-Segment-CTMR CT body, MRI body and T1 brain tasks after per-task technical acceptance. CT modes expose 117 foreground labels, MRI body 50 and brain 132. CTMR weights permit noncommercial research or evaluation only; see the [model ledger](EXTERNAL_MODELS.md).
- Generate NVIDIA NV-Reason-CT research draft reports for covered chest/abdomen CT, alone or after segmentation. Download JSON/Markdown/HTML with crop thumbnails and contour summaries. CBCT is refused; a separate technical acceptance marker controls availability.
- Route selected tasks to an SSH GPU worker with `REMOTE_GPU_TASKS` while other tasks use the local GPU.
- Crop axial CT and matching RTSTRUCTs for prostate SBRT on CPU. The standalone tool selects PTV*/CTV* contours with a default 150 mm superior/inferior margin. Optional local `total` postprocessing selects the prostate ROI. The crop keeps pixel data and creates matching new Study, Frame, Series and SOP UIDs by default.

The captured deployment catalog contains 253 tasks, 182 enabled. These are snapshot counts, not guarantees for a fresh installation. The installed TotalSegmentator version, model assets, enabled list and orientation markers determine `/api/v1/models`. Disabled models remain visible with their restrictions.

## Quick start: local GPU container

```bash
cp .env.example .env
# Set RTSEG_MODEL_BUNDLE and RTSEG_RUNTIME_BUNDLE to your licensed local assets.
docker compose build
docker compose up -d
curl -fsS http://localhost:8080/api/v1/health
```

Open `http://localhost:8080`. Configure authenticated TLS before accepting medical data. The application does not enforce user identity or `RTSEG_API_KEY`.

Model weights, runtimes, licenses and patient images are excluded from the source tree. The base image is not a complete external-model installation. See [deployment](docs/DEPLOYMENT.md) and [containers](docs/CONTAINERS.md) for mounts and SSH configuration.

## Local development and checks

Use Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m py_compile *.py scripts/*.py
.venv/bin/pytest -q
WORKER_MODE=local .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8080
```

`Licence key required` identifies TotalSegmentator tasks that require a key (`licensed`). `Non-commercial weights` uses the separate catalog field `noncommercial`. Full licence notes remain in model details; research-draft wording alone does not set either badge.

Run the browser UI regression checks with `python3 tests/ui_browser_server.py`, open `http://127.0.0.1:18743/__qa`, and choose **Run checks**. The harness serves synthetic API responses and checks the real UI at 1024, 1280 and 1440 pixels. It does not run models. Run it in a foreground tab or a headless browser: background tabs pause the ResizeObserver that reserves space for the selection tray, so the tray checks fail there.

CPU tests use synthetic images. Inference needs the separate model/runtime bundle and compatible GPU software. Colab and automatic fallback code are retained for compatibility; the documented deployment examples use local or SSH workers.

## Source and documentation

- `app.py`: API, static-page serving, upload sessions, cached catalog and job lifecycle.
- `static/`: offline HTML/CSS/JavaScript UI (bundled Open Sans webfont, SIL Open Font License in `static/fonts/OFL.txt`) with upload, interactive series preview, segmentation / AI report actions and review. Jobs opens images, contour overlays, reports and downloads; More tools contains the CPU crop tool.
- `run_task.py`, `run_*.py`: dispatch and model-specific inference wrappers.
- `convert_validate_rtstruct.py`, `convert_masks_validate_rtstruct.py`, `rtstruct_roundtrip.py`: direct conversion and geometry checks, including overlapping masks.
- `crop_prostate_sbrt.py`: standalone CPU crop tool.
- `catalog/`, `labels/`, `models/manifests/`, `models/patches/`: task definitions, labels and upstream compatibility records.
- [API](docs/API.md), [operations](docs/OPERATIONS.md), [model ledger](EXTERNAL_MODELS.md), [public release checklist](docs/PUBLIC_RELEASE_CHECKLIST.md).
