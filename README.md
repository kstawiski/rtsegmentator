# DICOM RT Segmentation Portal

Research portal for uploading DICOM image series, selecting compatible GPU segmentation models, and downloading the results as validated DICOM RT Structure Set (`RTSTRUCT`) files.

The WebUI can run inference in the same GPU container or send jobs over SSH to a dedicated GPU worker. Uploads are assessed locally before transfer: invalid files are ignored, distinct DICOM series are separated, and the user explicitly chooses one series and compatible models.

> **Research use only.** Model outputs require clinical review and are not suitable for unattended treatment-planning decisions.

## Features

- CT, MR and PET DICOM ingestion and series-level inspection.
- Modality-aware model selection. `total` and `total_mr` are never selected together.
- Current TotalSegmentator task catalog obtained from the installed worker version.
- External lymph-node, tumour and CTV models with descriptions, output structures, validation status and upstream references.
- Interactive slice viewer for PAM, SAT3D and SAM-Med2D point/box prompts.
- PET models can require a co-referenced CT series.
- Strict RTSTRUCT acceptance checks against the selected source series.
- Serialized GPU job execution and process-group termination.
- Per-job temporary-data cleanup so model processes release VRAM after completion.

The live catalog currently exposes 81 tasks. Models without reproducible weights, required metadata, authoritative label mappings or required paired inputs remain visible but disabled with a precise blocker status.

## Architecture

```text
Browser :8080
    |
    v
FastAPI portal host
  - upload assessment
  - DICOM series selection
  - prompt capture
  - job history/download
    |
    | SSH/SCP
    v
GPU worker
  - DICOM -> geometry-preserving NIfTI
  - model-specific preprocessing/inference
  - mask -> RTSTRUCT
  - strict DICOM validation
```

## Repository layout

- `app.py` — FastAPI application and WebUI.
- `run_task.py` — worker-side model dispatcher.
- `run_*.py` — compatibility and inference wrappers for external releases.
- `convert_validate_rtstruct.py` — mask conversion and strict RTSTRUCT validation.
- `dicom_series_to_nifti.py` — geometry-preserving DICOM conversion.
- `labels/` and `*-labels.json` — explicit output-label mappings.
- `EXTERNAL_MODELS.md` — model integration and validation evidence ledger.
- `docs/DEPLOYMENT.md` — host and worker deployment procedure.
- `docs/OPERATIONS.md` — operation, validation, privacy and troubleshooting.

Model weights, patient images, generated jobs, virtual environments and license keys are deliberately excluded from Git.

## Quick start with Docker and NVIDIA GPU

```bash
docker compose build
docker compose up -d
curl -fsS http://localhost:8080/api/v1/health
```

Open `http://localhost:8080`. This runs the WebUI and worker together with `WORKER_MODE=local`. See [Containers and offline deployment](docs/CONTAINERS.md) for a separate GPU server and air-gapped installation.

## Local application development

For local application development:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
WORKER_MODE=local .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080
```

The local runtime must contain the model environments and artifacts documented in `EXTERNAL_MODELS.md`. Set `WORKER_MODE=ssh` for a separately managed worker.

## Automation

The stable automation surface is `/api/v1`, with interactive documentation at `/api/docs`. It supports model discovery, multipart DICOM submission, job polling, health checks, and result download. See [API](docs/API.md).

## Documentation

- [Deployment](docs/DEPLOYMENT.md)
- [Containers and offline deployment](docs/CONTAINERS.md)
- [Automation API](docs/API.md)
- [Operations and safety](docs/OPERATIONS.md)
- [External-model validation ledger](EXTERNAL_MODELS.md)
