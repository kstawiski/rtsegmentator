# RTsegmentator

RTsegmentator is a research WebUI and HTTP API for assessing DICOM uploads,
running GPU segmentation, and exporting strictly validated DICOM RT Structure
Set (`RTSTRUCT`) files. It supports CT, MR and PET workflows, TotalSegmentator,
RT-focused lymph-node/tumour models, and interactive prompts for PAM, SAT3D and
SAM-Med2D.

> **Research use only.** This software and its model outputs are not medical
> devices. Every result must be reviewed before clinical use.

## What it does

- Inspects uploads locally before sending anything to a worker.
- Rejects invalid files and separates multiple studies, series and modalities.
- Lets the user select exactly one compatible series and model set.
- Keeps `total` (CT) and `total_mr` (MR) mutually exclusive.
- Supports a GPU on the WebUI host or a separate SSH-connected GPU worker.
- Exposes a versioned API and generated Swagger UI at `/api/docs`.
- Deletes input DICOM after completion and expires retained jobs automatically.
- Terminates each model process after inference, allowing GPU memory to be
  reclaimed between jobs.

The model picker includes descriptions, expected structures, validation state
and links to the upstream release. Models that cannot safely run remain visible
and disabled with their blocker.

## Deployment choices

### One offline GPU host

After preparing the authorized model payload described in
[Offline deployment](docs/OFFLINE.md):

```bash
docker compose build
docker compose up -d
```

Open `http://HOST:8080`. The same container runs the portal and GPU worker.

### Separate portal and GPU worker

Run `compose.remote-worker.yaml` on the GPU server and `compose.portal.yaml` on
the WebUI server. Only the selected series is transferred to the worker over
SSH. See [Deployment](docs/DEPLOYMENT.md).

### Source development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
WORKER_MODE=ssh SEGMENTATION_WORKER=user@gpu-host \
  .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080
```

Model weights, patient images, generated jobs, license keys and built offline
images are intentionally excluded from Git. Upstream model terms still apply;
see `models/manifest.json` and [the model ledger](EXTERNAL_MODELS.md).

## Automation

The recommended workflow is two-stage: upload and inspect, then submit one
returned `series_key`. This prevents accidental processing of mixed archives.

```bash
curl -H "X-API-Key: $RTSEG_API_KEY" \
  -F 'files=@study.zip' http://localhost:8080/api/v1/uploads

curl -H "X-API-Key: $RTSEG_API_KEY" \
  -F 'series_key=SERIES_KEY_FROM_FIRST_RESPONSE' \
  -F 'models=total' \
  http://localhost:8080/api/v1/uploads/UPLOAD_ID/jobs
```

Poll `/api/v1/jobs/JOB_ID`, then download
`/api/v1/jobs/JOB_ID/result`. Full examples are in [API](docs/API.md).

## Documentation

- [Offline image and model payload](docs/OFFLINE.md)
- [Portal/worker deployment](docs/DEPLOYMENT.md)
- [Automation API](docs/API.md)
- [Operations, privacy and validation](docs/OPERATIONS.md)
- [External-model validation ledger](EXTERNAL_MODELS.md)
