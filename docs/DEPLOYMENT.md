# Deployment

## Hosts

The application uses two systems:

1. **Portal host** — serves the WebUI on TCP port 8080 and temporarily stages uploads.
2. **GPU worker** — runs inference and RTSTRUCT conversion through SSH.

The default worker and paths are defined in `app.py`:

| Setting | Default |
|---|---|
| `WORKER_MODE` | `ssh` (`local` runs inference in the portal environment) |
| `SEGMENTATION_WORKER` | `worker@rtsegmentator-worker` |
| `REMOTE_JOB_ROOT` | `/home/worker/jobs` |
| `REMOTE_PYTHON` | `/opt/rtsegmentator/venv/bin/python` |
| `REMOTE_RUNNER` | `/app/run_task.py` |
| `PORTAL_DATA` | repository-local `data/` |
| `MAX_UPLOAD_BYTES` | 8 GiB |
| `RETENTION_HOURS` | 24 hours |

Override them with environment variables when the deployment differs.

## Portal host

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Install the systemd unit after adjusting its paths and account if necessary:

```bash
sudo install -m 0644 dicom-rt-portal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dicom-rt-portal.service
sudo systemctl status dicom-rt-portal.service
```

Verify:

```bash
curl -fsS http://127.0.0.1:8080/api/v1/health
```

## SSH worker access

Configure non-interactive key-based SSH from the portal account:

```bash
ssh worker@rtsegmentator-worker true
```

The portal copies only the selected DICOM series, job metadata and prompt data. Unselected uploaded series are not transferred.

## Worker installation

Create these directories on the worker:

```text
/home/worker/dicom-rt-seg/
/home/worker/dicom-rt-seg/.venv/
/home/worker/dicom-rt-jobs/
/models/
```

Copy the worker runtime code:

```bash
scp run_task.py run_*.py *.json worker@rtsegmentator-worker:/home/worker/dicom-rt-seg/
```

The worker uses isolated environments for incompatible historical releases, including nnU-Net v1, current nnU-Net v2, PAM/SAT3D/SAM-Med2D, Bouget and Raidionics. Exact release-specific repairs and validation results are recorded in `EXTERNAL_MODELS.md`.

Model weights are intentionally not stored in this repository. Download them from the upstream references displayed in the WebUI and place them at the paths used by `run_task.py`.

## TotalSegmentator

Install the current supported TotalSegmentator release in the worker runtime. The portal queries the installed package for its complete task and structure catalog.

Store the TotalSegmentator license on the worker using its CLI or another protected runtime mechanism. Never commit the key to Git or place it in the systemd unit.

Confirm the installation:

```bash
/opt/rtsegmentator/venv/bin/TotalSegmentator --version
/opt/rtsegmentator/venv/bin/totalseg_set_license --help
```

## Updating application code

After changing portal code:

```bash
python3 -m py_compile app.py run_task.py run_*.py
scp run_task.py run_*.py *.json worker@rtsegmentator-worker:/home/worker/dicom-rt-seg/
sudo systemctl restart dicom-rt-portal.service
systemctl is-active dicom-rt-portal.service
```
