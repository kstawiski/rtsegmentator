# Deployment

The supported examples are a single GPU container and a portal connected to an SSH GPU worker. Keep uploads and outputs on controlled storage. Put the portal behind authenticated TLS; the code has no user authentication or API-key enforcement.

## Single host with local GPU

Use `compose.yaml`. Supply a compatible NVIDIA driver/container runtime, a model bundle and isolated runtime environments. Mount models at `/models`, runtimes at `/runtimes`, persistent jobs at `/data`, and writable temporary files at `/scratch`. Model and runtime mounts are read-only. Preserve bundle symlinks and interpreter dependencies when transferring runtimes; copying a virtual environment alone may not make it relocatable.

```bash
cp .env.example .env
# Edit the bundle locations and ENABLED_EXTERNAL_MODELS.
docker compose build
docker compose up -d
curl -fsS http://localhost:8080/api/v1/health
curl -fsS http://localhost:8080/api/v1/models
```

The source Dockerfile is a generic CUDA image, not an export of a site-specific production image. Check the built image's imports, drivers and model runtimes before use. See the model ledger for retained input and licensing restrictions.

## Portal and SSH GPU worker

Run `compose.remote-worker.yaml` on the GPU host. Supply `SSH_AUTHORIZED_KEY` from an external environment. The example publishes SSH on port 2222. Install the portal's private SSH key and known-hosts file outside the repository. Configure the portal account's SSH alias so ordinary `ssh worker@rtsegmentator-worker` and `scp` use the correct host, port and identity; the application does not consume `WORKER_SSH_PORT` itself.

On the portal, set:

```text
WORKER_MODE=ssh
SEGMENTATION_WORKER=worker@rtsegmentator-worker
REMOTE_JOB_ROOT=/home/worker/jobs
REMOTE_PYTHON=/opt/rtsegmentator/venv/bin/python
REMOTE_RUNNER=/app/run_task.py
REMOTE_APP_ROOT=/app
REMOTE_MODEL_ROOT=/models
REMOTE_RUNTIME_ROOT=/runtimes
PORTAL_DATA=/data
```

Start a single application process with `uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1`. Generic systemd and Supervisor examples are in `dicom-rt-portal.service` and `deploy/`. The worker receives only the selected series and required paired inputs/prompts. Configure writable scratch storage on the worker. SSH environment forwarding depends on the SSH server configuration; verify the worker runtime sees the intended variables.

## Configuration

| Variable | Behavior |
|---|---|
| `WORKER_MODE` | Public default `local`; also accepts `ssh`, `colab`, `auto` |
| `PORTAL_DATA` | Job/upload root; default `data` beside `app.py`, container `/data` |
| `RTSEG_APP_ROOT` | Runner scripts and catalogs; defaults to the script directory, examples `/app` |
| `RTSEG_MODEL_ROOT` | Model assets; default `/models` |
| `RTSEG_RUNTIME_ROOT` | Isolated runtimes; runner fallback is the model root, examples explicitly `/runtimes` |
| `RTSEG_SCRATCH_ROOT` | Runner temporary storage; fallback is `runtime-homes/tmp` in the model root, examples explicitly `/scratch` |
| `ENABLED_EXTERNAL_MODELS` | Comma-separated external tasks allowed in local mode |
| `ENABLED_MODELS_FILE` | Additional task names, one per line; default `<PORTAL_DATA>/enabled-models.txt`; blank/comment lines ignored |
| `CATALOG_TTL_SECONDS` | Cache lifetime, default 600; enabled-file changes also trigger refresh |
| `PREFLIGHT_TIMEOUT_SECONDS` | Model preflight timeout, default 900 |
| `TASK_TIME_LIMIT_SECONDS` | Local/SSH task time limit, default 43200 |
| `TASK_STOP_GRACE_SECONDS` | Local process-tree termination grace, default 45 |
| `MAX_UPLOAD_BYTES` | Upload limit, default 8589934592 bytes |
| `RETENTION_HOURS` | Job/upload retention, default 24 |
| `RTSEG_ORIENTATION_VALIDATION` | Portal fallback marker directory; default `<RTSEG_MODEL_ROOT>/orientation-validation` |
| `RTSEG_CERR_MODERN_PYTHON` | Optional override of `<RTSEG_RUNTIME_ROOT>/envs/segmentation-modern/bin/python` |
| `RTSEG_CERR_PROSTATE_PYTHON` | Optional override of `<RTSEG_MODEL_ROOT>/runtime-homes/cerr/env6-prostate/bin/python` |

Check the actual catalog after deployment. A source manifest or historical validation record does not install a model, enable it, or establish clinical suitability.

## Updates and rollback

Keep the preceding image and configuration until the replacement passes import, health, upload and synthetic geometry checks. Wait for active jobs and uploads to finish before recreation. Retain model/runtime checksums separately. If startup or verification fails, restore the previous image/configuration and inspect retained job state. Do not overwrite the model bundle while a job is using it.

## Selected tasks on a separate SSH GPU

For a local portal whose GPU cannot run NV-Reason-CT, retain `WORKER_MODE=local` and set `REMOTE_GPU_TASKS=nv_reason_ct_report`. `auto` mode uses the same routing; local tasks retain their existing Colab fallback. Listed tasks always use SSH and do not enter local OOM fallback. Pure `ssh` mode still sends the whole job to its worker. The portable `colab` mode does not support this report task and keeps it disabled.

Use the existing `SEGMENTATION_WORKER`, `REMOTE_JOB_ROOT`, `REMOTE_PYTHON`, `REMOTE_RUNNER`, `REMOTE_APP_ROOT`, `REMOTE_MODEL_ROOT` and `REMOTE_RUNTIME_ROOT` settings above. Install the same application version and the isolated report runtime on the worker. Configure writable scratch with `REMOTE_SCRATCH_ROOT=/scratch`, an optional persistent cache with `REMOTE_NV_REASON_CT_CACHE=/scratch/nv-reason-ct-cache`, and, if needed, `REMOTE_TOTALSEG_WEIGHTS_PATH` for the worker's existing coverage weights. `RTSEG_NV_REASON_CT_COVERAGE_MODE` is forwarded explicitly over SSH. Other worker runtime configuration belongs in the worker environment.

The report transfer includes selected `input/`, paired `input_ct/` and `prompt.json` when present, plus finished RTSTRUCTs. Retrieval copies `output/report/` only for a routed report, then removes the remote job directory. Other routed tasks retrieve their named RTSTRUCT. Whole SSH jobs retrieve the complete output directory. The existing task time limit and remote runner stop apply; failed retrieval retains remote results for recovery. Local contour files survive report failure and retry.

Remote preflight runs during catalog refresh and is cached with the catalog. Local markers cannot enable a remote task. A failed remote check disables only routed tasks. Health reports remote reachability separately. Install and verify the remote report acceptance marker for the selected coverage mode before enabling the task.

The local container needs an SSH client, rsync, a private key and a verified known-hosts file. The Dockerfile supplies the clients. Keep credentials outside Git and mount the portal account's SSH directory read-only with an external Compose override:

```yaml
services:
  rtsegmentator:
    volumes:
      - ${RTSEG_SSH_CONFIG_DIR}:/root/.ssh:ro
```

The directory must contain the private key (mode 0600), `known_hosts`, and an SSH `config` entry resolving the worker alias, port and identity. Populate host keys through your normal verification process; do not disable host-key checking. The application uses batch SSH and does not prompt for passwords. Restart the portal after changing route configuration. Enabling remote routing does not itself install assets or bypass acceptance.
