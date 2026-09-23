# GPU containers and offline deployment

## Prerequisites

- Linux host with an NVIDIA GPU supported by the selected models.
- NVIDIA driver, Docker Engine, Compose v2, and NVIDIA Container Toolkit.
- Sufficient storage. The validated external-model collection currently occupies hundreds of gigabytes and should be mounted, not copied into an image layer.
- Build on a dedicated build/GPU host with ample free space (at least 30 GB for the baseline image build, more for model bundles). Do not build on the portal host unless it is sized for Docker layers and large Python/CUDA wheels.

The Dockerfile pins PyTorch 2.6.0 from the official CUDA 12.4 wheel index before installing TotalSegmentator. This prevents dependency resolution from silently mixing a CUDA 13 PyTorch stack into the CUDA 12.4 base image.

## One-machine deployment

This is the simplest setup: the WebUI and GPU runner share one container and no SSH transfer is needed.

```bash
docker compose build
docker compose up -d
curl -fsS http://localhost:8080/api/v1/health
```

The compose file sets `WORKER_MODE=local`, persists jobs in a named volume, and mounts the model and runtime bundles read-only at `/models` and `/runtimes`. Set their host locations without editing Compose:

```bash
export RTSEG_MODEL_BUNDLE=/path/to/model-bundle
export RTSEG_RUNTIME_BUNDLE=/path/to/container-compatible/rtseg-runtimes
docker compose up -d
```

Do not point `RTSEG_RUNTIME_BUNDLE` at an arbitrary host virtual environment. Python environments can contain absolute interpreter paths and must be created or verified at their intended container mount location. Run `model_preflight.py` inside the container before enabling tasks.
Only the TotalSegmentator task family is enabled by default in this portable image. External models appear in the catalog as `model_bundle_required`. After installing and validating a compatible bundle, explicitly enable its task names with the comma-separated `ENABLED_EXTERNAL_MODELS` environment variable.

## Separate GPU worker

Build the same image on the GPU server and run its SSH worker profile:

```bash
export SSH_AUTHORIZED_KEY="$(cat ~/.ssh/id_ed25519.pub)"
docker compose -f compose.remote-worker.yaml up -d
```

Configure an SSH host alias on the portal host because the container listens on port 2222:

```sshconfig
Host rtsegmentator-worker
  HostName gpu-server.example.org
  Port 2222
  User worker
  IdentityFile ~/.ssh/id_ed25519
```

Portal settings for that worker are:

```text
WORKER_MODE=ssh
SEGMENTATION_WORKER=rtsegmentator-worker
REMOTE_JOB_ROOT=/home/worker/jobs
REMOTE_PYTHON=/opt/rtsegmentator/venv/bin/python
REMOTE_RUNNER=/app/run_task.py
REMOTE_APP_ROOT=/app
REMOTE_MODEL_ROOT=/models
REMOTE_RUNTIME_ROOT=/runtimes
```

`/api/v1/models` runs `model_preflight.py` against the worker and disables an
external model with `dependencies_missing` if its runtime, source tree, or
checkpoint directory is absent. `ENABLED_EXTERNAL_MODELS` is still required
for local mode, so merely mounting an unreviewed bundle never makes it
selectable.

## Offline transfer

On an internet-connected staging machine:

```bash
sh scripts/export-offline-bundle.sh /path/to/rtsegmentator-offline
```

Populate the bundle's `models/` directory only with checkpoints you are licensed to use and redistribute internally. Move the directory to the offline GPU host, then run:

```bash
sh scripts/import-offline-bundle.sh /path/to/rtsegmentator-offline
docker compose -f /path/to/rtsegmentator-offline/compose.yaml up -d
```

The image archive is checksum-verified before loading. Keep a software bill of materials and checksums for separately supplied weights in regulated environments.

## Model scope

The base image pins the publicly installable TotalSegmentator runtime validated by this project. External models have different licenses, download controls, CUDA/Python requirements, and in several cases mutually incompatible historical environments. They are therefore read-only model and runtime mounts, not silently downloaded or embedded during an image build. This prevents every model update from duplicating hundreds of gigabytes in Docker's layer store. `EXTERNAL_MODELS.md` records each model's validation and availability state.
