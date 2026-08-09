# Offline GPU image

The public repository contains build recipes, not model weights. Model licenses
and access terms differ, and some releases prohibit redistribution. Build the
payload only from assets you are authorized to use and keep the resulting image
private unless every included asset permits redistribution.

## Prerequisites

- Linux x86-64 GPU host with an NVIDIA driver compatible with CUDA 12.8.
- Docker Engine, Compose v2 and NVIDIA Container Toolkit.
- A validated RTsegmentator runtime tree and model root. The exporter expects
  `MODEL_ROOT/{weights,sources}` and `RUNTIME_ROOT/.venv/bin/python`.
- Sufficient space: the complete internal payload may exceed 150 GB, before
  Docker layer overhead.

## Create the payload

On the prepared GPU worker:

```bash
./scripts/export_offline_payload.sh \
  /path/to/model-root /path/to/runtime-root offline/payload
```

The exporter dereferences symlinks and creates SHA-256 inventory
`offline/payload/payload-manifest.json`. It excludes DICOM, test fixtures,
previous results, jobs, Git metadata, caches and credentials. Review both the
manifest and `models/manifest.json` before building.

## Build and transfer

```bash
./scripts/build_offline_image.sh rtsegmentator/offline-worker:latest \
  rtsegmentator-offline-worker.tar
```

Copy the tar and its `.sha256` to the isolated host, then:

```bash
./scripts/verify_offline_image.sh rtsegmentator-offline-worker.tar
docker load -i rtsegmentator-offline-worker.tar
nvidia-smi
docker run --rm --gpus all --entrypoint nvidia-smi \
  rtsegmentator/offline-worker:latest
docker compose up -d
```

TotalSegmentator commercial tasks require a separately provisioned license
configuration. Do not bake a license key into an image or commit it to Git.
Mount its configuration volume as shown in the compose files.

## Roles

The same image has two roles:

- `RTSEG_ROLE=combined`: WebUI/API and GPU inference run on one host, port 8080.
- `RTSEG_ROLE=worker`: SSH-only GPU worker for a separately deployed portal.

The image uses one portal process and the application serializes GPU jobs. Each
model runs as a child process; when it exits, its CUDA context and VRAM are
released. The worker container remains idle without retaining a model server.

## Updating

Recreate the payload from the newly validated environments, build a new tagged
image, run a smoke test for each enabled model family, and retain the prior image
until validation completes. `latest` should only point at an image that passed
your local DICOM-to-RTSTRUCT acceptance suite.
