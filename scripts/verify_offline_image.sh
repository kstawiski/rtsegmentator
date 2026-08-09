#!/usr/bin/env bash
set -euo pipefail

archive=${1:?usage: verify_offline_image.sh IMAGE.tar}
sha256sum -c "$archive.sha256"
docker load -i "$archive"
docker run --rm --gpus all --entrypoint nvidia-smi rtsegmentator/offline-worker:latest
