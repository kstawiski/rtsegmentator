#!/usr/bin/env bash
set -euo pipefail

tag=${1:-rtsegmentator/offline-worker:latest}
archive=${2:-offline/rtsegmentator-offline-worker.tar}

test -f offline/payload.tar || {
  echo "Run scripts/export_offline_archive.sh first" >&2
  exit 2
}
command -v docker >/dev/null || {
  echo "Docker is required on the connected build host" >&2
  exit 2
}

docker build -f Dockerfile.offline-worker -t "$tag" .
mkdir -p "$(dirname "$archive")"
docker save -o "$archive" "$tag"
sha256sum "$archive" > "$archive.sha256"
echo "Offline image: $archive"
echo "Checksum:      $archive.sha256"
