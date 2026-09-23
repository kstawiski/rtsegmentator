#!/bin/sh
set -eu
destination=${1:-rtsegmentator-offline}
mkdir -p "$destination"
docker build -t rtsegmentator:1.0 .
docker save rtsegmentator:1.0 | gzip > "$destination/rtsegmentator-image.tar.gz"
cp compose.yaml "$destination/compose.yaml"
cp -R models "$destination/models"
sha256sum "$destination/rtsegmentator-image.tar.gz" > "$destination/SHA256SUMS"
echo "Offline bundle written to $destination"
