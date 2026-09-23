#!/bin/sh
set -eu
bundle=${1:-.}
(cd "$bundle" && sha256sum -c SHA256SUMS)
gzip -dc "$bundle/rtsegmentator-image.tar.gz" | docker load
echo "Run: docker compose -f $bundle/compose.yaml up -d"
