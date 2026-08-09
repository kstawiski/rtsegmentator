#!/usr/bin/env bash
set -euo pipefail

model_root=${1:-/home/konrad/lymph-models}
runtime_root=${2:-/home/konrad/dicom-rt-seg}
destination=${3:-offline/payload}

if [[ ! -d "$model_root/weights" || ! -d "$model_root/sources" ]]; then
  echo "Model root must contain weights/ and sources/: $model_root" >&2
  exit 2
fi
if [[ ! -x "$runtime_root/.venv/bin/python" ]]; then
  echo "Runtime root does not contain .venv/bin/python: $runtime_root" >&2
  exit 2
fi
if [[ -e "$destination" ]]; then
  echo "Destination already exists; move it aside explicitly: $destination" >&2
  exit 2
fi

mkdir -p "$destination/model-root" "$destination/runtime"

# Dereference symlinks so the image is self-contained. Never copy fixtures,
# patient jobs, prior inference results, caches, Git metadata, or credentials.
rsync -aL --info=progress2 \
  --exclude='testdata/' --exclude='results/' --exclude='.git/' \
  --exclude='__pycache__/' --exclude='*.dcm' \
  "$model_root/" "$destination/model-root/"

# These two result subtrees are runtime compatibility assets, not case output.
mkdir -p "$destination/model-root/results/dbdmp" "$destination/model-root/results/ln-seg-fm"
rsync -aL "$model_root/results/dbdmp/model/" "$destination/model-root/results/dbdmp/model/"
for name in monai130 nnunet251 labels.json; do
  [[ -e "$model_root/results/ln-seg-fm/$name" ]] && \
    rsync -aL "$model_root/results/ln-seg-fm/$name" "$destination/model-root/results/ln-seg-fm/"
done

rsync -aL --exclude='.git/' --exclude='data/' --exclude='__pycache__/' \
  "$runtime_root/" "$destination/runtime/"

python3 scripts/payload_manifest.py "$destination" > "$destination/payload-manifest.json"
echo "Offline payload created at $destination"
echo "Review models/manifest.json and upstream licenses before distributing the image."
