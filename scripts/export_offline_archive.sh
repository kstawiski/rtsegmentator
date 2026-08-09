#!/usr/bin/env bash
set -euo pipefail

model_root=${1:-/home/konrad/lymph-models}
runtime_root=${2:-/home/konrad/dicom-rt-seg}
archive=${3:-offline/payload.tar}

if [[ ! -d "$model_root/weights" || ! -d "$model_root/sources" ]]; then
  echo "Model root must contain weights/ and sources/: $model_root" >&2
  exit 2
fi
if [[ ! -x "$runtime_root/.venv/bin/python" ]]; then
  echo "Runtime root does not contain .venv/bin/python: $runtime_root" >&2
  exit 2
fi
if [[ -e "$archive" || -e "$archive.sha256" ]]; then
  echo "Archive or checksum already exists; move it aside explicitly: $archive" >&2
  exit 2
fi

mkdir -p "$(dirname "$archive")"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/home/konrad/lymph-models/results/dbdmp" \
  "$stage/home/konrad/lymph-models/results/ln-seg-fm" "$stage/home/konrad"

for name in weights sources envs; do
  ln -s "$model_root/$name" "$stage/home/konrad/lymph-models/$name"
done
ln -s "$model_root/results/dbdmp/model" \
  "$stage/home/konrad/lymph-models/results/dbdmp/model"
for name in monai130 nnunet251 labels.json; do
  [[ -e "$model_root/results/ln-seg-fm/$name" ]] && \
    ln -s "$model_root/results/ln-seg-fm/$name" \
      "$stage/home/konrad/lymph-models/results/ln-seg-fm/$name"
done
ln -s "$runtime_root" "$stage/home/konrad/dicom-rt-seg"

tar --create --file "$archive" --dereference --sparse \
  --exclude='*.dcm' --exclude='*.dicom' --exclude='testdata' \
  --exclude='home/konrad/dicom-rt-seg/data' --exclude='.git' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='*.pyc' \
  -C "$stage" home
sha256sum "$archive" > "$archive.sha256"
echo "Offline payload archive created: $archive"
echo "Review models/manifest.json and upstream licenses before distribution."
