#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /absolute/runtime/directory" >&2
  exit 2
fi

runtime=$1
case "$runtime" in
  /*) ;;
  *) echo "runtime path must be absolute" >&2; exit 2 ;;
esac

command -v uv >/dev/null 2>&1 || {
  echo "uv is required to construct the isolated runtime" >&2
  exit 1
}

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uv venv --python 3.11 "$runtime"
uv pip install --python "$runtime/bin/python" \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$runtime/bin/python" \
  -r "$project_dir/requirements-segmentation-modern.txt"

"$runtime/bin/python" - <<'PY'
import kornia
import lungmask
import moosez
import mrisegmentator
import mrsegmentator
import nnunetv2
import torch
import totalspineseg
print("CUDA available:", torch.cuda.is_available())
print("torch:", torch.__version__, "kornia:", kornia.__version__)
PY
