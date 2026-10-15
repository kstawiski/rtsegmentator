#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
  echo "usage: $0 /absolute/runtime/directory" >&2
  exit 2
fi
runtime=$1
case "$runtime" in /*) ;; *) echo "runtime path must be absolute" >&2; exit 2 ;; esac
command -v uv >/dev/null
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uv venv --python 3.12 "$runtime"
uv pip install --python "$runtime/bin/python" -r "$project_dir/requirements-nv-reason-ct.txt"
"$runtime/bin/python" -c 'import torch, transformers, monai; print("torch:", torch.__version__, "transformers:", transformers.__version__, "monai:", monai.__version__, "CUDA:", torch.cuda.is_available())'
