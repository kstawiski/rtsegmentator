#!/usr/bin/env bash
set -euo pipefail

# Mirror the public CERR model-installer repositories, weight archives and
# published conda-pack runtimes into the shared RTsegmentator model bundle.
# Nothing is written outside MODEL_ROOT.
MODEL_ROOT=${RTSEG_MODEL_ROOT:-/models}
SOURCE_ROOT="$MODEL_ROOT/sources/cerr"
DOWNLOAD_ROOT="$MODEL_ROOT/downloads/cerr"
CONDA_ROOT="$MODEL_ROOT/downloads/cerr-condapack"
WEIGHT_ROOT="$MODEL_ROOT/weights/cerr"

models=(
  CT_cardiac_structures_deeplab
  CT_LungOAR_incrMRRN
  MR_Prostate_Deeplab
  CT_Lung_SMIT
  MRI_Pancreas_Fullshot_AnatomicCtxShape
  CT_HeadAndNeck_OARs
  CT_HN_SMIT
  CT_HeartSubstruct_SMIT
  CT_WHOLEBODY_SMITplus
  MR_Rectum_GTV_SMIT
  MR_HN_Nodule_SMIT
  CT_Lung_OAR_SMITplus
  MR_BrainMets_SMITplus
)

mkdir -p "$SOURCE_ROOT" "$DOWNLOAD_ROOT" "$CONDA_ROOT" "$WEIGHT_ROOT"

decode_field() {
  local descriptor=$1 field_prefix=$2
  python3 - "$descriptor" "$field_prefix" <<'PY'
import base64
import gzip
import pathlib
import sys

descriptor = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
for line in descriptor.read_text(errors="replace").splitlines():
    fields = line.split()
    if len(fields) >= 2 and fields[0].startswith(prefix):
        print(gzip.decompress(base64.b64decode(fields[1])).decode().strip())
        raise SystemExit(0)
raise SystemExit(1)
PY
}

download_archive() {
  local url=$1 output=$2
  if [[ -s $output ]]; then
    gzip -t "$output"
    return
  fi
  local partial="${output}.part"
  curl -fsSL --retry 4 --retry-delay 3 --continue-at - -o "$partial" "$url"
  gzip -t "$partial"
  mv "$partial" "$output"
  sha256sum "$output" >"${output}.sha256"
  printf '%s\n' "$url" >"${output}.url"
}

declare -A conda_archives=()
for model in "${models[@]}"; do
  repo="$SOURCE_ROOT/$model"
  if [[ ! -d $repo/.git ]]; then
    if ! git clone --filter=blob:none "https://github.com/cerr/${model}.git" "$repo"; then
      printf 'BLOCKED %s: repository is not publicly cloneable\n' "$model" >&2
      continue
    fi
  fi
  descriptor="$repo/model.txt"
  if [[ ! -f $descriptor ]]; then
    printf 'BLOCKED %s: repository has no model.txt descriptor\n' "$model" >&2
    continue
  fi

  if weight_url=$(decode_field "$descriptor" MODEL_WEIGHTS); then
    archive="$DOWNLOAD_ROOT/${model}.tar.gz"
    if download_archive "$weight_url" "$archive"; then
      destination="$WEIGHT_ROOT/$model"
      mkdir -p "$destination"
      digest=$(sha256sum "$archive" | awk '{print $1}')
      if [[ ! -f $destination/.extracted.sha256 ]] ||
         [[ $(<"$destination/.extracted.sha256") != "$digest" ]]; then
        tar -xzf "$archive" -C "$destination"
        printf '%s\n' "$digest" >"$destination/.extracted.sha256"
      fi
    else
      printf 'BLOCKED %s: published weight URL is unavailable: %s\n' "$model" "$weight_url" >&2
    fi
  else
    printf 'BLOCKED %s: model.txt has no decodable weight URL\n' "$model" >&2
  fi

  if conda_url=$(decode_field "$descriptor" CONDAPACK); then
    # Several SMIT repositories deliberately share one identical runtime.
    key=$(printf '%s' "$conda_url" | sha256sum | cut -c1-16)
    conda_archives[$key]=$conda_url
  fi
done

for key in "${!conda_archives[@]}"; do
  url=${conda_archives[$key]}
  if ! download_archive "$url" "$CONDA_ROOT/${key}.tar.gz"; then
    printf 'BLOCKED conda-pack %s: published URL is unavailable: %s\n' "$key" "$url" >&2
  fi
done
