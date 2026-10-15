#!/usr/bin/env bash
set -euo pipefail

model_root=${RTSEG_MODEL_ROOT:-/models}
state_root=${RTSEG_WORKER_STATE_ROOT:-/data/worker-state}
storage_root=${RTSEG_STORAGE_ROOT:-/data}
storage_fstype=${RTSEG_STORAGE_FSTYPE_PREFIX:-nfs}
audit_home=${RTSEG_AUDIT_HOME:-/data/worker-home}
app_root=${RTSEG_APP_ROOT:-/app}
runtime_root=${RTSEG_RUNTIME_ROOT:-/runtimes}
venv_root=${RTSEG_VENV_ROOT:-$app_root/.venv}
worker_jobs=${RTSEG_WORKER_JOB_ROOT:-/data/jobs}
rtseg_jobs=${RTSEG_JOB_ROOT:-/data/rtseg-jobs}
build_root=${RTSEG_BUILD_ROOT:-/data/build}
archive_file=${RTSEG_ARCHIVE_FILE:-/data/rtsegmentator-image.tar.gz}
fail=0

require_nfs_path() {
    local path=$1 resolved source fstype
    resolved=$(readlink -f "$path")
    source=$(findmnt -n -o SOURCE -T "$resolved")
    fstype=$(findmnt -n -o FSTYPE -T "$resolved")
    printf '%-36s -> %-80s [%s %s]\n' "$path" "$resolved" "$fstype" "$source"
    if [[ $resolved != "$storage_root"/* || $fstype != "$storage_fstype"* ]]; then
        echo "ERROR: $path is not backed by $storage_root ($storage_fstype)" >&2
        fail=1
    fi
}

require_nfs_path "$model_root"
require_nfs_path "$state_root"
require_nfs_path "$audit_home/.apptainer"
require_nfs_path "$audit_home/.cache/huggingface"
require_nfs_path "$audit_home/.cache/lnq-segmenter"
require_nfs_path "$audit_home/.cache/monai"
require_nfs_path "$audit_home/.cache/nnunet"
require_nfs_path "$audit_home/.cache/pip"
require_nfs_path "$audit_home/.cache/torch"
require_nfs_path "$audit_home/.cache/TotalSegmentator"
require_nfs_path "$audit_home/.totalsegmentator"
require_nfs_path "$worker_jobs"
require_nfs_path "$app_root/data"
require_nfs_path "$rtseg_jobs"
require_nfs_path "$build_root"
if [[ -e "$archive_file" ]]; then
    require_nfs_path "$archive_file"
fi

declare -A mount_images=(
    ["$runtime_root"]=rtseg-runtimes.squashfs
    ["$venv_root"]=dicom-rt-seg-venv.squashfs
)
for mounted_path in "${!mount_images[@]}"; do
    backing=$(findmnt -n -o SOURCE "$mounted_path")
    image_name=${mount_images[$mounted_path]}
    loop_backing=$(losetup -j "$state_root/$image_name" 2>/dev/null || true)
    printf '%-36s mounted from %s\n' "$mounted_path" "$backing"
    if [[ $backing != /dev/loop* || -z $loop_backing ]]; then
        echo "ERROR: $mounted_path is not mounted from $state_root/$image_name" >&2
        fail=1
    fi
done

echo "Large files in local RTsegmentator operational directories:"
local_hits=$(find "$app_root" "$worker_jobs" \
    "$rtseg_jobs" -xdev \
    -path "$venv_root" -prune -o \
    -type f -size +100M -printf '%s\t%p\n' 2>/dev/null | sort -nr || true)
if [[ -n $local_hits ]]; then
    printf '%s\n' "$local_hits"
    echo "ERROR: inspect the local heavyweight files listed above" >&2
    fail=1
else
    echo "none"
fi

echo "Large RTsegmentator artifacts elsewhere on the local home filesystem:"
home_hits=$(find "$audit_home" -xdev -type f -size +100M \
    \( -iname '*rtsegmentator*' -o -iname '*dicom-rt*' -o \
       -iname '*hippunfold*' -o -iname '*nninteractive*' -o \
       -iname '*brats-toolkit*' -o -iname '*segvol*' -o \
       -iname '*voxtell*' -o -iname '*vista3d*' \) \
    -printf '%s\t%p\n' 2>/dev/null | sort -nr || true)
if [[ -n $home_hits ]]; then
    printf '%s\n' "$home_hits"
    echo "ERROR: move the RTsegmentator artifacts listed above to ${storage_root}" >&2
    fail=1
else
    echo "none"
fi

docker_bytes=$(sudo -n du -sx --block-size=1 /var/lib/docker 2>/dev/null | awk '{print $1}' || true)
if [[ -z $docker_bytes ]]; then
    echo "ERROR: unable to audit local Docker storage" >&2
    fail=1
else
    printf '%-36s %s bytes\n' /var/lib/docker "$docker_bytes"
    if (( docker_bytes > 1073741824 )); then
        echo "ERROR: Docker is using more than 1 GiB on the local filesystem" >&2
        fail=1
    fi
fi

df -h "$audit_home" "$storage_root"
exit "$fail"
