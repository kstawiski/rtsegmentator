#!/usr/bin/env bash
set -euo pipefail

role=${RTSEG_ROLE:-combined}

if [[ "$role" == "combined" ]]; then
  export WORKER_MODE=local
  exec setpriv --reuid=konrad --regid=konrad --init-groups \
    /home/konrad/dicom-rt-seg/.venv/bin/uvicorn app:app \
    --app-dir /home/konrad/dicom-rt-seg --host 0.0.0.0 --port 8080 --workers 1
fi

if [[ "$role" == "worker" ]]; then
  if [[ ! -s /run/secrets/authorized_keys ]]; then
    echo "RTSEG_ROLE=worker requires /run/secrets/authorized_keys" >&2
    exit 2
  fi
  install -d -m 0700 -o konrad -g konrad /home/konrad/.ssh
  install -m 0600 -o konrad -g konrad /run/secrets/authorized_keys /home/konrad/.ssh/authorized_keys
  ssh-keygen -A
  exec /usr/sbin/sshd -D -e
fi

echo "RTSEG_ROLE must be combined or worker" >&2
exit 2
