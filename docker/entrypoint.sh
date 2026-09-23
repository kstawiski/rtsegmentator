#!/bin/sh
set -eu

case "${1:-portal}" in
  portal)
    exec /opt/rtsegmentator/venv/bin/uvicorn app:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
    ;;
  worker-ssh)
    test -n "${SSH_AUTHORIZED_KEY:-}" || { echo "SSH_AUTHORIZED_KEY is required" >&2; exit 2; }
    printf '%s\n' "$SSH_AUTHORIZED_KEY" > /home/worker/.ssh/authorized_keys
    chmod 700 /home/worker/.ssh
    chmod 600 /home/worker/.ssh/authorized_keys
    chown -R worker:worker /home/worker/.ssh
    printf '%s\n' 'Port 2222' 'PermitRootLogin no' 'PasswordAuthentication no' 'AllowUsers worker' >> /etc/ssh/sshd_config
    exec /usr/sbin/sshd -D -e
    ;;
  *) exec "$@" ;;
esac
