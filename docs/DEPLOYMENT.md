# Deployment

RTsegmentator supports a local GPU worker and a remote SSH worker.

## Same-host GPU

Prepare the offline payload and image per [OFFLINE.md](OFFLINE.md), then run:

```bash
cp .env.example .env
docker compose up -d
```

The portal listens on host port 8080 and runs inference inside the same
container with `WORKER_MODE=local`.

## Separate WebUI and GPU worker

On the GPU server, install Docker, Compose and NVIDIA Container Toolkit, load
the offline image, and create `secrets/authorized_keys` containing only the
portal's public SSH key:

```bash
docker compose -f compose.remote-worker.yaml up -d
```

On the portal server, place the private key at `secrets/id_ed25519`, pin the
worker host key in `secrets/known_hosts`, set `SEGMENTATION_WORKER` and
`WORKER_SSH_PORT` in `.env`, then:

```bash
docker compose -f compose.portal.yaml up -d --build
```

Port 8080 is exposed on the portal server—the GPU worker need not expose it.
The portal sends only the selected series and retrieves only resulting DICOM
RTSTRUCT files.

## Source configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `WORKER_MODE` | `local` or `ssh` | `ssh` |
| `PORTAL_DATA` | staged uploads/jobs | `./data` |
| `SEGMENTATION_WORKER` | SSH `user@host` | `konrad@cpd-konrad-worker` |
| `WORKER_SSH_PORT` | optional SSH port | `22`/SSH default |
| `REMOTE_JOB_ROOT` | remote transient jobs | `/home/konrad/dicom-rt-jobs` |
| `REMOTE_PYTHON` | remote runtime Python | `/home/konrad/dicom-rt-seg/.venv/bin/python` |
| `REMOTE_RUNNER` | remote dispatcher | `/home/konrad/dicom-rt-seg/run_task.py` |
| `LOCAL_PYTHON` | same-host runtime Python | value of `REMOTE_PYTHON` |
| `LOCAL_RUNNER` | same-host dispatcher | value of `REMOTE_RUNNER` |
| `RTSEG_MODEL_ROOT` | worker model tree | `/home/konrad/lymph-models` |
| `RTSEG_RUNTIME_ROOT` | worker scripts/environments | `/home/konrad/dicom-rt-seg` |
| `RTSEG_API_KEY` | optional `/api/v1` key | unset |
| `MAX_UPLOAD_BYTES` | aggregate upload limit | 8 GiB |
| `RETENTION_HOURS` | job/upload retention | 24 |

Use one Uvicorn worker. The in-process queue lock is intentionally responsible
for serializing GPU inference; multiple Uvicorn workers would create competing
queues.

## Network and data safeguards

- Terminate TLS at a trusted reverse proxy and restrict access by network or
  identity-aware proxy.
- The optional API key protects versioned automation endpoints; it is not a
  replacement for WebUI authentication or TLS.
- Use dedicated SSH keys, pinned host keys and a worker account limited to the
  container.
- Keep job volumes encrypted and backed up only if your data policy allows it.
- Never publish patient DICOM, RTSTRUCT output, credentials, licenses or the
  private offline payload.
