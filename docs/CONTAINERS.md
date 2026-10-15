# Containers and offline bundles

`Dockerfile` builds a generic CUDA runtime with the portal and worker scripts. `compose.yaml` runs local inference; `compose.remote-worker.yaml` runs an SSH GPU worker. Both mount models and runtimes read-only. Use [deployment](DEPLOYMENT.md) for configuration and network access.

## Storage

| Container path | Purpose |
|---|---|
| `/app` | Source, catalogs, labels and runners |
| `/models` | Separately licensed model bundle and validation markers |
| `/runtimes` | Isolated Python environments and upstream runtime trees |
| `/data` | Upload sessions, jobs, logs and enabled-model list |
| `/scratch` | Writable runner temporary files |
| `/root/.totalsegmentator` | Local container's persistent TotalSegmentator configuration/cache |

Set `RTSEG_MODEL_BUNDLE` and `RTSEG_RUNTIME_BUNDLE` in an external `.env` file. These are host mount sources. Set `ENABLED_EXTERNAL_MODELS` there, or place an enabled-model list in the mounted data volume. Mounting assets does not by itself bypass preflight or orientation gates.

Read-only model support uses per-job directory mirrors for CADS and selected nnU-Net v1 pipelines. Model files remain symlinks into the mounted bundle. Other upstream runtimes may still need external cache, license or interpreter configuration; inspect the model-specific ledger.

## Offline transfer

`scripts/export-offline-bundle.sh` and `scripts/import-offline-bundle.sh` transfer a separately prepared bundle. Run them on storage with enough space for the archive and extraction. Keep published source, private images, checkpoints, medical images and credentials in separate artifacts. Preserve symlinks and verify checksums before changing mounts. Inspect the scripts' arguments before use; they do not download missing weights or grant licenses.

The supplied source Dockerfile and tests have no GPU acceptance result from this synchronization. Existing model-ledger measurements describe earlier runs on their stated fixtures.
