# Model bundles

This directory is a mount point and contains no patient data, credentials, license keys, or redistributed checkpoints.

The base image installs TotalSegmentator 2.17.0. Configure its license through the official TotalSegmentator mechanism where required. External model checkpoints must be obtained under their upstream terms and placed in the paths documented in `EXTERNAL_MODELS.md` and `run_task.py`.

For an offline installation, populate this directory on a licensed online staging machine, verify the files, then transfer it alongside the saved container image. Never publish a populated bundle without checking every model's redistribution license.

The runner resolves this mount through `RTSEG_MODEL_ROOT` (default `/models`)
and resolves conversion scripts and label maps through `RTSEG_APP_ROOT`
(default: the directory containing `run_task.py`). Executable environments can
live on faster local storage under `RTSEG_RUNTIME_ROOT`; it defaults to the
model root for backward compatibility. Run the same readiness gate used by the
API before enabling external models:

```bash
RTSEG_MODEL_ROOT=/models RTSEG_RUNTIME_ROOT=/runtimes \
  RTSEG_APP_ROOT=/app python /app/model_preflight.py
```

Every requested model must report `"ready": true`. This is an installation
check; it complements, but does not replace, the model-specific inference and
RTSTRUCT validation recorded in `EXTERNAL_MODELS.md`.

The current shared modern runtime additionally pins `kornia==0.7.4` for
TotalSpineSeg/auglab compatibility. Do not upgrade Kornia independently without
rerunning the TotalSpineSeg two-stage acceptance test.
