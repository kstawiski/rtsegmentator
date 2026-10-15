# LNQ2023 atlas adapter

The public share was readable without credentials on 2026-10-15. The worker
verified its directory and small metadata files through Persistent Chrome, but
could not transfer the complete archive through the browser bridge. No full
archive SHA-256 is available from this run. The catalog task remains disabled.

`run_lnq_atlas.py` is a staged adapter for source revision
`e48a74f3223e5d9ef06f79fb08282450c29f7a2d`. It requires:

- All five released `checkpoint_best.pth` files, `plans.json`, `dataset.json`
  and the six images in `weights/data/Atlas`.
- A provenance JSON with `source_revision`, the archive `sha256`, and
  `files_sha256` for every path returned by `run_lnq_atlas.required_files()`.
  File keys are relative to the extracted `weights` directory. Hashes must be
  calculated from the downloaded files. The runner verifies every used file.
- The existing nnU-Net 2.8.1 runtime with SimpleITK, PyTorch and the source's
  registration dependencies. The original predictor targets an older API.
- A separate **TotalSegmentator 1.5.7** environment. Set
  `RTSEG_LNQ_TOTAL_PYTHON` to its Python for portal dispatch/preflight, or pass
  `--total-python` to the runner. Registration uses the v1 numeric organ IDs.
  A v2 mask cannot be substituted. Put caches in a new writable project path.

The legacy v1 launcher isolates `Path.home()` within its child process because
that release ignores `TOTALSEG_HOME_DIR`. It disables usage telemetry in its
private configuration and leaves the user's home and installed package unchanged.
Its large `nnunet_tmp_` imaging workspace stays in the supplied project directory;
multiprocessing sockets continue to use the short local `TMPDIR`.

The adapter preserves the three inputs (CT, atlas distance map, probability
map), upstream registration and lung crop, five-fold averaging of restored
foreground probabilities, atlas-dependent threshold (0.15–0.30), lung convex
hull restriction and removal of components with an equivalent ellipsoid
diameter below 5 mm on any axis. It uses the installed network builder because
the released trainer inherits that builder. Strict CPU loading passed for folds 0 and 2. The remaining three checkpoints
still need verification.

Input is reoriented losslessly to DICOM LPS. Oblique CTs are rejected because
the upstream registration assumes axis-aligned LPS. Final labels are restored
to the original CT grid using nearest-neighbour interpolation. The only target
is a visible mediastinal lymph-node candidate; this is not an elective nodal
region map.

```sh
python run_lnq_atlas.py --input ct.nii.gz --output prediction.nii.gz \
  --work NEW_PRIVATE_WORK_DIRECTORY --source PINNED_LNQ_SOURCE \
  --weights EXTRACTED_WEIGHTS --manifest provenance.json \
  --total-python TOTAL_V1_ENV/bin/python
```

Use a short local `TMPDIR` for multiprocessing sockets, with CTs, predictions,
registration intermediates and caches on project storage. The worker's CPU
checks verify weight/architecture compatibility for folds 0 and 2. They do not
verify the complete prediction pipeline. Complete download, checks of the three
remaining folds, GPU inference and RTSTRUCT geometry checks remain required
before enabling. Clinical readiness is not established.

Sources: [pinned repository](https://github.com/MICAI-IMI-UzL/LNQ2023/tree/e48a74f3223e5d9ef06f79fb08282450c29f7a2d),
[public share](https://cloud.imi.uni-luebeck.de/s/24q2qr4ycA7CgSP),
[repository licence, CC BY-NC-ND 4.0](https://github.com/MICAI-IMI-UzL/LNQ2023/blob/e48a74f3223e5d9ef06f79fb08282450c29f7a2d/LICENSE).
Separate archive-specific terms were not found.
