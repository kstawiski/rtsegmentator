# Operations and safety

## Upload and model selection

Upload DICOM files, a folder, or ZIP archives. The portal groups images by Study, Series and modality. Select one assessed series and compatible models. CT defaults to `total`; MR defaults to `total_mr`. PET1 ISRT tasks require CT in the same Study and Frame of Reference. Do not treat this check as image registration.

The browser uses resumable sessions with requests up to 32 MiB. A resumed chunk must use the server's expected byte offset. The receiving request times out after 60 seconds without new data. Session assessment runs in the background. Restart recovery marks interrupted jobs and sessions for review or retry; retention cleanup removes expired data.

## Prompts

Prompt coordinates are `[column, row, slice]` on the selected series. Keep prompt data with its source series. The catalog describes the required prompt type and supported modality for each model.

- PAM propagates a positive point or axial box through a volume.
- SAT3D needs a positive 3D point and supports negative points within its prompt-centred ROI.
- SAM-Med2D is slice-wise; prompt every intended slice.
- MedSAM, SAM-Med3D, MedSAM2/Medical-SAM2, Medical-SAM3, SegVol, nnInteractive, VISTA3D and VoxTell have separate prompt, license and field-of-view restrictions. Read their catalog entries and the model ledger before selecting them.
- BiomedParse remains blocked by its approval-gated checkpoint.

The assessed-upload endpoint accepts `prompt_json`. Do not use the direct multipart job endpoint for paired or prompted models. See [API](API.md) for route differences.

## RTSTRUCT geometry gate

The converter checks prediction shape and affine against the model input, requires an explicit name for every nonzero label, maps column/row/slice arrays directly to RTSTRUCT row/column/slice masks, and checks DICOM SOP class and source image references. It compares physical LPS bounds and independently rasterizes contours onto the NIfTI grid. The physical round-trip minimum Dice is 0.95. Bounding-box tolerance is the larger of 2.1 mm and 2.1 times the largest voxel spacing. Overlapping-mask conversion preserves separate ROIs.

External model availability also requires orientation acceptance markers. These technical checks detect conversion errors; they do not measure segmentation accuracy in a new patient. Empty candidate predictions may complete with a no-candidates note; contours are not fabricated.

## Prostate SBRT crop

The CPU tool crops single-frame axial CT and matching RTSTRUCTs along patient z only. It accepts slice normals within 1 degree of patient z. It uses contours named PTV* or CTV*, excluding subtraction helpers such as `Bowel - PTV`, with a default 150 mm margin. The optional local `total` postprocess selects the ROI named `prostate` and retains the original full RTSTRUCT.

Pixel data are unchanged. Whole contours outside the retained slice range are removed. New Study, Frame, Series and SOP UIDs are the default, with RTSTRUCT references rewritten to match. `keep_uids` is a debugging option. Downloads include a crop report. Review the cropped image range and contours before any clinical use.

## Catalog and job lifecycle

A cold model catalog waits for its first preflight. Subsequent requests use the cache while background refresh runs. The default cache lifetime is 600 seconds; editing the enabled-model file also requests refresh. Enabled flags depend on installation, orientation markers and local policy.

Jobs run under one worker lock. Local and SSH inference use a default 12-hour per-task limit. The portal snapshots the local process tree, sends SIGTERM, waits up to 45 seconds, then kills survivors with PID start-time checks. The SSH path also requests termination of the matching remote runner. Colab compatibility paths retain their separate timeout behavior.

## Public audit fixtures and private evidence

`catalog/orientation-audit-fixtures.json` contains public fixtures from MSD, KiTS19, TotalSegmentator MRI, TemplateFlow and other named releases. Missing private-case entries are intentional. Supply a complete external JSON task-to-fixture map with `RTSEG_ORIENTATION_FIXTURES`; values resolve under the model bundle's `testdata`. `RTSEG_ORIENTATION_PROMPTS` can select an external prompt map. Absolute prompt paths are supported by the existing path resolution.

Historical acceptance metadata retains measured counts and model limitations. Private fixture names, job identifiers and private artifact hashes are withheld; a public fixture has not been substituted into those historical results. Revalidation aliases can be supplied with `RTSEG_LEGACY_ARTIFACT_ALIASES`, and `RTSEG_RAIDIONICS_RESULT_DIR` selects the legacy result directory. These configurations belong outside Git.

`scripts/audit-worker-storage.sh` is a specialized NFS/loop-image audit, not a prerequisite for local-container deployment. Configure its `RTSEG_STORAGE_ROOT`, `RTSEG_STORAGE_FSTYPE_PREFIX`, `RTSEG_AUDIT_HOME`, `RTSEG_WORKER_STATE_ROOT`, application/runtime/venv roots and job/build/archive paths for that layout. It retains the existing 1 GiB Docker-storage threshold.

## Privacy and checks

Restrict the portal to authenticated users over TLS. Job IDs are not credentials. Do not commit uploads, patient prompts, model weights, license files or logs. Retention defaults to 24 hours; set `RETENTION_HOURS` and review storage permissions and backup policies.

```bash
curl -fsS http://localhost:8080/api/v1/health
curl -fsS http://localhost:8080/api/v1/models
```

See [EXTERNAL_MODELS.md](../EXTERNAL_MODELS.md) for the retained numerical results and model-specific restrictions.
