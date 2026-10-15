# Operations and safety

## Upload and model selection

Upload DICOM files, a folder, or ZIP archives. The portal groups images by Study, Series and modality. Select one assessed series and compatible models. The recommended starting models are `total` for CT and `total_mr` for MR. Select each model explicitly in the guided UI; recommendations are never selected or run automatically. PET1 ISRT tasks require CT in the same Study and Frame of Reference. Do not treat this check as image registration.

The browser uses resumable sessions with requests up to 32 MiB. A resumed chunk must use the server's expected byte offset. The receiving request times out after 60 seconds without new data. Session assessment runs in the background. Restart recovery marks interrupted jobs and sessions for review or retry; retention cleanup removes expired data.

## Prompts

Prompt coordinates are `[column, row, slice]` on the selected series, sorted along the DICOM slice normal. The upload preview and viewer share this order and the stored pixel grid. Recreate prompts saved with the older raw-z preview for flipped or non-axial series. Keep prompt data with its source series. The catalog describes the required prompt type and supported modality for each model.

- PAM propagates a positive point or axial box through a volume.
- SAT3D needs a positive 3D point and supports negative points within its prompt-centred ROI.
- SAM-Med2D is slice-wise; prompt every intended slice.
- MedSAM, SAM-Med3D, MedSAM2/Medical-SAM2, Medical-SAM3, SegVol, nnInteractive, VISTA3D and VoxTell have separate prompt, license and field-of-view restrictions. Read their catalog entries and the model ledger before selecting them.
- BiomedParse remains blocked by its approval-gated checkpoint.

The assessed-upload endpoint accepts `prompt_json` and validates the required prompt type and coordinate bounds. Direct multipart and direct session-job submission reject paired or prompted models with HTTP 400. Paired series must have present, matching Study and Frame of Reference identifiers. See [API](API.md) for route differences.

## RTSTRUCT geometry gate

The converter checks prediction shape and affine against the model input, requires an explicit name for every nonzero label, maps column/row/slice arrays directly to RTSTRUCT row/column/slice masks, and checks DICOM SOP class and source image references. It compares physical LPS bounds and independently rasterizes contours onto the NIfTI grid. The physical round-trip minimum Dice is 0.95. Bounding-box tolerance is the larger of 2.1 mm and 2.1 times the largest voxel spacing. Overlapping-mask conversion preserves separate ROIs.

External model availability also requires orientation acceptance markers. These technical checks detect conversion errors; they do not measure segmentation accuracy in a new patient. Empty candidate predictions may complete with a no-candidates note; contours are not fabricated.

The viewer's synthetic tests found an existing `rt-utils==1.2.7` writer limitation: it pairs unequal row/column spacings incorrectly and assumes uniform slice spacing. The unchanged production converter rejected the tested unequal-spacing masks at its physical geometry gate. Viewer tests use the production converter for supported fixtures and independently specified patient-space contours in an rt-utils RTSTRUCT container for unequal-spacing and gapped fixtures. Passing those viewer tests does not validate the writer for these inputs.

## Prostate SBRT crop

The CPU tool crops single-frame axial CT and matching RTSTRUCTs along patient z only. It accepts slice normals within 1 degree of patient z. It uses contours named PTV* or CTV*, excluding subtraction helpers such as `Bowel - PTV`, with a default 150 mm margin. The optional local `total` postprocess selects the ROI named `prostate` and retains the original full RTSTRUCT.

Pixel data are unchanged. Whole contours outside the retained slice range are removed. New Study, Frame, Series and SOP UIDs are the default, with RTSTRUCT references rewritten to match. `keep_uids` is a debugging option. Downloads include a crop report. Review the cropped image range and contours before any clinical use.

## Catalog and job lifecycle

A cold model catalog waits for its first preflight. Subsequent requests use the cache while background refresh runs. The default cache lifetime is 600 seconds; editing the enabled-model file also requests refresh. Enabled flags depend on installation, orientation markers and local policy. A local task that passes its technical gates but is absent from the enabled list reports `not_enabled_on_this_server`. Missing dependencies, pending markers and declared blockers keep their specific status.

Refresh exceptions are logged with a traceback. The portal continues serving the last good catalog, and health exposes `catalog_built_at`, `catalog_refreshing` and `catalog_error`. A successful refresh clears the error. An initial build failure returns HTTP 503 from the models endpoint. Inspect service logs for the full exception; the health response omits exception messages and paths.

Jobs run under one worker lock. Local and SSH inference use a default 12-hour per-task limit. The portal snapshots the local process tree, sends SIGTERM, waits up to 45 seconds, then kills survivors with PID start-time checks. The SSH path also requests termination of the matching remote runner. Colab compatibility paths retain their separate timeout behavior.

## Public audit fixtures and private evidence

`catalog/orientation-audit-fixtures.json` contains public fixtures from MSD, KiTS19, TotalSegmentator MRI, TemplateFlow and other named releases. Missing private-case entries are intentional. Supply a complete external JSON task-to-fixture map with `RTSEG_ORIENTATION_FIXTURES`; values resolve under the model bundle's `testdata`. `RTSEG_ORIENTATION_PROMPTS` can select an external prompt map. Absolute prompt paths are supported by the existing path resolution.

Historical acceptance metadata retains measured counts and model limitations. Private fixture names, job identifiers and private artifact hashes are withheld; a public fixture has not been substituted into those historical results. Revalidation aliases can be supplied with `RTSEG_LEGACY_ARTIFACT_ALIASES`, and `RTSEG_RAIDIONICS_RESULT_DIR` selects the legacy result directory. These configurations belong outside Git.

`scripts/audit-worker-storage.sh` is a specialized NFS/loop-image audit, not a prerequisite for local-container deployment. Configure its `RTSEG_STORAGE_ROOT`, `RTSEG_STORAGE_FSTYPE_PREFIX`, `RTSEG_AUDIT_HOME`, `RTSEG_WORKER_STATE_ROOT`, application/runtime/venv roots and job/build/archive paths for that layout. It retains the existing 1 GiB Docker-storage threshold.

## Privacy and checks

Restrict the portal to authenticated users over TLS. Job IDs are not credentials. Do not commit uploads, patient prompts, model weights, license files or logs. Retention defaults to 24 hours; set `RETENTION_HOURS` and review storage permissions and backup policies. `/api/v1/config` exposes the active retention and upload limits.

Completed segmentation and draft-report jobs retain the selected input images on the portal host in `job/input` for research viewing until retention expiry or `DELETE /api/v1/jobs/{jid}`. Retention is measured from job creation; the existing hourly cleanup removes the whole job. Viewer access returns 404 after expiry even before cleanup runs. Plan storage for these retained images as well as outputs. Companion CT input and remote worker copies keep their existing cleanup behaviour. Older completed jobs whose inputs were deleted cannot be viewed without a new upload. The separate prostate crop tool retains its existing output package and input cleanup.

Deletion is available for terminal jobs and removes local images, contours, reports and logs. Queued or running jobs return 409. SSH cleanup must succeed before local deletion; a 502 response means the local job remains for a retry. Contours are displayed on acquired slice planes without interpolation or image registration. These synthetic geometry checks and local API changes do not establish clinical or manuscript readiness.

```bash
curl -fsS http://localhost:8080/api/v1/health
curl -fsS http://localhost:8080/api/v1/models
```

See [EXTERNAL_MODELS.md](../EXTERNAL_MODELS.md) for the retained numerical results and model-specific restrictions.

## NV-Reason-CT draft reports

Use diagnostic CT only. Known CBCT device names, explicit CBCT/cone-beam text, and a Varian/Elekta acquisition with a Ram-Lak kernel plus either 100–150 kV or a 450–480 mm reconstruction diameter are refused. The HU rule requires the 0.1 percentile between −1100 and −900 HU and a maximum above 300 HU. These rules can refuse tightly cropped diagnostic images and do not establish image quality.

Coverage uses `total --fast` (3 mm) by default, with no ROI subset. Set `RTSEG_NV_REASON_CT_COVERAGE_MODE=full` to use full resolution. Chest requires lung volume ≥1000 mL and heart ≥100 mL; abdomen requires liver ≥500 mL and at least one kidney ≥50 mL. If neither region qualifies, the report records `not_generated`. Compare fast and full decisions on the acceptance series before accepting the faster configuration.

Mapped statements are withheld if the organ group is absent. Truncated masks, insufficient volumes or missing required components receive “[partly in the scanned volume]”. Regional bones are checked for presence, without a truncation warning at expected scan boundaries. Pleura maps to lungs; mediastinum/hila to heart, aorta, esophagus and trachea; great vessels to aorta and the total task's pulmonary-vein mask; chest wall/breast to ribs, sternum and costal cartilages; bowel/appendix to small bowel, colon and duodenum. These group checks do not directly segment every structure named in a heading. Medical devices, CHEST/ABDOMEN labels and other unmapped headings are listed once under “Not checked against the segmentation”. Peritoneum/omentum, lymph nodes and abdominal wall/soft tissues stay unmapped. Impression/label mentions of absent anatomy remain flagged, with raw output available in a collapsed appendix.

The publisher crop is checked for fallback and ≥90% containment of regional organ masks. Failure triggers a mask-centred 384 mm cube, sampled on a 192³ LPS grid at 2 mm with air padding. Actual model tensors must match the expected normalized crop within absolute tolerance 1e-5. Larger anatomical extents can still exceed this crop and are flagged. Midplane thumbnails do not show the entire volume.

Contour volumes are approximate polygon area × source slice spacing / 1000; holes and overlaps are not subtracted. Non-CLOSED_PLANAR contours are omitted. Slice ranges are zero based. These are not dosimetric measurements.

Install the isolated runtime with `scripts/bootstrap-nv-reason-ct-runtime.sh /runtimes/envs/nv-reason-ct`. Runtime resolution checks `RTSEG_NV_REASON_CT_PYTHON`, then `RTSEG_RUNTIME_ROOT/envs/nv-reason-ct/bin/python`, then `RTSEG_MODEL_ROOT/runtime-envs/nv-reason-ct/bin/python`. Store the pinned snapshot under `RTSEG_MODEL_ROOT/weights/nv-reason-ct/<revision>/`. Set `TOTALSEG_WEIGHTS_PATH` to installed total weights: Dataset297 for fast coverage, Dataset291–295 for full. Missing coverage weights fail before a download attempt. Usage reporting is disabled in a job-local TotalSegmentator configuration. Model loading uses local files, offline Hub settings and pinned custom model code.

Use sm_80-class or newer CUDA hardware. The worker waits up to 120 seconds for 14 GiB free before coverage/model loading and checks for 4 GiB free before generation. Resource refusal returns a reason. Model caches use `RTSEG_NV_REASON_CT_CACHE`, defaulting to `RTSEG_SCRATCH_ROOT/nv-reason-ct-cache`; use writable persistent storage for warm Triton/Inductor caches. This supports read-only model/runtime mounts. When `RTSEG_SCRATCH_ROOT` is unset, the dispatcher creates a short per-job `rtseg-*` directory under system temp and removes it on exit. Configured scratch longer than 60 bytes uses a separate short directory for child `TMPDIR`, `TEMP` and `TMP`; an unwritable setting also redirects scratch there. The runner logs each configured-path fallback. This leaves room for nnU-Net's multiprocessing Unix sockets. Output locations are unchanged; configured scratch and persistent caches are not deleted. Forced termination with SIGKILL cannot run exit cleanup, so inspect abandoned `rtseg-*` directories after a forced stop.

Enable the task through `ENABLED_EXTERNAL_MODELS` or the enabled-model file in local mode. The marker at `validation/nv_reason_ct_report.json` requires `status: "passed"`, the pinned `revision`, checks `chest`, `abdomen`, `pelvis`, `cbct`, and `coverage_mode` matching the configured mode. Older markers without a mode attest only `full`. Remote tasks use the remote worker's marker. Reports and input share job retention; retained remote results after failed retrieval also need operator retention. See the [ledger](../EXTERNAL_MODELS.md) for observed diagnostic failures and historical GPU measurements.


## Automatic NV-Segment tasks

`nv_segment_ct_auto` and `nv_segment_ctmr_ct_body` accept a single CT series. `nv_segment_ctmr_mri_body` accepts one body MR series. `nv_segment_ctmr_mri_brain` uses T1 brain MR; skull stripping, normalization, registration and inverse mapping run internally. No prompt is needed. The dispatcher checks DICOM Modality before conversion; that tag does not establish the MR sequence. Review the sequence and field of view before selecting a task. CT modes exclude the publisher's tumour heads. Tiny spurious labels may occur. CTMR weights are restricted to noncommercial research or evaluation.

Each task requires installed assets, a matching `validation/<task>.json` acceptance marker and an `orientation-validation/<task>.json` marker. Acceptance must contain `task`, `result: "passed"`, `source_commit`, `checkpoint_revision` and `checkpoint_sha256` matching `nv_segment_logic.py`. Missing, malformed or mismatched markers block selection and ordinary dispatch. The runner also verifies the actual checkpoint, source files and ROI map. The marker is a technical execution gate, not clinical approval.

In local mode, add accepted tasks to `ENABLED_MODELS_FILE` or `ENABLED_EXTERNAL_MODELS`. No enabled-models file is changed by this integration. To route any of them to an SSH worker, use the existing comma-separated `REMOTE_GPU_TASKS` setting. Install this same code and the pinned assets on that worker; its preflight and markers are authoritative. Keep the existing task timeout and process-tree cancellation enabled. Pascal sm_61 compatibility remains untested.

Run the orientation audit only on the chosen GPU worker, with writable marker directories. Set `RTSEG_APP_ROOT`, `RTSEG_MODEL_ROOT`, `RTSEG_RUNTIME_ROOT`, and `RTSEG_WORKER_PYTHON` (the worker interpreter with DICOM/conversion dependencies). Set `RTSEG_ORIENTATION_RUN_ROOT` to a fresh directory on project storage **outside** the model bundle, which is read-only during ordinary jobs. Use a short task-owned `RTSEG_SCRATCH_ROOT`/`TMPDIR` such as a `mktemp -d /tmp/nvs.XXXXXX` directory for multiprocessing sockets, and remove it after the audit. Keep images, copied bundles and logs on project storage.

```bash
WORKER_MODE=local REMOTE_GPU_TASKS= "$RTSEG_WORKER_PYTHON" \
  "$RTSEG_APP_ROOT/scripts/run-orientation-audit.py" --nv-segment-acceptance \
  nv_segment_ct_auto nv_segment_ctmr_ct_body \
  nv_segment_ctmr_mri_body nv_segment_ctmr_mri_brain
```

This operator-only audit option runs fresh checks even if a marker already exists. It passes an explicit bootstrap argument to the local dispatcher; normal portal and SSH jobs do not pass it. Old markers for the audited task are removed before the run. New markers are written only after GPU inference, exact-grid uint16 output, RTSTRUCT conversion/round-trip validation and orientation auditing succeed. Failure returns a nonzero exit status and leaves that task unaccepted. `TASK_TIME_LIMIT_SECONDS` applies to each inference task; the default is 12 hours. Logs and retained predictions are under `RTSEG_ORIENTATION_RUN_ROOT`. The audit does not edit the enabled-models list.

The four fixture assignments are in `catalog/orientation-audit-fixtures.json`. Brain uses the existing public OpenNeuro-derived T1 DICOM fixture; the publisher's NIfTI-only brain example cannot check DICOM contour orientation. No entries are needed in the prompt map for these automatic tasks. Model-agreement, laterality and GPU-compatibility results are recorded in `EXTERNAL_MODELS.md`.
