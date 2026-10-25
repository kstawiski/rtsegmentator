# Automation API

The versioned API is available under `/api/v1`; interactive OpenAPI documentation is at `/api/docs` and the schema at `/api/openapi.json`.

## Submit and follow a job

Submit one DICOM image series as repeated multipart `files` fields. Repeat `models` for each requested model:

```bash
curl -fsS -X POST http://localhost:8080/api/v1/jobs \
  -F 'models=total' \
  -F 'files=@CT.1.dcm' \
  -F 'files=@CT.2.dcm'
```

The response contains the job `id`. Poll it and download the result after `status` becomes `complete`:

```bash
curl -fsS http://localhost:8080/api/v1/jobs/JOB_ID
curl -fLo rtstruct.zip http://localhost:8080/api/v1/jobs/JOB_ID/result
curl -fsS http://localhost:8080/api/v1/jobs/JOB_ID/manifest
```

Every segmentation result ZIP contains `rtsegmentator_manifest.json`, returned alone by the `manifest` route: the source series, each model's outcome and exact ROI names, the catalog's licence and limitation text where recorded, and the size and SHA-256 of every packaged file. See [result manifest](result-manifest.md). The route returns HTTP `409` until the job completes and `404` for packages without a manifest, such as the prostate SBRT crop tool. [`rtseg_client.py`](client.md) wraps these routes and verifies packages.

Other routes:

- `GET /api/v1/config` (alias `/api/config`) — public configuration, without credentials or storage paths. The response has exactly these fields: `{"retention_hours": 24, "max_upload_bytes": 8589934592, "worker_mode": "local", "remote_gpu_tasks": [], "version": "1.0.0", "report_task": "nv_reason_ct_report"}`. Retention and upload limits are integers reflecting the configured values; mode/version are strings and remote tasks are a sorted list of strings.
- `GET /api/v1/health` — portal mode, worker reachability, `catalog_built_at` (Unix timestamp of the last successful build, or `null`), `catalog_refreshing` (boolean), and `catalog_error` (`null` or a short failure summary). A refresh error makes health `degraded`; the last good catalog remains available. Errors clear after a successful refresh. The models endpoint returns HTTP `503` if the first build fails.
- `GET /api/v1/models` — model catalog, modalities, outputs, availability, and curated `clinical_tags` for tumor-site discovery. Tags mean a model may be useful in that planning workflow; they do not imply tumor-segmentation capability or a clinical indication.
  Each task includes `structure_kind` (`fixed`, `variable`, `prompted`, `report`, or `unresolved`),
  `structure_names` (exact exported ROI names), and `structure_count` (an integer for `fixed`, null otherwise).
  `structure_source` and `structure_note` describe the label source and export semantics. `structures` remains
  the short human-readable output summary. Fixed names are listed in numeric label-index order after the writer's
  name normalization (ROI order inside an RTSTRUCT can differ); only structures present in a scan are written. Prompted outputs give the target ROI name;
  variable outputs describe their naming pattern in the note. Disabled tasks with unavailable exporter metadata
  are `unresolved`. A worker task missing from the generated catalog is disabled as `structure_metadata_unavailable`.
  The public contract is generated from exporter inputs in `catalog/structures.json`, without loading model weights.
  Regenerate it with `python scripts/build_structure_catalog.py --bundle-root MODEL_BUNDLE` using the worker's
  TotalSegmentator installation; add `--totalsegmentator-root PACKAGE_DIR` to select another metadata copy.
  Use `--check` for a nonzero exit and unified diff on drift. `--resolution-log PRIVATE_LOG.json` writes detailed
  provenance outside the public repository. This metadata check does not run models or establish clinical readiness.

- `GET /api/v1/jobs` — recent jobs.

## Viewer

The viewer reads the selected series on the portal host in every worker mode. It does not run a model.

| Route | Result |
| --- | --- |
| `GET /api/v1/jobs/{jid}/viewer` | Series geometry, window presets, all output RTSTRUCT ROI lists, and report availability. |
| `GET /api/v1/jobs/{jid}/viewer/slices/{index}.png` | An 8-bit grayscale PNG in the stored DICOM pixel grid. |
| `GET /api/v1/jobs/{jid}/viewer/contours/{index}` | ROI paths in that PNG's pixel coordinates. |
| `GET /api/v1/uploads/{uid}/series/{series_key}/viewer` | The same metadata envelope, with empty `structure_sets` and `report: false`. |
| `GET /api/v1/uploads/{uid}/series/{series_key}/viewer/slices/{index}.png` | An image from an assessed upload, before job submission. |
| `DELETE /api/v1/jobs/{jid}` | Deletes a terminal job, its images, output and logs; returns `{"id":"JOB_ID","deleted":true}`. |

Use the exact `series_key` returned by upload assessment (`SeriesInstanceUID|Modality`) and URL-encode it as one path segment. Upload viewing expires when the selected files move into a job or upload retention expires. Job IDs accept only letters, digits and hyphens; upload/session IDs cannot be used as job IDs. Identifiers are not filesystem paths or authorization credentials.

Example job metadata (illustrative values):

```json
{
  "series": {
    "modality": "CT", "rows": 512, "columns": 512, "slices": 96,
    "pixel_spacing": [0.8, 0.8], "slice_spacing": 2.0, "orientation": "axial",
    "plane": "axial", "column_direction": [1, 0, 0], "row_direction": [0, 1, 0],
    "window_presets": [
      {"id": "soft-tissue", "label": "Soft tissue", "center": 40, "width": 400},
      {"id": "lung", "label": "Lung", "center": -600, "width": 1500},
      {"id": "bone", "label": "Bone", "center": 400, "width": 1800},
      {"id": "brain", "label": "Brain", "center": 40, "width": 80}
    ],
    "default_window": "soft-tissue"
  },
  "structure_sets": [{
    "model": "total", "label": "Structures",
    "rois": [{"key": "a2b3c4d5e6f78901:1", "number": 1, "name": "Example ROI",
              "color": "#112233", "slices": [10, 11, 12]}]
  }],
  "report": false
}
```

`pixel_spacing` is `[row, column]` in mm. `slice_spacing` is the median distance between acquired planes (the spacing/thickness tag for a single image); it does not imply a regular stack. Missing slices are not inserted or interpolated. `orientation` is `axial`, `coronal` or `sagittal` when the normal is within 1 degree of that patient axis, otherwise `oblique`. `plane` repeats this derived plane label; the existing `orientation` field remains available. `column_direction` is the first `ImageOrientationPatient` triplet (increasing stored column/x); `row_direction` is the second triplet (increasing stored row/y). Both use DICOM patient LPS coordinates: positive x is left, positive y posterior, positive z head. These fields are identical in the job and upload viewer metadata.

PNG pixels and contour coordinates remain in the stored grid. The UI enables **Standard orientation** by default. For signed patient-axis permutations (each cosine component within `1e-4` of 0 or ±1), it flips/transposes the display: axial A at top/R at left; coronal H at top/R at left; sagittal H at top/A at left. Oblique edges, including rotation within an axial plane, stay as stored. Disabling the option restores stored order. Image, contours, clipping and inverse cursor mapping use one affine; the prompt-placement preview uses it for point/box overlays and clicks, while JSON coordinates retain stored column/row/slice indices; pan and anchored zoom use that same mapping. There is no physical-aspect correction or resampling. Four edge markers show patient R/L, A/P and H/F after the display transform; an oblique edge adds its second strongest direction when its normalized component is at least 0.3 (at most two letters).

All image routes accept either `?window=lung` (or another returned preset ID), or `?center=40&width=400`. Omission selects `default_window`. A valid first DICOM window takes precedence and appears as `dicom`. CT without that window defaults to soft tissue. MR/PT without a DICOM window use `auto`, computed once from the middle slice's 1st and 99th intensity percentiles. The selected window is applied after rescale slope/intercept, using DICOM LINEAR windowing; MONOCHROME1 is inverted. Width must be finite and at least 1; center must be finite. Do not combine a preset with numeric window parameters. PNG headers include `X-Slice-Index`, `X-Slice-Count`, `X-Columns` and `X-Rows`. Viewer JSON and images use `Cache-Control: no-store`.

Example contour response:

```json
{"index":10,"rois":[{"key":"a2b3c4d5e6f78901:1","paths":[[[27.0,3.0],[30.0,3.0],[30.0,15.0],[40.0,15.0],[40.0,18.0],[27.0,18.0]]]}]}
```

Coordinates are floats: `x = column`, `y = row`; `(0,0)` is the centre of the top-left stored pixel. Indices are zero based. Each path is a closed polygon; close its last point to its first when drawing. Use an even-odd fill rule for multiple paths belonging to one ROI, including `CLOSEDPLANAR_XOR`. `key` identifies output-relative RTSTRUCT path plus ROI number, so repeated numbers/names from different models remain distinct. `model` is the filename stem with `_RTSTRUCT` removed; `label` is `StructureSetLabel` or that model name. Colours use `ROIDisplayColor`, with a fixed palette when absent or invalid. RTSTRUCT discovery includes nested DICOM output files. ROIs with no matching source planes have `slices: []`; they are not overlaid on a different series. Unsupported non-closed contours return 422. Report-only and empty-prediction jobs have no structure sets. `report` means the completed job's JSON report is available through the existing report routes, including a report that records `not_generated`; it does not assert a successful diagnostic report.

Geometry and prompt contract:

- Sort slices by `dot(IPP, cross(IOP[0:3], IOP[3:6]))`, increasing along the slice normal. Do not sort by raw patient z, filename or `InstanceNumber`.
- Match a contour's `ContourImageSequence` SOP Instance UID first. A foreign UID never falls back to distance. Without image references, select the nearest acquired plane within half the smallest acquired spacing (capped by `SpacingBetweenSlices`, if present). Equidistant planes are ambiguous and omitted. This prevents a large gap from widening the matching tolerance. Conflicting references, off-plane referenced contours and nonplanar contours return 422.
- For patient point `P`, slice origin `IPP`, first IOP triplet `C`, second IOP triplet `R`, and DICOM spacing `[dr, dc]`, compute `x = dot(P - IPP, C) / dc` and `y = dot(P - IPP, R) / dr`. Each slice supplies its own origin. A different Frame of Reference is not treated as registration.
- The existing `/api/uploads/{uid}/preview?series_key=...&index=...` now shares this order and rendering. It retains its index-clamping behaviour and dimension headers. Its former raw-z ordering and per-image percentile window fallback have changed. New image routes reject out-of-range indices. Existing prompted runners receive `[column,row,slice]` on the selected series through `dicom_series_to_nifti.py` (GDCM/SimpleITK) and preserve those array indices. Prompts saved against the old preview can refer to a different slice in flipped/non-axial series; recreate those prompts. An irregular stack retains acquired slice indices, although NIfTI conversion represents its slice spacing with an average.

Missing, expired or already deleted data return 404, including completed jobs from before input retention was introduced. Unsupported/malformed geometry, multiframe images, colour images, decoding failures and invalid windows return 422. Compressed transfer syntaxes require an appropriate installed pydicom decoder. The selected series must have consistent dimensions, in-plane spacing, orientation and Frame of Reference, with distinct slice planes and SOP Instance UIDs. No resampling is performed. The separate prostate crop tool still has its own output package and input cleanup; it does not retain a selected viewer series.

Deletion returns 409 for queued, transferring, processing, retrieving or unknown/nonterminal states. SSH jobs, routed SSH tasks and worker-orchestrated Colab jobs use the existing remote cleanup helper before local removal. Remote failure returns 502 and keeps local data for a deletion retry. Delete and retry requests share a process-local lock; this does not add distributed job coordination. Retention still uses the existing cleanup schedule. Deletion cannot cancel a running job.

Series headers and mapped overlays use bounded in-process caches. File paths, modification times, sizes and inode changes invalidate cached entries; RTSTRUCT parsing is reused across metadata and contour requests. PNG requests decode one slice. Cache eviction, deletion and application restart can require rebuilding an index.

HTTP `202` means accepted. HTTP `409` from the result endpoint means the job is not complete. Validation failures use `400`/`422`.

## Security

The service handles medical data and does not implement user identity or tenant isolation. Do not expose it directly to the public internet. Put it behind authenticated TLS termination, restrict request size and network access, and use a separate data volume with an appropriate retention policy. Job identifiers are not authorization credentials.

## Resumable uploads

1. `POST /api/v1/upload-sessions` returns `session_id`, `max_request_bytes` (33554432) and `max_total_bytes`.
2. `PUT /api/v1/upload-sessions/{session_id}/files/{file_index}?name=CT.dcm&offset=0&size=TOTAL_BYTES` sends raw bytes. Use a stable numeric file index, original total size and basename. Continue at the returned `received` offset. A `409` response with `expected_offset` indicates where to resume.
3. `POST /api/v1/upload-sessions/{session_id}/assess` starts assessment; poll `GET /api/v1/upload-sessions/{session_id}` for `ready` or `failed`. A ready result contains the assessed `upload_id` and series list.
4. `POST /api/v1/uploads/{upload_id}/jobs` selects repeated `series_keys` and `models`, optional `prompt_json` and optional `postprocess`. This route supplies paired-series/prompt contracts and creates one job per series. The single-series legacy `/api/uploads/{upload_id}/start` remains available.

For one unambiguous series without prompts or pairing, `/api/v1/upload-sessions/{session_id}/jobs` accepts repeated `models`. It has a four-second catalog wait and can return `503` while the catalog warms. Use assessed uploads for paired/prompted tasks; the direct and session-job routes return HTTP `400` for them because they do not supply those inputs. All creation routes check the catalog's prompt and paired-input contract. Assessed submission requires a JSON object in `prompt_json` with the task's required points, box and/or text. Coordinates must be finite numbers within the selected series. Paired CT requires present, matching Study and Frame of Reference identifiers. Request chunks over 32 MiB return `413`; stalled bodies return `408`; a busy session lock returns `429`. Finalization is idempotent for the same operation/options.

## Prostate SBRT crop

`POST /api/v1/tools/prostate-sbrt-crop` accepts repeated multipart `files`, `margin_mm` (default 150) and `keep_uids` (default false). Upload axial CT with matching RTSTRUCTs. It returns a job ID and runs on CPU.

For resumable upload, use `POST /api/v1/upload-sessions/{session_id}/prostate-sbrt-crop` with the same crop options after all files arrive. Poll the job and download its result normally. The ZIP contains cropped CT, matching RTSTRUCTs and `crop_report.txt`.

Segmentation jobs can request `postprocess=prostate_sbrt_crop` only with local `total` inference. This retains the full RTSTRUCT and adds a crop package or an explicit no-selected-ROI note.

`POST /api/v1/jobs/{id}/retry` retries a failed job using retained input. Retention and cleanup can remove those inputs. There is no public cancellation endpoint.

## MRI sequence roles

Models declaring `input_roles` and `reference_role` use one job per selected primary
series. The primary supplies the reference role, usually axial T2W. Other selected
models without roles continue to use that primary alone. Use assessed uploads for
these models; direct single-series submission rejects them.

The batch endpoint accepts an optional `roles` form field containing a JSON object
keyed by primary `series_key`. Its values map companion role names to a series key
or selector object. For example (illustrative keys):

```json
{
  "T2_SERIES_KEY": {
    "adc": "ADC_SERIES_KEY",
    "hbv": {"series_key": "DWI_SERIES_KEY", "b_value": 1400}
  }
}
```

```bash
curl -fsS -X POST http://localhost:8080/api/v1/uploads/UPLOAD_ID/jobs \
  -F 'series_keys=T2_SERIES_KEY' -F 'models=MODEL_WITH_INPUT_ROLES' \
  -F 'roles={"T2_SERIES_KEY":{"adc":"ADC_SERIES_KEY","hbv":{"series_key":"DWI_SERIES_KEY","b_value":1400}}}'
```

All roles must have the declared modality and share a present StudyInstanceUID with
the primary. Each role uses a different series. FrameOfReferenceUID may differ;
the review UI warns about this and `input_roles` records both identifiers and
`frame_of_reference_differs`. These checks do not establish inter-sequence alignment.
Unknown primary/companion keys, undeclared roles, wrong modality/study or invalid
b-value selectors return HTTP 400 before any job is created.

An omitted role uses a suggestion only when the highest rank is unique. Automatic
ADC selection requires ADC evidence in ImageType. ADC/apparent-diffusion description
or protocol matches remain candidates but require an explicit choice.
An explicit ADC choice can select another same-study MR series. HBV candidates
require a documented b-value of at least 800 s/mm²; the highest b-value ranks first.
Each `input_roles` record includes `selection_method`: `image_type` for automatic
ADC, `b_value` for automatic HBV, or `user` for explicit choices and the selected
reference. The UI submits only choices the user changed; displaying a suggestion
does not convert it to a user choice.
A `null` role explicitly leaves that role unselected. Missing or tied required
roles skip only models requiring them, with reasons in `jobs[].skipped_models`
and `job.json`. If a primary has no runnable models, the whole batch returns 400
and retains the upload. The review page shows each skipped model before submission.

MR assessment exposes `image_type`, `is_adc`, `protocol_name`, acquisition time/date-time,
Study/Frame identifiers, orientation, spacing, dimensions, `instances`, `b_values`,
`b_value_sources`, `hbv_options` and `hbv_unavailable_reason`. Dimensions are
`[columns, rows, frames]`; an HBV option reports only the selected volume's frames.
Each instance records its SOP UID and per-frame diffusion values/geometry. Public
assessment omits source filenames. `b_value_source` is `(0018,9087)`;
`b_value_path` distinguishes top-level, shared and per-frame functional groups.
No vendor private tags or b-values inferred from descriptions are accepted.
If any instance/frame lacks a standard b-value, the series cannot supply HBV.
Enhanced MR containing several b-values inside one instance is assessed but cannot
be selected as HBV: frame extraction is not implemented. Separate single-b Enhanced
MR instances are supported. Known repeated slice positions and inconsistent volume
geometry are rejected; no frame rewriting, resampling or ADC intensity conversion
occurs during staging.
Any rejected or unreadable instance makes its series unavailable for every role,
including the reference. `assessment_errors` and `unavailable_reason` explain the
block. An unreadable DICOM whose series cannot be recovered blocks MR selection
for that upload. Sources are reassessed before manifest creation and staged MRI
is reassessed before queueing.

Jobs store the reference in `input/`, companions in `input_<role>/`, and the chosen
series and b-value/source in `input_roles`. For mixed-b series, only whole instances
matching the selected b-value are copied to `input_hbv/`. Shared companions are
copied before primary files are moved. `input_manifest` covers every role directory;
`input_identity` binds that manifest to the role selections. Partial directories or
changed selectors prevent dispatch, recovery and retry. Retry keeps the original
choices. SSH transport carries the role directories and job record, including for
single-series jobs. Result packages retain declared prostate lesion/provenance JSON
and spatial maps on positive and empty results. `MAX_RESULT_BYTES` bounds total
uncompressed result files (default 8 GiB). Companions are removed after
successful result packaging; reference images remain for the existing viewer until
deletion or retention expiry.

Multi-sequence jobs support local and SSH execution. Colab execution/fallback is
blocked for these jobs because the separate Colab runners retain a single-series
transport contract. Existing single-series Colab jobs are unchanged.

## CT draft reports

Select `nv_reason_ct_report` alone or with segmentation tasks. It runs last and produces a research AI draft requiring radiologist review. The catalog identifies `output: "report"`, license and limitations. Availability requires the report acceptance marker on the task's execution worker; segmentation orientation gates remain separate.

- `GET /api/v1/jobs/{jid}/report` returns JSON.
- `GET /api/v1/jobs/{jid}/report/html` and `GET /jobs/{jid}/report` return self-contained HTML.
- The existing result route returns a ZIP with `report/nv_reason_ct_report.{json,md,html}` and any RTSTRUCTs. Report-only jobs have no RTSTRUCT.

Job metadata includes `report_status` (`pending`, `running`, `generated`, `not_generated`, `failed`) and `report_reason`. Expected input/resource refusals produce a downloadable `not_generated` report. Worker crashes fail the job but retain completed contours for retry. Failed SSH retrieval retains the remote copy. Report routes return HTTP 409 until the job completes.

Inspection adds `cbct` and `cbct_reason` to each series. Single-series MR, PET and detected CBCT submissions with a report model are rejected with HTTP 400 through direct/legacy assessed routes. Batch submission applies the report only to non-CBCT CT series and rejects models applying to no selected series. Resumable direct-job submissions are accepted asynchronously; incompatible inputs then fail during staging before inference. HU and coverage refusals are recorded in the report.

JSON includes revision/hashes, package versions, geometry/HU checks, coverage mode and volumes, region decisions, raw output, filtered statements, unchecked headings, flags, crop QC/thumbnails, timings and peak model VRAM. DICOM demographic/identifier fields are excluded. ROI names and approximate contour volumes remain medical data.

When remote tasks are configured, health adds `remote_gpu_tasks`, `remote_worker` and `remote_worker_reachable`; failed reachability makes overall health `degraded` without changing local worker reachability. Routed catalog entries expose `execution_backend: "ssh"`. Unreachable remote preflight disables those tasks with `remote_worker_unreachable_or_preflight_failed`; missing task results use `remote_preflight_missing`. Catalog caching and refresh rules remain the same.


## Several series from one assessed upload

After assessment, submit repeated `series_keys` and `models` form fields to
`POST /api/v1/uploads/{uid}/jobs`. Optional fields are `prompt_json` and `postprocess`.
Models form one selection: each job receives the subset supporting its series modality.
For example, select `total`, `nv_segment_ctmr_ct_body`, `total_mr` and
`nv_segment_ctmr_mri_body` for a CT/MR upload. The report model runs only on non-CBCT CT.
PET models requiring CT use a copy of a series with matching, present Study and Frame of
Reference identifiers. The matching CT can also be selected for its own job.

The response is HTTP 202:

```json
{"batch_id":"batch-example","ids":["JOB_CT","JOB_MR"],"jobs":[
  {"id":"JOB_CT","series_key":"CT_KEY","models":["total","nv_segment_ctmr_ct_body"]},
  {"id":"JOB_MR","series_key":"MR_KEY","models":["total_mr","nv_segment_ctmr_mri_body"]}
]}
```

All selection validation precedes job creation. HTTP 400 identifies duplicate/unknown
series keys, unavailable models, models applying to no selected series, a series with
nothing to run, missing paired CT, or invalid prompts/postprocessing. Interactive models
require exactly one selected series because prompt coordinates refer to that series.
`postprocess=prostate_sbrt_crop` applies only to jobs supporting local `total` inference;
requesting it when no job qualifies returns 400. A rejected selection leaves the staged
upload available for correction. Successful submission moves the selected series into
separate jobs and removes the staged upload, including unselected series.

`GET /api/v1/jobs` and `GET /api/v1/jobs/{id}` expose `batch_id`, `batch_index` (one based),
`batch_size`, and `series` with `key`, `series_uid`, `modality`, `description`, `slices`,
`rows`, `columns`, `study_date`, and `cbct`. `report_status` is `pending` only in jobs
including the report model; it is null in the others. Older jobs can lack the new fields.
The recent-jobs list groups batches newest first and keeps each batch in series order. Its
100-job limit includes the whole last batch so sibling links remain available.
Use `GET /api/v1/jobs?batch_id=BATCH_ID` to retrieve an older batch for sibling navigation.
Jobs execute sequentially in FIFO acceptance order; each submission or manual retry
receives a new `queue_sequence`. A manual retry joins behind jobs already waiting.
A batch reserves consecutive queue positions in the submitted series order.
Waiting `detail` reports the number of earlier
jobs and updates as jobs finish: "Next in queue", "Queued behind 1 earlier job", or
"Queued behind N earlier jobs". Restart and watchdog recovery re-enter this dispatcher
in original creation order among waiting jobs without reordering existing reservations.
FIFO reservations and jobs awaiting input staging are live to the watchdog.
A task stall retry keeps the running job’s queue position.

The legacy `POST /api/uploads/{uid}/start` accepts one `series_key` with its existing
strict model/series validation and still returns HTTP 200 with `{"id":"JOB_ID"}`.
It uses the shared creation code and records batch fields with `batch_size=1`.
Direct and resumable single-series segmentation submissions also record size-one batches;
resumable submissions fill the series metadata after staging and before inference.
Input preparation uses a non-runnable `staging` status. New jobs record expected input
file names and sizes in `input_manifest`, including PET CT companions. Only complete
inputs become `queued`; interrupted assessed batches are restored to their upload when
safe, and their job records fail with an explicit preparation error. Incomplete retained
input cannot be retried; submit the upload again.
Jobs created by this version always carry an `input_manifest` key (null until staging completes), so a failed
or interrupted staging can never be retried or recovered from a partial input. Jobs created before input manifests
existed (no `input_manifest` key and no staging marker) were recorded only after their series was in place; they
keep the earlier rule: retry and recovery accept a nonempty input that holds one image series (and the recorded
slice count, when present).

## Job activity and task logs

`GET /api/v1/jobs/{id}` includes these optional progress fields in `job.json`:

| Field | Meaning |
| --- | --- |
| `current_task` | Model/task identifier currently being executed. |
| `stage`, `stage_key` | Human-readable stage and stable stage identifier. |
| `stage_done`, `stage_total` | Completed and total items in the current stage, or null when unknown. |
| `stage_started` | Unix timestamp when this stage began. |
| `task_started` | Unix timestamp when the current task attempt began. |
| `last_activity` | Unix timestamp of the latest observed log growth/progress. |
| `last_cpu_activity` | Unix timestamp of the latest observed increase in local process-tree CPU ticks; initialized at attempt start. |
| `task_remote` | True for an SSH-monitored task; activity is based on log growth. |
| `task_timings` | Object keyed by task with `started`, `attempt_started`, `attempts`, `status` (`running`, `done`, `failed`), and, after completion, `finished` and `duration` in seconds. Duration includes retries. |
| `completed_models` | Successfully completed tasks; retained outputs can be reused during recovery. |
| `stall_threshold`, `stall_warning` | Configured inactivity interval in seconds and warning state before recovery. |
| `stall_events` | Event array with `task`, human-readable `stage`, Unix `time` and `action` (`retry`, `failed`, `orphan_recovery`). |
| `task_pid`, `task_pid_started` | Internal local runner/SSH-client process identity for liveness checks; cleared after the attempt. |
| `recovery_count`, `interrupted_status` | Existing interrupted-job recovery count and prior state. |

`detail` remains available and names the actual task, stage and structure count when known. `updated` records metadata writes, including monitor polls; use the activity fields for liveness. Older jobs can lack these fields.

`GET /api/v1/jobs/{id}/log?task=total&lines=40` returns `{"task":"total","lines":["..."]}`. Omitting `task` selects `current_task`. Only a task in the job's `models` list is accepted. Invalid job/task identifiers, unknown jobs/tasks and symlinked task logs return 404. A task without a log returns an empty `lines` array. `lines` is clamped to 1–200 and the file read is capped at 256 KiB; the response can contain fewer lines. ANSI escape sequences are removed and carriage-return progress updates collapse to their latest rendition. The endpoint reads the log only and accepts no file paths.

`RTSEG_STALL_SECONDS` (default 1200) and `RTSEG_STALL_RETRIES` (default 1) configure automatic task recovery. A monitor publishes a warning for one polling interval before stopping a stalled task. A silent local task with increasing process-tree CPU time remains active. SSH monitoring currently relies on log growth. See [Operations](OPERATIONS.md#task-progress-and-stall-recovery) for retry and orphan-watchdog behavior.
## Viewer loading and contour batches

Job viewer metadata returns series geometry without waiting for the overlay index.
`contours_ready` is a boolean; `contours_state` is `building`, `ready` or `failed`.
Automatic metadata polling stops at `failed`. The viewer shows the error in its
overlay controls. **Retry viewer** reloads metadata with `retry_contours=true`,
which discards only failed background index entries and starts another build.
While building, `structure_sets` can be empty. Poll the metadata route to obtain
ROI names, colours and available slice indices after the build finishes. A failed
index leaves the image viewer available and supplies `contours_error`; contour
requests then return the geometry/metadata error (422).

Both `GET /api/v1/jobs/{jid}/viewer/contours/{index}` and
`GET /api/v1/jobs/{jid}/viewer/contours?start=K&count=N` return HTTP 202 with
`Retry-After: 1` and `{"contours_ready":false,"contours_state":"building"}`
while the single-flight index build is pending (including a busy build queue).
Retry these responses; they are not empty contour sets. A completed range returns
`{"start":K,"slices":{"K":[ROIS],"K+1":[]}}`, with one entry for every requested
zero-based index, including empty planes. `start` is required and nonnegative;
`count` is required and between 1 and 32 (422 otherwise). The entire requested
range must exist (404 otherwise). Range and single-slice entries are identical.
All these responses retain `Cache-Control: no-store`.

The browser retains decoded images in a per-viewer memory LRU (96 MiB estimated
RGBA plus PNG transfer bytes, at most 256 entries), and contour payloads in a
separate LRU (16 MiB estimated JavaScript payload size, at most 256 entries).
Loading another series or destroying the viewer clears both. There is no patient
image disk cache. Cache hits display immediately; other slice requests wait for a
60 ms scrolling pause. Up to four image/contour/metadata requests run at once,
with prefetch covering up to 16 slices in the scroll direction and six behind.
Contour prefetch uses range requests. Requests outside the new window are aborted.
The displayed image and contours always share an index; the label retains that
index and identifies a pending slice. Contours can arrive after the image, with a
small loading status. Window changes also retain the displayed window readout
until the replacement image arrives. Background builds and their caches remain
process-local and are invalidated by output/source file versions and job deletion;
restart or eviction can require rebuilding. This implementation does not persist
an overlay index to disk.
