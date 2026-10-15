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
```

Other routes:

- `GET /api/v1/health` — portal mode and worker reachability.
- `GET /api/v1/models` — model catalog, modalities, outputs, availability, and curated `clinical_tags` for tumor-site discovery. Tags mean a model may be useful in that planning workflow; they do not imply tumor-segmentation capability or a clinical indication.
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

`pixel_spacing` is `[row, column]` in mm. `slice_spacing` is the median distance between acquired planes (the spacing/thickness tag for a single image); it does not imply a regular stack. Missing slices are not inserted or interpolated. `orientation` is `axial`, `coronal` or `sagittal` when the normal is within 1 degree of that patient axis, otherwise `oblique`. This label does not change image rotation or laterality. Render the PNG without an implicit radiological flip, rotation, transpose or physical-aspect correction; apply any display transform equally to image and paths.

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
4. `POST /api/uploads/{upload_id}/start` selects `series_key`, repeated `models`, optional `prompt_json` and optional `postprocess`. This route supplies paired-series/prompt contracts.

For one unambiguous series without prompts or pairing, `/api/v1/upload-sessions/{session_id}/jobs` accepts repeated `models`. It has a four-second catalog wait and can return `503` while the catalog warms. Use assessed uploads for paired/prompted tasks; the direct and session-job routes do not supply those inputs. Request chunks over 32 MiB return `413`; stalled bodies return `408`; a busy session lock returns `429`. Finalization is idempotent for the same operation/options.

## Prostate SBRT crop

`POST /api/v1/tools/prostate-sbrt-crop` accepts repeated multipart `files`, `margin_mm` (default 150) and `keep_uids` (default false). Upload axial CT with matching RTSTRUCTs. It returns a job ID and runs on CPU.

For resumable upload, use `POST /api/v1/upload-sessions/{session_id}/prostate-sbrt-crop` with the same crop options after all files arrive. Poll the job and download its result normally. The ZIP contains cropped CT, matching RTSTRUCTs and `crop_report.txt`.

Segmentation jobs can request `postprocess=prostate_sbrt_crop` only with local `total` inference. This retains the full RTSTRUCT and adds a crop package or an explicit no-selected-ROI note.

`POST /api/v1/jobs/{id}/retry` retries a failed job using retained input. Retention and cleanup can remove those inputs. There is no public cancellation endpoint.

## CT draft reports

Select `nv_reason_ct_report` alone or with segmentation tasks. It runs last and produces a research AI draft requiring radiologist review. The catalog identifies `output: "report"`, license and limitations. Availability requires the report acceptance marker on the task's execution worker; segmentation orientation gates remain separate.

- `GET /api/v1/jobs/{jid}/report` returns JSON.
- `GET /api/v1/jobs/{jid}/report/html` and `GET /jobs/{jid}/report` return self-contained HTML.
- The existing result route returns a ZIP with `report/nv_reason_ct_report.{json,md,html}` and any RTSTRUCTs. Report-only jobs have no RTSTRUCT.

Job metadata includes `report_status` (`pending`, `running`, `generated`, `not_generated`, `failed`) and `report_reason`. Expected input/resource refusals produce a downloadable `not_generated` report. Worker crashes fail the job but retain completed contours for retry. Failed SSH retrieval retains the remote copy. Report routes return HTTP 409 until the job completes.

Inspection adds `cbct` and `cbct_reason` to each series. MR, PET and detected CBCT submissions are rejected with HTTP 400 through direct/assessed routes. Resumable direct-job submissions are accepted asynchronously; incompatible inputs then fail during staging before inference. HU and coverage refusals are recorded in the report.

JSON includes revision/hashes, package versions, geometry/HU checks, coverage mode and volumes, region decisions, raw output, filtered statements, unchecked headings, flags, crop QC/thumbnails, timings and peak model VRAM. DICOM demographic/identifier fields are excluded. ROI names and approximate contour volumes remain medical data.

When remote tasks are configured, health adds `remote_gpu_tasks`, `remote_worker` and `remote_worker_reachable`; failed reachability makes overall health `degraded` without changing local worker reachability. Routed catalog entries expose `execution_backend: "ssh"`. Unreachable remote preflight disables those tasks with `remote_worker_unreachable_or_preflight_failed`; missing task results use `remote_preflight_missing`. Catalog caching and refresh rules remain the same.
