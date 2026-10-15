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
