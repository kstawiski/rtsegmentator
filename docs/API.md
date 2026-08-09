# Automation API

Swagger UI is available at `/api/docs`; the OpenAPI document is
`/api/openapi.json`. Set `RTSEG_API_KEY` to require an `X-API-Key` header on all
versioned `/api/v1` endpoints. Put the service behind TLS and network access
control whenever it handles identifiable data.

## Recommended two-stage submission

1. `POST /api/v1/uploads` with one or more `files` parts. Files may be DICOM or
   ZIP archives. The response reports rejected files and every detected CT, MR
   or PET series. No data has reached the GPU worker yet.
2. `POST /api/v1/uploads/{upload_id}/jobs` with one returned `series_key`, one
   or more repeated `models` fields, and optional `prompt_json`.
3. Poll `GET /api/v1/jobs/{job_id}` until `complete` or `failed`.
4. Download `GET /api/v1/jobs/{job_id}/result`.

```bash
base=http://localhost:8080
auth="X-API-Key: ${RTSEG_API_KEY}"

curl -sS -H "$auth" -F 'files=@study.zip' \
  "$base/api/v1/uploads" > assessment.json

curl -sS -H "$auth" \
  -F 'series_key=SERIES_KEY' -F 'models=total' \
  "$base/api/v1/uploads/UPLOAD_ID/jobs"

curl -sS -H "$auth" "$base/api/v1/jobs/JOB_ID"
curl -fL -H "$auth" -o result.zip "$base/api/v1/jobs/JOB_ID/result"
```

`GET /api/v1/models` returns supported modality/modalities, structures,
description, prompt type, upstream reference, availability and validation
status for every model. Clients must not assume disabled models can be run.

## Prompted models

Coordinates are `[column, row, zero_based_slice]` in the assessed series.

```json
{
  "positive_points": [[210, 164, 72]],
  "negative_points": [[180, 140, 72]],
  "box": [[190, 145, 72], [240, 190, 72]],
  "text": "lymph node"
}
```

- PAM requires a positive point or same-slice box and propagates in 3D.
- SAT3D requires at least one positive point and accepts negative points.
- SAM-Med2D is slice-wise; prompt every slice that should be contoured.

Send the JSON as the `prompt_json` form field. Model and coordinate validation
occurs before any transfer or inference.

## Direct submission

`POST /api/v1/jobs` accepts an already isolated, single DICOM series through
repeated `files` parts plus repeated `models` fields. Mixed uploads are rejected.
Prompted and paired PET/CT models require the two-stage API, which is safer for
all arbitrary archives.

`GET /api/v1/health` is a liveness endpoint. A successful response does not
prove that every model weight or GPU path is healthy; use a locally validated
smoke series after installation.
