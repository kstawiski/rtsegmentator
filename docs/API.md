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
