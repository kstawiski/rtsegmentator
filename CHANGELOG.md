# Changelog

## Unreleased

- Prepare reconstruction-field fill for the 12 listed CT tasks on acquired slices
  only. Replace each slice's most frequent exterior value with air, preserve other
  low values, and log and warn about residual exterior low voxels. NV-Segment retains
  its base head-foot-axis detection and replacement of every recognised exterior voxel.
- Stage RTSTRUCTs outside result collection, validate them and write the export record
  before atomic publication. Clear task output before reruns while preserving completed
  results needed for report retry. Exclude empty models' RTSTRUCTs and orientation records
  from packages, keep the tables and maps those runs wrote, and record empty outcomes before checking for files in the manifest.
- Treat confirmed task-scoped remote file absence as empty retrieval while preserving
  transport failures. Report named empty inputs that reach either converter in conversion
  and manifest notes, one line per model, outside the job status line; earlier task exits
  may supply only model-level empty reporting. Clear a task's remote output without relying
  on the worker's login shell.
- Check saved study, series, frame, image and SOP class references
  and require every non-empty input structure. Validate contour borders using the accepted
  source geometry, including rounded oblique direction cosines, with unchanged tolerances.
- Require new acceptance markers for the two GTRC tasks and BiomedParse after source changes.

## 1.0.0 — 2026-10-25

First public release.

### Portal and API

- Upload of folders, extensionless DICOM files and ZIP archives through resumable
  sessions (requests up to 32 MiB); assessment groups images by study, series and
  modality and reports why a series cannot be used.
- Versioned automation API under `/api/v1` with OpenAPI documentation at
  `/api/docs`. One job per selected series; jobs run in a serialized GPU queue with
  progress, per-task time limits, retries from retained input and deletion.
- Local GPU and SSH GPU workers, with per-task routing of selected models to an SSH
  worker.
- Model catalog built from the installed TotalSegmentator runtime and committed
  external-model catalogs. Each task exposes its modality, licence, exact exported
  ROI names, structure kind and the reason it is disabled when it cannot run.
- Slice viewer for uploaded and completed series with patient orientation markers
  and RTSTRUCT overlays; prompted models take points, boxes or text from the viewer.
- MRI sequence roles (for example T2W, ADC and high-b-value DWI) selected from
  standard DICOM evidence for multi-sequence prostate models.
- NV-Reason-CT research draft reports for covered chest and abdomen CT, and a CPU
  crop tool for prostate SBRT CT and RTSTRUCT.

### RTSTRUCT export

- Masks are mapped directly onto the source image grid. Export checks source SOP
  references, physical bounds and a contour-to-mask round trip (Dice ≥ 0.95,
  plane tolerance 0.05 voxel) before an RTSTRUCT is released.
- A shared, lossless orientation contract converts DICOM to NIfTI for every model
  and restores predictions to the acquired grid; external tasks are enabled only
  with recorded orientation evidence.

### Result manifest and client

- Every segmentation result ZIP contains `rtsegmentator_manifest.json`
  (schema `rtsegmentator.result-manifest` 1.0, published in
  `schemas/result-manifest-1.0.schema.json`). It lists each file with its size and
  SHA-256, the source series, and per model the outcome, the exact ROI names
  written, the catalog label contract, and the catalog's licence and limitation
  text where recorded.
  `GET /api/v1/jobs/{id}/manifest` returns it for completed jobs.
- `rtseg_client.py`, a standard-library Python client and command-line tool,
  uploads series, submits jobs, waits, streams results to disk and verifies every
  package against its manifest and against the manifest the portal serves for the
  job. It never replays a job submission after an ambiguous failure, refuses
  redirects to another origin so proxy credentials stay with the portal, and
  rejects noncanonical or colliding member names. RTpipeline vendors the same file
  to import RTsegmentator results as a segmentation source.

### Distribution

- Citation metadata (`CITATION.cff`, `codemeta.json`, `.zenodo.json`), a
  documentation website built with MkDocs, and a generated model catalog page.
- The Apache-2.0 licence covers original RTsegmentator source code only. Model
  weights, runtimes and datasets are not distributed and remain under their own
  licences.
