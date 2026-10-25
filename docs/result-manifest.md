# Result manifest

Every segmentation result ZIP contains `rtsegmentator_manifest.json` at its root.
Software that consumes RTsegmentator output should read this file instead of
inferring meaning from file names. `GET /api/v1/jobs/{id}/manifest` returns the same
document for a completed job.

The structure is published as a JSON Schema in
[`schemas/result-manifest-1.0.schema.json`](https://github.com/kstawiski/rtsegmentator/blob/main/schemas/result-manifest-1.0.schema.json)
(schema identifier `rtsegmentator.result-manifest`, version `1.0`). Additions within
major version 1 are backward compatible; a change that removes or redefines a field
requires version 2.

## Fields

| Field | Content |
| --- | --- |
| `schema`, `schema_version` | `rtsegmentator.result-manifest` and `1.0`. |
| `generator` | `{"name": "RTsegmentator", "version": "1.0.0"}` |
| `created_utc` | When the package was written. |
| `job` | Job ID, creation time, batch ID and position, requested models and post-processing. |
| `source_series` | SeriesInstanceUID, modality, slice count, rows, columns and the CBCT flag of the segmented series. Descriptions and dates are omitted. |
| `input_roles` | For multi-sequence MRI jobs: the series used for each role and how it was selected. |
| `models` | One entry per requested model (below). |
| `files` | Every other member of the ZIP with its `size` in bytes and `sha256`. |
| `notes` | Conversion and post-processing notes, for example labels a model predicts but the exporter omits. |

Each `models` entry contains:

| Field | Content |
| --- | --- |
| `name` | Portal task name. |
| `outcome` | `rtstruct` (an RTSTRUCT was written), `no_structures_found` (the model ran and found nothing in this series), `report` (the task produces a document) or `no_rtstruct`. |
| `rtstruct` | For `rtstruct`: the member path, SOPInstanceUID, the exact ROI names in RTSTRUCT order, the referenced Frame of Reference UIDs and the referenced SeriesInstanceUIDs. |
| `structure_kind`, `catalog_structures` | The task's label contract from `catalog/structures.json`. For a *fixed* task, a catalog label missing from `roi_names` was not written: the model did not find it, or the exporter omitted it, in which case `notes` names it. |
| `sidecars` | Files that share the `<task>_RTSTRUCT.dcm.` prefix: orientation evidence, lesion tables, provenance records and spatial maps. |
| `license`, `code_license`, `weights_license`, `noncommercial`, `licensed`, `limitations`, `reference`, `description` | The catalog's descriptive text for the model, copied when present. |
| `requested` | `false` for an RTSTRUCT whose task was not in the job's model list. Absent otherwise. |

## Example

```json
{
  "schema": "rtsegmentator.result-manifest",
  "schema_version": "1.0",
  "generator": {"name": "RTsegmentator", "version": "1.0.0"},
  "created_utc": "2026-10-25T10:15:02Z",
  "job": {"id": "20261025-101203-...-1a2b3c4d", "created_utc": "2026-10-25T10:12:03Z",
          "batch_id": "batch-20261025-101203-...", "batch_index": 1, "batch_size": 1,
          "models": ["total", "lung_nodules"], "postprocess": []},
  "source_series": {"series_instance_uid": "1.2.826.0.1....", "modality": "CT",
                    "slices": 142, "rows": 512, "columns": 512, "cbct": false},
  "models": [
    {"name": "total", "structure_kind": "fixed", "catalog_structures": ["spleen", "kidney_right", "..."],
     "outcome": "rtstruct",
     "rtstruct": {"path": "total_RTSTRUCT.dcm", "sop_instance_uid": "2.25....",
                  "roi_names": ["spleen", "kidney_right", "liver", "..."], "roi_count": 61,
                  "frame_of_reference_uids": ["1.2.826.0.1...."],
                  "referenced_series_instance_uids": ["1.2.826.0.1...."]},
     "sidecars": ["total_RTSTRUCT.dcm.orientation.json"],
     "reference": "https://github.com/wasserth/TotalSegmentator"},
    {"name": "lung_nodules", "structure_kind": "fixed", "catalog_structures": ["lung", "lung_nodules"],
     "outcome": "no_structures_found"}
  ],
  "files": [
    {"path": "NO_STRUCTURES_FOUND.txt", "size": 112, "sha256": "..."},
    {"path": "total_RTSTRUCT.dcm", "size": 4183022, "sha256": "..."},
    {"path": "total_RTSTRUCT.dcm.orientation.json", "size": 1204, "sha256": "..."}
  ],
  "notes": []
}
```

## Verifying a package

```bash
python rtseg_client.py verify result.zip
```

Verification fails for a missing manifest, a missing or altered file, an unlisted
file, an unsafe member name or an unsupported schema version. RTpipeline runs the
same check, and additionally refuses an RTSTRUCT that does not reference the course's
planning CT series or whose ROI names differ from the manifest.

The manifest identifies DICOM series by UID. RTSTRUCT files themselves carry the
patient identifiers of their source images, so treat result packages as medical data.
