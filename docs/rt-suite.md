# RTsegmentator and RTpipeline

RTsegmentator and [RTpipeline](https://github.com/kstawiski/rtpipeline) form one
toolkit for radiotherapy research data:

| Tool | Role |
| --- | --- |
| **RTsegmentator** | Runs segmentation models on single DICOM series and returns checked RTSTRUCTs, interactively or through the API. |
| **RTpipeline** | Organizes raw treatment-planning exports into patient courses and computes dose-volume metrics, radiomics, robustness and quality-control tables for whole cohorts. |

RTpipeline already runs TotalSegmentator and locally installed nnU-Net models. The
portal adds the rest of its catalog (nodal, prostate, PET, brain, prompted and other
models) without installing each model's runtime next to the pipeline, and lets one GPU
host serve several projects.

```text
 TPS export ──► RTpipeline organize ──► planning CT per course
                                             │
                                             ▼  upload (rtseg_client)
                                       RTsegmentator ──► RTSTRUCT + manifest
                                             │
                     verify checksums, series and ROI inventory
                                             ▼
               Segmentation_CustomModels/rtseg_<model>/rtstruct.dcm
                                             │
                     DVH · radiomics · robustness · QC tables
```

## Online: RTpipeline calls the portal

Add to the RTpipeline project configuration:

```yaml
rtsegmentator:
  enabled: true
  url: "https://rtseg.example.org"
  models: ["total", "lung_nodules"]
  headers_env: "RTSEG_CLIENT_HEADERS"   # proxy credentials, read from the environment
```

The pipeline's `segmentation_custom` stage then uploads each course's contracted
planning CT, waits for the job, verifies the package and writes each model's RTSTRUCT
to `Segmentation_CustomModels/rtseg_<model>/`. The directory's `manifest.json` fixes
the expected ROI inventory to the names the portal wrote and records the portal URL,
job ID, RTsegmentator version, model licence and checksums. Results that match the
current planning CT and portal are reused on later runs. Output tables label these
structures with the segmentation source `CustomModel:rtseg_<model>`.

## Offline: import downloaded packages

When the pipeline host cannot reach the portal, download results from the portal UI
or with the client and import them:

```bash
rtpipeline rtsegmentator-import --course-dir Output/PATIENT/COURSE results/JOB_ID.zip
```

The import applies the same checks and writes the same files as the online path.

## Checks at the boundary

A package is imported only when every check passes; otherwise nothing is written:

1. The ZIP verifies against its manifest (no missing, altered or unlisted file).
2. The manifest's source series is the course's planning CT series.
3. Each RTSTRUCT references that series and its ROI names equal the manifest's.
4. ROI names are present and unique.

A model that ran and found nothing (`no_structures_found`) removes any earlier import
for that model instead of leaving stale structures.
