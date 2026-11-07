# Technical checks

RTsegmentator separates two questions. Whether an exported RTSTRUCT is geometrically
consistent with its source images, and whether a model ran under its documented input
contract, are checked by software for every job. Whether a model's contours are
anatomically or clinically correct is not, and remains the user's and the model
authors' responsibility.

## Every export

| Check | Rule |
| --- | --- |
| Grid mapping | Masks are restored to the acquired DICOM grid before contours are built. A shifted, cropped, scaled or sheared prediction grid is refused; the affine tolerance is 0.001 mm. |
| Orientation | A lossless orientation contract records the DICOM-to-NIfTI transform. Each published RTSTRUCT includes a `.orientation.json` export sidecar in its result package. |
| References | Every contour references exactly one SOP instance of the selected series. The saved file's top-level FrameOfReferenceUID (when present), every ReferencedFrameOfReferenceSequence entry and every ROI's ReferencedFrameOfReferenceUID must match the source frame. The top-level StudyInstanceUID must equal the source study. Patient name and ID are not compared, because the writer saves them under a fixed Latin-1 character set and replaces characters outside it. Every referenced study must match the source StudyInstanceUID and every RTReferencedSeriesSequence entry must match the source SeriesInstanceUID. Hierarchy image references must belong to that selected series, and contour images must also appear in that hierarchy. Image SOP classes must match the source dataset SOPClassUID or file-meta MediaStorageSOPClassUID. Missing frame, study, series or hierarchy image reference sequences are refused. |
| ROI completeness | Both writers require exactly one saved ROI and contour set, with the expected name and a unique matching ROI number, for each non-empty input. Each must contain contour coordinates. Deliberately filtered out-of-contract labels remain reported omissions. |
| Physical bounds | Every saved vertex is mapped onto its referenced acquired image by solving with the accepted in-plane basis, including rounded direction cosines. Column and row coordinates must lie between −0.5 and N−0.5, the outer voxel edges, with 0.0001 voxel tolerance for numeric serialization and projection. Normal distance must remain within half the nearest adjacent plane spacing (SliceThickness for a single image), with the same 0.0001 voxel allowance. A single image without a positive declared thickness defines a plane, with 0.0001 mm numeric allowance; negative or non-finite thickness is refused. The round-trip rasterizer also checks every in-plane vertex before rounding or clipping; exterior fragments cannot be hidden by clipping. |
| ROI bounding box | Each ROI's patient-space coordinate minima and maxima must match the input mask's extrema within max(2.1 mm, 2.1 × largest NIfTI voxel spacing). This separate check retains its existing tolerance. |
| Round trip | Contours are rasterized back onto the source grid and compared with the model mask: Dice ≥ 0.95 and plane tolerance 0.05 voxel. |
| Labels | Auxiliary TotalSegmentator labels are filtered against the task's label map. A label a model predicts outside its published label list is omitted and reported in the conversion notes (for example LNCTVSeg class 7). |
| Empty structures | A named label or independent mask with zero foreground voxels is omitted, reported as zero voxels in structured conversion notes, and emits a warning. Successful exports also include omissions in validation JSON. Log collection lists them per model as not present in the prediction, in `CONVERSION_NOTES.txt` and the result manifest's schema 1.0 `notes` array; the job status line names only structures dropped from a non-empty mask. Other non-empty structures still export successfully. Entirely empty inputs that reach either converter report each named empty input in the log and manifest and create no RTSTRUCT. Tasks that exit earlier have only model-level empty reporting unless their runner supplies a separate note. |

Both writers remove any existing final RTSTRUCT and its orientation sidecar before processing. They save
in a private staging directory beside the output directory, require the same filesystem, validate that file,
write the export record, then publish the RTSTRUCT with an atomic rename. Handled failures remove the
staging file and export sidecar and raise the original failure. A killed converter can leave a staging
directory outside the result, manifest and report readers. Before each task that needs to run, the portal
clears that task's output; completed tasks with a retained result are skipped and preserved for report retry.
Finalization excludes the RTSTRUCT and its orientation record for models declared empty, keeps the lesion tables, maps and provenance that run wrote, and the manifest
records their outcome as `no_structures_found`. Task-scoped remote retrieval treats a confirmed absence of
matching files as empty; SSH and scp errors still fail retrieval.

## CT reconstruction-field fill

For the 12 CT tasks listed under [CT input preparation](../EXTERNAL_MODELS.md#ct-input-preparation),
`run_task.py` prepares the job-local NIfTI immediately after DICOM conversion and before the runner reads it.
The selected source-series headers must all identify CT; MR and PET inputs to the same prompted task IDs bypass it.
The shared `ct_padding.py` function finds border-connected values below −1500 HU that match the exterior
of a large, roughly centred reconstruction circle on acquired slices only (array axis 2 under the orientation
contract, including oblique series). In each slice, only exterior voxels equal to that slice's most frequent
exterior value become −1024 HU. Different low values, including in-field artefacts next to fill, remain
unchanged. Low regions that fail the circle test also remain. The job log records `ct_padding_replaced_voxels`,
`ct_padding_residual_low_voxels` (recognised exterior voxels below −1500 HU left after replacement), and
`ct_padding_fill_hu` (the replacement value, −1024 HU), including zero counts when nothing is recognised. A positive residual count emits a warning.

The preparation preserves the acquired grid, affine and orientation. Source DICOM files, the orientation
sidecar and RTSTRUCT references are unchanged. With no recognised exterior fill, the NIfTI is not written
and remains byte-identical. Unlisted CT tasks and every MR and PET task bypass this extension. NV-Segment
CT modes retain their base runner-local rule: inspect only the axis closest to head-foot and replace every recognised exterior voxel, through a separate function with shared writing code.

This is intensity preparation, not an anatomical check or a general HU refusal gate. Fully padded slices
and unexplained interior low values are not repaired. Segmentation tasks have no separate refusal for a mostly padded volume. The existing NV-Reason-CT
report HU refusal rules remain unchanged. Passing the export checks does not detect missing model structures caused by intensity scaling.
Model inference and acceptance on the deployed GPUs are required before release.

## Before a model is enabled

External tasks are enabled on a deployment only with recorded orientation evidence
for that installation. Report generation and some model families require a separate
acceptance marker on the worker that runs them. The catalog keeps disabled tasks
visible with the reason, for example `orientation_revalidation_pending` or
`requires_registered_pet_ct`. GTRC's two tasks and BiomedParse bind acceptance markers to dispatch and
converter source hashes; this release requires new acceptance evidence before those tasks can be enabled.

## Continuous integration

The test suite runs on every push and pull request with synthetic DICOM data only. It
covers upload and assessment, job lifecycle and recovery, RTSTRUCT conversion and
geometry checks, viewer geometry on axial, coronal, sagittal and oblique synthetic
series, catalog generation, the result manifest and the client. Model inference is
not part of continuous integration: it needs licensed weights and a GPU. Browser UI
checks run separately with `tests/ui_browser_server.py`.

## What is not established

Passing these checks does not show that a model segments a given anatomy, disease,
scanner or protocol correctly. Model-specific evidence, known failure modes and
restrictions are recorded in the [model ledger](external-models.md). Each result's
manifest copies the catalog's licence and limitation fields where they are recorded;
the [model catalog](models.md) shows which tasks lack them.
