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
| Orientation | One lossless orientation contract converts DICOM to NIfTI for every model and records the transform in a `.orientation.json` sidecar that is packaged with the result. |
| References | Every contour references a SOP instance of the selected series. |
| Physical bounds | Contours lie inside the physical extent of the referenced images. |
| Round trip | Contours are rasterized back onto the source grid and compared with the model mask: Dice ≥ 0.95 and plane tolerance 0.05 voxel. |
| Labels | Auxiliary TotalSegmentator labels are filtered against the task's label map. A label a model predicts outside its published label list is omitted and reported in the conversion notes (for example LNCTVSeg class 7). |

## Before a model is enabled

External tasks are enabled on a deployment only with recorded orientation evidence
for that installation. Report generation and some model families require a separate
acceptance marker on the worker that runs them. The catalog keeps disabled tasks
visible with the reason, for example `orientation_revalidation_pending` or
`requires_registered_pet_ct`.

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
restrictions are recorded in the [model ledger](external-models.md) and travel with
every result in its manifest.
