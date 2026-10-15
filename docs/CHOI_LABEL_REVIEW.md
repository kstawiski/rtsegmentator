# Choi TMLI label review

The public Choi release does not document the numeric order of its 18 regions.
The task stays disabled. Anatomical proposals do not establish clinical or
manuscript readiness.

## Generate evidence

Use the original CT in HU, a numeric Choi prediction, and separate binary
TotalSegmentator `total` masks of that same CT. A Python environment with
`nibabel`, `numpy`, `scipy` and `Pillow` is required.

```sh
python scripts/map-choi-labels.py --ct case.nii.gz \
  --prediction choi.nii.gz --total total_masks --output review
```

`review/anatomy.json` records all 18 labels, including absent ones: volume in mL,
centroids in RAS and LPS mm, cranio-caudal extent, nearest vertebral centroids,
vertebral z ranges, local spinal midline, side fractions, and distances to
anatomical masks. Input files are identified by SHA-256. The measured grids
must match after lossless axis permutations/flips; no registration is inferred.
CT spatial units must explicitly be millimetres. Masks with unspecified units
are accepted only when their grid exactly matches that CT; this inference is
recorded in the report. Other unit declarations fail. Oblique measurements use
the full affine. Review images sample true axial and coronal physical planes.

Patient left is negative x in RAS and positive x in DICOM LPS. DICOM LPS is
`(-RAS x, -RAS y, RAS z)`. Both review panels place patient right at image left.
A 3 mm band around the local midline is considered central; a side requires at
least 80% of label voxels on that side. Cord centroids define the local midline
when available, extended by vertebral centroids. Extrapolated segments are flagged.

`review.html` is self-contained, with 18 embedded PNGs and a review form.
The PNGs are also saved separately. Each pair passes through a voxel nearest
the centroid of the largest 6-neighbour component, with a close view of that
component. The report retains full-label extents and the three largest component
centroids/volumes. If the largest component holds less than 80% of label voxels and the two
largest component centroids are more than 80 mm apart, confidence is capped at
0.35. It is a sample of the contour, not a full-volume
review. The HTML displays explicit proposal rules and competing candidates.
Scores measure heuristic evidence strength; they are not calibrated probabilities.
No unique assignment is forced when multiple labels suggest the same region.

TotalSegmentator does not supply a separate obturator fossa, femoral head,
carina or internal/external iliac vessel split. The report labels these proxies
explicitly. Mediastinal and pelvic subdivisions require review in the original
volume. Missing labels require another fold or case. Compare fold 0 with the
five-fold ensemble, which previously suppressed several low-support labels.

## Apply a confirmed mapping

1. A clinician reviews all 18 labels, resolves ambiguities, and records the cases
   or review artifacts used. The form exports a JSON with reviewer, date,
   evidence identifiers and 18 distinct region assignments.
2. Replace `labels/choi-tmli.json` with that reviewed JSON. Keep private reviewer
   identities and case details in the private review archive; use a review role
   or identifier and opaque evidence identifiers in any public copy.
3. Run technical inference, RTSTRUCT conversion and orientation checks. Preflight
   requires the installed runtime, all five checkpoints, the confirmed mapping,
   and the existing technical-validation marker contract. The portal separately
   enforces its orientation and server configuration gates.
4. After manager confirmation, change the single `choi_tmli` catalog line in
   `app.py` to `enabled=True` and an accurate reviewed status. No other code
   change is needed for naming. This does not establish clinical readiness.

Both preflight and dispatch reject unconfirmed, incomplete, duplicate or
unattributed mappings. The inference runner can produce numeric output for
review without named RTSTRUCT export:

```sh
python run_choi_tmli.py --input case.nii.gz --model RELEASE_TRAINER_FOLDER \
  --work NEW_PRIVATE_WORK_DIRECTORY --output choi.nii.gz --folds 0
```

The runner adds the release-specific +1024 offset exactly once to the original
HU CT. Do not supply an already shifted CT. Omitting `--folds` uses all five
folds; test-time mirroring stays disabled. Checkpoints are read through links;
metadata and inference outputs are private to the supplied work directory.

Sources: [Choi et al., 2024](https://doi.org/10.1371/journal.pone.0299448),
[Zenodo release](https://zenodo.org/records/7839889). Checked 2026-10-15.
