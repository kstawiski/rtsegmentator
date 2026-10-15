# Choi TMLI geometry and label review

The task stays disabled. The axis variant and numeric mapping of the 18 regions
remain unconfirmed. CPU geometry tests and local review sheets do not establish
clinical or manuscript readiness.

## Voxel-space geometry finding

[Choi et al. (2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0299448)
report non-contrast CT, mean pixel size 1.113 × 1.113 mm, and 5 mm slices.
The [released model metadata](https://zenodo.org/records/7839889) record training
sizes of 219–275 × 512 × 512 in array z/y/x order, 1 × 1 × 1 mm spacing for all
20 cases, and `transpose_forward=[0,1,2]`. The full-resolution stage uses 1 mm.
These observations support dummy-spacing, voxel-space training. They do not
identify the authors' storage-axis directions. The mean pixel size also does
not establish each training case's exact physical spacing.

The previous runner supplied real CT spacing. nnU-Net therefore resampled a
1.5 mm slice stack to 1 mm: a 735-slice scan became about 1102 slices, compared
with about 221 slices at the paper's 5 mm scale. This is approximately a
fivefold cranio-caudal scale mismatch and a 1.113-fold in-plane scale mismatch.
The mismatch is verified; whether it explains the incoherent predictions
requires the manager's variant and fold comparisons.

## Input and inverse geometry contract

`run_choi_tmli.py` accepts a 3D CT in HU with explicit millimetre units.
It uses all eight voxel-face corners to cover the physical field of view,
including feet-first, prone and oblique acquisitions:

1. Resample HU linearly at 1.113 × 1.113 × 5 mm in physical LPS axes. Outside HU
   is −1024. Centre each target grid on the physical bounding box. Each in-plane
   dimension is at least 512; pad smaller fields and enlarge a dimension when
   its field exceeds 512 × 1.113 = 569.856 mm. Preserve spacing and never crop.
   The slice count is the ceiling of physical superior–inferior coverage / 5,
   allowing 0.0001 voxel tolerance for NIfTI float32 header rounding.
2. Specify three flip bits using `--axis-variant` or `CHOI_AXIS_VARIANT`.
   The bits refer to storage x/y/z relative to identity LPS directions, as read
   by SimpleITK from an HFS DICOM stack. `000` means L/P/S positive directions;
   `100` reverses x, `010` reverses y, `001` reverses z. All eight combinations
   are available. These are physical directions, independent of input storage.
3. Add 1024 once after HU resampling. Write the same array with 1 mm spacing,
   zero origin and identity **LPS** direction. Its NIfTI **RAS** affine is
   `diag(-1,-1,1,1)`. nnU-Net's SimpleITK reader thus sees the training header;
   it may crop background but performs no spacing change at full resolution.
4. Restore the prediction's true target geometry from `geometry.json`, then
   resample integer labels with nearest neighbour onto the original input grid.
   Export the source shape, affine and millimetre units for converter checks.

The default is `unconfirmed`; inference fails before creating a workspace unless
an explicit variant is supplied. The following selects an evaluation variant,
not a confirmed default:

```sh
python run_choi_tmli.py --input case.nii.gz --model RELEASE_TRAINER_FOLDER \
  --work NEW_PRIVATE_WORK_DIRECTORY --output choi.nii.gz \
  --axis-variant 000 --folds 0
```

Do not supply an already shifted CT. Omitting `--folds` uses all five folds.
The normal nnU-Net v1 predictor averages fold probabilities before argmax;
restored discrete label maps are never averaged. Test-time mirroring is disabled.
Checkpoints are read through links; mutable metadata and outputs use the work
directory. No confirmed default is selected by the evaluation script.

## Generate anatomical evidence

Use the original CT in HU, the restored numeric prediction, and separate binary
TotalSegmentator `total` masks on the same CT grid. Dependencies are `nibabel`,
`numpy`, `scipy` and `Pillow`.

```sh
python scripts/map-choi-labels.py --ct case.nii.gz \
  --prediction choi.nii.gz --total total_masks --output review
```

Add `--fold-agreement fold-agreement.json` when the manager script has generated
pairwise fold Dice. Agreement describes the individual folds, including on an
ensemble sheet; it is not a comparison against a clinical reference contour.
Both-absent labels have undefined Dice and one-absent labels have zero Dice.

`review.html` starts with an 18-label summary: volume, component count, largest
component share, its side and vertebral span, the top two candidate regions with
scores, and fold agreement when available. Per-label axial/coronal panels follow.
Raw measurements are inside collapsed details. `choi-review-template.csv` has
columns `label`, `proposed region`, `reviewer region`, `confidence`, `comment`;
reviewer fields start blank. Completing a CSV does not confirm the label map.

The largest 6-neighbour component determines the displayed side, span and
candidate-region evidence. A separate span covers the union of components
**individually at least 1 mL**. Full-label extents remain in `anatomy.json` so
fragments remain inspectable. Vertebral spans use nearest centroids and
intersecting physical z ranges, not vertebral boundary assignments. The report
retains the three largest component centroids and volumes. Widely separated
substantial components, competing rules and duplicate proposals lower confidence.
Scores are heuristic evidence strengths, not calibrated probabilities.

All measurements use the full NIfTI RAS affine. Patient left is negative RAS x,
positive LPS x. Side uses the local spinal midline with a 3 mm central band and
an 80% side threshold. Midline extrapolation is flagged. Review panels place
patient right at image left; true axial and coronal planes also handle obliquity.
The panels sample the largest component; inspect the full volume for extent.

Input files are identified by SHA-256. Measured grids must match after lossless
axis permutations/flips. A mask may inherit millimetre units only when its
unspecified-unit grid matches the explicitly millimetre CT. No registration is
inferred. TotalSegmentator lacks separate obturator fossa, femoral head, carina,
and internal/external iliac vessel masks; these proxies cannot settle pelvic
or mediastinal subdivisions. Missing labels require further folds or cases.

## Manager checks and mapping

The private manager script first compares fold 0 across eight variants using
largest-component share, with absent labels scored zero. It then evaluates all
five folds of the best two, ranks by mean inter-fold Dice followed by coherence,
and runs the best variant's five-fold probability ensemble. Runtime and sampled
process-group GPU memory, per-label volumes/components/spans and all pairwise
fold Dice are recorded. Exploratory coherence thresholds do not select a
clinically validated orientation or establish label identity.

After manager geometry review, a clinician must resolve all 18 numeric labels
and record review evidence. The HTML can export a complete, attributed mapping
JSON; `labels/choi-tmli.json` accepts no partial or duplicate assignments.
Preflight and dispatch reject unconfirmed mappings. Fresh inference, converter,
and orientation checks remain necessary. Keep patient data and reviewer identities
in private storage. Catalog enablement remains a separate manager decision after
the axis contract and label mapping are resolved.

Sources checked 2026-10-15: [PLOS ONE paper](https://doi.org/10.1371/journal.pone.0299448),
[Zenodo release](https://zenodo.org/records/7839889).
