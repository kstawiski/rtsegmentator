# Prostate MRI runners

These four integrations use the released inference methods below. GPU inference and reference-case acceptance are pending. Synthetic tests establish software contracts only.

| Task | Required series, in network channel order | Output | Weight terms |
|---|---|---|---|
| `picai_nnunet_supervised_bpmri` | T2W, ADC, HBV | Zero to five csPCa candidates | CC BY-NC-SA 4.0 |
| `picai_nnunet_semisupervised_bpmri` | T2W, ADC, HBV | Zero to five csPCa candidates | CC BY-NC-SA 4.0 |
| `umu_prostate_zones_urethra_t2` | T2W | PZ, CZ, TZ, AFS, constructed 6-mm urethra | Weight licence not stated; repository MIT |
| `prostate158_tumor_bpmri` | T2W, ADC, DWI (`hbv` role) | Largest tumour component | CC BY 4.0 |

Every transported job includes `job.json`, including single-series Umeå jobs. The multi-series tasks require `job.json.input_roles` with `t2w`, `adc` and `hbv`, and a complete `input_manifest`. T2 is in `input/`; ADC and HBV are in `input_adc/` and `input_hbv/`. The runner checks one MR volume per role, matching StudyInstanceUID, selected series identity, slice geometry and recorded HBV b-value. Different frame identifiers are recorded as a warning. The input adapter owns b-value selection and documented vendor-tag decoding. Enhanced MR must first be staged as supported scalar slices.

A series with any rejected image instance is unavailable, including as the reference. Assessment is checked before a manifest is created and again on staged inputs. Automatic ADC selection requires ImageType ADC evidence. Description and protocol matches require an explicit user choice; the role record retains the selection method.

Each role passes through `dicom_series_to_nifti.py`. The adapter undoes that converter's lossless storage permutation before applying the model's preprocessing. ADC values retain DICOM rescale handling; the runner does not infer a change of ADC units. PI-CAI's fixed ADC normalization therefore requires a P3 check of the scanner's stored units. No generic registration is added. Prostate158 resamples ADC and HBV to the T2 physical grid before its published transforms.

## PI-CAI baselines

The [supervised wrapper](https://github.com/DIAGNijmegen/picai_nnunet_gc_algorithm/tree/fbea30efc5f57adab0445e554f3de3c0a2dca64a) and [semi-supervised wrapper](https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm/tree/6129f7e0fe3fc3994064672aa2c7f629584004a3) are baseline checkpoints. Their model cards do not report performance. They are not the final consortium system.

`run_picai.py` uses `picai-prep==2.0.0` (`Sample.preprocess`), nnU-Net 1.7.0, the authors' `nnUNetTrainerV2_Loss_FL_and_CE_checkpoints`, tasks 2201/2203, folds 0–4 and `model_best`. It retains each release's plans and ADC constants. TTA and patch overlap use the upstream defaults. Custom trainers and model folders are mirrored into the job; the immutable environment and weights are not modified.

The foreground softmax is restored with `picai-baseline==0.5.2` to T2 before the published central crop/pad to `(20,384,384)` **voxels**, followed by crop/pad back to the original array size. `report-guided-annotation==0.2.4` performs dynamic extraction: maximum five candidates, peak divisor 2.5, 26-connectivity, removal of components with 10 voxels or fewer, one-voxel adjacency suppression, and a remaining-peak stopping value of 0.01. The indexed component output keeps equal-score lesions distinct. RGA can emit a zero-score component after removing tiny peaks; the adapter excludes that component from ROIs and preserves the detection map exactly.

ROI names such as `csPCa_L01_p0.73` contain a rounded likelihood. `<task>_RTSTRUCT.dcm.lesions.json` retains the full-precision candidate and case likelihoods, voxel counts, mm³/ml volumes, extraction parameters, input roles, package versions, source commit and weight SHA-256 hashes. Case likelihood is the maximum detection-map value. The probability and detection NIfTIs remain separate artifacts. No candidates is a completed empty result using the portal's existing `empty prediction: no RTSTRUCT created` marker; no contour is fabricated.

## Umeå zones and urethra

The [pinned Umeå source](https://github.com/UMU-DDI/prostate-segmentation/tree/eacfe6436635f43b74dadf4d31e8424a856014e5) specifies labels 1=PZ, 2=CZ, 3=TZ, 4=AFS, 5=urethra. `run_umu_prostate.py` checks this mapping against the native Dataset077 export, runs all five `checkpoint_best.pth` folds with nnU-Net v2 2.2, and requests probabilities. It uses the export's NRRD/SimpleITK convention without additional RAS reorientation.

Postprocessing selects background/prostate components, fits probability-weighted quadratic x/y centrelines, draws a 3-mm-radius circle on each slice and clips it to the gland. It retains each zone's largest 18-connected component and fills unassigned gland voxels by physical distance to the remaining zones. The published circle assumes equal in-plane spacing; unequal x/y spacing is rejected. Slice spacing may be anisotropic. Zone filling calls the unchanged pinned upstream function, including its distance transform for absent zones. A nonempty gland with fewer than three nonzero urethra-probability slices raises an error. Empty glands remain empty.

`Urethra_6mm_model` is a constructed 6-mm-diameter structure, not the lumen. The publication reports median centreline distances of **2.8 and 2.9 mm** against two raters, compared with **3.6 mm** between raters. The study included 40 double-read test cases and 55 external cases. These are published results, not local measurements. [Publication](https://pubmed.ncbi.nlm.nih.gov/42011184/).

## Prostate158 tumour

`run_prostate158.py` follows the [original source](https://github.com/kbressem/prostate158/tree/8187638887409c95ba0b091f249a40c0822c89e0): three-channel MONAI UNet, widths 16/32/64/128/256/512, five stride-2 stages, four residual units, PReLU, batch normalization and dropout 0.15. The author README states that sequences in each study share orientation, direction and spacing. Before the published transforms, ADC and HBV are resampled onto the T2 grid with SimpleITK, an identity transform, linear interpolation, zero outside the source field of view and float32 output. T2 is unchanged. This reproduces the dataset input convention; it does not estimate registration or correct patient motion. Provenance records the source and output grids, interpolation and transform in `input_grid_resampling`. Deterministic transforms are 0.5-mm spacing, RAS orientation, intensity scaling and normalization per channel. Inference uses 96³ windows, batch size 4 and overlap 0.5. Argmax and the largest 26-connected foreground component precede nearest-neighbour restoration to T2. This preserves the release's single-component output.

The [author model description](https://prostate158.grand-challenge.org/) reports tumour Dice **0.453/0.398** against two readers on **19 test cases**. [Weights and CC BY 4.0 terms](https://zenodo.org/records/7040585).

Inference follows the author's evaluator precision policy: CUDA autocast is enabled
for Torch ≥1.6, with sliding-window aggregation on the input device. Provenance
records the policy, autocast state and aggregation device. No local comparison of
model predictions or GPU memory use has been performed.

## Bundle installation

All paths below, including `runtime-envs/`, are relative to `RTSEG_MODEL_ROOT`. These environments are model-bundle assets, as for NV-Segment. `RTSEG_RUNTIME_ROOT` selects the separate worker `envs/` assets for other tasks; it does not relocate prostate environments. Dispatch and preflight use the same model-root paths. The worker runtime image is read-only.

```text
weights/picai_supervised/results/nnUNet/3d_fullres/Task2201_picai_baseline/
  nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/
    plans.pkl
    fold_0..fold_4/model_best.model[.pkl]
weights/picai_semi_supervised/results/nnUNet/3d_fullres/Task2203_picai_baseline/
  nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/
    plans.pkl
    fold_0..fold_4/model_best.model[.pkl]
weights/umu_prostate_zones/Dataset077_ProstateZones/
  nnUNetTrainer__nnUNetPlans__3d_fullres/
    dataset.json
    plans.json
    fold_0..fold_4/checkpoint_best.pth
weights/prostate158_tumor/tumor.pt
sources/picai_supervised/
sources/picai_semi_supervised/
sources/umu_prostate_zones/
sources/prostate158_tumor/
runtime-envs/picai/
runtime-envs/umu-nnunetv2/
runtime-envs/prostate158/
```

Keep the source licences, the PI-CAI weight licence and the full acquisition manifests with the bundles. Extract the Umeå native export under its weight directory, preserving its internal paths. The structure generator reads the Umeå `dataset.json` at the final path; it never reads checkpoint tensors. `model_preflight.py` requires each task's existing-style `validation/<task>.json` and reports orientation validation separately. This implementation creates neither acceptance record.

The environment lock files and build evidence are delivered with the implementation report. Python 3.11 and Torch 2.5.1 CUDA 12.4 replace the older Torch/CUDA versions to target both Pascal and Hopper. Import checks and compiled architecture metadata do not establish that either GPU completes inference. MONAI 0.9.1 is the stable replacement for Prostate158's development snapshot; compatibility must be checked against the released checkpoint and public engineering inputs in P3.

## P3 engineering checks

Use the author-shipped PI-CAI public example converted to synthetic-identifiable DICOM series, retaining input intensities and physical geometry. Check each role manifest, HBV selector, ADC rescale/units, all five folds, upstream probability/detection results, lesion IDs and full-precision scores, negative results, T2 geometry, and RTSTRUCT SOP/frame references. Run both supported GPUs separately and record device, wheel architecture targets, peak memory and runtime. Use synthetic inputs for Umeå/Prostate158 geometry and preprocessing until an authorised single case is available. No cohort download or accuracy estimate is part of these checks.

Result downloads retain the `.dcm.lesions.json`, `.dcm.provenance.json` and map artifacts, including on completed empty runs. Their names share the task's RTSTRUCT prefix for task-specific transport. Top-level artifacts use a per-task allowlist; the complete download has a configurable uncompressed size limit (`MAX_RESULT_BYTES`, default 8 GiB).
