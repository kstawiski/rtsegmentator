# Prostate MRI runners

These four integrations use the released inference methods below. GPU inference and reference-case acceptance are pending. Synthetic tests establish software contracts only.

| Task | Required series, in network channel order | Output | Weight terms |
|---|---|---|---|
| `picai_nnunet_supervised_bpmri` | T2W, ADC, HBV | Zero to five csPCa candidates | CC BY-NC-SA 4.0 |
| `picai_nnunet_semisupervised_bpmri` | T2W, ADC, HBV | Zero to five csPCa candidates | CC BY-NC-SA 4.0 |
| `umu_prostate_zones_urethra_t2` | T2W | PZ, CZ, TZ, AFS, constructed 6-mm urethra | Weight licence not stated; repository MIT |
| `prostate158_tumor_bpmri` | T2W, ADC, DWI (`hbv` role) | Largest tumour component | CC BY 4.0 |

Every transported job includes `job.json`, including single-series Umeå jobs. The multi-series tasks require `job.json.input_roles` with `t2w`, `adc` and `hbv`, and a complete `input_manifest`. T2 is in `input/`; ADC and HBV are in `input_adc/` and `input_hbv/`. The runner checks one MR volume per role, matching StudyInstanceUID, selected series identity, slice geometry and recorded HBV b-value. Different frame identifiers are recorded as a warning. The input adapter owns b-value selection and documented vendor-tag decoding. Enhanced MR must first be staged as supported scalar slices.

A series with any rejected image instance is unavailable, including as the reference. An image identified by its DICOM preamble, image SOP class or pixel-data tag that has no recoverable series identity blocks every MR series in that upload; unrelated non-DICOM files remain ignored. Assessment is checked before a manifest is created and again on staged inputs. Automatic ADC selection requires ImageType ADC evidence. Description and protocol matches require an explicit user choice; the role record retains the selection method.

Each role passes through `dicom_series_to_nifti.py`. The adapter undoes that converter's lossless storage permutation before applying the model's preprocessing. ADC values retain DICOM rescale handling; the runner does not infer a change of ADC units. PI-CAI's fixed ADC normalization therefore requires a P3 check of the scanner's stored units. No generic registration is added. Prostate158 resamples ADC and HBV to the T2 physical grid before its published transforms.

## PI-CAI baselines

The [supervised wrapper](https://github.com/DIAGNijmegen/picai_nnunet_gc_algorithm/tree/fbea30efc5f57adab0445e554f3de3c0a2dca64a) and [semi-supervised wrapper](https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm/tree/6129f7e0fe3fc3994064672aa2c7f629584004a3) are baseline checkpoints. Their model cards do not report performance. They are not the final consortium system.

`run_picai.py` uses `picai-prep==2.0.0` (`Sample.preprocess`), nnU-Net 1.7.0, the authors' `nnUNetTrainerV2_Loss_FL_and_CE_checkpoints`, tasks 2201/2203, folds 0–4 and `model_best`. It retains each release's plans and ADC constants. TTA and patch overlap use the upstream defaults. Custom trainers and model folders are mirrored into the job; the immutable environment and weights are not modified.

`picai_predict.py` calls the same nnU-Net prediction entry point with unchanged prediction arguments. After `load_model_and_checkpoint_files` restores and initializes the trainer, it sets `torch.backends.cudnn.deterministic=True` and `torch.backends.cudnn.benchmark=False`. This overrides the training-time flag stored with the trainer. The lesion sidecar records these settings under `provenance.cudnn`. CPU tests check the ordering and arguments; fresh reference and DICOM GPU acceptance must establish numerical repeatability. No shared reproduction tolerance is changed.

The supervised task remains explicitly blocked: the lowest-scoring candidate in the authors' public example contained 390 voxels versus 1,132 in the reference output, outside the reproduction tolerance. The semi-supervised task is assessed separately. Deviation records cannot enable either task.

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

Keep the source licences, the PI-CAI weight licence and the full acquisition manifests with the bundles. Extract the Umeå native export under its weight directory, preserving its internal paths. The structure generator reads the Umeå `dataset.json` at the final path; it never reads checkpoint tensors. `model_preflight.py` validates each task's `validation/<task>.json` and `orientation-validation/<task>.json` contents. Acceptance must identify the task, pass both reference and DICOM modes, record the required checks and bound reference evidence, and match the pinned installed source commit and every installed checkpoint, plans and metadata hash. Source identity comes from the checkout HEAD or the acquisition `RTSEG_MANIFEST.json` when Git metadata is absent. Failed checks, deviations, incomplete records and stale hashes block selection. Orientation must identify the same task and pass. Missing preflight also blocks selection. This implementation creates neither acceptance record.

The environment lock files and build evidence are delivered with the implementation report. Python 3.11 and Torch 2.5.1 CUDA 12.4 replace the older Torch/CUDA versions to target both Pascal and Hopper. Import checks and compiled architecture metadata do not establish that either GPU completes inference. MONAI 0.9.1 is the stable replacement for Prostate158's development snapshot; compatibility must be checked against the released checkpoint and public engineering inputs in P3.

## P3 engineering checks

Use the author-shipped PI-CAI public example converted to synthetic-identifiable DICOM series, retaining input intensities and physical geometry. Check each role manifest, HBV selector, ADC rescale/units, all five folds, upstream probability/detection results, lesion IDs and full-precision scores, negative results, T2 geometry, and RTSTRUCT SOP/frame references. Run both supported GPUs separately and record device, wheel architecture targets, peak memory and runtime. Use synthetic inputs for Umeå/Prostate158 geometry and preprocessing until an authorised single case is available. No cohort download or accuracy estimate is part of these checks.

Result downloads retain the `.dcm.lesions.json`, `.dcm.provenance.json` and map artifacts, including on completed empty runs. Their names share the task's RTSTRUCT prefix for task-specific transport. Top-level artifacts use a per-task allowlist; the complete download has a configurable uncompressed size limit (`MAX_RESULT_BYTES`, default 8 GiB).

## NCI/NVIDIA prostate lesion bpMRI

`nci_monai_prostate_lesion_bpmri` requires T2W, ADC and one high-b DWI volume.
High-b selection accepts the standard diffusion b-value tag or Siemens B_value
with the `SIEMENS MR HEADER` private creator on every selected slice. The
ProstateX-0004 scanner-computed b=1400 series uses this Siemens encoding.
Explicit selection remains available. Series descriptions never supply b-values.
It exports `NCI_Prostate` and one `NCI_L01_PIRADS4`-style ROI per lesion. The
PI-RADS class is a model output. Research use only: the licence excludes
diagnosis and treatment. Owner accepted these terms on 2026-10-22.

The [external study](https://pubmed.ncbi.nlm.nih.gov/40753327/) evaluated 151
patients. Examination-level csPCa AUROC was 0.78 (95% CI 0.71–0.86), sensitivity
0.87 and specificity 0.53. These are published results, not results of this integration.

The [pinned release](https://github.com/Project-MONAI/research-contributions/tree/21ed8e57c7256834d4fbaf19579ca25ad3d135ee/prostate-mri-lesion-seg)
is called through its unchanged numerical operators. The adapter bypasses the
Holoscan scheduler and uses the SDK DICOM loader and volume operator on each
selected role directory. Original int16 arrays and double-precision affine metadata
are retained separately from the NIfTI geometry companions. The organ model
calls the SDK `compute_impl`. It uses the upstream RAS,
1-mm spacing, nonzero normalization, 128×128×16 sliding window, batch size 4,
overlap 0.5, softmax, inverse transforms and argmax. Lesion inference retains
nearest-neighbour diffusion resampling to T2, 0.5-mm resizing, gland bounding-box
crop with 32-voxel margin, per-channel normalization and all five models.
The final mean probability map is saved before gland multiplication. Thresholding
uses ≥0.6344772701607316 after that multiplication.

This commit defines **two** lesion output channels and writes foreground channel
1 once. An earlier design note's channel-2 overwrite does not apply. The classifier
uses upstream connected components on its 0.5-mm grid and 64³ crops. Argmax
indices 0–3 give PI-RADS 2–5; index 2 is promoted to PI-RADS 5 when the major axis
exceeds 40 voxels (20 mm). Softmax of the observed four-class logits is included
in JSON without changing classification. These probabilities describe the four
model classes; no calibrated cancer likelihood is claimed.

The upstream gland report selects background through `regions[0]` and uses
0.125 mm³ per voxel on the native gland grid. Those values are retained under
explicit `upstream_*` names. Exported gland and lesion volumes use native T2
voxel volume; classifier-grid lesion volume and major-axis length have separate
fields. Classifier components are restored individually with nearest-neighbour
interpolation. A component that disappears on T2 causes a clear error. Empty
lesion maps skip the upstream invalid empty-crop classification path and produce
a gland-only RTSTRUCT with an empty lesion list.

The adapter rejects an empty gland, nonfinite normalization, undersized crops,
nonbinary gland output and output-grid mismatch. It does not repair the upstream
numerical method. Probability, binary detection and gland maps accompany the
lesion sidecar. Both P6000 and H100 acceptance remain required on their respective
hosts; no GPU result is supplied by local drafting or synthetic tests.

## U-MambaMTL bpMRI

`umambamtl_cspca_bpmri` requires T2W, ADC and one high-b DWI volume. It exports
zero to three `UMMTL_L01_p0.73`-style lesion ROIs, with full-precision lesion and
examination likelihoods in JSON. The auxiliary zone head gates lesion probability;
it is not exported as a zonal label map. Weight licence not stated. Repository
code is Apache-2.0. Owner accepted these terms on 2026-10-22.

The [MIDL paper, Tables 2 and 3](https://proceedings.mlr.press/v301/larsen26a.html)
reports its strongest result for **MTL-Single**: AUROC 0.867, average precision
0.696 and combined score 0.781 in 100 PI-CAI hidden tuning cases; AUROC
0.932 ± 0.041, average precision 0.705 ± 0.101 and score 0.818 ± 0.062 in 200
institutional examinations. The ± values are the larger distance from the point
estimate to the two 95% bootstrap confidence bounds, not standard deviations.
The released wrapper constructs two decoders, matching the paper's **MTL-Dual**
architecture. That row reports AUROC/AP 0.843/0.622 (100 cases) and
0.925 ± 0.043 / 0.685 ± 0.097 (200 cases). The single-decoder results must not be
assigned to this release. This architectural correspondence is inferred from the
pinned network code; it does not independently verify the checkpoint's study identity.

The [pinned wrapper](https://github.com/mokkalokka/U-MambaMTL/tree/bc657e9f45d7d1ed457b071dd099686e6e90a742)
assigns ADC paths to the key `hbv` and HBV paths to `adc`, in both the crop and
detection stages. Its keys concatenate as T2W/HBV/ADC, but its **actual modality
order is T2W/ADC/HBV**. The integration preserves this released behavior and
records both orders. No silent channel correction is made.

Only exact input/output path literals are relocated into the job. The upstream
crop model, RAS orientation, diffusion resampling, 0.5×0.5×3-mm spacing, reflected
padding, gland foreground crop, 256×256×20 centre crop, percentile clipping,
normalization, five-fold sigmoid predictions, mean zone-foreground gating >0.5
and T2 inversion remain unchanged. RGA 0.2.4 uses dynamic extraction, maximum
3 candidates, minimum-voxel parameter 72 (components ≤72 are removed) and peak
divisor 3.0. Indexed candidates keep equal-score lesions separate. Zero-score
empty components do not become ROIs. Empty detections are completed empty results.

The shipped test script selects a Swin UNETR archive; its saved map is not an
established U-MambaMTL author reference. Acceptance uses the unchanged U-MambaMTL
wrapper functions on the same arrays instead. It predeclares the upstream test's
maximum absolute map/likelihood difference of 0.001, without outlier allowances,
and requires exact candidate mask identity. NCI has no published numeric test
tolerance, so its harness requires exact equality rather than inventing one.

U-MambaMTL requires CUDA compute capability ≥7.0. Its published extensions have
no P6000 sm_61 target. Runtime and preflight reject that device; configure this
task in `REMOTE_GPU_TASKS` for the H100 worker. Routing configuration is outside
this change.

## Held MRI bundle and acceptance contracts

All paths below are relative to `RTSEG_MODEL_ROOT`:

```text
sources/nci_monai_lesion/prostate-mri-lesion-seg/  # full pinned subtree and licence
sources/umambamtl/                              # full pinned checkout and licence
weights/nci_monai_lesion/organ/model.ts
weights/nci_monai_lesion/fold0..fold4/model_best_fold0..4.pth.tar
weights/nci_monai_lesion/classifier/model_best.pth.tar
weights/umambamtl/detector/f0..f4.ckpt
weights/umambamtl/prostate_crop/prostate_weights.ckpt
runtime-envs/nci-monai/bin/python
runtime-envs/nci-monai/requirements.lock
runtime-envs/umamba/bin/python
runtime-envs/umamba/requirements.lock
validation/<task>.json
orientation-validation/<task>.json
```

The existing dispatcher and result allowlist obtain both tasks from
`prostate_models.TASKS`; no production routing or enabled-model list is changed.
The two marker files must pass the same validation: both reference and DICOM
modes, two repeat runs per mode, required geometry/RTSTRUCT/reference checks,
predeclared tolerances, supported GPU, installed source commit and source-file
hashes, every checkpoint hash, checkpoint revision and environment-lock hash.
A failed or waived check cannot enable a task. Local tests do not create markers.
The task report contains environment build evidence and manager-run GPU commands.


### Acceptance and installed runtime binding

For these two tasks, each `validation/<task>.json` and
`orientation-validation/<task>.json` contains separate `devices.H100` and
`devices.P6000` records under schema version 2. U-MambaMTL accepts H100 only.
Preflight queries the model interpreter's selected visible GPU. The runner
checks it again at startup. Device name, capability, CUDA build, compiled
architectures and inference policy must match that device's record.

Acceptance includes installed-file manifests of both the model interpreter and
the worker interpreter used for DICOM conversion, dispatch-time image processing
and RTSTRUCT creation. Each inventories the whole environment (including all
site-packages), standard library, extension binaries and additional site import
paths. The worker executable and effective import environment are bound in each
device record. Changes to worker dependencies invalidate both tasks' evidence.
Hashes are cached by resolved path, `st_dev`, `st_ino`, `st_size`, `st_mtime_ns`
and `st_ctime_ns`. `model_preflight.py` maintains a persistent, preflight-only
digest cache (`digests.json`) for the two held-MRI tasks in the per-user
writable directory resolved in order from `RTSEG_HELD_MRI_PREFLIGHT_CACHE` if
set, else `$XDG_CACHE_HOME/rtsegmentator/held-mri-preflight-cache`, else
`~/.cache/rtsegmentator/held-mri-preflight-cache`, falling back to
`<RTSEG_SCRATCH_ROOT>/held-mri-preflight-cache` (where `RTSEG_SCRATCH_ROOT`
defaults to `/scratch`) only when the home cache directory cannot be created or
validated. The cache directory must lie outside the application root, model
bundle and worker runtime root, and both the directory chain and cache file must
be regular non-symlink paths owned by the current UID and not writable by group
or others; updates acquire an exclusive lock and write a mode-`0600` temporary
file followed by `os.replace`. New entries are flushed incrementally (at least
every 30 s of hashing), at session end, and on `SIGTERM` or `SIGHUP` before
process exit (with previous signal handlers restored afterwards). Unsafe,
unreadable or corrupt cache files are ignored and recomputed. Trust boundary:
the persistent and in-memory digest caches are used **only by preflight**.
Acceptance publication (`store_device_marker`), production dispatch
(`prostate_models.run_task`) and runner execution startup
(`held_mri.runtime_start`) always read and hash all bound bytes uncached
(`uncached=True`), so a stale or poisoned cache entry can at worst cause
preflight to report ready while execution and publication still reject modified
bytes. Unchanged lock text alone is insufficient.
All production Python modules, catalogue/label contracts and upstream source
modules are bound, including DICOM conversion and RTSTRUCT geometry code.

Model and worker processes start with `-E -s -B -X
pycache_prefix=<empty-private-directory>`. The cache directory is new for each
process and removed after it exits. This redirects cache reads before imports;
`-B` alone only prevents cache writes. Startup checks enforce the policy.
Ordinary `__pycache__` files cannot affect execution or invalidate evidence.
Standalone `.pyc` files outside those directories can execute without source,
so they remain inventoried. The dispatcher restarts with these flags before
importing conversion/export packages; its child processes use the same policy.

The pinned NCI lesion operator has a directory-creation race: five parallel
folds each test for `lesion/` and call `os.makedirs` without `exist_ok=True`.
It waits for futures without reading their exceptions, so failures can first
appear during merging. The adapter and unmodified-application reference now
pre-create that job-owned directory. They refuse existing fold probability
files and require five nonempty new files after inference, naming missing folds
on failure. Parallel inference, transforms, thresholds and all numerics remain
unchanged; the upstream source is not patched.

NCI retains the unchanged application's default cuDNN policy: deterministic
selection disabled, benchmark disabled, cuDNN TF32 enabled and matmul TF32
disabled. U-Mamba retains deterministic cuDNN. Exact NCI map and mask equality
is required. DICOM acceptance invokes `run_task.py` with the explicit
`--held-mri-acceptance-audit` bootstrap option and compares NCI outputs and
PI-RADS classes with a separate unmodified application run on the same study.
This operator-only CLI option permits acceptance before markers exist; it is
not a portal input. Production requires both markers. The acceptance command
invalidates all covered tasks for that GPU before fixture preparation and
removes them on failure or interruption, preserving other-device evidence.
