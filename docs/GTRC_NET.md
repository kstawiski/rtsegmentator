# GTRC-Net Total Tumour Burden Integration (`gtrc_psma_petct_ttb`, `gtrc_fdg_petct_ttb`)

## 1. Overview and Published Evidence

GTRC-Net (Global Threshold Regional Consensus Network) segments whole-body metastatic prostate cancer total tumour burden (TTB) on paired PET/CT examinations:

- **`gtrc_psma_petct_ttb`**: PSMA-avid total tumour burden (`GTRC_PSMA_TTB`) on `[68Ga]PSMA-11` or `[18F]DCFPyL` PET/CT (`Dataset881_PSMA_PET`).
- **`gtrc_fdg_petct_ttb`**: FDG-avid metabolic total tumour burden (`GTRC_FDG_TTB`) on `[18F]FDG` PET/CT (`Dataset882_FDG_PET`).

### Primary Reference and Cohorts
- **Publication**: Jackson P, Buteau JP, McIntosh L, et al. *Deep Learning for Automated Measures of SUV and Molecular Tumor Volume in [68Ga]PSMA-11 or [18F]DCFPyL, [18F]FDG, and [177Lu]Lu-PSMA-617 Imaging with Global Threshold Regional Consensus Network.* J Nucl Med. 2025;66(11):1811–1817. PMID: `40935568`. DOI: `https://doi.org/10.2967/jnumed.125.270077`.
- **Source repository**: `https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained` at pinned commit `9db02ddd569fd4e82a2caf8973aba6741d5cd4a8`.
- **Public engineering fixture**: Zenodo record `18150034` (`train_0001.zip`, CC BY 4.0).
- **Development and external test cohorts** (mCRPC patients staged for `[177Lu]Lu-PSMA-617` across the ProsTIC registry, LUCAB, LuPSMA, LuPARP, and PRINCE studies; only scans with quantifiable disease were included; exact values from Table 1, p. 1814):

| Task | Tracer | 5-Fold CV Cohort | External Holdout Test Cohort | Mean Volumetric DSC (CV / Holdout, Table 1) | Median Surface DSC (nnU-Net → GTRC-Net) | Biomarker Pearson *r* (Volume / SUVmean / SUVmax / VIP) | Liver Involvement Sens. / Spec. (Median Hepatic DSC) |
|---|---|---:|---:|---|---|---|
| `gtrc_psma_petct_ttb` | `[68Ga]PSMA-11` (567) + `[18F]DCFPyL` (109) | 676 | 56 | 0.9365 (0.9333 Ga-68, 0.9553 DCFPyL) / 0.9520 | 0.85 → 0.96 | 0.98 / 0.98 / 0.96 / 0.97 | 73.7% / 98.4% (0.564) |
| `gtrc_fdg_petct_ttb` | `[18F]FDG` | 390 | 56 | 0.8448 / 0.8447 | 0.80 → 0.94 | 0.94 / 0.88 / 0.91 / 0.96 | 81.8% / 98.1% (0.759) |

### Patient-Specific SUV Threshold Contract
In JNM 2025;66(11):1811–1817 (Materials and Methods, *Cohort Description and Ground Truth Reference*), the authors state verbatim:
> *"Cases were delineated by nuclear medicine physicians and showed the extent of whole-body metastatic disease using either a fixed threshold (SUV of ≥3 for PSMA PET and SPECT imaging) or a variable liver-based threshold (1.5x mean + 2x SD of liver background for [18F]FDG PET imaging), consistent with the methods reported by Ferdinandus et al. (8), Buteau et al. (7), and Wahl et al. (20)."*

(Note: the DEEP-PSMA Grand Challenge pages describe the FDG reference threshold as liver mean + 2 SD.)
- **`gtrc_fdg_petct_ttb`**: There is **no fixed FDG default**. The user must enter the patient-specific FDG SUVbw threshold (`0.1` to `100.0` g/mL) for each job via the portal UI (`#gtrcFdgThreshold`) or API (`task_options.gtrc_fdg_petct_ttb.suv_threshold` or `fdg_suv_threshold`). It is persisted in `job.json` with `"source": "user-entered"`, passed to `run_gtrc.py --suv-threshold`, and recorded in the sidecar JSON. Jobs missing a valid FDG threshold fail closed with HTTP 400 / `ValueError`.
- **`gtrc_psma_petct_ttb`**: Uses the published `3.0` g/mL threshold by default (`"source": "published-default"`), with optional per-job override in `[0.1, 100.0]` g/mL (`"source": "user-entered"`). The API preserves `"published-default"` only for `gtrc_psma_petct_ttb` at `3.0` and rejects `"published-default"` for any other PSMA value or any FDG threshold.
- **Implementation threshold range rationale (`MIN_SUV_THRESHOLD = 0.1`, `MAX_SUV_THRESHOLD = 100.0`)**: `[0.1, 100.0]` is an RTsegmentator implementation input-validation range (not a range stated in Jackson et al.). It prevents division by zero, negative numbers, `NaN`/`Inf`, or unscaled raw `Bq/mL` activity values (typically $10^3\text{–}10^5\text{ Bq/mL}$) accidentally entered in place of `SUVbw`, while accommodating any positive clinical PET `SUVbw` threshold.

### Licence Status
Neither the upstream GitHub repository nor the weight files state a software or weight licence (`"No licence stated by the authors"`). Per owner decision on 2026-10-22, these terms are accepted and recorded in `catalog/gtrc.json` and sidecar provenance.

---

## 2. Paired PET/CT Input Contract and QIBA SUVbw Conversion

Both tasks register `"modality": "PT"`, `"requires_ct": true`, and `"requires_preflight": true` in `catalog/gtrc.json`:
1. **Reference series (`job/input`)**: Attenuation- and decay-corrected 3D PET series (`Modality == "PT"`). Exported RTSTRUCT contours reference the PET SOP Instance UIDs and physical grid.
2. **Paired series (`job/input_ct`)**: Co-registered 3D CT series (`Modality == "CT"`) in the same `StudyInstanceUID` and `FrameOfReferenceUID`.
   - Every CT slice is inspected (`inspect_ct_series_metadata`) to verify that all slices share one `StudyInstanceUID` and `FrameOfReferenceUID` matching the PET series.
   - If an upload contains multiple CT series matching the PET series's `StudyInstanceUID` and `FrameOfReferenceUID`, the job fails closed with HTTP 400 unless the user explicitly selects the companion CT series (`#gtrcCtCompanion` / `ct_series_key`).
   - Before running inference, `gtrc_models.check_physical_overlap` enforces: (a) 3D physical LPS bounding-box overlap fraction `>= 0.25`, (b) CT physical axial (`Z`, Inferior–Superior) extent covering the PET axial (`Z`) extent within `max(5.0 mm, pet_spacing_z, ct_spacing_z)` (`AXIAL_COVERAGE_MIN_TOLERANCE_MM = 5.0`), and (c) 100% of PET voxels with `SUVbw >= suv_threshold` lying inside the actual `sitk.sitkLinear` interpolator support of the CT volume (decided directly from the resampled all-ones CT support image using the same identity transform, `sitkLinear` interpolator, and `0.0` outside default as the upstream CT resampling in `preprocess_pet_ct`, without a narrower `[0, size - 1]` index restriction) so no above-threshold PET voxel receives the `-1000` HU outside-buffer default value.

### Vendor-Neutral PET-to-SUVbw Converter (`pet_to_suvbw.py`)
`pet_to_suvbw.py` implements the RSNA Quantitative Imaging Biomarker Alliance (QIBA) SUVbw profile (*FDG-PET/CT Technical Committee SUVbw Pseudocode*; DICOM PS3.3 C.8.9.1 / C.8.9.2 / C.7.6.16.2.11):

$$\text{decay\_factor} = 2^{-(t_{\text{ref}} - t_{\text{inj}}) / T_{1/2}}, \quad \text{SUV}_{\text{bw}}\;[\text{g/mL}] = C_{\text{PET}}\;[\text{Bq/mL}] \times \frac{W_{\text{patient}}\;[\text{kg}] \times 1000}{A_{\text{injected}}\;[\text{Bq}] \times \text{decay\_factor}}$$

Explicit header checks enforced across all ordered slices:
- **`Units (0054,1001)` and `RealWorldValueMappingSequence (0040,9096)`**:
  - Validated on **every slice**.
  - `"BQML"`: converted to float32 `SUVbw` using the verified radiopharmaceutical and patient-weight tags below.
  - Explicit body-weight SUV (`Units == "GML"` with `SUVType (0054,1006)` equal to `"BW"` or absent, or `Units == "SUVBW"`, or `Units == "SUV"` with `SUVType == "BW"`, or `Units` paired with a verified UCUM `g/ml{SUVbw}` mapping): retained in float32 `SUVbw` (`conversion_mode = "already_suvbw"`, `suv_scale_factor = 1.0`) and **never converted twice**.
  - Every `RealWorldValueMappingSequence` item must carry a 1-item UCUM `MeasurementUnitsCodeSequence (0040,08EA)`. `LUTExplanation (0028,3003)` and `LUTLabel (0040,9210)` are allowed as descriptive text on linear mappings; only actual lookup-table data (`LUTData (0028,3006)` or `RealWorldValueLUTData (0040,9212)`) is rejected as a nonlinear LUT. For integer `PixelData`, `RealWorldValueFirstValueMapped (0040,9216)` and `RealWorldValueLastValueMapped (0040,9211)` are required (DICOM PS3.3 Table C.7.6.16-12b) and verified against each slice's actual decoded stored pixels and `SmallestImagePixelValue`/`LargestImagePixelValue`; out-of-range pixels or contradictory extrema are rejected. Linear RWVM `RealWorldValueSlope`/`RealWorldValueIntercept` must agree with `RescaleSlope`/`RescaleIntercept` within `RWVM_SLOPE_RTOL = 1e-5` and bound the resulting SUV error across the verified stored-value domain within `RWVM_MAX_SUV_ERROR_TOL = 1e-4` g/mL.
  - Non-body-weight SUV types (`"BSA"`, `"LBM"`, `"LBMJAMES"`, `"LBMJANMA"`, `"IBW"`), non-activity units (`"CNTS"`, `"CPS"`, `"PROPCNTS"`, etc.), missing/non-UCUM `MeasurementUnitsCodeSequence`, nonlinear LUT data, or unapplied/contradictory linear mappings are rejected on every slice with a `ValueError`.
- **Rescale Tags**:
  - Requires explicit `RescaleSlope (0028,1053) > 0` and finite `RescaleIntercept (0028,1052)` on every slice (missing tags are rejected rather than defaulted).
- **Decay and Image Corrections (`Units == "BQML"`)**:
  - Requires `DecayCorrection (0054,1102) == "START"` and `CorrectedImage (0028,0051)` containing both `"DECY"` and `"ATTN"` on every slice.
- **Dose, Half-Life, Weight, and QIBA Scan-Start Timing (`Units == "BQML"`)**:
  - Requires `PatientWeight (0010,1030)` in `(0, 500]` kg, `RadionuclideTotalDose (0018,1074) >= 10,000` Bq (an explicit heuristic floor `MIN_PLAUSIBLE_DOSE_BQ = 10_000.0` to reject accidental MBq/mCi entries while acknowledging that DICOM specifies Bq without a formal lower bound), and `RadionuclideHalfLife (0018,1075) > 0` s, consistent across slices.
  - Parses explicit DICOM `DT` timezone suffixes (`&ZZXX`), cross-checks them against `TimezoneOffsetFromUTC (0008,0201)` when present, retains each slice's acquisition UTC offset (`s_acq_tz`) in every timestamp comparison (including `SeriesTime (0008,0031)` vs `AcquisitionTime`/`AcquisitionDateTime`), converts timezone-aware timestamps to UTC, and rejects any unresolved aware/naive or conflicting timezone offset combination (such as an explicit `AcquisitionDateTime` or `RadiopharmaceuticalStartDateTime` `DT` offset paired with a naive `SeriesTime` or `RadiopharmaceuticalStartTime` `TM` without `TimezoneOffsetFromUTC`).
  - Resolves the decay reference scan-start timestamp (`_resolve_qiba_scan_start`) across all slices following the QIBA profile while requiring consistency across all present acquisition timing records (`SeriesDate/SeriesTime`, `AcquisitionDate/AcquisitionTime`, `FrameReferenceTime` + `ActualFrameDuration`, GE `(0009,100D)` with private creator `(0009,0010) == "GEMS_PETD_01"`, and Siemens `(0071,1022)` with private creator `(0071,0010) == "SIEMENS MED PT"`), rejecting conflicting evidence (`> 2.0 s` discrepancy or scan start later than `AcquisitionTime + 1.0 s`) rather than preferring one source.
  - **Midnight and Date Crossing Guard**: Never guesses a `+86400` s rollover. Requires paired date+time for both scan start and injection whenever dates differ across tags or times cross midnight.

---

## 3. Published Inference Method and Static Blockers Fixed

### Role of `rigid.tfm`, `threshold.json`, and `totseg_24.nii.gz` in Upstream Source
Inspection of all upstream scripts (`00_copy_deep_psma_training_data.py` through `06_Train_Consensus_Evaluator.py`, `GTRC.py`, `GTRC_Infer.py`, `gtrc_utils.py`) and the Zenodo `train_0001` archive confirmed:
- `rigid.tfm` (inter-study rigid transform between the PSMA CT and FDG CT in the DEEP-PSMA dataset) and `totseg_24.nii.gz` (TotalSegmentator organ labels) are **never read in either training or inference**.
- `threshold.json` (`{"suv_threshold": 2.999862}` in `train_0001/PSMA/threshold.json`; `{"suv_threshold": 2.9650267106980666}` in `train_0001/FDG/threshold.json`) is read **only during training data preparation** (`00_copy_deep_psma_training_data.py`), where `01_Preprocess_Dataset.py` defines training class `2` (`norm`) as `(PET / suv_threshold >= 1.0) & (TTB == 0)`.
- At inference time (`GTRC.py` / `GTRC_Infer.py`), normal physiologic uptake (`label 2`) is predicted directly by the 3-class nnU-Net (`0=background`, `1=ttb`, `2=norm`), and the SUV threshold is supplied via `--suv-threshold` (`resolve_job_suv_threshold`):
  - `gtrc_psma_petct_ttb`: `3.0` g/mL (`"source": "published-default"`, or `"user-entered"` when overridden in `[0.1, 100.0]`; acceptance uses `train_0001/PSMA/threshold.json = 2.999862`).
  - `gtrc_fdg_petct_ttb`: required per-job patient-specific threshold (`"source": "user-entered"`, finite in `[0.1, 100.0]`, no default; acceptance uses `train_0001/FDG/threshold.json = 2.9650267106980666`).

### Preprocessing, nnU-Net v2, and Subregion Consensus Postprocessing
1. **CT Resampling and Threshold Division**:
   - CT is resampled to the PET physical grid using `sitk.ResampleImageFilter()` (`SetReferenceImage(pt_suv)`, identity transform, linear interpolation, `SetDefaultPixelValue(-1000)`).
   - Channel 0 (`gtrc_0000.nii.gz`) is `pt_rs = pt_suv / suv_threshold` (`NoNormalization` in `plans.json`).
   - Channel 1 (`gtrc_0001.nii.gz`) is `ct_rs` in HU (`CTNormalization`: PSMA clip `[-1000, 920]`, mean `32.071888`, SD `212.490005`; FDG clip `[-1000, 1039]`, mean `67.151413`, SD `188.757507`).
2. **5-Fold 3D nnU-Net v2 with Deterministic cuDNN**:
   - Runs `nnUNetv2_predict -d <881|882> -c 3d_fullres` across all 5 folds (`fold_0..4/checkpoint_final.pth`), producing `gtrc.nii.gz` (`0=background`, `1=ttb`, `2=norm`).
   - Because `nnUNetPredictor.__init__` sets `torch.backends.cudnn.benchmark = True`, `run_gtrc.py` wraps `nnUNetPredictor.__init__` via `DETERMINISTIC_NNUNET_ENTRYPOINT` so `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True` are enforced after nnU-Net initializes its predictor, and writes the child process's effective torch/cuDNN flags to `nnunet_runtime_flags.json`.
   - **Documented H100 `train_0001` cuDNN policy deviation**: On H100 (`sm_90`) `train_0001/PSMA`, a controlled repeatability experiment (`repeatability-20261022/controlled-comparison.txt`) established that deterministic cuDNN (`cudnn.deterministic=True`, `cudnn.benchmark=False`, 4 runs with 0 intra-group differing voxels) and unmodified autotuned `nnUNetv2_predict` (`cudnn.benchmark=True`, 3 runs with 0 intra-group differing voxels) each reproduce their own raw nnU-Net label predictions identically while differing from each other by 1 raw nnU-Net label voxel, with 0 differing subregion or final TTB voxels. On `train_0001/FDG` (`acceptance_h100_r1b`), deterministic reference vs autotuned runner differed by 2 raw nnU-Net label voxels with 0 differing subregion or final TTB voxels (same mechanism expected, not separately ablated). Both the production runner and the upstream reference wrapper in `acceptance_gtrc.py` enforce deterministic cuDNN.
   - **Acceptance contract by mode**: reference mode gives the runner and unmodified `GTRC_Infer.py` the same NIfTI inputs and requires exact agreement of raw nnU-Net labels, subregions and the final TTB mask. DICOM mode reads PET back from int16 DICOM, which carries the declared SUV encoding error (at most 0.000385 SUVbw on `train_0001`), so raw nnU-Net labels and subregions can differ at a few boundary voxels; those counts are recorded as diagnostics. The exported TTB mask must still agree exactly with reference mode, and the RTSTRUCT round trip must return it. On the H100 (`acceptance_h100_r2`) DICOM mode differed from reference mode by 1 (PSMA) and 4 (FDG) raw nnU-Net label voxels and 10 (PSMA) and 0 (FDG) subregion voxels, with 0 differing TTB voxels.
3. **Watershed Candidate Subregion Partitioning**:
   - Threshold mask: `tar = (pt_ar >= 1.0).astype('int8')` (voxels with `SUV >= suv_threshold`).
   - Calls `gtrc_utils.create_subregion_labels(pt_rs, tar, local_maxima_threshold=0.16666, sphere_radius=8.0)`:
     - Builds an 8.0 mm physical sphere footprint on the PET voxel spacing (`np.meshgrid` with `indexing='ij'` followed by `np.swapaxes(sphere, 0, 2)`).
     - Detects local maxima via `skimage.morphology.h_maxima(qs_ar, 0.16666, sphere)`, labels peaks with 6-connected `scipy.ndimage.label`, and partitions `tar` via `skimage.segmentation.watershed(-qs_ar, numbered_peaks, mask=threshold_array)`.
     - **Preserved upstream indexing quirk**: Above-threshold connected components without an `h_maxima` peak (`missing_peaks_ar`) are assigned `labels_ws[missing_peaks_ar_numbered == (i+1)] = n_peaks + i` for `i in range(n_missing)` and cast to `sitkInt16`. Because `i` starts at `0`, the first peakless component receives label `n_peaks` (merging with peak `n_peaks` when `n_peaks >= 1`, or remaining `0`/background when `n_peaks == 0`). `run_gtrc.py` preserves this exact upstream function so outputs match unmodified `GTRC_Infer.py` bit-for-bit.
4. **Six Subregion Features, 5-Fold Keras MLP Voting, and Final TTB Re-painting**:
   - For each candidate subregion `1..total_subregions`, computes `[total_volume, suv_max, suv_mean, ct_hu_mean, pred_ttb_overlap, pred_norm_overlap]` (where `suv_max` and `suv_mean` are computed on `pt_rs = pt_suv / suv_threshold`, matching `GTRC_Infer.py`).
   - Loads `fold_0..4/consensus_model.keras` on CPU (`CUDA_VISIBLE_DEVICES="-1"`, `tf.config.set_visible_devices([], "GPU")`). Each model votes positive if `score > 0.5`; a subregion is accepted if `votes > 2.5` (strict majority ≥3 of 5).
   - Final re-painting sets `output_ar[(pred_ttb_ar > 0) & (tar > 0)] = 1`.

### Static Blockers Fixed
1. `GTRC_Infer.py` line 27 hardcoded `nn_predict_exe = r"C:\GTRC\GTRC_venv\Scripts\nnUNetv2_predict.exe"` and used `os.system()`; `run_gtrc.py` invokes `nnUNetv2_predict` from the isolated `gtrc` environment via `subprocess.run(..., check=True)`.
2. `GTRC_Infer.py` lines 157/167 referenced `data/LuPSMA/...` while the directory on disk is `data/LUPSMA/...`.
3. `nnunet_config_paths.py` created directories relative to the working directory and `../GTRC_baseline/` at import time; `run_gtrc.py` confines `nnUNet_raw` and `nnUNet_preprocessed` inside the job workspace and points `nnUNet_results` to the read-only bundle weights.

---

## 4. Outputs, Sidecar Schema, and Empty-Result Handling

- **RTSTRUCT ROI**: `GTRC_PSMA_TTB` (`gtrc_psma_petct_ttb`) or `GTRC_FDG_TTB` (`gtrc_fdg_petct_ttb`), exported on the reference PET grid via `convert_validate_rtstruct.py`.
- **Declared Output Artifacts (`OUTPUT_ARTIFACTS`)**:
  - `<task>_RTSTRUCT.dcm` (RTSTRUCT on reference PET grid)
  - `<task>_RTSTRUCT.dcm.lesions.json` (biomarker, connected-component, subregion, threshold, and provenance sidecar)
  - `<task>_RTSTRUCT.dcm.ttb_mask.nii.gz` (binary NIfTI TTB mask on reference PET grid)
  - Naming every declared output artifact with the `<task>_RTSTRUCT.dcm` prefix guarantees automatic transfer by `REMOTE_OUTPUT_GLOBS` (`f"{task}_RTSTRUCT.dcm*"`) on routed remote GPU jobs. The intermediate `<task>_pet_suvbw.json` conversion metadata stays in the scratch work directory and is embedded in `lesions.json["job_context"]["pet_suvbw_conversion"]`.
- **Empty Predictions**: If 0 voxels are predicted as TTB, `gtrc_models.run_task` copies `<task>_RTSTRUCT.dcm.lesions.json` and `<task>_RTSTRUCT.dcm.ttb_mask.nii.gz` to `job/output/` and exits with `"empty prediction: no RTSTRUCT created"`, which `app.py` (including routed SSH retrieval) records as a valid empty result (`NO_STRUCTURES_FOUND.txt` + sidecars in `rtstruct-results.zip`).

---

## 5. Out-of-Scope Note: Lu-PSMA SPECT/CT (`Dataset883_LUPSMA_SPECT`)

In Jackson et al. (J Nucl Med 2025;66(11):1811–1817, Table 1, p. 1814), the `[177Lu]Lu-PSMA-617` SPECT/CT model was evaluated on 477 scans in 5-fold cross-validation (mean DSC `0.9696`), with no external SPECT holdout test cohort. `Dataset883_LUPSMA_SPECT` and `data/LUPSMA/subregion_metrics/fold_0..4/consensus_model.keras` are included in the bundle layout proposal (`LUPSMA_WEIGHT_FILES` in `gtrc_models.py`), but no portal task is enabled because:
1. Portal upload assessment currently ingests `CT`, `MR`, and `PT` single-frame slice series, not multi-frame `NM` (Nuclear Medicine) SPECT tomograms.
2. Quantitative Lu-PSMA SPECT in `GTRC_Infer.py` requires input voxel values calibrated in quantitative SUV (`Bq/mL` with camera sensitivity calibration factor, decay correction, injected activity, and patient weight, or vendor quantitative SPECT SUV maps), whereas clinical `NM` DICOM exports frequently store uncalibrated counts (`CNTS`).
