# External model integration ledger

This ledger distinguishes automatic models from prompted tools and records whether a release can be reproduced. The records below describe historical integration runs. Current selection depends on runtime preflight, segmentation orientation evidence or separate report acceptance, and the enabled-model policy. Research use only; execution and geometry results do not establish clinical accuracy.


## NVIDIA NV-Segment automatic tasks (2026-10-15)

Four automatic tasks are integrated behind separate installation, acceptance and orientation gates. No acceptance marker is included in this change. The existing `vista3d_prompted_ct` source, runner and acceptance remain unchanged.

| Task | Input | Foreground labels | Scope |
|---|---|---:|---|
| `nv_segment_ct_auto` | CT | 117 | NV-Segment-CT automatic selection |
| `nv_segment_ctmr_ct_body` | CT | 117 | NV-Segment-CTMR `CT_BODY` |
| `nv_segment_ctmr_mri_body` | MR | 50 | NV-Segment-CTMR `MRI_BODY`, TotalSegmentator-MRI classes |
| `nv_segment_ctmr_mri_brain` | T1 MR | 132 | NV-Segment-CTMR `MRI_BRAIN`, LUMIR regions |

The publisher's automatic CT selection excludes 15 of 132 CT classes, including the solid-tumour heads: IDs 2, 16, 18, 20, 21, 23–27 and 128–132. Renal cyst labels use “Lesion candidate”. This mode does not generate automatic tumour targets. Brain's stated 133 classes include background; label 255 is a foreground region and is preserved. All exported labels use uint16 and retain the publisher's sparse IDs.

**Provenance and licences.** [NV-Segment-CT](https://huggingface.co/nvidia/NV-Segment-CT/tree/afb51518689f71e6abb367ee6301b2cd0225c66a) uses the NVIDIA Open Model License. Its checkpoint revision is `afb51518689f71e6abb367ee6301b2cd0225c66a`; `weights/vista3d/NV-Segment-CT/model.pt` is 871,970,895 bytes, SHA-256 `c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b`. The [source](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/f9f5f51b589e5dc9c23c453cf5138398e4084056) is pinned to `f9f5f51b589e5dc9c23c453cf5138398e4084056` under `sources/NV-Segment-CTMR/NV-Segment-CT`.

[NV-Segment-CTMR](https://huggingface.co/nvidia/NV-Segment-CTMR/tree/4fb8b4a6b2532be9f1c449a3726fe5440ab4213a) uses checkpoint revision `4fb8b4a6b2532be9f1c449a3726fe5440ab4213a`. `weights/vista3d/NV-Segment-CTMR/model.pt` is 871,950,375 bytes, SHA-256 `6c6d86a48a7a9d377ddb41d663d487cfedd6456df3b828ecd931e5cba5e71f99`. The [source](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/cb921f5c58837c0f42a713855d68b32af88e1cdd) is pinned separately to `cb921f5c58837c0f42a713855d68b32af88e1cdd` under `sources/NV-Segment-CTMR-cb921f5c/NV-Segment-CTMR`. Code is Apache-2.0. The [weight licence, section 3.3](https://huggingface.co/nvidia/NV-Segment-CTMR/blob/4fb8b4a6b2532be9f1c449a3726fe5440ab4213a/LICENSE) restricts use to noncommercial research or evaluation, with the stated NVIDIA/affiliate exception. CTMR weights do not use the CT Open Model licence. Source-file and label-map hashes for all modes are recorded in `nv_segment_manifest.json` and checked before inference.

**Runtime and brain preprocessing.** Segmentation uses `envs/lnq` (prepared with Python 3.11.14, PyTorch 2.13.0+cu130 and MONAI 1.6.0). The brain pipeline also needs `runtime-envs/nv-segment` (NumPy 1.26.4, SciPy 1.15.3, antspyx 0.6.1, Surfa 0.6.3 and pymedio 0.2.14), with its recorded package lock and lnq package linkage. These prepared runtime versions are provenance, not a claim of Pascal support. No packages or checkpoints are downloaded during a job.

Brain mode runs SynthStrip internally, then the publisher's N4/intensity-normalization and affine registration to the LUMIR template. It inverts the transform with nearest-neighbour labels and restores the exact original grid. Repairs are applied only to a job copy: MONAI point conversion, NaN background 0, strict checkpoint loading, uint16 export and exact inverse-grid dimensions. The output writer rejects noninteger, unknown or empty labels and grid mismatches.

SynthStrip 1.8 is extracted under `runtime-homes/nv-segment/synthstrip-1.8`; Docker is not needed for inference. Image digest: `sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06`. The standard checkpoint SHA-256 is `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33`; the script SHA-256 is `fa18eb301568dc7c99bd9f69805545a7ce6d68e3fa3e1775c68053c9069cc4a1`. The [publisher](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/#code) offers weights under MIT or CC BY 4.0; cite [Hoopes et al., NeuroImage 260 (2022), 119474](https://doi.org/10.1016/j.neuroimage.2022.119474).

**MRI sequence scope.** MRI_BODY selects the publisher's 50 TotalSegmentator-MRI classes. Sequence and field of view affect output; this integration establishes no sequence-specific performance. For brain, the [pinned model card](https://huggingface.co/nvidia/NV-Segment-CTMR/blob/4fb8b4a6b2532be9f1c449a3726fe5440ab4213a/README.md) specifies standard T1, while the [pinned source documentation](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/cb921f5c58837c0f42a713855d68b32af88e1cdd/NV-Segment-CTMR/docs/README.md) also names T2 and FLAIR. The configured audit uses T1 only. Non-T1 brain acceptance remains unresolved.

**Reported GPU observations.** The manager supplied these preliminary H100 measurements for `nv_segment_ct_auto`; this implementation worker did not repeat GPU inference.

| Public fixture | Runtime | Peak VRAM | RTSTRUCT ROIs |
|---|---:|---:|---:|
| KiTS19 case00000 | 60.7 s | 21.3 GiB | 68 |
| MSD lung_053 | 28.7 s | 16.2 GiB | 100 |

The manager reported valid output grids and correct lung, kidney, spleen and liver sides. The chest CT also contained spurious left/right femur labels of 277/1,068 voxels and hip labels of 4/327 voxels. Small false labels are therefore an observed limitation; review every contour in its anatomical context. The manager's complete rerun and final evidence remain pending. Compatibility and memory use on Pascal sm_61 with 24 GiB are unknown. These measurements and technical markers do not establish clinical or manuscript readiness.

**Orientation fixtures and acceptance.** The configured public DICOM fixtures are MSD lung for CT automatic, KiTS19 case00000 for CTMR CT body, TotalSegmentator-MRI s0052 for MRI body, and the existing `hippunfold-ds002168-T1w/dicom` fixture for brain. The latter contains 256 DICOM files and derives from the public [OpenNeuro ds002168 dataset](https://openneuro.org/datasets/ds002168); the publisher's separate brain NIfTI example is not used as a substitute for a DICOM round trip. The audit checks execution, source-grid conversion, physical contour round trips and model identity. It does not assess segmentation accuracy. See [operations](docs/OPERATIONS.md) for marker generation and local/remote activation.

## NVIDIA NV-Reason-CT draft reports (2026-10-15)

Task `nv_reason_ct_report` produces JSON, Markdown and HTML for diagnostic chest/abdomen CT; it does not produce RTSTRUCTs. [Publisher and license](https://huggingface.co/nvidia/NV-Reason-CT): OpenMDW-1.1, revision `386b93e034983f6c1fc841a43833a1b6a0cd9c13`. The checkpoint SHA-256 is `655c15e59dfa1957f7d5f475b1702d689fb7107160f21256028391bb5f6e6ba5` (10,644,167,648 bytes).

Manager-run technical acceptance of the earlier implementation on an H100 produced a chest report for public MSD lung CT with `total` segmentation and an abdomen report for public KiTS19 case 00000 with a report-only job. The chest used the publisher crop; the abdomen used a segmentation-guided crop after publisher fallback. Both crops contained 100% of the segmented regional organs. Full-resolution coverage took 275–313 s, model loading 21–25 s, and warm-cache generation 57–64 s for about 400 tokens. Peak model-process allocation was 11.2 GiB; plan for approximately 12 GiB model VRAM, with the worker requiring 14 GiB free before loading. The production-tree port and default fast coverage require new GPU checks; these historical results are not acceptance of this version.

Known limitations:

- Without anatomical gating, the model describes organs outside the field of view. It can generate confident findings from unreadable input. Coverage masks can also be wrong.
- On public KiTS19 case 00000, it described a 7.8 mL (about 2.6 cm) renal tumour as a hypodensity “too small to characterize”. Segmentation checks do not validate diagnostic findings.
- CBCT is refused. A negative CBCT header check does not establish diagnostic image quality.
- Use an sm_80-class or newer CUDA GPU. A 24 GiB Pascal GPU could not run the model: bf16 SDPA exhausted memory, fp16 failed in cuDNN conv3d, and disabling cuDNN also exhausted memory. Per-task SSH routing supports a separate compatible worker.

TotalSegmentator `total --fast` (3 mm) is the default coverage mode, without an ROI subset; `RTSEG_NV_REASON_CT_COVERAGE_MODE=full` restores full resolution. Fast/full region and group-status comparison is pending GPU execution. Report availability uses `validation/nv_reason_ct_report.json`, independently of segmentation orientation markers. Radiologist review is required; technical acceptance and local drafting do not establish clinical or manuscript readiness.

## Spinal Cord Toolbox 7.3 (2026-08-13)

- The official SCT 7.3 runtime and all 15 requested catalog-task checkpoint sets are installed.
- Fourteen single-series tasks are represented in `catalog/sct.json`. The T1+T2
  tumor/edema/cavity task is visible but blocked by its registered two-series input contract.
- `sct_spinalcord` passed H100 inference, non-empty RTSTRUCT conversion, source-SOP
  validation, model-specific DICOM identity, and patient-space orientation audit.
  The other installed tasks remain acceptance-gated until suitable sequence-specific
  public fixtures produce non-empty predictions; installation is not presented as proof.
- The complete runtime is a checksum-protected read-only SquashFS image on NFS at
  `runtime-homes/sct/sct-7.3.squashfs`, mounted at `runtime-homes/sct/sct_7.3`.
  The 17 GB worker-local staging tree was removed after mounted-runtime validation.

## GOUHFI 2.0 (2026-08-13)

- Official tag 2.0.0 source and both public Zenodo model archives were downloaded
  directly to NFS and matched the publisher's MD5 values. Each task uses all five folds.
- `gouhfi_v2_brain_mr` emitted all 35 released labels; its strictly validated
  RTSTRUCT has 35 non-empty ROIs and references 78 images from the selected series.
- `gouhfi_v2_dkt_mr` emitted all 62 bilateral DKT labels; its strictly validated
  RTSTRUCT has 62 non-empty ROIs and references 67 selected-series images.
- The portal path applies brain extraction and the official LIA/intensity conformation,
  runs official inference and postprocessing, restores results to native DICOM geometry,
  and applies the model-specific DICOM identity fields. Both orientation audits passed.
  This is a research model, not a clinically approved contouring system.

## SIAM v3 (2026-08-13)

- The current SIAM source and official `pred_DS108_LcsfP_Ano` five-fold archive
  are installed on NFS; the 5.1 GB archive matches Zenodo MD5
  `0c3ca6d0730a3dcb924e079af04474c3`.
- `siam_v3_head_mr_ct` ran all five folds on the H100 and produced all 17
  checkpoint labels. The 4,704,902-byte RTSTRUCT contains 17 non-empty ROIs,
  references 91 selected-series images, carries model-specific DICOM identity,
  and passed patient-space orientation validation.
- SIAM accepts MRI contrasts and CT, but the acceptance fixture was T1 MRI.
  Its anomaly output is retained as `Anomaly_Candidate`; it is not presented as
  a definitive tumour, lesion diagnosis, GTV or CTV.

## FastSurferVINN (2026-08-13)

- The official stable source and all three VINN view checkpoints are installed
  on NFS. The shared CUDA runtime needs only a small NFS-hosted `h5py` overlay.
- `fastsurfervinn_t1_mr` ran axial, coronal and sagittal inference on the H100,
  emitted 95 non-empty cortical/subcortical labels, and produced a 1,704,354-byte
  RTSTRUCT referencing 77 selected-series images.
- The wrapper requests native orientation, converts the upstream MGZ result to
  NIfTI without changing its affine, and nearest-neighbour restores labels to the
  original DICOM grid. Model-specific identity and strict orientation audits passed.
- The model expects a whole-head T1-weighted MRI. Outputs are research anatomy
  labels and do not define glioma, metastasis, GTV or CTV boundaries.

## TractSeg 2.10 (2026-08-13)

- Official source and all nine public pretrained archives are installed and
  checksum-indexed on NFS (1.3 GB total).
- The released peaks fixture ran successfully through the official CUDA path:
  all 72 tract channels were non-empty with 237,552 foreground voxels.
- The portal entry remains disabled with status
  `requires_diffusion_peaks_or_dwi_gradients`. TractSeg requires a 9-channel
  CSD/bedpostX peaks NIfTI, or raw 4D DWI plus matching b-values and b-vectors.
  A selected scalar 3D DICOM series cannot satisfy that contract, so no false
  RTSTRUCT acceptance is claimed.

## BraTS Toolkit legacy ensemble (2026-08-13)

- Official source commit `86e7fa89b48e965b6c23151931f88e074930f75e` is
  installed on NFS. All legacy image artifacts are stored under
  `images/brats-toolkit`; model preflight requires every one before reporting
  the family installed.
- The MIC-DKFZ image is an intact 2018 CUDA 9/PyTorch 0.4 environment. Its
  embedded CUDA kernels cannot execute on the H100, although the runtime sees
  the device. A documented compatibility patch maps its released checkpoint to
  CPU and makes the otherwise read-only `/params` tree writable through an NFS
  bind mount.
- Released fold-0 inference on `Brats18_TCIA10_639_1` passed. Output shape and
  affine exactly matched the 240x240x155 source, with labels 1, 2 and 4 all
  non-empty (103,221; 79,248; and 1,009 voxels respectively). Output SHA-256 is
  `ca13836cbeda6fff78546985dabf62fc9f0b37b75c5e6af381b8a1661d642150`.
- This is not selectable: all toolkit algorithms require skull-stripped,
  co-registered FLAIR, T1, T1c and T2 NIfTI inputs in SRI-24 space. The portal
  currently selects one DICOM series, so silently substituting or guessing the
  other channels would be unsafe.

## CEREBRUM-7T and DeepBrainSeg (2026-08-13)

- CEREBRUM-7T source and its included 14.9 MB HDF5 checkpoint are installed and
  checksum-indexed. The published inference performs voxelwise z-scoring using
  training-cohort mean and standard-deviation volumes, but those two volumes are
  absent from the repository/release. The model is therefore visible with the
  precise `missing_voxelwise_training_normalization` blocker; invented scalar
  normalization is not accepted as reproduction.
- The catalog's stale `mobarakol/DeepBrainSeg` URL was corrected to the actual
  `koriavinash1/DeepBrainSeg` upstream. Source, BraTS fixtures and all four 0.0.1
  checkpoints are on NFS; all four state dictionaries load successfully in the
  current PyTorch runtime. It remains input-contract blocked because inference
  requires registered T1, T1ce, T2 and FLAIR volumes plus a brain mask.

| Release | Input | Download state | Execution state |
|---|---|---|---|
| HNLNL 2D+3D ensemble | H&N planning CT | Complete 2D+3D five-fold release downloaded | **PASS; PORTAL ENABLED:** found and isolated a 516-slice 1.5-mm planning H&N CT. The exact portal branch was smoke-tested through all ten folds, probability-level 2D+3D averaging, published component postprocessing, RTSTRUCT serialization and cleanup. All 20 classes were nonempty and the strictly valid 20-ROI RTSTRUCT references 125 source images; the multi-gigabyte temporary probabilities were removed afterward. The exact numeric label order was recovered from `LevelNameList` in the paper's Europe PMC `DataSheet_2.zip` supplement: Ia, bilateral Ib/II/III/IVa/IVb/V, VIa/VIb/VIIa, and bilateral VIIb/VIII. |
| Choi TMLI | whole-body/planning CT | Complete Zenodo archive downloaded; missing `plans.pkl` reconstructed from release checkpoint metadata | **TECHNICAL PASS; ontology pending:** the release plans reveal training intensity centred near stored pixel value 977, while our converter correctly emitted HU. Restoring the release-specific +1024 stored-pixel convention fixed the all-background failure. Fold 0 on a 735-slice `CT WB 1.5 Br38` produced 16 contourable classes/513 referenced images. The corrected five-fold ensemble also passed strict validation, producing 9 nonempty numeric regions, 9 contour sets and 188 source-image references; its suppression of weak fold-0 classes is recorded rather than hidden. The Zenodo page names the ten regional groups but neither it nor the archive maps numeric labels 1–18 to region/laterality, so clinical names are not guessed and the portal entry remains disabled. |
| CL-Net 235-class checkpoint | CT | Source plus complete 940 MB seven-decoder checkpoint archive downloaded and extracted | **UPSTREAM WEIGHT BLOCKER:** all seven public decoder label files were inspected and contain no lymph-node class. The repository's training configuration names separate `Task1007_HeadNeck_LNS_18` and `Task1008_Chest_LNS_15` decoders, but their weights and station-index maps are absent from the public checkpoint. The advertised 33 stations therefore cannot be reproduced from the downloadable release. |
| ISRT CT-only (ResUNet, SegResNet, SwinUNETR) | planning CT | All 9 weights downloaded; all three ensembles portal-integrated | **PASS:** each three-checkpoint ensemble produced a nonempty mask and valid RTSTRUCT: ResUNet 152,010 foreground voxels/112 referenced images; SegResNet 38,659/114; SwinUNETR 236,098/128 |
| ISRT PET fusion variants | registered planning CT + PET1, optionally PET2 | All 36 weights downloaded; four PET1-only ablation ensembles portal-integrated | **TECHNICAL PASS, DOMAIN WARNING:** same-Frame-of-Reference CT/PSMA-PET with BQML converted to SUVbw×1000. PET1 early/deform (48,177 voxels), early/rigid (13,996), late/deform (21,746), and late/rigid (29,898) each produced a valid one-ROI RTSTRUCT. The fixture lacks a distinct planning CT, so the AC CT was supplied for both CT channels; it is also outside the pediatric Hodgkin/FDG domain. The PET1+PET2 releases remain input-blocked because no longitudinal PET2 is available |
| Pediatric thoracic lymphoma | contrast CT | Full nnU-Net v1 export downloaded, MD5 verified, extracted, and portal-integrated | **PASS:** nonempty mask; valid one-ROI RTSTRUCT references 192 source images. Tested source is low-dose AC CT rather than pediatric contrast CT, so protocol warning is shown |
| SegRap 2023 Task 2 | paired noncontrast + contrast H&N CT | All five folds downloaded | **TECHNICAL PASS, INPUT WARNING:** the wrapper was repaired to use TotalSegmentator 2.17 for body cropping (upstream silently continued after its pinned TotalSegmentator failed). All five folds produced GTVp 34,720 and GTVnd 536 voxels; valid 2-ROI RTSTRUCT/56 refs. Only one appropriate planning H&N CT was available, so it was deliberately duplicated into native and contrast channels; this validates execution/geometry but not paired-contrast clinical performance |
| UWLAIR HNTS-MRG | pre/mid-RT T2 MRI | Full 28.5 GB, 20-model Dropbox ensemble downloaded, integrity-tested, and extracted | **PASS:** public HNTS-MRG24 patient 101. Pre-RT produced GTVp 3,529 and GTVn 21,767 voxels; valid 2-ROI RTSTRUCT/33 refs. Mid-RT produced GTVn 15,327 voxels; valid 1-ROI RTSTRUCT/29 refs. The single-series pre-RT 10-model ensemble is portal-integrated as `uwlair_hntsmrg_pre_t2`; a portal job completed with GTVp/GTVn RTSTRUCT and HTTP download. The source fixture is NIfTI-only, so RTSTRUCT geometry was validated against explicitly synthetic, geometry-faithful MR DICOM references. Mid-RT remains validation-only because it requires paired pre-RT image/mask registration inputs, not one selected series |
| WltyBY HNTS-MRG | T2 MRI | Source cloned; all five Task 1 and all five Task 2 weights downloaded and reconstructed into runnable nnU-Net exports | **PASS:** public HNTS-MRG24 patient 101. Task 1 produced GTVp 3,252 and GTVn 21,322 voxels; valid 2-ROI RTSTRUCT/32 refs. Task 2 (mid-RT T2 + registered pre-RT T2 + registered pre-RT mask) produced GTVn 13,311 voxels; valid 1-ROI RTSTRUCT/27 refs. RTSTRUCT geometry was tested against synthetic, geometry-faithful MR DICOM references because the public source is NIfTI/MHA only |
| elitap HNTS-MRG | T2 MRI | Source cloned; pre-RT and mid-RT checkpoints downloaded | **PASS:** pre-RT produced GTVp 2,982 and GTVn 22,897 voxels (valid 2-ROI RTSTRUCT/33 refs); mid-RT produced GTVn 13,738 voxels (valid 1-ROI RTSTRUCT/27 refs). The repository omitted the checkpoint's required `3d_fullres_oversample_4c` plan, recovered exactly from checkpoint metadata. Geometry was tested against synthetic DICOM references because the public fixture is NIfTI/MHA only |
| STU-Net HNTS-MRG | T2 MRI | Both Task 1 B and S release checkpoints downloaded and deserialized successfully | **REPRODUCTION BLOCKED BY RELEASE PACKAGING:** repository contains only raw nnU-Net-v1 training checkpoints. It omits task plans/properties, preprocessing configuration, inference wrapper, fold/export metadata, and a compatible trained-model directory; these cannot be recovered from either checkpoint. The generic STU-Net repository supplies architecture code but not the HNTS task preprocessing. Running with invented plans would not validate the released models and was intentionally rejected |
| lnq-segmenter inguinal-v1 | CT | Complete, checksum verified; portal-integrated fold 0 | **PASS:** complete 702-slice CT produced 494 candidate voxels (max probability 0.999660). Valid one-ROI RTSTRUCT references 35 source images. |
| lnq-segmenter abdominopelvic-v1 | CT | Complete, checksum verified; portal-integrated fold 0 | **PASS:** complete 702-slice CT produced 106 candidate voxels (max probability 0.997274). Valid one-ROI RTSTRUCT references 16 source images. |
| lnq-segmenter axillary-v1 | CT | Complete, checksum verified; portal-integrated fold 0 | **PASS:** complete 702-slice CT produced 704 candidate voxels (max probability 0.999568) and a strictly valid one-ROI RTSTRUCT. This demonstrates that the prior empty five-fold result was ensemble/fixture-sensitive rather than a broken checkpoint. |
| lnq-segmenter mediastinal-v1 | CT | Complete, checksum verified | **PASS:** five-fold inference on a geometry-preserving thoracic crop found 92 candidate voxels; restored full geometry and produced a valid one-ROI RTSTRUCT with 14 contour-image references |
| Bouget CT_Lungs / UNet-SW32 / AGUNet / AGUNet-APG | contrast CT | Source, exact 1.0 GB archive and isolated Python 3.7/TensorFlow 1.14 environment installed | CT_Lungs executes; UNet-SW32 and AGUNet each produced valid RTSTRUCTs (65 and 128 referenced images). APG executes with explicit zero priors and produced a valid RTSTRUCT (38 refs), but clinically meaningful priors cannot be generated because that model is unreleased. Upstream automatic CT_Lungs handoff also has a filename bug (`pred_` vs `labels_`) |
| Raidionics Bouget deployment | contrast CT | Lungs + lymph-node ONNX weights downloaded and portal-integrated | **PASS:** nonempty mask; valid one-ROI RTSTRUCT references 127 source images from the 702-slice DICOM CT |
| LNQ2023 probabilistic atlas | CT | Source cloned | Published IMI Cloud share currently requires credentials (HTTP 401); weights/atlas blocked; CC BY-NC-ND |
| CompAI LNQ Model 7 | thoracic CT | GitLab source and exact 249 MB LFS checkpoint downloaded; Apache-2.0 | **PASS; PORTAL ENABLED:** release-specific lung normalization and Model 7 checkpoint produced 494 candidate voxels in standalone validation. The exact portal branch was then smoke-tested on the 735-slice CT: TotalSegmentator lung cropping, custom nnU-Net preprocessing, inference, full-geometry restoration and strict RTSTRUCT conversion all passed, yielding one ROI with 76 source-image references. PyTorch 2.6 release-checkpoint loading is handled explicitly and temporary data are cleaned. |
| DBDMP LNQ | CT | Source and 298 MB Google Drive checkpoint downloaded | **PASS; PORTAL ENABLED:** deeper inspection found that public `crop2lung` assigns z/y/x voxel indices directly as an x/y/z physical origin—twice and without spacing/direction. The repaired geometry-preserving implementation on the 735-slice whole-body CT produced 44,550 raw foreground voxels, of which 44,524 met the release intensity gate and 44,271 survived official component refinement. The strictly valid RTSTRUCT has one ROI and 64 referenced CT images. PyTorch 2.6 checkpoint loading was also made explicit for this trusted release. The checkpoint names an unavailable trainer (`nnUNetTrainer_LNQ4Tversky3`), so the compatible VNetv2 implementation is selected from its documented inference export. |
| WCODE-PIA LNQ | CT | Source cloned | Weights are distributed through Baidu; download/access remains blocked outside that service |
| LN-Seg-FM variants | H&N CT | Source plus 4.5 GB Google Drive weights for original nnU-Net, ResEnc-M, ResEnc-L, and SwinUNETR downloaded | **PASS:** all four fold-0 automatic checkpoints ran on the 516-slice planning H&N CT and produced nonempty visible-node masks: original 2,649, ResEnc-M 5,881, ResEnc-L 2,561, and SwinUNETR 4,033 voxels. Each produced a valid one-ROI RTSTRUCT referencing respectively 51, 61, 60, and 89 source images. All four are portal-integrated; visible job (private job identifier withheld) completed and returned four downloadable RTSTRUCTs. The Swin release required its documented nnU-Net 2.5.1/MonAI 1.3.0 runtime, isolated from the shared environment. These are visible-node candidate contours, not named elective neck levels |
| PAM | CT/MR/PET + user prompt | Source plus both public checkpoints downloaded; portal-integrated | **PASS:** interactive DICOM slice viewer supplies a positive point or 2D box. On HNTS-MRG24 T2 MR, PAM propagated 83,487 foreground voxels over 51 slices. A geometry bug in the upstream tutorial was corrected by restoring origin and direction, and the resulting one-ROI RTSTRUCT passed strict validation. A portal job completed and is downloadable. |
| BiomedParse v1/v2 | text-prompted CT/MR/PET | Source cloned | Hugging Face repository now approval-gated (HTTP 401); weights unavailable to worker |
| SAT3D | CT/MR/PET + 3D prompt | Source plus model and critic checkpoints downloaded | **PASS; PORTAL ENABLED:** repaired the public release's stale `build_sam3D_swin` alias, isolated its optional `timm`/`torchvision` dependency issue, and added prompt-centred 128³ inference with positive/negative point support. A T2 H&N MRI WebUI job produced 25,017 foreground voxels across 27 slices and a strictly valid one-ROI RTSTRUCT with 27 referenced source images. Requires at least one positive 3D point. |
| SAM-Med2D | CT/MR/PET slices + box/point prompts | Source plus 2.56 GB official checkpoint downloaded; portal-integrated | **PASS, SLICE-WISE:** fixed the upstream eager torchvision/NMS dependency and PyTorch 2.6+ checkpoint-loading incompatibility. A box on MR slice 41 produced 1,720 foreground pixels and a valid one-ROI/one-referenced-image RTSTRUCT. A portal job completed. The portal explicitly states that users must prompt every desired slice; no unvalidated 3D propagation is implied. |

## Local modality fixtures

A private secondary fixture pool was used only after DICOM-level modality, anatomy, geometry, and Frame of Reference checks. The available MR fixture was pelvic T1 and the PET fixture was whole-body PET. They establish MR/PET ingestion and inspection, but are not valid substitutes for head-and-neck T2 MRI or for a model requiring registered planning CT plus PET. Public HNTS-MRG24 patient 101 supplies appropriate pre/mid-treatment T2 MRI in NIfTI/MHA form. Multi-series and multi-modality uploads remain separated into explicit series selections; registration-dependent models require compatible Frame of Reference/geometry or a deliberate registration workflow.

## RTSTRUCT acceptance gate

Every automatic output must satisfy all of the following:

1. prediction shape and affine match the geometry-preserving model input;
2. every nonzero label has an explicit clinical/research ROI name;
3. the result is DICOM RT Structure Set Storage;
4. the output contains at least one nonempty ROI and contour set;
5. every referenced SOP Instance UID belongs to the selected source DICOM series.

An empty candidate-detection mask is reported as “no candidates detected”; contours are never fabricated merely to create a file.
# Additional catalog integrations (2026-08-10)

- The nine CADS `cads-model-open_v1.0.0` checkpoints (tasks 551–559) are
  installed under the CC BY-SA 4.0 model license and exposed as separately
  selectable CT tasks. Together they contain 167 fixed labels spanning organs,
  vertebrae, vessels, bones and muscles, ribs, RT OARs, brain tissues,
  head-and-neck OARs and body regions. Tasks 557 and 558 retain their official
  prerequisite inference/postprocessing (553, and 552+553 respectively). Every
  checkpoint passed DICOM-to-GPU-inference-to-RTSTRUCT validation on abdominal,
  thoracic or head CT; outputs contained 6–22 non-empty ROIs and referenced
  only the selected source series. Archive hashes are recorded in
  `models/manifests/cads-open.json`, with exact RTSTRUCT hashes and counts in
  `models/validation/cads_open_*.json`.
- Both official CLIP-Driven UniversalModel checkpoints are installed and
  selectable: U-Net and Swin UNETR. The U-Net produced all 32 released organ,
  tumor and cyst ROIs; Swin UNETR produced 28 non-empty ROIs on the same
  abdominal CT. Both referenced all 167 source images and passed strict
  RTSTRUCT validation. The release's MONAI 0.9 inversion restores arrays to the
  original voxel grid but leaves a stale 1.5-mm RAS affine; the runner replaces
  that header with the original model-input affine after direct-order spatial
  verification against CADS (liver Dice 0.9765). Exact provenance and evidence
  are in `models/manifests/universal-model.json` and
  `models/validation/universal_model_*.json`.
- All four official MRAnnotator fold-0 checkpoints are installed and enabled:
  abdomen (8 labels), shoulder/knee (10), pelvis/prostate (4), and spine (22).
  Each completed DICOM-to-GPU-inference-to-RTSTRUCT validation after the
  release-required LAS reorientation and restoration to source geometry. The
  pelvic test used an official Medical Segmentation Decathlon T2 prostate case;
  the shoulder/knee acceptance exercised the shoulder portion of that combined
  checkpoint. Provenance and exact output hashes are recorded in
  `models/manifests/mrannotator.json` and
  `models/validation/mrannotator_*.json`.
- All four PYCAD Model Zoo releases are installed and enabled: vertebrae CT,
  skull CT, cardiac MRI and knee-joint MRI. Each official fold-0 checkpoint
  produced a nonempty, strictly valid RTSTRUCT. The cardiac release documents
  its classes only as `h1`/`h2`/`h3`, so the portal preserves those names and
  warns that the anatomical ontology is unavailable. The knee test proves
  execution and geometry on an out-of-domain MRI fixture, not knee accuracy.
  Archive and output hashes are in `models/manifests/pycad-model-zoo.json` and
  `models/validation/pycad_*.json`.
- The official standalone SynthSeg v1 checkpoint is installed and enabled as
  `synthseg_v1_brain`. It produced all 31 released non-background brain labels
  on the TemplateFlow MNI152 T1 template and a strictly valid RTSTRUCT with 80
  referenced images. SynthSeg emits a canonical 1-mm segmentation, so the
  runner restores it to the selected DICOM geometry with nearest-neighbour
  resampling before contour conversion. Its exact TensorFlow 2.2/CUDA 10.1
  environment is incompatible with the H100's CUDA 12 stack and is therefore
  explicitly CPU-isolated with eight threads. Provenance is in
  `models/manifests/synthseg.json`.
- The MARS brainstem five-fold nnU-Net release is installed and enabled for
  whole-brain T1 MRI. Its three label identities—mesencephalon, pons and
  medulla oblongata—were verified from the upstream volume-extraction script,
  and all three were nonempty in a strictly valid RTSTRUCT. The companion
  MARS-WMH five-fold archive and source are also installed, but remain disabled
  because the official pipeline requires separate FLAIR and T1 series plus
  registration of T1 into FLAIR space; duplicating one selected series would
  not satisfy that contract. Checksums and provenance are recorded in
  `models/manifests/mars-brain-models.json`.

- `lungmask_r231`, `lungmask_ltrc_lobes`, and
  `lungmask_ltrc_lobes_r231` are installed from the official v0.0 release,
  checksum-verified, and enabled. All three completed DICOM-to-RTSTRUCT
  acceptance tests on thoracic CT. The strict validator reported respectively
  2, 5, and 5 non-empty ROIs, with all contour image references belonging to
  the source DICOM series. Exact hashes and result sizes are recorded in
  `models/manifests/lungmask.json`.
- `mrsegmentator` and `mrsegmentator_body_comp` are installed with the
  upstream checksum-aware downloader and enabled. The base model produced 21
  non-empty ROIs on thoracic CT. The MRI-only body-composition release was
  tested on an in-domain upper-abdominal/thoracic MRI fixture from the official
  TotalSegmentator MRI dataset and produced five non-empty tissue ROIs. Both
  outputs passed strict RTSTRUCT geometry/reference validation; provenance and
  result sizes are in `models/manifests/mrsegmentator.json`.
- `totalspineseg` is installed from both official `r20260730` model archives
  and enabled. The two-stage pipeline produced 16 non-empty, anatomically
  labelled spine/cord/canal ROIs on cervical T2 MRI and a strictly valid
  RTSTRUCT referencing 69 source images. The shared runtime pins Kornia 0.7.4
  because upstream `auglab 20260109` is incompatible with Kornia 0.8.3.
  Archive hashes and the acceptance result are recorded in
  `models/manifests/totalspineseg.json`.
- All ten exposed MOOSE clinical-CT groups are installed from the official
  release registry and enabled: body, cardiac, digestive, lungs, muscles,
  organs, peripheral bones, ribs, vertebrae, and body composition. Every group
  produced a non-empty, strictly valid RTSTRUCT; together the tests exercised
  117 output ROIs. The body-composition pipeline also uses the upstream fast
  vertebra model as its localization dependency. Per-task results and
  checkpoint-tree hashes are recorded in `models/manifests/moose.json`.
- The BAMF LiTS and Cervix-Cancer-CBCT-Segmentator nnU-Net v2 archives were
  downloaded from Zenodo, verified against both published MD5 values and local
  SHA-256 hashes, and run as full five-fold ensembles. BAMF produced seven
  non-empty RTSTRUCT ROIs on abdominal CT. The cervix release produced all four
  checkpoint labels and a valid RTSTRUCT on pelvic CT; this proves execution
  and geometry only, because an appropriate public CBCT acceptance fixture was
  not supplied by the release. Exact upstream label strings are retained (the
  ambiguous `CP` and `tumsomething` names are not silently reinterpreted).
- `mrisegmentator` 0.4.2 and its official five-fold Box checkpoint are
  installed and portal-integrated. On the 208-slice MRI fixture it produced
  16 non-empty structures from its 62-class label map; the RTSTRUCT references
  all 208 source images and passed strict validation. The upstream CLI help
  currently says 68 structures, but its distributed `dataset.json` contains
  62 foreground labels, so the portal reports the checkpoint-owned value.
- `vibesegmentator` dataset 100 is installed from all three official release
  archives/folds and runs through an explicit offline model root. Its 72-class
  output produced 33 non-empty ROIs on the 208-slice MRI fixture. The resulting
  37,750,432-byte RTSTRUCT references all source images and passed strict
  validation. Source commit, checkpoint-tree hash, and output hash are recorded
  in `models/manifests/vibesegmentator.json`.
- `dentalsegmentator` is installed through the MOOSE-compatible distribution
  of the public DentalSegmentator checkpoint. All five checkpoint-owned labels
  (upper skull, mandible, upper/lower teeth and mandibular canal) were nonempty
  on the head CT fixture. Its 2,930,026-byte RTSTRUCT references 126 source
  images and passed strict validation.
- The legacy nnU-Net v1 `Task004_Hippocampus` five-fold 3D-full-resolution
  ensemble is installed from the checksum-verified public model-zoo archive.
  It was tested on an official Medical Segmentation Decathlon T1 MRI converted
  to geometry-faithful research DICOM and produced both anterior and posterior
  hippocampal ROIs in a valid RTSTRUCT.
- The legacy nnU-Net v1 `Task006_Lung` five-fold 3D-full-resolution ensemble
  is installed from the checksum-verified public model-zoo archive. The full
  portal branch was tested on official MSD training CT `lung_053`, converted
  to a 252-slice CT DICOM series with signed HU storage. It produced one
  non-empty lung-tumor ROI, referenced 77 source images, and passed strict
  RTSTRUCT validation. The temporary 5.0 GB model and 9.2 GB fixture archives
  were removed after their checksums and the retained artifacts were recorded.
- The legacy nnU-Net v1 `Task002_Heart` five-fold 3D-full-resolution ensemble
  is installed from the checksum-verified public model-zoo archive. On official
  MSD cardiac MRI `la_007`, converted to a 130-slice research DICOM series, it
  produced one left-atrium ROI and a strictly valid 159,588-byte RTSTRUCT that
  references 76 source images.
- The legacy nnU-Net v1 `Task009_Spleen` five-fold 3D-full-resolution ensemble
  is installed from the checksum-verified public model-zoo archive. Official
  MSD CT `spleen_19` produced one non-empty spleen ROI and a strictly valid
  71,702-byte RTSTRUCT referencing 18 source images.
- The legacy nnU-Net v1 `Task003_Liver` five-fold 3D-full-resolution ensemble
  is installed from the checksum-verified public model-zoo archive. On a
  167-slice abdominal CT fixture it produced a non-empty liver ROI and a
  strictly valid 872,904-byte RTSTRUCT referencing 110 source images. The
  checkpoint's tumor label was empty on this non-tumor-specific fixture and
  was not fabricated.
- The legacy nnU-Net v1 `Task007_Pancreas` five-fold 3D-full-resolution
  ensemble is installed from the checksum-verified public model-zoo archive.
  On the abdominal CT fixture it produced a non-empty pancreas ROI and a
  strictly valid 217,706-byte RTSTRUCT referencing 57 source images. The tumor
  label was empty on this fixture and was not fabricated.
- The legacy nnU-Net v1 `Task008_HepaticVessel` five-fold 3D-full-resolution
  ensemble is installed from the checksum-verified public model-zoo archive.
  It produced a non-empty hepatic-vessel ROI and a strictly valid RTSTRUCT on
  the abdominal CT fixture. The tumor label was empty and was not fabricated.
- The legacy nnU-Net v1 `Task027_ACDC` five-fold 3D-full-resolution ensemble
  is installed from the checksum-verified public model-zoo archive. A cardiac
  MRI technical fixture produced all three checkpoint labels and a strictly
  valid 1,173,466-byte RTSTRUCT. The fixture is not an ACDC short-axis cine
  case, so the portal retains the model's sequence-domain warning.
- The legacy nnU-Net v1 `Task024_Promise` five-fold 3D-full-resolution
  ensemble is installed from the checksum-verified public model-zoo archive.
  It completed the full portal path and produced a structurally valid prostate
  RTSTRUCT on a whole-torso MRI fixture. This is an execution/geometry test
  only: the fixture is not PROMISE12 prostate T2 MRI, and the portal explicitly
  preserves that sequence requirement.
- The legacy nnU-Net v1 `Task055_SegTHOR` five-fold 3D-full-resolution
  ensemble is installed from the checksum-verified public model-zoo archive.
  On a 179-slice thoracic CT it produced all four checkpoint labels (esophagus,
  heart, trachea and aorta) and a strictly valid 1,037,898-byte RTSTRUCT that
  references all source images.
- The legacy nnU-Net v1 `Task017_AbdominalOrganSegmentation` BTCV five-fold
  3D-full-resolution ensemble is installed from the checksum-verified public
  model-zoo archive. Its exact 13-label ordering comes from the official
  nnU-Net dataset-conversion source. It produced 12 non-empty ROIs and a
  strictly valid 2,726,606-byte RTSTRUCT referencing all 167 source images;
  only the gallbladder label was empty on this fixture.
- The distinct legacy nnU-Net v1 `Task029_LITS` five-fold 3D-full-resolution
  ensemble is installed from its own checksum-verified public archive. It
  produced a non-empty liver ROI and a strictly valid RTSTRUCT referencing 110
  source images. Its tumor label was empty on this fixture and was not
  fabricated; it remains a separate task from `Task003_Liver`.
- The legacy nnU-Net v1 KiTS archive is installed under its actual upstream
  identifier, `Task048_KiTS_clean` (older catalogs and the conversion script
  call the dataset Task040). On public KiTS19 `case_00000`, both the kidney and
  kidney-tumor labels were non-empty. The resulting 1,421,284-byte RTSTRUCT
  references the complete 611-image source series, contains contours on 264
  images, and passed strict validation. This test also exposed and fixed the
  research NIfTI-to-DICOM helper's handling of files whose stored slice axis is
  not axis 2; it now canonicalizes from the NIfTI affine before export.
- The legacy nnU-Net v1 `Task010_Colon` five-fold 3D-full-resolution ensemble
  is installed from the checksum-verified public model-zoo archive. On public
  Medical Segmentation Decathlon CT `colon_053`, it produced a non-empty colon
  tumor ROI and a strictly valid 45,168-byte RTSTRUCT referencing the complete
  137-image source series, with contours on eight images.
- The MONAI Model Zoo `spleen_ct_segmentation` bundle version 0.6.1 is
  installed with its weights, TorchScript export, training/evaluation configs,
  metadata and license. The portal executes the bundle-owned preprocessing,
  sliding-window inference and inverse transforms through MONAI 1.6.0. On MSD
  `spleen_19`, it produced a non-empty spleen ROI and a strictly valid
  69,122-byte RTSTRUCT referencing the complete 51-image source series, with
  contours on 18 images.
- The MONAI Model Zoo `multi_organ_segmentation` DiNTS bundle version 0.0.6
  is installed with its model, TorchScript export, architecture-search code,
  configs and metadata. A narrowly scoped compatibility override permits the
  checksum-pinned official NumPy architecture pickle under PyTorch 2.6 while
  retaining safe checkpoint loading elsewhere. It produced six non-empty ROIs
  (artery, portal vein, liver, spleen, stomach and pancreas) and a strictly
  valid 1,240,254-byte RTSTRUCT on the abdominal CT fixture. Gallbladder was
  empty on this fixture and was not fabricated.
- The MONAI Model Zoo `pancreas_ct_dints_segmentation` bundle version 0.5.2
  is installed and uses the same checksum-scoped DiNTS compatibility path. On
  the abdominal CT fixture it produced non-empty pancreas and pancreatic-tumor
  ROIs and a strictly valid 244,048-byte RTSTRUCT with contours on 59 of the
  167 source images. This proves execution and geometry; target accuracy still
  requires pancreas-protocol local validation.
- The MONAI Model Zoo `pediatric_abdominal_ct_segmentation` DynUNet bundle
  version 0.4.6 is installed with its training/inference configuration and
  A100 TensorRT export. It produced all three labels (liver, spleen and
  pancreas) and a strictly valid 1,248,946-byte RTSTRUCT on the abdominal CT
  fixture. This is an execution/geometry test on an adult scan, not pediatric
  performance validation; the portal preserves the pediatric domain warning.
- The MONAI Model Zoo `renalStructures_UNEST_segmentation` bundle version
  0.2.7 is installed. On public KiTS19 `case_00000`, all three renal
  substructures—cortex, medulla and pelvicalyceal system—were non-empty. Its
  strictly valid 4,768,052-byte RTSTRUCT references the complete 611-image
  source series and contains contours on 352 images.
- The complete MONAI `renalStructures_CECT_segmentation` bundle version 0.2.3
  is also retained and checksum-recorded, but remains disabled. Inspection of
  its actual inference config proves that it requires co-registered arterial,
  venous and excretory CT volumes as three channels; the current one-series
  portal contract cannot satisfy that requirement without fabricating phases.
- The MONAI Model Zoo `swin_unetr_btcv_segmentation` bundle version 0.5.8 is
  installed. Its MONAI 1.4-era config is adapted per job by removing only the
  now-obsolete `SwinUNETR.img_size` constructor argument; the checksum-recorded
  upstream bundle remains unchanged. All 13 checkpoint labels were non-empty
  on the abdominal CT fixture, producing a strictly valid 3,272,662-byte
  RTSTRUCT with the complete 167-image source series referenced.
- The MONAI Model Zoo `wholeBody_ct_segmentation` bundle version 0.2.7 is
  installed with both official 1.5 mm and 3 mm checkpoints. Full-resolution
  inference completed but its 105-channel softmax exceeded the 48 GB GPU; the
  bundle-provided low-resolution checkpoint completed the uncropped 735-slice
  whole-body CT. It produced 102 non-empty ROIs and a strictly valid
  18,842,824-byte RTSTRUCT. Only face and gallbladder were empty and were not
  fabricated. Label names are loaded directly from bundle metadata.
- The MONAI Model Zoo `ventricular_short_axis_3label` bundle version 0.3.5 is
  installed with both its state-dict and TorchScript checkpoints. On the
  130-slice MSD cardiac MRI fixture it produced all three checkpoint labels
  (LV pool, myocardium and RV pool). The 1,974,990-byte RTSTRUCT references
  all source images and passed strict validation.
- The MONAI Model Zoo `wholeBrainSeg_Large_UNEST_segmentation` bundle version
  0.2.7 is installed. Its metadata defines 132 foreground labels (not 133
  foreground structures). A cropped MSD hippocampus T1 technical fixture
  produced 38 non-empty ROIs and a strictly valid 121,882-byte RTSTRUCT. This
  proves bundle execution and DICOM geometry only; full-brain anatomical
  performance remains subject to an in-domain, full-field-of-view validation.
- The MONAI Model Zoo `prostate_mri_anatomy` bundle version 0.3.6 is installed
  and enabled. Its actual metadata confirms a single-channel MRI contract and
  two foreground zones. Both central gland and peripheral zone were non-empty
  on a 208-slice whole-torso MRI technical fixture, producing a strictly valid
  2,000,172-byte RTSTRUCT. This is an execution/geometry test, not validation
  against the model's Prostate158 3T T2 acquisition domain.
- The complete MONAI `brats_mri_segmentation` bundle version 0.5.4 is retained
  and checksum-recorded but disabled. Its config and metadata require four
  co-registered inputs in exact T1c, T1, T2 and FLAIR order. The current portal
  selects one DICOM series, so enabling it would require fabricating or
  misidentifying channels.
- HD-BET 2.0.1 and its automatically distributed release-2.0.0 checkpoint are
  installed with the runtime home and weights redirected to the NFS model
  bundle. On an 82-slice head-and-neck T2 MRI fixture it produced a non-empty
  brain mask and a strictly valid 370,834-byte RTSTRUCT. The contour references
  28 source images; no geometry or image outside the selected series is used.
- HD-CTBET 1.1 is installed from upstream commit `78da973b` with its official
  fold-0 CT checkpoint. Upstream's exporter compared a 3-D array directly to a
  three-element target shape and failed with a broadcasting error; the retained
  one-line patch compares `seg_old_size.shape`, with the exact patch and hashes
  recorded in `models/manifests/hd-ctbet.json`. On a 142-slice head CT it
  produced a non-empty brain ROI and a strictly valid 598,312-byte RTSTRUCT
  referencing 92 source images.
- LungTumorMask v1.3.1 is installed from upstream commit `d1bca4b8` with the
  released `dc_student.pth` checkpoint stored on NFS. Its 2021 MONAI imports
  are no longer available, so the compatibility runner preserves the released
  49,117,073-parameter network (all 52 checkpoint tensors match) and delegates
  lung localization to the installed lungmask R231 model. On the 179-slice
  thoracic CT fixture it produced a non-empty lung-tumor candidate ROI and a
  strictly valid 84,664-byte RTSTRUCT with contours referencing 45 source
  images. This proves execution and geometry, not tumor accuracy on the
  technical fixture.
- TrackRAD Track'n'Treat source and its checksum-verified 468 MB averaged
  MedSAM2 checkpoint are installed on NFS. It remains deliberately disabled:
  upstream inference requires sagittal 2D cine-MRI plus an explicit first-frame
  target mask and returns a temporal MHA mask sequence, whereas the portal's
  current contract is a static DICOM series producing RTSTRUCT.
- MedSAM2 source and all six public WangLab checkpoints are stored on NFS. The
  FLARE25 RECIST CT-lesion checkpoint loads strictly and is portal-integrated
  for a positive/negative point set or 2-D box on one axial seed slice. On the
  released FLARE23 public fixture, box propagation scored Dice 0.9098 and
  single-point propagation scored Dice 0.9073. The exact DICOM branch on MSD
  lung case 53 produced a one-ROI RTSTRUCT referencing 92 source images with
  zero physical bounding-box delta. The model is restricted to research and
  education use, and every propagated lesion contour requires review.
- The separate ImprintLab Medical-SAM2 GPL release and its official 38.9M-
  parameter tiny checkpoint are also installed. All 468 checkpoint tensors
  load strictly. Box propagation ran on the public FLARE lesion technical
  fixture (Dice 0.4613, explicitly out of its BTCV domain), and the complete
  DICOM branch produced a one-ROI RTSTRUCT referencing 73 source images with
  zero physical bounding-box delta. It remains clearly distinguished from the
  higher-performing WangLab CT-lesion checkpoint in the model picker.
- nnInteractive 2.5.1 source commit
  `bbe12fdccc876cb2d4e0a47133811e362608e000` and the official v1.0 fold-0
  checkpoint are installed on NFS. The checkpoint SHA-256 is
  `b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9`
  and its model license is CC BY-NC-SA 4.0 (distinct from the Apache-2.0
  source license). A positive point on the public MNI T1 fixture produced a
  non-empty source-grid mask and a strictly valid, orientation-audited
  RTSTRUCT with model-specific DICOM identity. The portal supports positive
  and negative points and the checkpoint's required 2-D box interaction.
- VISTA3D/NV-Segment-CT source commit
  `f9f5f51b589e5dc9c23c453cf5138398e4084056` and Hugging Face revision
  `afb51518689f71e6abb367ee6301b2cd0225c66a` are installed on NFS. The
  871,970,895-byte checkpoint has SHA-256
  `c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b`.
  A positive-point run on the public MSD spleen CT fixture produced a non-empty
  3-D mask and a one-ROI RTSTRUCT referencing 18 source images; strict
  orientation validation passed. The runner records and applies a one-line
  MONAI 1.6 tensor compatibility repair without modifying upstream source.
- VoxTell source commit `8ef0332aec5d5dff925d6f6370386c00334b57ab`
  and Hugging Face revision `0809ac94c5ed594198bf4e85d63245cf222464d7`
  are installed on NFS with the official v1.1 checkpoint, its 14,194-label
  embedding bank, and the Qwen3-Embedding-4B fallback for unseen free text.
  The prompt `spleen` produced 101,225 foreground voxels on the public MSD
  fixture and a one-ROI RTSTRUCT referencing 19 source images. Direct
  source-grid conversion matched the ROI physical bounding box exactly.
- SegVol source commit `5768c9795794a327a2629ce7f137ab03d5ea8378`
  and Hugging Face revision `516701d5bef430604a13f5db7ce896149d075663`
  are installed on NFS. Its 723,726,667-byte checkpoint has SHA-256
  `500f2758a8f989339b2b2baf09a819169bc87549795193d3cfe505726ac0b399`.
  The exact released MONAI 0.9 architecture loads all 180,890,064 parameters;
  a documented one-line seed-range repair makes MONAI 0.9 compatible with
  NumPy 2 without changing model computation. Text-prompted `spleen` inference
  achieved Dice 0.9801 on the publisher's fixture. The same prompt on the MSD
  DICOM fixture produced a one-ROI RTSTRUCT referencing 18 source images with
  zero per-ROI physical bounding-box delta. Text, point and 3-D box prompts are
  exposed by the portable runner.
- UniverSeg source commit `833a0c34c65e38d675e21bd48ddec6797cc03259`
  and its sole official release checkpoint are installed on NFS. The
  4,748,449-byte OpenRAIL++-M research checkpoint has SHA-256
  `62c98c48c08bc05f84ed19187844d5eb7e6b7a48d2f690f2033fd8de04543884`;
  all 1,182,785 parameters load strictly. With 64 support image-mask pairs it
  achieved Dice 0.9101 for label 1 on the publisher-linked OASIS fixture. The
  model is visible but disabled because its class is defined by uploaded 2-D
  support images and masks, an input type the current portal does not accept.
  Reusing a fixed unrelated support set would not implement few-shot contouring.
- HippUnfold source commit `544fdb6e37eb42567b1ad3c7f4901a180bdb19b8`
  and both catalog checkpoint releases are staged on NFS. The ADNI T1 archive
  is 2,452,807,680 bytes (SHA-256
  `55be492ccc65b4fd60bdd18dde2e63c0c5676060cb950b2bbbd2fd12ea57ebf8`)
  and contains all five nnU-Net-v1 folds. The Maguire T2 archive is
  1,138,683,578 bytes (SHA-256
  `ee4e28d246cb1acb86df6b99be50c66b9d762356385643092877a9723a552de7`)
  and contains all five nnU-Net-v2 folds. Both are visible as acceptance-pending
  until the matching development runtime produces native-space subfields and
  a strictly validated, orientation-audited RTSTRUCT. Limited-field T2 input
  may require a separately registered T1 series and must not be treated as a
  valid single-series case.
- HD-GLIO-AUTO source and checksum-verified v2 checkpoint are installed, and
  its published inference code confirms a
  mandatory native T1, post-contrast T1, T2 and FLAIR input set. The pipeline
  then performs HD-BET brain extraction and rigid registration before tumor
  inference. It remains visible but disabled until multi-series selection is
  implemented; supplying one series four times would be invalid.
- All twelve MATTO-GBM TRANSCAN archives match their published Zenodo MD5
  checksums and are extracted on NFS. The single-channel enhancing-tumor model
  is enabled with the publisher's HD-BET and 0.1/99.9-percentile preprocessing.
  Its five-fold best-checkpoint ensemble produced a non-empty enhancing-tumor
  ROI and a strictly valid 271,356-byte RTSTRUCT on the axial T1c fixture,
  referencing 44 source images. The other eleven models remain visible as an
  installed family while their FLAIR, SUV PET, derived-mask, atlas-orientation,
  or eyes-masking contracts are implemented and tested.
- MultiTalent V1 source commit
  `0bc7b2c32e31cb00008354f2420b65b85ef09578` and the complete official
  Zenodo Task100 archive are installed on NFS. The 4,634,439,232-byte archive
  matches publisher MD5 `a86617d4c71ef7b97d9eaf931df390cd` and contains
  standard U-Net and residual-encoder checkpoints for all five folds plus the
  all-data checkpoints. The release implements 47 independent, overlapping
  dataset-specific heads, not the newer repository branch's 117-class claim.
  The standard and residual all-data checkpoints have SHA-256
  `03851e77590fcd89b96320b666068971a484593ee948573feae6f88dc3c43fd0`
  and `506b91701e0957d10b67c0375275053a1f4364d3febe81289f34b5a1237bb775`.
  Both loaded and ran on the H100. Their overlap-preserving RTSTRUCTs contained
  47 and 44 non-empty ROIs respectively, referenced all 51 source CT images,
  had zero per-ROI physical bounding-box delta, and carry distinct model names
  in StructureSetLabel, StructureSetName and SeriesDescription. The Zenodo
  record is non-commercial and the constituent public datasets may impose
  additional terms.
- HECKTOR 2020 winning-solution source commit
  `5ebc6774d178139b52abebbe926b4cafabe3e29d` and all eight official PET/CT
  ensemble checkpoints are installed on NFS. The 696,677,684-byte Dropbox
  archive has SHA-256
  `1666bad2c4355bb059a1339e6fc605f65838da5bd931139030c24ccab52d02d7`.
  After removing the publisher's DataParallel `module.` prefix where present,
  every 310-tensor state dictionary loads strictly into the released
  `FastSmoothSENormDeepUNet_supervision_skip_no_drop` architecture. The family
  remains disabled because published inference requires registered PET and CT
  plus a patient-specific 144 mm physical head-and-neck bounding box. That
  bounding-box CSV is part of the HECKTOR challenge data, whose official page
  currently says access cannot be granted while hospital agreements are being
  extended. Guessing a crop or running a pelvic PET fixture would not prove
  valid inference.

## Source synchronization and interpretation of evidence

The source now includes cached catalogs, resumable uploads, per-task process-tree termination, read-only bundle mirrors, TotalSegmentator auxiliary-label filtering and the CPU prostate SBRT crop tool. Direct source-grid RTSTRUCT conversion and independent physical round-trip checks replace the earlier orientation workaround. See [operations](docs/OPERATIONS.md) for limits and configuration.

The captured deployment returned 253 tasks, 182 enabled. These counts describe that installation; fresh installations must pass their own preflight and orientation gates. All model-specific warnings and numerical results above are retained. Private fixture/job identifiers and private artifact hashes are withheld without relabeling the associated measurements as public-fixture validation. Public fixture maps and external private override configuration are documented in [operations](docs/OPERATIONS.md).

The dated SCT statements above describe successive historical states. Current task definitions are in `catalog/sct.json`, with additional recorded validation in `models/validation/`. Neither a copied marker nor this source synchronization establishes clinical or manuscript readiness.
