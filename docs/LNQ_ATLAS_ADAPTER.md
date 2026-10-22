# LNQ2023 atlas adapter

`run_lnq_atlas.py` runs source revision `e48a74f3223e5d9ef06f79fb08282450c29f7a2d`.
The five-fold archive is installed at `weights/lnq-atlas-20261015/`. Its
`provenance.json` records archive SHA-256
`ae0735e8444179deb51114dbf3fc469c854758f06849d69747985bbbaa3fecba` and a
SHA-256 for every file returned by `required_files()`. Folds 0–4 passed strict
CPU loading with no missing or unexpected keys. H100 acceptance passed on 2026-10-22 on a public CT (MSD Task06 lung_053): two production runs and the released predictor gave identical masks (17,234 voxels).
The task stays off unless that marker and the orientation marker pass.

The adapter requires:

- All five `checkpoint_best.pth` files, `plans.json`, `dataset.json`, and the
  six atlas images under `weights/lnq-atlas-20261015/weights`.
- nnU-Net 2.2.1 and dynamic-network-architectures 0.2, in
  `runtime-envs/lnq-atlas-ref`. The adapter calls
  `nnUNetTrainer.build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=False)`.
  Output heads come from the label manager. `determine_num_input_channels`
  must return 3. The predictor is constructed with `perform_everything_on_gpu=True`.
- A separate TotalSegmentator 1.5.7 runtime. Set `RTSEG_LNQ_TOTAL_PYTHON`, or
  pass `--total-python`. Registration uses the v1 organ IDs. A v2 mask is not
  a substitute.
- Task 256 fast weights at `weights/totalsegmentator-v1.5.7` (override
  `RTSEG_LNQ_TOTAL_WEIGHTS` or pass `--total-weights`). The launcher refuses
  to start if `TOTALSEG_WEIGHTS_PATH` is unset, the pinned trainer directory
  `nnUNetTrainerV2_ep8000_nomirror__nnUNetPlansv2.1` is missing or ambiguous,
  or `model_final_checkpoint.model`, `model_final_checkpoint.model.pkl`, or
  `plans.pkl` is missing or does not match its pin. Inference does not
  download weights.

`run_total_v1.py` isolates `Path.home()` in the child process because
TotalSegmentator 1.5.7 ignores `TOTALSEG_HOME_DIR`. It turns usage telemetry
off in that private configuration. Large `nnunet_tmp_` files stay in the
supplied project directory. Multiprocessing sockets use the short `TMPDIR`.

The adapter keeps the published steps: CT, atlas distance map and probability
map; registration and lung crop; five-fold average of restored foreground
softmax probabilities; atlas threshold 0.15–0.30 from
`get_probability_threshold_probmap` with t=0.3; lung convex hull; removal of
components whose equivalent ellipsoid diameter is below 5 mm on any axis.
Input is reoriented to DICOM LPS. Oblique CTs are rejected. The last step
resamples the mask onto the original CT with nearest-neighbour interpolation.
Upstream `process.py` returns the mask on the registration grid and does not
do that resample. The GPU harness applies the same resample before Dice.

The output is one ROI, `Mediastinal_Lymph_Node_Candidate`: a visible
mediastinal lymph-node candidate, not an elective nodal region or a station map.
No remaining candidate completes with no RTSTRUCT. The exporter reports
`empty prediction: no RTSTRUCT created`, and the portal records that as a
completed task with no contour.

This task is routed to the H100 worker. On the P6000 portal (compute
capability 6.1), keep `lnq_atlas` in `REMOTE_GPU_TASKS`. Preflight probes
`runtime-envs/lnq-atlas-ref/bin/python` on the execution host. An ordinary
`sm_XY` cubin covers a device with the same major capability and a minor
capability greater than or equal to Y. The reference lock's torch 2.0.1+cu118
build lists `sm_60`, so that cubin covers compute capability 6.1. That is a
compatibility rule, not a completed P6000 inference run. `sm_90a` is not
treated as `sm_90`. The acceptance marker stores the GPU name and compute
capability. An H100 marker does not enable the task on a P6000. `envs/lnq`
is not used for this task.

Code licence: CC BY-NC-ND 4.0. Archive weight terms are not stated. Owner
accepted these terms on 2026-10-22.

Engelson et al., MELBA 2024, sections 4.1–4.3 and Tables 2 and 4. Unless a
result is marked otherwise, the paper evaluates on the LNQ test set: 100
patients and 861 reference nodes. The ablation that uses all modifications,
a binarization threshold of 0.2 and no minimum-diameter filter had Dice
0.6033 and found 57% of reference nodes (LN-found 0.5654), compared with 27%
(LN-found 0.2749) when training on CT only. The challenge submission, a
five-model ensemble with threshold 0.3 and a 5 mm minimum diameter, had Dice
0.5690 and ASSD 6.8976 mm on that same test set. The authors state that
post-processing fit on the small validation set did not carry over to the
test set. The installed adapter uses the submission settings (threshold 0.3,
5 mm), not the 0.2 ablation.

```sh
python run_lnq_atlas.py --input ct.nii.gz --output prediction.nii.gz \
  --work NEW_PRIVATE_WORK_DIRECTORY --source PINNED_LNQ_SOURCE \
  --weights EXTRACTED_WEIGHTS --manifest provenance.json \
  --total-python TOTAL_V1_ENV/bin/python \
  --total-weights WEIGHTS/totalsegmentator-v1.5.7
```

The network contract, set in the adapter and recorded in
`deterministic_contract.json`, is `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`nnUNet_compile=false`, and, after `nnUNetPredictor.__init__`,
`cudnn.deterministic=True` with `cudnn.benchmark` off. nnU-Net 2.2.1 does not
set `cudnn.benchmark` in that constructor. The recorded value is the string
`off`. The task-256 contract is separate. TotalSegmentator 1.5.7 restores
`nnUNetTrainerV2_ep8000_nomirror` with `deterministic=False` and `fp16=True`,
so the trainer turns `cudnn.benchmark` on. `predict_cases_fastest` does not
clear that flag. `run_total_v1.py` sets deterministic cuDNN and benchmark off
after `load_model_and_checkpoint_files` returns, then writes
`total_deterministic_contract.json`. Mixed precision, autocast, fastest mode,
TTA off, one resampling thread and the published 3 mm resample stay as
released. Thread caps are set in that child process only.

The acceptance harness runs the portal job twice and requires identical
task-256 masks and identical final masks before the reference comparison.
It keeps both pre-threshold foreground probability maps on the original CT
grid. Their maximum absolute difference, and the probability values at
mask voxels that differ, are diagnostics. The pass criterion remains exact
mask agreement.

The reference process uses `lnq-upstream-ref` (nnU-Net v2.2.1, commit
`84512916ff61ff027245ac175071bf7e0abbc29b`, the last release that still
provides `training.dataloading.data_loader_2d` and
`perform_everything_on_gpu`). Production uses `runtime-envs/lnq-atlas-ref`,
built from that environment's lock. The reference imports the released
predictor unmodified. The harness does not patch torch, cuDNN, or the
predictor in the adapter process. Agreement compares the mask `run_task.py`
passes to the RTSTRUCT exporter with the reference mask on the CT grid.
Foreground Dice must be exactly 1. A failure writes no marker.

Remaining reference adaptations, and why they do not change the network
numerics: storage paths are redirected; the reference reuses the adapter's
TotalSegmentator mask and runs registration again (that stage is not an
independent comparison); the reference output is resampled with the same
nearest-neighbour call the adapter uses to leave the registration grid
(grid only, default value 0, uint8). `torch.load(..., weights_only=False)`
loads the trusted checkpoints. Torch 2.0.1 accepts that argument. It does
not change tensor values.

Sources: [pinned repository](https://github.com/MICAI-IMI-UzL/LNQ2023/tree/e48a74f3223e5d9ef06f79fb08282450c29f7a2d),
[public share](https://cloud.imi.uni-luebeck.de/s/24q2qr4ycA7CgSP),
[MELBA paper](https://www.melba-journal.org/papers/2024:009.html),
[repository licence, CC BY-NC-ND 4.0](https://github.com/MICAI-IMI-UzL/LNQ2023/blob/e48a74f3223e5d9ef06f79fb08282450c29f7a2d/LICENSE).
