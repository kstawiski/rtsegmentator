"""Structured, flushed task progress shared by CPU converters and the runner."""
from pathlib import Path

import json
import time


class Progress:
    def __init__(self):
        self.stage = None
        self.emitted = float("-inf")

    def emit(self, stage, done=None, total=None, *, force=False, **fields):
        now = time.monotonic()
        if not force and stage == self.stage and now - self.emitted < 1:
            return
        payload = {"stage": stage, **fields}
        if done is not None: payload["done"] = done
        if total is not None: payload["total"] = total
        print("RTSEG_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)
        self.stage, self.emitted = stage, now


def command_stage(command):
    """Describe dispatcher subprocesses without changing their invocation."""
    names = {Path(str(arg)).name for arg in command[:3]}
    if 'dicom_series_to_nifti.py' in names:
        return 'conversion'
    if names & {'convert_validate_rtstruct.py', 'convert_masks_validate_rtstruct.py'}:
        return 'rtstruct_contours'
    if names & {'keep_labels.py', 'resample_label_to_reference.py', 'threshold_probability.py',
                'average_probabilities.py', 'combine_sct_outputs.py', 'combine_universal_model.py',
                'nrrd_to_nifti.py', 'labels_from_segmentation_lut.py', 'lut_to_labels.py',
                'totalseg_combine_masks', 'nnUNet_ensemble'}:
        return 'postprocessing'
    if 'resample_image_to_reference.py' in names:
        return 'resampling'
    if names & {'crop_to_mask.py', 'normalize_masked_nifti.py', 'pet_bqml_to_suv1000.py',
                'prepare_isrt_pet_config.py', 'reorient_nifti.py', 'orientation_contract.py'}:
        return 'staging'
    return 'inference'
