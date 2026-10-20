#!/usr/bin/env python3
"""Original Prostate158 tumour UNet and deterministic inference transforms."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np

from prostate_models import CHANNEL_ROLES, model_folder, runtime_provenance
from prostate_postprocess import write_json

TASK = 'prostate158_tumor_bpmri'
MODEL_CONFIG = {'spatial_dims': 3, 'in_channels': 3, 'out_channels': 2,
                'channels': (16, 32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2, 2),
                'num_res_units': 4, 'act': 'PRELU', 'norm': 'BATCH', 'dropout': 0.15}


def preprocessing():
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
                                  Orientationd, ScaleIntensityd, NormalizeIntensityd, EnsureTyped)
    keys = list(CHANNEL_ROLES)
    return Compose([LoadImaged(keys), EnsureChannelFirstd(keys),
                    Spacingd(keys, pixdim=(0.5, 0.5, 0.5), mode='bilinear'),
                    Orientationd(keys, axcodes='RAS'), ScaleIntensityd(keys, minv=0, maxv=1),
                    NormalizeIntensityd(keys), EnsureTyped(keys)])


def prepare_channels(paths):
    """Published independent transforms require physically corresponding grids."""
    data = preprocessing()({role: str(paths[role]) for role in CHANNEL_ROLES})
    reference = data['t2w']
    for role in CHANNEL_ROLES[1:]:
        if (tuple(data[role].shape) != tuple(reference.shape)
                or not np.allclose(np.asarray(data[role].affine), np.asarray(reference.affine), rtol=0, atol=1e-3)):
            raise ValueError('Prostate158 requires prealigned T2/ADC/DWI grids after published transforms; no registration is supplied')
    return data


def restore_segmentation(labels_xyz, transformed_affine, reference_path, destination):
    import nibabel as nib
    import SimpleITK as sitk
    from prostate_postprocess import restore_to_reference, assert_same_grid
    native = Path(destination).with_name('transformed-label.nii.gz')
    nib.save(nib.Nifti1Image(labels_xyz.astype(np.uint8), np.asarray(transformed_affine)), native)
    try:
        reference = sitk.ReadImage(str(reference_path))
        restored = restore_to_reference(sitk.ReadImage(str(native)), reference)
        assert_same_grid(restored, reference)
        sitk.WriteImage(restored, str(destination), True)
        return restored
    finally:
        native.unlink(missing_ok=True)


def predict_logits(image, model, *, amp, torch_module=None, inferer=None):
    """Match the author's evaluator: no gradients, CUDA autocast, input-device aggregation."""
    if torch_module is None:
        import torch as torch_module
    if inferer is None:
        from monai.inferers import sliding_window_inference as inferer
    with torch_module.no_grad():
        with torch_module.cuda.amp.autocast() if amp else nullcontext():
            return inferer(image, roi_size=(96, 96, 96), sw_batch_size=4,
                           predictor=model, overlap=0.5, mode='constant')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=[TASK])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import SimpleITK as sitk
    from monai.networks.nets import UNet
    from monai.transforms import KeepLargestConnectedComponent
    from monai.utils import get_torch_version_tuple
    # Pinned prostate158/utils.py USE_AMP and train.py SupervisedEvaluator.
    amp = get_torch_version_tuple() >= (1, 6)
    work, output = args.work, args.output
    provenance = runtime_provenance(TASK, args.model_root)
    provenance.update(json.loads((work / 'context.json').read_text()))
    provenance.update({'weights_licence': 'CC BY 4.0', 'model_config': MODEL_CONFIG,
                       'channel_mapping': {'t2w': 'T2W', 'adc': 'ADC', 'hbv': 'DWI'},
                       'spacing_mm': [0.5, 0.5, 0.5], 'orientation': 'RAS',
                       'sliding_window': {'roi_size': [96, 96, 96], 'batch_size': 4, 'overlap': 0.5, 'mode': 'constant'},
                       'precision': {'autocast_enabled': amp, 'device_type': 'cuda',
                                     'autocast_dtype': 'float16' if amp else None,
                                     'policy': 'Author evaluator USE_AMP: torch version >= 1.6',
                                     'aggregation_device': 'input device (cuda)'},
                       'postprocessing': 'argmax then largest foreground component, 26-connectivity; nearest-neighbour restoration'})
    data = prepare_channels({role: work / 'native' / f'{role}.nii.gz' for role in CHANNEL_ROLES})
    image = torch.cat([data[role].as_tensor() for role in CHANNEL_ROLES], dim=0).unsqueeze(0)
    model = UNet(**MODEL_CONFIG).to('cuda').eval()
    checkpoint = torch.load(args.model_root / model_folder(TASK) / 'tumor.pt', map_location='cpu', weights_only=True)
    # The author saves the plain state_dict; do not silently accept other releases.
    model.load_state_dict(checkpoint, strict=True)
    logits = predict_logits(image.to('cuda'), model, amp=amp)
    labels = KeepLargestConnectedComponent(applied_labels=[1])(logits.argmax(1))[0].cpu().numpy()
    restored = restore_segmentation(labels, data['t2w'].affine, work / 'native/t2w.nii.gz', work / 'restored.nii.gz')
    sitk.WriteImage(restored, str(output) + '.segmentation.nii.gz', True)
    provenance['status'] = 'nonempty' if sitk.GetArrayViewFromImage(restored).any() else 'empty'
    write_json(str(output) + '.provenance.json', provenance)
    write_json(work / 'result.json', {'empty': provenance['status'] == 'empty'})


if __name__ == '__main__':
    main()
