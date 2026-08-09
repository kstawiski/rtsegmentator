import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    args = parser.parse_args()

    sys.path.insert(0, args.source)
    sys.path.insert(0, os.path.join(os.path.dirname(args.source), "Algorithm"))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from lung_crop_without_seg import (
        crop_ND_volume_with_bounding_box,
        get_ND_bounding_box,
        get_largest_k_components,
        lungmask,
    )
    from process_utils import label_refine
    from scipy import ndimage

    image = sitk.ReadImage(args.input)
    # The released crop2lung assigns z/y/x voxel indices directly as an x/y/z
    # physical origin (twice, and without spacing/direction).  Reproduce its
    # crop while carrying the combined array offset back through the source
    # image geometry.
    full = sitk.GetArrayFromImage(image)
    body_mask = ndimage.binary_opening(full > -600, np.ones((3, 3, 3)), iterations=2)
    body_mask = get_largest_k_components(body_mask, 1)
    body_min, body_max = get_ND_bounding_box(body_mask, margin=[5, 10, 10])
    body = crop_ND_volume_with_bounding_box(full, body_min, body_max)
    body_image = sitk.GetImageFromArray(body)
    body_image.SetSpacing(image.GetSpacing())
    lung = lungmask(body_image)
    lung_min, lung_max = get_ND_bounding_box(lung)
    centre = np.asarray(body.shape) // 2
    for axis in range(1, 3):
        if (centre[axis] - lung_min[axis]) * 0.5 > (lung_max[axis] - centre[axis]) and lung_min[axis] < centre[axis]:
            lung_max[axis] = body.shape[axis] - lung_min[axis]
        elif lung_min[axis] > centre[axis]:
            lung_min[axis] = body.shape[axis] - lung_max[axis]
    crop_array = crop_ND_volume_with_bounding_box(body, lung_min, lung_max)
    total_min_zyx = np.asarray(body_min) + np.asarray(lung_min)
    cropped = sitk.GetImageFromArray(crop_array)
    cropped.SetSpacing(image.GetSpacing())
    cropped.SetDirection(image.GetDirection())
    cropped.SetOrigin(image.TransformIndexToPhysicalPoint(tuple(int(v) for v in total_min_zyx[::-1])))
    properties = {
        "sitk_stuff": {
            "spacing": cropped.GetSpacing(),
            "origin": cropped.GetOrigin(),
            "direction": cropped.GetDirection(),
        },
        "spacing": list(cropped.GetSpacing())[::-1],
    }
    array = sitk.GetArrayFromImage(cropped).astype(np.float32)

    # PyTorch 2.6 changed torch.load's default to weights_only=True, while the
    # authors' trusted checkpoint contains NumPy scalar training metadata.
    trusted_torch_load = torch.load
    def load_release_checkpoint(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return trusted_torch_load(*load_args, **load_kwargs)
    torch.load = load_release_checkpoint

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        args.model, use_folds=("all",), checkpoint_name="checkpoint_final.pth"
    )
    prediction = predictor.predict_single_npy_array(array[None], properties, None, None, False)
    raw_values, raw_counts = np.unique(prediction, return_counts=True)
    print("raw", dict(zip(raw_values.tolist(), raw_counts.tolist())))
    foreground = prediction != 0
    in_range = (array <= 157 * 1.5) & (array >= -36 * 1.5)
    print("raw foreground in intensity range", int(np.count_nonzero(foreground & in_range)),
          "of", int(np.count_nonzero(foreground)))
    if args.raw_output:
        raw_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
        raw_image.CopyInformation(cropped)
        sitk.WriteImage(sitk.Resample(raw_image, image, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8),
                        args.raw_output, True)
    prediction = label_refine(array, prediction, np.prod(properties["spacing"]))
    cropped_seg = sitk.GetImageFromArray(prediction.astype(np.uint8))
    cropped_seg.CopyInformation(cropped)
    restored = sitk.Resample(
        cropped_seg, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    sitk.WriteImage(restored, args.output, True)
    values, counts = np.unique(sitk.GetArrayFromImage(restored), return_counts=True)
    print(dict(zip(values.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
