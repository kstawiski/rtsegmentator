#!/usr/bin/env python3
"""Combine UniversalModel's 32 overlapping binary masks into one label map."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

NAMES = [
    "Spleen", "Right Kidney", "Left Kidney", "Gall Bladder", "Esophagus",
    "Liver", "Stomach", "Aorta", "Postcava", "Portal Vein and Splenic Vein",
    "Pancreas", "Right Adrenal Gland", "Left Adrenal Gland", "Duodenum",
    "Hepatic Vessel", "Right Lung", "Left Lung", "Colon", "Intestine",
    "Rectum", "Bladder", "Prostate", "Left Head of Femur",
    "Right Head of Femur", "Celiac Truck", "Kidney Tumor", "Liver Tumor",
    "Pancreas Tumor", "Hepatic Vessel Tumor", "Lung Tumor", "Colon Tumor",
    "Kidney Cyst",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_directory", type=Path)
    parser.add_argument("model_input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reference = nib.load(args.model_input)
    combined = np.zeros(reference.shape, dtype=np.uint8)
    counts = {}
    for label, name in enumerate(NAMES, 1):
        candidates = list(args.mask_directory.glob(f"*_{name}.nii.gz"))
        if len(candidates) != 1:
            raise SystemExit(f"expected one UniversalModel mask for {name!r}, found {len(candidates)}")
        image = nib.load(candidates[0])
        if image.shape != reference.shape:
            raise SystemExit(f"UniversalModel output shape mismatch for {name}")
        # MONAI 0.9 Invertd restores the voxel array to the original grid but
        # SaveImaged retains the temporary 1.5-mm RAS affine. The released
        # pipeline therefore writes stale metadata despite a correctly inverted
        # array. We deliberately take geometry from model_input below. This was
        # independently checked against CADS on the acceptance CT (liver Dice
        # 0.9765 in direct voxel order; axis flips fail the comparison).
        mask = np.asanyarray(image.dataobj) > 0.5
        counts[label] = int(mask.sum())
        # Later classes deliberately win overlaps. This preserves the released
        # lesion classes over their containing organs and kidney cysts over kidney.
        combined[mask] = label
    if not np.any(combined):
        raise SystemExit("UniversalModel produced no foreground masks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(combined, reference.affine, reference.header), args.output)
    print({"foreground_voxels": int(np.count_nonzero(combined)), "source_mask_voxels": counts})


if __name__ == "__main__":
    main()
