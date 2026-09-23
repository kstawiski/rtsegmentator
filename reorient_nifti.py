#!/usr/bin/env python3
"""Reorient a NIfTI voxel grid while preserving its world-space geometry."""

import argparse
from pathlib import Path

import nibabel as nib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("orientation", choices=("LAS", "RAS", "LPS"))
    args = parser.parse_args()
    image = nib.load(args.input)
    start = nib.orientations.io_orientation(image.affine)
    target = nib.orientations.axcodes2ornt(tuple(args.orientation))
    transform = nib.orientations.ornt_transform(start, target)
    result = image.as_reoriented(transform)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, args.output)
    print(f"{''.join(nib.aff2axcodes(image.affine))} -> {''.join(nib.aff2axcodes(result.affine))}; {image.shape} -> {result.shape}")


if __name__ == "__main__":
    main()
