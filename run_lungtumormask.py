#!/usr/bin/env python3
"""Run the LungTumorMask checkpoint with current PyTorch/MONAI releases."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from scipy.ndimage import binary_closing, zoom
from torch import nn


class DoubleUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (64, 128, 256, 512, 1024)
        strides = (2, 2, 2, 2)

        def down(inc: int, outc: int, stride: int) -> nn.Module:
            return Convolution(3, inc, outc, strides=stride, kernel_size=3,
                               act=Act.PRELU, norm=Norm.INSTANCE)

        def up(inc: int, outc: int, stride: int, top: bool) -> nn.Module:
            return Convolution(3, inc, outc, strides=stride, kernel_size=3,
                               act=Act.PRELU, norm=Norm.INSTANCE,
                               conv_only=top, is_transposed=True)

        def block(inc: int, outc: int, remaining: tuple[int, ...],
                  remaining_strides: tuple[int, ...], top: bool):
            current, stride = remaining[0], remaining_strides[0]
            if len(remaining) > 2:
                branch1, branch2 = block(current, current, remaining[1:],
                                         remaining_strides[1:], False)
                up_channels = current * 2
            else:
                bottom = down(current, remaining[1], 1)
                branch1 = branch2 = bottom
                up_channels = current + remaining[1]
            return (
                nn.Sequential(down(inc, current, stride), SkipConnection(branch1),
                              up(up_channels, outc, stride, top)),
                nn.Sequential(down(inc, current, stride), SkipConnection(branch2),
                              up(up_channels, outc, stride, top)),
            )

        self.model1, self.model2 = block(1, 1, channels, strides, True)
        self.activation = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.model1(value))


def bounds(mask: np.ndarray, value: int):
    points = np.argwhere(mask == value)
    if not len(points):
        return None
    lo = points.min(axis=0)
    hi = points.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def infer_crop(model: nn.Module, image: np.ndarray, region, spacing, device):
    crop = np.clip(image[region], -1024, 1000).astype(np.float32)
    std = float(crop.std())
    crop = (crop - float(crop.mean())) / (std if std > 1e-8 else 1.0)
    factors = tuple(float(s) / target for s, target in zip(spacing, (1.0, 1.0, 1.5)))
    resized = zoom(crop, factors, order=1)
    pads = tuple((0, (-size) % 16) for size in resized.shape)
    padded = np.pad(resized, pads)
    tensor = torch.from_numpy(padded[None, None]).to(device)
    with torch.inference_mode():
        probability = model(tensor).squeeze().cpu().numpy()
    probability = probability[tuple(slice(0, size) for size in resized.shape)]
    restored = zoom(probability,
                    tuple(a / b for a, b in zip(crop.shape, probability.shape)), order=1)
    return restored[tuple(slice(0, size) for size in crop.shape)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lung-mask", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    source = nib.load(args.input)
    image = np.asarray(source.dataobj)
    lungs = np.asarray(nib.load(args.lung_mask).dataobj).astype(np.uint8)
    if image.shape != lungs.shape:
        raise SystemExit(f"Input/lung-mask shape mismatch: {image.shape} != {lungs.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleUNet().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    probability = np.zeros(image.shape, dtype=np.float32)
    for label in (1, 2):
        region = bounds(lungs, label)
        if region is None:
            continue
        predicted = infer_crop(model, image, region, source.header.get_zooms()[:3], device)
        probability[region] = np.maximum(probability[region], predicted)
    result = probability >= args.threshold
    result[lungs == 0] = False
    structure = np.ones((3, 3, 1), dtype=bool)
    result = binary_closing(result, structure=structure).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, source.affine, source.header), args.output)


if __name__ == "__main__":
    main()
