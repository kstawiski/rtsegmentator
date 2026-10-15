"""Propagate a 2-D CT lesion prompt through a volume with MedSAM2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--implementation", choices=("wanglab", "imprintlab"),
                        default="wanglab")
    parser.add_argument("--window-low", type=float, default=-200.0)
    parser.add_argument("--window-high", type=float, default=300.0)
    args = parser.parse_args()

    sys.path[:0] = [args.site_packages, args.source]
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from scipy import ndimage  # noqa: PLC0415
    if args.implementation == "wanglab":
        from sam2.build_sam import build_sam2_video_predictor_npz as build_predictor  # noqa: PLC0415
    else:
        from sam2_train.build_sam import build_sam2_video_predictor as build_predictor  # noqa: PLC0415

    image = nib.load(args.input)
    xyz = np.asanyarray(image.dataobj).astype(np.float32)
    if xyz.ndim != 3 or not np.any(np.isfinite(xyz)):
        raise SystemExit(f"MedSAM2 requires a finite 3-D CT, got {xyz.shape}")
    prompt = json.loads(Path(args.prompt).read_text())
    box = prompt.get("box")
    positive = prompt.get("positive_points", [])
    if box:
        if not isinstance(box, list) or len(box) != 2:
            raise SystemExit("MedSAM2 box requires two [column,row,slice] corners")
        corners = [[int(round(float(v))) for v in point] for point in box]
        if corners[0][2] != corners[1][2]:
            raise SystemExit("MedSAM2 requires a 2-D box on one axial slice")
        x0, x1 = sorted((corners[0][0], corners[1][0]))
        y0, y1 = sorted((corners[0][1], corners[1][1]))
        z = corners[0][2]
        box_xyxy = np.asarray([x0, y0, x1, y1], dtype=np.float32)
        point_coords = point_labels = None
    elif positive:
        points = [[int(round(float(v))) for v in point] for point in positive]
        slices = {point[2] for point in points}
        if len(slices) != 1:
            raise SystemExit("MedSAM2 positive points must lie on one axial seed slice")
        z = slices.pop()
        point_coords = np.asarray([[p[0], p[1]] for p in points], dtype=np.float32)
        point_labels = np.ones(len(points), dtype=np.int32)
        negative = [[int(round(float(v))) for v in point]
                    for point in prompt.get("negative_points", [])]
        if any(point[2] != z for point in negative):
            raise SystemExit("MedSAM2 negative points must lie on the seed slice")
        if negative:
            point_coords = np.concatenate(
                [point_coords, np.asarray([[p[0], p[1]] for p in negative], dtype=np.float32)]
            )
            point_labels = np.concatenate([point_labels, np.zeros(len(negative), dtype=np.int32)])
        box_xyxy = None
    else:
        raise SystemExit("MedSAM2 requires a positive point or 2-D box")
    if z < 0 or z >= xyz.shape[2]:
        raise SystemExit(f"MedSAM2 seed slice {z} is outside image depth {xyz.shape[2]}")

    zyx = xyz.transpose(2, 1, 0)
    clipped = np.clip(zyx, args.window_low, args.window_high)
    scaled = np.uint8((clipped - args.window_low) /
                      (args.window_high - args.window_low) * 255.0)
    depth, height, width = scaled.shape
    target_size = 512 if args.implementation == "wanglab" else 1024
    frames = np.empty((depth, 3, target_size, target_size), dtype=np.float32)
    for index, frame in enumerate(scaled):
        resized = np.asarray(Image.fromarray(frame).resize(
            (target_size, target_size), Image.Resampling.BILINEAR
        ))
        frames[index] = np.repeat(resized[None], 3, axis=0) / 255.0
    means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    stds = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    tensor = torch.from_numpy((frames - means) / stds).cuda()

    if args.implementation == "wanglab":
        config = "//" + str(Path(args.source).resolve() /
                             "sam2/configs/sam2.1_hiera_t512.yaml")
    else:
        config = "sam2_hiera_t"
    predictor = build_predictor(
        config, str(Path(args.checkpoint).resolve()), device="cuda",
        apply_postprocessing=args.implementation == "wanglab",
    )
    output_zyx = np.zeros((depth, height, width), dtype=np.uint8)

    def seed(state):
        kwargs = {"inference_state": state, "frame_idx": z, "obj_id": 1}
        if box_xyxy is not None:
            scale = (np.asarray([target_size / width, target_size / height,
                                 target_size / width, target_size / height])
                     if args.implementation == "wanglab" else np.ones(4))
            scaled_box = box_xyxy * scale
            if args.implementation == "wanglab":
                predictor.add_new_points_or_box(**kwargs, box=scaled_box)
            else:
                predictor.add_new_bbox(**kwargs, bbox=scaled_box)
        else:
            scale = (np.asarray([target_size / width, target_size / height])
                     if args.implementation == "wanglab" else np.ones(2))
            scaled_points = point_coords * scale
            method = (predictor.add_new_points_or_box if args.implementation == "wanglab"
                      else predictor.add_new_points)
            method(**kwargs, points=scaled_points, labels=point_labels)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if args.implementation == "wanglab":
            state = predictor.init_state(tensor, height, width)
        else:
            state = predictor.val_init_state(tensor, height, width)
        seed(state)
        for frame_index, _, logits in predictor.propagate_in_video(state):
            output_zyx[frame_index, logits[0, 0].cpu().numpy() > 0] = 1
        predictor.reset_state(state)
        seed(state)
        for frame_index, _, logits in predictor.propagate_in_video(state, reverse=True):
            output_zyx[frame_index, logits[0, 0].cpu().numpy() > 0] = 1
        predictor.reset_state(state)
    if np.any(output_zyx):
        components, count = ndimage.label(output_zyx)
        if count:
            sizes = np.bincount(components.ravel()); sizes[0] = 0
            output_zyx = np.uint8(components == sizes.argmax())
    if not np.any(output_zyx):
        raise SystemExit("MedSAM2 produced an empty mask")

    result = output_zyx.transpose(2, 1, 0)
    header = image.header.copy(); header.set_data_dtype(np.uint8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, image.affine, header), output)
    print(json.dumps({"voxels": int(result.sum()), "seed_slice": z,
                      "prompt": "box" if box_xyxy is not None else "points"}))


if __name__ == "__main__":
    main()
