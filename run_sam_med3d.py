"""Run the official SAM-Med3D Turbo checkpoint from 3-D point prompts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spacing", type=float, default=1.5)
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    source = Path(args.source).resolve()
    package = types.ModuleType("segment_anything")
    package.__path__ = [str(source / "segment_anything")]
    sys.modules["segment_anything"] = package
    builder = importlib.import_module("segment_anything.build_sam3D")

    original = nib.load(args.input)
    data = np.asanyarray(original.dataobj)
    if data.ndim != 3 or not np.any(np.isfinite(data)):
        raise SystemExit(f"SAM-Med3D requires a finite 3-D image, got {data.shape}")
    prompt = json.loads(Path(args.prompt).read_text())
    points = prompt.get("positive_points", [])
    labels = [1] * len(points)
    negative = prompt.get("negative_points", [])
    points += negative; labels += [0] * len(negative)
    if not points and prompt.get("box"):
        box = prompt["box"]
        if not isinstance(box, list) or len(box) != 2:
            raise SystemExit("SAM-Med3D box requires two [column,row,slice] corners")
        points = [[(float(box[0][axis]) + float(box[1][axis])) / 2 for axis in range(3)]]
        labels = [1]
    if not points or 1 not in labels:
        raise SystemExit("SAM-Med3D requires a positive 3-D point or box")
    points_xyz = np.asarray(points, dtype=np.float64)
    if np.any(points_xyz < 0) or np.any(points_xyz >= np.asarray(original.shape)):
        raise SystemExit(f"SAM-Med3D prompt outside image shape {original.shape}")

    # Match the released 1.5-mm canonical preprocessing, then choose a 128^3
    # prompt-centred field of view. This replaces its GT-centred CropOrPad,
    # which is unavailable in an interactive deployment.
    canonical = nib.as_closest_canonical(
        resample_to_output(original, voxel_sizes=(args.spacing,) * 3, order=1)
    )
    world = nib.affines.apply_affine(original.affine, points_xyz)
    canonical_points = nib.affines.apply_affine(np.linalg.inv(canonical.affine), world)
    positive_center = canonical_points[np.asarray(labels) == 1].mean(axis=0)
    start = np.rint(positive_center).astype(int) - 64
    source_start = np.maximum(start, 0)
    source_end = np.minimum(start + 128, np.asarray(canonical.shape))
    target_start = source_start - start
    target_end = target_start + source_end - source_start
    crop = np.zeros((128, 128, 128), dtype=np.float32)
    canonical_data = np.asanyarray(canonical.dataobj).astype(np.float32)
    crop[tuple(slice(a, b) for a, b in zip(target_start, target_end))] = canonical_data[
        tuple(slice(a, b) for a, b in zip(source_start, source_end))
    ]
    foreground = crop > 0
    if np.any(foreground):
        mean, std = float(crop[foreground].mean()), float(crop[foreground].std())
    else:
        mean, std = float(crop.mean()), float(crop.std())
    crop = (crop - mean) / max(std, 1e-8)
    crop_points = canonical_points - start
    inside = np.all((crop_points >= 0) & (crop_points < 128), axis=1)
    crop_points = crop_points[inside]
    crop_labels = np.asarray(labels, dtype=np.int64)[inside]

    model = builder.build_sam3D_vit_b_ori().cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    image_tensor = torch.from_numpy(crop)[None, None].cuda()
    point_tensor = torch.from_numpy(crop_points.astype(np.float32))[None].cuda()
    label_tensor = torch.from_numpy(crop_labels)[None].cuda()
    with torch.inference_mode():
        embedding = model.image_encoder(image_tensor)
        # The released Turbo inference path always conditions the first click
        # on an empty 32^3 low-resolution mask.  Passing ``None`` selects a
        # different prompt-encoder embedding and produces near-empty masks.
        previous_mask = torch.zeros((1, 1, 32, 32, 32), device="cuda")
        sparse, dense = model.prompt_encoder(
            points=[point_tensor, label_tensor], boxes=None, masks=previous_mask
        )
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        mask = functional.interpolate(
            logits, size=(128, 128, 128), mode="trilinear", align_corners=False
        )[0, 0].cpu().numpy() > 0
    if not np.any(mask):
        raise SystemExit("SAM-Med3D produced an empty mask")

    canonical_mask = np.zeros(canonical.shape, dtype=np.uint8)
    canonical_mask[tuple(slice(a, b) for a, b in zip(source_start, source_end))] = mask[
        tuple(slice(a, b) for a, b in zip(target_start, target_end))
    ]
    restored = resample_from_to(
        nib.Nifti1Image(canonical_mask, canonical.affine),
        (original.shape, original.affine), order=0,
    )
    result = np.uint8(np.asanyarray(restored.dataobj) > 0)
    if not np.any(result):
        raise SystemExit("SAM-Med3D mask became empty after source-grid restoration")
    header = original.header.copy(); header.set_data_dtype(np.uint8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, original.affine, header), output)
    print(json.dumps({"voxels": int(result.sum()), "prompts": len(crop_points),
                      "checkpoint_parameters": sum(value.numel() for value in state.values())}))


if __name__ == "__main__":
    main()
