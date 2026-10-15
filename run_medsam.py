"""Run the official MedSAM ViT-B checkpoint on explicitly box-prompted slices."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as functional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    prompt = json.loads(Path(args.prompt).read_text())
    box = prompt.get("box")
    if not isinstance(box, list) or len(box) != 2:
        raise SystemExit("MedSAM requires a box with two [column,row,slice] corners")
    if int(box[0][2]) != int(box[1][2]):
        raise SystemExit("MedSAM is 2-D; both box corners must be on the same slice")

    source = Path(args.source).resolve()
    package = types.ModuleType("segment_anything")
    package.__path__ = [str(source / "segment_anything")]
    sys.modules["segment_anything"] = package
    registry = importlib.import_module("segment_anything.build_sam").sam_model_registry
    model = registry["vit_b"](checkpoint=args.checkpoint).cuda().eval()

    image = sitk.ReadImage(args.input)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    z = int(box[0][2])
    if z < 0 or z >= volume.shape[0]:
        raise SystemExit(f"MedSAM slice {z} outside volume with {volume.shape[0]} slices")
    x0, x1 = sorted((float(box[0][0]), float(box[1][0])))
    y0, y1 = sorted((float(box[0][1]), float(box[1][1])))
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("MedSAM box must have non-zero width and height")

    frame = volume[z]
    finite = frame[np.isfinite(frame)]
    low, high = np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    frame = np.clip((frame - low) / (high - low), 0, 1)
    rgb = np.repeat(frame[None, None], 3, axis=1)
    image_tensor = torch.from_numpy(rgb).float().cuda()
    image_tensor = functional.interpolate(
        image_tensor, size=(1024, 1024), mode="bicubic", align_corners=False
    )
    height, width = frame.shape
    box_1024 = torch.tensor(
        [[[x0 / width * 1024, y0 / height * 1024,
           x1 / width * 1024, y1 / height * 1024]]],
        dtype=torch.float32, device="cuda",
    )
    with torch.inference_mode():
        image_embedding = model.image_encoder(image_tensor)
        sparse, dense = model.prompt_encoder(points=None, boxes=box_1024, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        probability = torch.sigmoid(functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        ))[0, 0].cpu().numpy()
    mask = probability > 0.5
    if not np.any(mask):
        raise SystemExit("MedSAM produced an empty mask")
    result = np.zeros(volume.shape, dtype=np.uint8)
    result[z] = mask
    output = sitk.GetImageFromArray(result)
    output.CopyInformation(image)
    sitk.WriteImage(output, args.output)
    print(json.dumps({"slice": z, "voxels": int(mask.sum()),
                      "checkpoint_parameters": sum(v.numel() for v in model.state_dict().values())}))


if __name__ == "__main__":
    main()
