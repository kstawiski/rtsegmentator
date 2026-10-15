"""Run the official SegVol checkpoint with text, point, or box prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def _point(value: object, shape: tuple[int, int, int]) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise SystemExit("SegVol points must be [column,row,slice]")
    result = [int(round(float(item))) for item in value]
    if any(result[i] < 0 or result[i] >= shape[i] for i in range(3)):
        raise SystemExit(f"SegVol point {result} is outside image shape {shape}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--monai-site", required=True)
    parser.add_argument("--transformers-site", required=True)
    parser.add_argument("--no-zoom", action="store_true")
    args = parser.parse_args()

    for path in (args.input, args.model_dir, args.monai_site, args.transformers_site):
        if not Path(path).exists():
            raise SystemExit(f"required SegVol path does not exist: {path}")
    # The released checkpoint is architecture-exact with MONAI 0.9. Put that
    # overlay before the modern runner environment and keep all model loading offline.
    sys.path[:0] = [str(Path(args.monai_site).resolve()),
                    str(Path(args.transformers_site).resolve())]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    source = nib.load(args.input)
    raw = np.asarray(source.dataobj, dtype=np.float32)
    if raw.ndim != 3:
        raise SystemExit(f"SegVol requires one 3-D CT image, got {raw.shape}")
    prompt = json.loads(Path(args.prompt).read_text())
    text_prompt = prompt.get("text")
    if text_prompt is not None:
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise SystemExit("SegVol prompt.text must be a non-empty string")
        text_prompt = " ".join(text_prompt.split())
        if len(text_prompt) > 256 or any(ord(char) < 32 for char in text_prompt):
            raise SystemExit("SegVol prompt.text must be at most 256 printable characters")
    positive = [_point(item, raw.shape) for item in prompt.get("positive_points", [])]
    negative = [_point(item, raw.shape) for item in prompt.get("negative_points", [])]
    box_raw = prompt.get("box")
    box = None
    if box_raw is not None:
        if not isinstance(box_raw, list) or len(box_raw) != 2:
            raise SystemExit("SegVol box must contain two [column,row,slice] corners")
        box = [_point(item, raw.shape) for item in box_raw]
    if not text_prompt and not positive and box is None:
        raise SystemExit("SegVol requires prompt.text, a positive point, or a box")
    if box is not None and (positive or negative):
        raise SystemExit("SegVol cannot combine box and point prompts")

    model_dir = str(Path(args.model_dir).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True, test_mode=True
    )
    model.model.text_encoder.tokenizer = tokenizer
    model.eval().cuda()

    # Match the released processor exactly: foreground normalization, Z/Y/X
    # transpose, nonzero crop, then nearest-neighbour zoom to 32x256x256.
    image = np.expand_dims(raw, 0)
    image = model.processor.ForegroundNorm(image).astype(np.float32, copy=False)
    dummy = np.zeros_like(image, dtype=np.uint8)
    item = model.processor.zoom_transform(image, dummy)
    full = item["image"].unsqueeze(0).cuda()
    zoomed = item["zoom_out_image"].unsqueeze(0).cuda()
    start = np.asarray(item["foreground_start_coord"], dtype=np.float64)
    end = np.asarray(item["foreground_end_coord"], dtype=np.float64)
    crop_shape = end - start
    zoom_shape = np.asarray(zoomed.shape[-3:], dtype=np.float64)

    def to_zoom_xyz(point_xyz: list[int]) -> list[float]:
        transformed = np.asarray([point_xyz[2], point_xyz[1], point_xyz[0]], dtype=np.float64)
        scaled = (transformed - start) * zoom_shape / crop_shape
        scaled = np.clip(scaled, 0, zoom_shape - 1)
        return scaled.tolist()

    point_group = None
    if positive or negative:
        coords = torch.tensor([to_zoom_xyz(p) for p in positive + negative],
                              dtype=torch.float32, device="cuda").unsqueeze(0)
        labels = torch.tensor([1] * len(positive) + [0] * len(negative),
                              dtype=torch.float32, device="cuda").unsqueeze(0)
        point_map = model.__class__.__module__  # retain remote module before lookup
        remote = sys.modules[point_map]
        binary = remote.build_binary_points(coords[0].cpu(), labels[0].cpu(),
                                            tuple(int(x) for x in zoom_shape))
        point_group = [(coords, labels), binary.unsqueeze(0).unsqueeze(0)]

    box_group = None
    if box is not None:
        corners = np.asarray([to_zoom_xyz(p) for p in box])
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        box_tensor = torch.tensor(np.r_[lower, upper], dtype=torch.float32,
                                  device="cuda").unsqueeze(0)
        remote = sys.modules[model.__class__.__module__]
        binary = remote.build_binary_cube(box_tensor.cpu(),
                                          binary_cube_shape=tuple(int(x) for x in zoom_shape))
        box_group = [box_tensor, binary.unsqueeze(0).unsqueeze(0)]

    with torch.inference_mode():
        logits = model.forward_test(
            image=full, zoomed_image=zoomed,
            text_prompt=[text_prompt] if text_prompt else None,
            bbox_prompt_group=box_group, point_prompt_group=point_group,
            use_zoom=not args.no_zoom,
        )[0, 0]
    # Reverse crop and Z/Y/X transpose while retaining the exact source affine.
    cropped = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
    restored_zyx = np.zeros(raw.shape[::-1], dtype=np.uint8)
    restored_zyx[tuple(slice(int(a), int(b)) for a, b in zip(start, end))] = cropped
    mask = restored_zyx.transpose(2, 1, 0)
    if not np.any(mask):
        raise SystemExit("SegVol returned an empty mask")
    header = source.header.copy(); header.set_data_dtype(np.uint8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, source.affine, header), output)
    print(json.dumps({"voxels": int(mask.sum()), "text": text_prompt,
                      "positive_points": len(positive), "negative_points": len(negative),
                      "box": box is not None, "zoom": not args.no_zoom}))


if __name__ == "__main__":
    main()
