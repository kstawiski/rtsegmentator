"""Run the released Medical-SAM3 3-D checkpoint on its 8-slice protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--site-packages", required=True)
    args = parser.parse_args()

    sys.path[:0] = [args.site_packages, args.source]
    from sam3.model_builder import build_sam3_video_model  # noqa: PLC0415

    prompt = json.loads(Path(args.prompt).read_text())
    text = prompt.get("text")
    box = prompt.get("box")
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("Medical-SAM3 requires a non-empty medical concept text prompt")
    if not isinstance(box, list) or len(box) != 2 or box[0][2] != box[1][2]:
        raise SystemExit("Medical-SAM3 requires a 2-D box on one axial seed slice")

    image = sitk.ReadImage(args.input)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    seed = int(box[0][2])
    if seed < 0 or seed >= volume.shape[0]:
        raise SystemExit(f"Medical-SAM3 seed slice {seed} outside {volume.shape[0]} slices")
    # The released 3-D training/evaluation contract is an eight-frame window.
    start = max(0, min(seed - 4, volume.shape[0] - min(8, volume.shape[0])))
    stop = min(volume.shape[0], start + 8)
    local_seed = seed - start
    frames = []
    for frame in volume[start:stop]:
        finite = frame[np.isfinite(frame)]
        low, high = np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 1.0)
        if high <= low:
            high = low + 1.0
        gray = np.clip((frame - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(np.repeat(gray[..., None], 3, axis=2)))

    model = build_sam3_video_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        bpe_path=str(Path(args.source) / "assets/bpe_simple_vocab_16e6.txt.gz"),
        strict_state_dict_loading=False,
        training_mode=False,
        device="cuda",
        compile=False,
        apply_temporal_disambiguation=True,
    ).eval()
    model.hotstart_delay = 0
    model.hotstart_unmatch_thresh = 0
    model.hotstart_dup_thresh = 0

    height, width = volume.shape[1:]
    x0, x1 = sorted((float(box[0][0]), float(box[1][0])))
    y0, y1 = sorted((float(box[0][1]), float(box[1][1])))
    box_xywh = np.asarray([[x0 / width, y0 / height,
                            (x1 - x0) / width, (y1 - y0) / height]], dtype=np.float32)
    state = model.init_state(resource_path=frames, offload_video_to_cpu=True)
    with torch.inference_mode():
        model.add_prompt(
            inference_state=state,
            frame_idx=local_seed,
            text_str=text.strip(),
            boxes_xywh=box_xywh,
            box_labels=torch.ones(1, dtype=torch.long),
        )
        masks: dict[int, np.ndarray] = {}
        for reverse, frame_idx in ((False, 0), (True, len(frames) - 1)):
            for index, output in model.propagate_in_video(
                state, start_frame_idx=frame_idx,
                max_frame_num_to_track=len(frames), reverse=reverse
            ):
                if output is None or output.get("out_binary_masks") is None:
                    continue
                candidate = np.any(np.asarray(output["out_binary_masks"]), axis=0)
                if candidate.shape != (height, width):
                    import cv2  # noqa: PLC0415
                    candidate = cv2.resize(candidate.astype(np.uint8), (width, height),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                if index not in masks or candidate.sum() > masks[index].sum():
                    masks[index] = candidate
    result = np.zeros(volume.shape, dtype=np.uint8)
    for local, mask in masks.items():
        result[start + int(local)] = mask
    if not np.any(result):
        raise SystemExit("Medical-SAM3 produced an empty mask")
    out = sitk.GetImageFromArray(result)
    out.CopyInformation(image)
    sitk.WriteImage(out, args.output)
    print(json.dumps({"window": [start, stop], "seed_slice": seed,
                      "predicted_slices": int(np.any(result, axis=(1, 2)).sum()),
                      "voxels": int(result.sum())}))


if __name__ == "__main__":
    main()
