"""Run the official NV-Segment-CT/VISTA3D checkpoint from point prompts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def point(value, shape):
    if not isinstance(value, list) or len(value) != 3:
        raise SystemExit("VISTA3D points must be [column,row,slice]")
    result = [int(round(float(item))) for item in value]
    if any(result[axis] < 0 or result[axis] >= shape[axis] for axis in range(3)):
        raise SystemExit(f"VISTA3D point {result} is outside image shape {shape}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()

    image = nib.load(args.input)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise SystemExit(f"VISTA3D requires one 3-D CT image, got {data.shape}")
    prompt = json.loads(Path(args.prompt).read_text())
    positive = [point(item, data.shape) for item in prompt.get("positive_points", [])]
    negative = [point(item, data.shape) for item in prompt.get("negative_points", [])]
    if not positive:
        raise SystemExit("VISTA3D requires at least one positive point")

    source = Path(args.source)
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    (work / "models").mkdir(parents=True)
    shutil.copytree(source / "configs", work / "configs")
    shutil.copytree(source / "scripts", work / "scripts")
    (work / "models/model.pt").symlink_to(Path(args.checkpoint).resolve())

    # MONAI >=1.6 returns a Tensor from transform_points; the released bundle
    # calls torch.from_numpy unconditionally. Keep the upstream source immutable
    # and apply the documented one-line compatibility repair to the job copy.
    evaluator = work / "scripts/evaluator.py"
    text = evaluator.read_text()
    old = "points = torch.from_numpy(points).to(inputs.device)"
    if old not in text:
        raise SystemExit("VISTA3D evaluator compatibility anchor was not found")
    evaluator.write_text(text.replace(old, "points = torch.as_tensor(points).to(inputs.device)", 1))

    config_path = work / "configs/inference.json"
    config = json.loads(config_path.read_text())
    config["initialize"] = config["initialize"][1:]  # checkpoint is already local; no network touch
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    input_dict = {"image": str(Path(args.input).resolve()),
                  "points": positive + negative,
                  "point_labels": [1] * len(positive) + [0] * len(negative)}
    output_dir = work / "output"
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    subprocess.run([args.runtime_python, "-m", "monai.bundle", "run",
                    "--config_file", "configs/inference.json",
                    "--input_dict", json.dumps(input_dict),
                    "--output_dir", str(output_dir)], cwd=work, env=env, check=True)
    candidates = list(output_dir.rglob("*_trans.nii.gz"))
    if len(candidates) != 1:
        raise SystemExit(f"VISTA3D created {len(candidates)} predictions; expected one")
    pred_img = nib.load(candidates[0])
    prediction = np.asanyarray(pred_img.dataobj)
    mask = ((prediction != 0) & (prediction != 255)).astype(np.uint8)
    if mask.shape != data.shape or not np.any(mask):
        raise SystemExit(f"VISTA3D returned invalid/empty mask {mask.shape}")
    header = image.header.copy(); header.set_data_dtype(np.uint8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, image.affine, header), output)
    print({"voxels": int(mask.sum()), "positive_points": len(positive),
           "negative_points": len(negative)})


if __name__ == "__main__":
    main()
