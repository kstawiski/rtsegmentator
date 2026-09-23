#!/usr/bin/env python3
"""Run a standardized single-channel nnU-Net v2 model folder."""

import argparse
import json
import nibabel as nib
import numpy as np
import subprocess
import sys
from pathlib import Path


def dataset_json(model_folder: Path) -> Path:
    for candidate in (model_folder / "dataset.json", *(p / "dataset.json" for p in model_folder.parents)):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No dataset.json found for {model_folder}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--keep-label", type=int)
    parser.add_argument("--keep-name")
    parser.add_argument("--lowres-model-folder")
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    folds = sorted(p.name.removeprefix("fold_") for p in model_folder.glob("fold_*") if p.is_dir())
    if not folds:
        raise SystemExit(f"No fold directories found in {model_folder}")
    def prediction_command(folder: Path, destination: str) -> list[str]:
        folder_folds = sorted(p.name.removeprefix("fold_") for p in folder.glob("fold_*") if p.is_dir())
        if not folder_folds:
            raise SystemExit(f"No fold directories found in {folder}")
        return [
        str(Path(sys.executable).parent / "nnUNetv2_predict_from_modelfolder"),
        "-i", args.input_dir, "-o", destination, "-m", str(folder),
        "-f", *folder_folds, "-chk", args.checkpoint, "-npp", "1", "-nps", "1",
        "-device", "cuda", "--disable_progress_bar",
        ]
    command = prediction_command(model_folder, args.output_dir)
    if args.lowres_model_folder:
        lowres = Path(args.lowres_model_folder)
        lowres_output = str(Path(args.output_dir).parent / "lowres_prediction")
        subprocess.run(prediction_command(lowres, lowres_output), check=True)
        command.extend(["-prev_stage_predictions", lowres_output])
    subprocess.run(command, check=True)

    if args.keep_label is not None:
        outputs = sorted(Path(args.output_dir).glob("*.nii.gz"))
        if len(outputs) != 1:
            raise SystemExit(f"Expected one nnU-Net output, found {len(outputs)}")
        image = nib.load(outputs[0])
        mask = (np.asanyarray(image.dataobj) == args.keep_label).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, image.affine, image.header), outputs[0])
        Path(args.labels).write_text(json.dumps({"1": args.keep_name or f"label_{args.keep_label}"}, indent=2) + "\n")
        return

    raw = json.loads(dataset_json(model_folder).read_text()).get("labels", {})
    labels = {}
    for name, value in raw.items():
        if str(name).lower() == "background":
            continue
        if isinstance(value, list):
            raise SystemExit(f"Overlapping region label {name!r} requires a dedicated exporter")
        labels[str(int(value))] = str(name)
    if not labels:
        raise SystemExit("dataset.json contains no scalar foreground labels")
    Path(args.labels).write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
